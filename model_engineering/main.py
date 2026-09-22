#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import json
import psutil
import numpy as np
import random
import cv2
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import mlflow
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

import hydra

from application.preprocessing.PreProcessing import ImageProcessing, OpenCVPreprocessing, Letterbox
from application.cmd.Training import Training
import subprocess
from application.dataset.CustomDataset import CustomDataset
from domain.SetupModel import SetupModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def setup_ddp():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[DDP] Rank {local_rank} of {dist.get_world_size()} (backend={backend})")
        return local_rank, dist.get_world_size()
    return 0, 1


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def log_preprocessed_samples(dataset, writer, save_path, num_samples=8):
    import torchvision.utils as vutils
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    images = []
    for idx in indices:
        img, _ = dataset[idx]
        images.append(img.cpu())

    grid = vutils.make_grid(images, nrow=4, normalize=True, scale_each=True)
    writer.add_image("preprocessed/samples", grid, 0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        if i >= 8:
            break
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(dataset.labels[idx])
    plt.tight_layout()
    plot_path = os.path.join(save_path, "preprocessed_samples.png")
    fig.savefig(plot_path)
    plt.close(fig)
    mlflow.log_artifact(plot_path)


def save_model_checkpoint(model, optimizer, epoch, save_path):
    raw_model = model.module if hasattr(model, 'module') else model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_path, 'model.pt'))
    print(f"Modelo salvo em: {save_path}")
    mlflow.log_artifact(os.path.join(save_path, 'model.pt'))


def save_training_metrics(metrics, save_path, results_dir="results"):
    path = os.path.join(save_path, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    metrics_copy = os.path.join(results_dir, "metrics",
                                f"metrics_{metrics.get('model', 'unknown')}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(metrics_copy), exist_ok=True)
    with open(metrics_copy, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {path}")
    mlflow.log_artifact(path)


def monitor_resources():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)
    cpu_usage = process.cpu_percent(interval=1)
    print(f"Memory: {mem_usage:.2f} GB, CPU: {cpu_usage}%")
    return mem_usage, cpu_usage


def _prepare_cam_sample(test_loader, input_size=512, opencv=True, grayscale=True, crop="center"):
    subset = test_loader.dataset
    idx = subset.indices[0]
    custom_dataset = subset.dataset
    img_path = custom_dataset.data.iloc[idx]['img_name']

    cv_img = cv2.imread(img_path)
    if cv_img is None:
        return None
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    if crop == "letterbox":
        spatial = transforms.Compose([Letterbox(input_size)])
    else:
        spatial = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
        ])
    cropped = spatial(pil_img)

    processed = OpenCVPreprocessing(grayscale=grayscale)(cropped) if opencv else cropped
    raw_np = np.array(processed).astype(np.float32) / 255.0

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = to_tensor(processed).unsqueeze(0)

    return raw_np, input_tensor


def run_training(model, train_loader, test_loader, epochs, device, optimizer,
                 loss_fn, scheduler, model_name=None, cam_sample=None,
                 rank=0, world_size=1, parallelism="none",
                 patience=7, min_delta=0.001, results_dir="..",
                 input_size=512):
    cam_interval = max(1, epochs // 5)
    timestamp = int(time.time())
    save_path = os.path.join(results_dir, "tensorboard",
                             f"ml-model-{model_name}-{timestamp}")

    writer = SummaryWriter(save_path) if rank == 0 else None
    start_time = time.time()

    if writer:
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        try:
            graph_model = model.module if hasattr(model, 'module') else model
            writer.add_graph(graph_model, dummy_input)
        except Exception:
            pass
        writer.add_text("Config/parallelism", parallelism)
        writer.add_text("Config/world_size", str(world_size))
        writer.add_text("Config/model", model_name)

    training = Training(train_loader, test_loader, model, writer, None,
                        model_name=model_name, cam_sample=cam_sample,
                        cam_interval=cam_interval, rank=rank, world_size=world_size,
                        patience=patience, min_delta=min_delta)

    if rank == 0:
        train_dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
        log_preprocessed_samples(train_dataset, writer, save_path)

    for epoch in range(epochs):
        if rank == 0:
            if epoch > 0 and len(training.epoch_times) > 0:
                avg_epoch = sum(training.epoch_times) / len(training.epoch_times)
                remaining = (epochs - epoch) * avg_epoch
                print(f"\nEpoch {epoch + 1}/{epochs}  "
                      f"| media epoca: {avg_epoch:.1f}s  "
                      f"| ETA: {remaining:.0f}s ({remaining/60:.1f}min)")
            else:
                print(f"\nEpoch {epoch + 1}/{epochs}\n{'-' * 30}")
        training.train(loss_fn, optimizer, device, epoch)
        test_loss = training.test(loss_fn, epoch, device)
        training.visualize_gradcam(epoch, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        if training.should_stop:
            print(f"Early stopping ativado na época {epoch + 1}")
            break

    total_time = time.time() - start_time

    if rank == 0:
        save_model_checkpoint(model, optimizer, epoch, save_path)

        total_images = len(train_loader.dataset) * (epoch + 1)
        avg_img_per_sec = total_images / total_time if total_time > 0 else 0
        avg_epoch_time = (sum(training.epoch_times) / len(training.epoch_times)
                          if training.epoch_times else 0)

        metrics = {
            "model": model_name,
            "parallelism": parallelism,
            "epochs": epoch + 1,
            "total_time_seconds": round(total_time, 2),
            "avg_epoch_time_seconds": round(avg_epoch_time, 2),
            "total_images_processed": total_images,
            "avg_throughput_img_per_sec": round(avg_img_per_sec, 2),
            "gpu_count": world_size if world_size > 1 else (torch.cuda.device_count() if torch.cuda.is_available() else 0),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "early_stopped": bool(training.should_stop),
        }

        if training.throughputs:
            metrics["avg_epoch_throughput"] = round(np.mean(training.throughputs), 2)
            metrics["peak_throughput"] = round(max(training.throughputs), 2)

        mlflow.log_params({
            "model": model_name,
            "epochs": epoch + 1,
            "early_stopped": training.should_stop,
        })
        mlflow.log_metrics({
            "total_time_seconds": total_time,
            "avg_throughput": avg_img_per_sec,
        })

        save_training_metrics(metrics, save_path, results_dir=results_dir)
        writer.add_hparams({"model": model_name, "parallelism": parallelism},
                           {"final_test_loss": test_loss, "total_time": total_time})

        print(f"\n{'='*40}")
        print(f"{'='*40}")
        print(f"Treinamento concluido em {total_time:.1f}s")
        print(f"Throughput medio: {avg_img_per_sec:.1f} imagens/s")
        print(f"Metricas finais:")
        print(f"  Accuracy:    {training.accuracy:.2f}%")
        print(f"  AUROC:       {training.auroc:.4f}")
        print(f"  MCC:         {training.mcc:.4f}")
        print(f"  F1 Score:    {training.f1:.4f}")
        print(f"  Precision:   {training.precision:.4f}")
        print(f"  Recall:      {training.recall:.4f}")
        print(f"  Specificity: {training.specificity:.4f}")
        print(f"  Kappa:       {training.kappa:.4f}")
        print(f"Resultados salvos em: {save_path}")
        print(f"MLflow:  mlflow ui --port 5000 --backend-store-uri sqlite:///{results_dir}/mlflow/mlflow.db")
        print(f"TB:      tensorboard --logdir {results_dir}/tensorboard")
        print(f"{'='*40}")

        writer.flush()
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()

    return metrics, training.should_stop


def train_fold(cfg, fold, local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    pre = cfg.preprocessing
    dataset = ImageProcessing(
        preprocessed_dir=cfg.data.preprocessed_dir or None,
        input_size=pre.input_size,
        crop=pre.crop,
        random_erasing=pre.random_erasing,
        opencv=pre.opencv,
        grayscale=pre.grayscale,
    )
    train_loader, test_loader = dataset.pre_processing(
        fold=fold, batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        rank=local_rank, world_size=world_size
    )

    cam_sample = _prepare_cam_sample(
        test_loader, input_size=pre.input_size,
        opencv=pre.opencv, grayscale=pre.grayscale,
        crop=pre.crop,
    ) if is_main_process() else None

    setup = SetupModel(cfg.model, scheduler=cfg.training.scheduler)
    model, loss_fn, optimizer, scheduler = setup.setup_model(
        device, lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
        optimizer_name=cfg.training.optimizer,
        scheduler_name=cfg.training.scheduler,
        epochs=cfg.training.epochs,
        unfreeze_blocks=cfg.training.unfreeze_blocks,
        pos_weight=cfg.training.pos_weight,
        frontend=cfg.frontend,
        frontend_input_size=pre.input_size,
    )

    parallelism = "DDP" if world_size > 1 else ("DataParallel" if cfg.dp else "none")
    if cfg.dp and world_size == 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        parallelism = "DataParallel"
    elif world_size > 1:
        if torch.cuda.is_available():
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            model = nn.parallel.DistributedDataParallel(model)
            parallelism = "DDP (CPU)"

    metrics, early_stopped = run_training(
        model, train_loader, test_loader, cfg.training.epochs, device, optimizer,
        loss_fn, scheduler, model_name=cfg.model, cam_sample=cam_sample,
        rank=local_rank, world_size=world_size, parallelism=parallelism,
        patience=cfg.training.patience, min_delta=cfg.training.min_delta,
        results_dir=cfg.results_dir, input_size=cfg.preprocessing.input_size,
    )
    metrics["fold"] = fold
    return metrics


CFG_DEFAULTS = {
    "preprocessing": {
        "opencv": True,
        "grayscale": True,
        "input_size": 512,
        "crop": "center",
        "random_erasing": False,
    },
    "frontend": "none",
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # exps antigos podem nao ter as secoes novas: preenche defaults
    cfg = OmegaConf.merge(OmegaConf.create(CFG_DEFAULTS), cfg)
    set_seed(cfg.seed)

    results_dir = cfg.results_dir
    os.makedirs(os.path.join(results_dir, "mlflow"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(results_dir, 'mlflow', 'mlflow.db')}")

    mlflow.set_experiment(f"cnn-skin-{cfg.model}")

    dataset_full = CustomDataset(cfg.data.csv)

    with mlflow.start_run(run_name=f"fold-{cfg.data.fold}") as run:
        OmegaConf.save(cfg, "hydra_config.yaml")
        mlflow.log_artifact("hydra_config.yaml")
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        dataset_full.log_to_mlflow()
        local_rank, world_size = setup_ddp()

        metrics = train_fold(cfg, cfg.data.fold, local_rank, world_size)

        if is_main_process():
            results_dir_abs = os.path.abspath(results_dir)
            print(f"\n{'='*40}")
            print(f"Fold {cfg.data.fold} concluido")
            print(f"Resultados salvos em: {results_dir_abs}")
            print(f"MLflow:  mlflow ui --port 5000 --backend-store-uri sqlite:///{results_dir}/mlflow/mlflow.db")
            print(f"TB:      tensorboard --logdir {results_dir}/tensorboard")
            print(f"{'='*40}\n")

            try:
                subprocess.run(["git", "add", "-A"], cwd=results_dir_abs,
                               capture_output=True, timeout=30)
                result = subprocess.run(
                    ["git", "commit", "-m", f"feat: {cfg.model} fold {cfg.data.fold}"],
                    cwd=results_dir_abs, capture_output=True, timeout=30, text=True,
                )
                if result.returncode == 0:
                    push = subprocess.run(
                        ["git", "push"], cwd=results_dir_abs,
                        capture_output=True, timeout=60, text=True,
                    )
                    if push.returncode != 0:
                        print(f"[AVISO] git push falhou: {push.stderr.strip()}")
                        print(f"  Para fazer push manual:")
                        print(f"  cd {results_dir_abs} && git push")
                elif "nothing to commit" not in result.stdout:
                    print(f"[AVISO] git commit falhou: {result.stderr.strip()}")
            except Exception as e:
                print(f"[AVISO] auto-commit/push: {e}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
