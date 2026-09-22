#!/usr/bin/env python
# coding: utf-8
"""
Overfit check - sanidade do pipeline.

Se o modelo NAO consegue memorizar ~20 imagens, ha bug no pipeline
(transforms, labels, loss, arquitetura). Use ANTES de treinar por horas.

Uso:
    # com OpenCV (comportamento atual - controle):
    python overfit_check.py model=vgg16 training.unfreeze_blocks=2

    # sem OpenCV (hipotese principal - RGB original):
    python overfit_check.py model=vgg16 training.unfreeze_blocks=2 preprocessing.opencv=false
"""

import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from application.preprocessing.PreProcessing import ImageProcessing
from application.dataset.CustomDataset import CustomDataset
from domain.SetupModel import SetupModel

PER_CLASS = 10
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3


def build_mini_csv(csv_file: str, per_class: int = PER_CLASS) -> str:
    full = pd.read_csv(csv_file)
    parts = [
        full[full["labels"] == c].sample(n=min(per_class, full["labels"].eq(c).sum()),
                                         random_state=42)
        for c in full["labels"].unique()
    ]
    mini = pd.concat(parts, ignore_index=True)
    mini_csv = os.path.join(os.getcwd(), "mini_dataset.csv")
    mini.to_csv(mini_csv, index=False)
    return mini_csv


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


def main(cfg: DictConfig):
    cfg = OmegaConf.merge(OmegaConf.create(CFG_DEFAULTS), cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    mini_csv = build_mini_csv(cfg.data.csv)
    print(f"Mini-dataset: {os.path.abspath(mini_csv)}")

    proc = ImageProcessing(
        preprocessed_dir=None,
        input_size=cfg.preprocessing.input_size,
        crop=cfg.preprocessing.crop,
        random_erasing=False,
        opencv=cfg.preprocessing.opencv,
        grayscale=cfg.preprocessing.grayscale,
    )
    dataset = CustomDataset(csv_file=mini_csv, transform=proc.train_transforms)
    dist = dataset.class_distribution
    print(f"Amostras: {len(dataset)} | Distribicao: {dist}")
    print(f"Pipeline: opencv={cfg.preprocessing.opencv} "
          f"grayscale={cfg.preprocessing.grayscale} "
          f"input_size={cfg.preprocessing.input_size} "
          f"crop={cfg.preprocessing.crop} "
          f"frontend={cfg.frontend}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    setup = SetupModel(cfg.model, scheduler="plateau")
    model, loss_fn, optimizer, _ = setup.setup_model(
        device, lr=LR, optimizer_name="adam",
        unfreeze_blocks=cfg.training.unfreeze_blocks,
        pos_weight=0,
        frontend=cfg.frontend,
        frontend_input_size=cfg.preprocessing.input_size,
    )

    last_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze(1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(X)
            correct += int(((torch.sigmoid(pred) > 0.5).float() == y).sum())
            total += len(X)
        last_acc = correct / total
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | loss: {loss_sum / total:.4f} | train acc: {last_acc:.2%}")

    print("\nVeredito:")
    if last_acc >= 0.90:
        print("  OK - pipeline memoriza o mini-dataset. Causa do colapso esta fora daqui.")
    else:
        print("  FALHOU - modelo nao memoriza 20 imagens. Bug no pipeline "
              "(transforms/labels/loss/arquitetura). Nao treinar ainda.")


@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
