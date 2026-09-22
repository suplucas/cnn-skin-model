import os
import re
import hashlib
import torch
import mlflow
from PIL import Image, ImageOps
from pandas import read_csv
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


def _resolve_csv(csv_file):
    """Resolve o caminho do CSV.

    Hydra muda o cwd para o job dir durante a execucao; se o CSV nao
    estiver no cwd, tenta a raiz do projeto (model_engineering/).
    """
    if os.path.isabs(csv_file) or os.path.exists(csv_file):
        return csv_file
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    candidate = os.path.join(project_root, csv_file)
    if os.path.exists(candidate):
        return candidate
    return csv_file


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None, data_dir=None):
        csv_file = _resolve_csv(csv_file)
        print(f"[DEBUG] Lendo CSV de: {csv_file}")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Arquivo {csv_file} não encontrado.")

        self.data = read_csv(csv_file)
        if self.data is None or "labels" not in self.data.columns:
            raise ValueError("CSV inválido ou coluna 'labels' ausente.")

        self.data_dir = data_dir or os.getenv('DATA_DIR', '')
        if self.data_dir:
            self.data['img_name'] = self.data['img_name'].apply(
                lambda p: os.path.join(self.data_dir, p) if not os.path.isabs(p) else p
            )

        self.transform = transform
        self.target_transform = target_transform
        self.labels = [str(label) for label in self.data['labels']]
        self.class_to_idx = {"psoriasis": 0, "dermatite": 1}

        self.patient_ids = []
        for path in self.data['img_name']:
            match = re.search(r'\((\d{15,})\)', str(path))
            patient_id = match.group(1) if match else str(path)
            self.patient_ids.append(patient_id)

    @property
    def dataset_hash(self):
        caminhos = sorted(self.data['img_name'].tolist())
        raw = "\n".join(caminhos).encode()
        return hashlib.md5(raw).hexdigest()

    @property
    def class_distribution(self):
        return self.data['labels'].value_counts().to_dict()

    def log_to_mlflow(self):
        mlflow.log_param("dataset_hash", self.dataset_hash)
        mlflow.log_param("dataset_size", len(self.data))
        dist = self.class_distribution
        for cls, count in dist.items():
            mlflow.log_param(f"class_{cls}_count", count)
        mlflow.log_param("num_patients", len(set(self.patient_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[int(idx)]['img_name']
        image = Image.open(image_path)
        # aplica rotacao EXIF (fotos de celular: ~7% com orientacao 90/180 graus)
        image = ImageOps.exif_transpose(image)
        label = self.data.iloc[idx]['labels']
            
        if self.transform:
            image = self.transform(image)

        label_numeric = torch.tensor(self.class_to_idx[label], dtype=torch.float32)

        
        return image, label_numeric
    
    transforms = transforms.Compose([
        transforms.RandomRotation(50,fill=1),
        transforms.RandomResizedCrop((224,224)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),  # Converte para tensor
    ])
