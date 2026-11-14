"""Shared utilities: class names, dataset, transforms, model, and metrics."""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import Dataset
from torchvision import models, transforms

# Label convention used everywhere in this project.
CLASSES = ["chihuahua", "muffin"]  # index 0 = chihuahua, index 1 = muffin
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageCSV(Dataset):
    """Reads a CSV with columns `path` (relative to data_root) and `label` (int)."""

    def __init__(self, df: pd.DataFrame, data_root: str, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(f"{self.root}/{self.paths[i]}").convert("RGB")
        return self.transform(img), self.labels[i]


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        ps.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    per_class = cm.diagonal() / cm.sum(axis=1)
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),  # rows = true class, cols = predicted class
    }
    for name, acc in zip(CLASSES, per_class):
        out[f"acc_{name}"] = float(acc)
    return out
