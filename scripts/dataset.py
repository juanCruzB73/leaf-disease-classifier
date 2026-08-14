"""Shared dataset/transform helpers for train.py and evaluate.py."""
import json
from pathlib import Path

from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_label_map(models_dir: Path):
    with open(models_dir / "label_map.json", encoding="utf-8") as f:
        return json.load(f)


def train_transforms(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_dataset(processed_dir: Path, split: str, label_map: dict, image_size=224):
    """ImageFolder, but forces the class->index mapping to match label_map.json
    instead of whatever alphabetic order ImageFolder would derive on its own."""
    tfm = train_transforms(image_size) if split == "train" else eval_transforms(image_size)
    ds = datasets.ImageFolder(processed_dir / split, transform=tfm)
    ds.class_to_idx = label_map
    ds.samples = [(path, label_map[Path(path).parent.name]) for path, _ in ds.samples]
    ds.targets = [s[1] for s in ds.samples]
    return ds
