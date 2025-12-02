import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class DataConfig:
    """Configuration for CarDD data loading."""
    root_dir: str = "../data/raw"
    num_classes: int = 6
    batch_size: int = 16
    num_workers: int = 0


class CarDDMultiLabelDataset(Dataset):
    """
    PyTorch Dataset for CarDD multi-label classification.

    Each sample is a dict with:
      - "image_path": path to the image file
      - "label": np.ndarray or list of length num_classes with 0/1 entries
    """

    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        label = sample["label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # multi-label -> float tensor (for BCEWithLogitsLoss)
        label = torch.as_tensor(label, dtype=torch.float32)

        return image, label


def load_coco_multilabel(
    json_path: str,
    img_dir: str,
    num_classes: int,
) -> Tuple[List[Dict], List[str]]:
    """
    Load COCO-style annotations for a multi-label classification task.

    Assumes:
      - `images` list with entries: {"id": int, "file_name": str, ...}
      - `annotations` list with entries: {"image_id": int, "category_id": int, ...}
      - `categories` list with entries: {"id": int, "name": str, ...}

    Returns:
      samples: list of {"image_path": str, "label": np.ndarray[num_classes]}
      class_names: list of class names in category-id order used to build labels.
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Map image_id -> file_name
    id_to_filename = {
        img["id"]: img["file_name"] for img in coco.get("images", [])
    }

    # Map category index (0..num_classes-1) to name, and category_id -> idx
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    catid_to_idx = {c["id"]: i for i, c in enumerate(categories)}

    # Initialize label matrix: for each image, we will set 1 for each category present
    imageid_to_label = {
        img_id: np.zeros(num_classes, dtype=np.int32)
        for img_id in id_to_filename.keys()
    }

    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        if img_id not in imageid_to_label:
            continue
        if cat_id not in catid_to_idx:
            continue

        idx = catid_to_idx[cat_id]
        imageid_to_label[img_id][idx] = 1

    samples = []
    for img_id, file_name in id_to_filename.items():
        image_path = os.path.join(img_dir, file_name)
        label = imageid_to_label[img_id]
        samples.append(
            {
                "image_path": image_path,
                "label": label,
            }
        )

    return samples, class_names


def create_dataloaders(
    config: DataConfig,
    train_transforms: transforms.Compose | None = None,
    val_transforms: transforms.Compose | None = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and val DataLoaders from a root directory with COCO-style splits.

    Expects:
      root_dir/
        train/
          annotations.json
          images/
        val/
          annotations.json
          images/
    """
    root = Path(config.root_dir)

    train_json_path = root / "train" / "annotations.json"
    train_img_dir = root / "train" / "images"

    val_json_path = root / "val" / "annotations.json"
    val_img_dir = root / "val" / "images"

    train_samples, class_names = load_coco_multilabel(
        str(train_json_path),
        str(train_img_dir),
        num_classes=config.num_classes,
    )
    val_samples, _ = load_coco_multilabel(
        str(val_json_path),
        str(val_img_dir),
        num_classes=config.num_classes,
    )

    if train_transforms is None:
        train_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    if val_transforms is None:
        val_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    train_dataset = CarDDMultiLabelDataset(
        train_samples, transform=train_transforms
    )
    val_dataset = CarDDMultiLabelDataset(
        val_samples, transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, class_names
