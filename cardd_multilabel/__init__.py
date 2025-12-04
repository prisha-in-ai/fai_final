"""
cardd_multilabel


Utilities for training and evaluating a multi-label car damage classifier
on the CarDD dataset.
"""

from .data import (
    CarDDMultiLabelDataset,
    load_coco_multilabel,
    create_dataloaders,
)
from .model import build_resnet50_multilabel
from .train import (
    train_model,
    train_one_epoch,
    evaluate_model,
)
from .metrics import (
    compute_classification_report,
    compute_confusion_summary,
)

__all__ = [
    "CarDDMultiLabelDataset",
    "load_coco_multilabel",
    "create_dataloaders",
    "build_resnet50_multilabel",
    "train_model",
    "train_one_epoch",
    "evaluate_model",
    "compute_classification_report",
    "compute_confusion_summary",
]
