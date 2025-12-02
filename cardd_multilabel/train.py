from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


from .metrics import (
    compute_classification_report,
    compute_confusion_summary,
    compute_f1_scores,
)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    num_epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    threshold: float = 0.5
    device: str = "cuda"  # or "mps" / "cpu"
    checkpoint_path: str = "resnet50_multilabel_cardd_best.pt"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Train for a single epoch, returning average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate model on a dataloader.

    Returns a dict with:
      - val_loss
      - y_true (np.ndarray)
      - y_pred (np.ndarray)
      - micro_f1
      - per_class_f1
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    y_trues: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int()

        y_trues.append(labels.cpu().numpy().astype(int))
        y_preds.append(preds.cpu().numpy().astype(int))

    val_loss = total_loss / max(num_batches, 1)
    y_true = np.vstack(y_trues)
    y_pred = np.vstack(y_preds)

    micro_f1, per_class_f1 = compute_f1_scores(y_true, y_pred)

    return {
        "val_loss": val_loss,
        "y_true": y_true,
        "y_pred": y_pred,
        "micro_f1": micro_f1,
        "per_class_f1": per_class_f1,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    config: TrainConfig,
) -> Dict:
    """
    Full training loop.

    Tracks best validation micro-F1 and saves the best checkpoint.
    """
    device = torch.device(config.device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    best_val_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        print(f"  Train loss: {train_loss:.4f}")

        eval_results = evaluate_model(
            model,
            val_loader,
            device,
            criterion,
            threshold=config.threshold,
        )

        val_loss = eval_results["val_loss"]
        micro_f1 = eval_results["micro_f1"]
        per_class_f1 = eval_results["per_class_f1"]
        y_true = eval_results["y_true"]
        y_pred = eval_results["y_pred"]

        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val micro F1: {micro_f1:.4f}")

        for i, name in enumerate(class_names):
            print(f"    {name:15s} F1: {per_class_f1[i]:.4f}")

        # Save best checkpoint based on micro-F1
        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, config.checkpoint_path)
            print(f"  Saved new best model → {config.checkpoint_path}")

        scheduler.step()

    print("\nTraining complete. Best val micro F1:", best_val_f1)

    # Also compute and print final classification report & confusion summary
    report = compute_classification_report(
        y_true, y_pred, class_names
    )
    print("\nClassification report (per-class):\n")
    print(report)

    summary_df = compute_confusion_summary(
        y_true, y_pred, class_names
    )
    print("\nPer-class confusion summary:\n")
    print(summary_df)

    return {
        "best_val_micro_f1": best_val_f1,
        "classification_report": report,
        "confusion_summary": summary_df,
    }
