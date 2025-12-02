from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    zero_division: int = 0,
) -> str:
    """
    Compute a text classification report for multi-label predictions.

    y_true, y_pred: (N, C) arrays of 0/1.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=zero_division,
    )
    return report


def compute_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute micro-F1 and per-class F1 scores for multi-label predictions.
    """
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return micro_f1, per_class_f1


def compute_confusion_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Compute confusion matrices per class and summarize TP, FP, FN, TN.

    Returns:
        DataFrame indexed by class name with columns [TP, FP, FN, TN].
    """
    rows = []

    for i, name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        # Ensure 2x2 shape
        cm2 = np.zeros((2, 2), dtype=int)
        cm2[: cm.shape[0], : cm.shape[1]] = cm
        tn, fp, fn, tp = cm2.ravel()
        rows.append(
            {
                "class": name,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )

    df = pd.DataFrame(rows).set_index("class")
    return df
