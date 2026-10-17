"""Classification metrics in plain numpy."""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predictions."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(cm.sum(), 1))


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.mean())
