"""Metrics for multiclass image classification."""

import numpy as np


def confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    """Count actual/predicted pairs; rows are true classes, columns predicted."""
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("True and predicted labels must have matching shapes")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if (
        np.any(true_labels < 0)
        or np.any(true_labels >= num_classes)
        or np.any(predicted_labels < 0)
        or np.any(predicted_labels >= num_classes)
    ):
        raise ValueError("Labels must be between 0 and num_classes - 1")

    counts = np.zeros((num_classes, num_classes), dtype=np.int64)  # (C, C)
    np.add.at(counts, (true_labels, predicted_labels), 1)
    return counts


def per_class_accuracy(counts: np.ndarray) -> np.ndarray:
    """Return correct fraction per true class for a (C, C) confusion matrix."""
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("Confusion matrix must be square with shape (C, C)")
    examples_per_class = np.sum(counts, axis=1)  # (C,)
    correct_per_class = np.diag(counts)  # (C,)
    return np.divide(
        correct_per_class,
        examples_per_class,
        out=np.zeros_like(correct_per_class, dtype=np.float64),
        where=examples_per_class != 0,
    )
