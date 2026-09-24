"""Multiclass softmax regression implemented with vectorized NumPy."""

from __future__ import annotations

import numpy as np

from src.data import RANDOM_SEED


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels (N,) to one-hot targets shaped (N, C)."""
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shaped (N,), got {labels.shape}")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Labels must be between 0 and num_classes - 1")

    targets = np.zeros((len(labels), num_classes), dtype=np.float32)  # (N, C)
    targets[np.arange(len(labels)), labels] = 1.0  # selected positions in (N, C)
    return targets


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return class probabilities shaped (N, C), stable across large scores."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shaped (N, C), got {logits.shape}")
    row_maxima = np.max(logits, axis=1, keepdims=True)  # (N, 1)
    shifted_logits = logits - row_maxima  # (N, C) - (N, 1) -> (N, C)
    exponentials = np.exp(shifted_logits)  # (N, C)
    row_sums = np.sum(exponentials, axis=1, keepdims=True)  # (N, 1)
    return exponentials / row_sums  # (N, C) / (N, 1) -> (N, C)


def cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Return mean multiclass cross-entropy for one-hot targets (N, C)."""
    if targets.shape != probabilities.shape:
        raise ValueError("Targets and probabilities must have matching shapes")
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)  # (N, C)
    per_example = -np.sum(targets * np.log(safe_probabilities), axis=1)  # (N,)
    return float(np.mean(per_example))


def compute_gradients(
    images: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return loss and gradients for W (784, C) and b (C,)."""
    logits = images @ weights + bias  # (N, 784) @ (784, C) -> (N, C)
    probabilities = softmax(logits)  # (N, C)
    errors = probabilities - targets  # (N, C)
    loss = cross_entropy(targets, probabilities)
    weight_gradient = images.T @ errors / images.shape[0]  # (784, N) @ (N, C) -> (784, C)
    bias_gradient = np.mean(errors, axis=0)  # (C,)
    return loss, weight_gradient, bias_gradient


def predict_probabilities(images: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return class probabilities shaped (N, C)."""
    logits = images @ weights + bias  # (N, 784) @ (784, C) -> (N, C)
    return softmax(logits)  # (N, C)


def predict(images: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return predicted class indices shaped (N,)."""
    probabilities = predict_probabilities(images, weights, bias)  # (N, C)
    return np.argmax(probabilities, axis=1)  # (N,)


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return the fraction of integer class predictions that are correct."""
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have matching shapes")
    return float(np.mean(labels == predictions))


def train(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 20,
    batch_size: int = 128,
    num_classes: int = 10,
) -> dict[str, object]:
    """Train with mini-batch SGD and return weights, bias, and epoch history."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_images.ndim != 2 or val_images.ndim != 2:
        raise ValueError("Image arrays must have shape (N, features)")
    if train_images.shape[1] != val_images.shape[1]:
        raise ValueError("Training and validation images need matching features")

    train_targets = one_hot(train_labels, num_classes)  # (N_train, C)
    val_targets = one_hot(val_labels, num_classes)  # (N_val, C)
    feature_count = train_images.shape[1]
    weights = np.zeros((feature_count, num_classes), dtype=np.float32)  # (784, C)
    bias = np.zeros(num_classes, dtype=np.float32)  # (C,)
    rng = np.random.default_rng(RANDOM_SEED)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for _ in range(epochs):
        shuffled_indices = rng.permutation(len(train_images))  # (N_train,)
        for start in range(0, len(train_images), batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]  # (batch,)
            batch_images = train_images[batch_indices]  # (batch, 784)
            batch_targets = train_targets[batch_indices]  # (batch, C)
            _, weight_gradient, bias_gradient = compute_gradients(
                batch_images,
                batch_targets,
                weights,
                bias,
            )
            weights -= learning_rate * weight_gradient  # (784, C)
            bias -= learning_rate * bias_gradient  # (C,)

        # Epoch metrics summarize the complete training and validation splits.
        train_probabilities = predict_probabilities(train_images, weights, bias)
        val_probabilities = predict_probabilities(val_images, weights, bias)
        history["train_loss"].append(cross_entropy(train_targets, train_probabilities))
        history["val_loss"].append(cross_entropy(val_targets, val_probabilities))
        history["val_accuracy"].append(
            accuracy(val_labels, np.argmax(val_probabilities, axis=1))
        )

    return {"weights": weights, "bias": bias, "history": history}
