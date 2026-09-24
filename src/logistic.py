"""Binary logistic regression implemented with vectorized NumPy operations."""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert scores shaped (N,) to probabilities shaped (N,) in [0, 1]."""
    safe_logits = np.clip(logits, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-safe_logits))  # (N,) -> (N,)


def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return mean binary cross-entropy for labels and probabilities shaped (N,)."""
    epsilon = 1e-12
    safe_probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    # Both terms are (N,); mean reduces the per-example losses to one scalar.
    losses = -(
        labels * np.log(safe_probabilities)
        + (1.0 - labels) * np.log(1.0 - safe_probabilities)
    )
    return float(np.mean(losses))


def compute_gradients(
    images: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> tuple[float, np.ndarray, float]:
    """Return loss, weight gradient (784,), and bias gradient (scalar)."""
    logits = images @ weights + bias  # (N, 784) @ (784,) -> (N,)
    probabilities = sigmoid(logits)  # (N,) -> (N,)
    errors = probabilities - labels  # (N,)
    loss = binary_cross_entropy(labels, probabilities)
    weight_gradient = images.T @ errors / images.shape[0]  # (784, N) @ (N,) -> (784,)
    bias_gradient = float(np.mean(errors))
    return loss, weight_gradient, bias_gradient


def predict_probabilities(images: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Return positive-class probabilities shaped (N,)."""
    logits = images @ weights + bias  # (N, 784) @ (784,) -> (N,)
    return sigmoid(logits)


def accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return fraction of correct predictions using 0.5 as the threshold."""
    predictions = probabilities >= 0.5  # (N,)
    return float(np.mean(predictions == labels.astype(bool)))


def train(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    learning_rate: float,
    epochs: int = 100,
) -> dict[str, object]:
    """Fit full-batch logistic regression and return parameters and histories."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    weights = np.zeros(train_images.shape[1], dtype=np.float64)  # (784,)
    bias = 0.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for _ in range(epochs):
        train_loss, weight_gradient, bias_gradient = compute_gradients(
            train_images, train_labels, weights, bias
        )
        weights -= learning_rate * weight_gradient  # (784,)
        bias -= learning_rate * bias_gradient

        # Report metrics after the update so they describe the current model.
        train_probabilities = predict_probabilities(train_images, weights, bias)
        val_probabilities = predict_probabilities(val_images, weights, bias)
        history["train_loss"].append(
            binary_cross_entropy(train_labels, train_probabilities)
        )
        history["val_loss"].append(binary_cross_entropy(val_labels, val_probabilities))
        history["val_accuracy"].append(accuracy(val_labels, val_probabilities))

    return {
        "weights": weights,
        "bias": bias,
        "history": history,
    }
