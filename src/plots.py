"""Plot small visual summaries of Fashion-MNIST data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def save_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    count: int = 25,
) -> Path:
    """Save a labeled grid of flattened (N, 784) images and return its path."""
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected images shaped (N, 784), got {images.shape}")
    if len(images) != len(labels):
        raise ValueError("Images and labels must have the same number of rows")
    if count <= 0 or count > len(images):
        raise ValueError("count must be positive and no greater than dataset size")

    columns = 5
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    flat_axes = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(flat_axes):
        axis.axis("off")
        if index >= count:
            continue
        label = int(labels[index])
        axis.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set_title(CLASS_NAMES[label])

    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    output_path: str | Path,
) -> Path:
    """Save training and validation loss curves for one or more runs."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for run_name, history in histories.items():
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axis.plot(epochs, history["train_loss"], label=f"{run_name} train")
        axis.plot(
            epochs,
            history["val_loss"],
            linestyle="--",
            label=f"{run_name} validation",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy")
    axis.set_title("Logistic regression loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
