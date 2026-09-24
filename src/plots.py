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
    columns = 2
    rows = (len(histories) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 6, rows * 3.5),
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)

    for axis, (run_name, history) in zip(flat_axes, histories.items()):
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
        axis.set_title(f"Learning rate {run_name}")
        axis.legend()
        axis.grid(True, alpha=0.3)

    for axis in flat_axes[len(histories) :]:
        axis.axis("off")

    figure.suptitle("Logistic regression loss")
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_confusion_matrix(
    counts: np.ndarray,
    output_path: str | Path,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> Path:
    """Save an annotated confusion matrix; rows are true classes."""
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if len(class_names) != counts.shape[0]:
        raise ValueError("Need one class name per matrix row and column")

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(counts, cmap="Blues")  # (C, C)
    figure.colorbar(image, ax=axis, label="Number of images")
    axis.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Validation confusion matrix")

    threshold = counts.max() / 2 if counts.size else 0
    for true_class in range(counts.shape[0]):
        for predicted_class in range(counts.shape[1]):
            value = counts[true_class, predicted_class]
            text_color = "white" if value > threshold else "black"
            axis.text(
                predicted_class,
                true_class,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
