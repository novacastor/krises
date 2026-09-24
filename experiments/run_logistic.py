"""Compare three learning rates for T-shirt/top vs Trouser classification."""

import numpy as np

from src.data import load_training_validation
from src.logistic import train
from src.plots import save_loss_curves


def keep_two_classes(
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep labels 0 and 1, mapping T-shirt/top to 0 and Trouser to 1."""
    selected = (labels == 0) | (labels == 1)  # (N,)
    binary_labels = (labels[selected] == 1).astype(float)  # (selected N,)
    return images[selected], binary_labels


def main() -> None:
    train_x, train_y, val_x, val_y = load_training_validation()
    train_x, train_y = keep_two_classes(train_x, train_y)
    val_x, val_y = keep_two_classes(val_x, val_y)

    train_counts = np.bincount(train_y.astype(int), minlength=2)
    print(
        f"Binary train shape: {train_x.shape}; "
        f"class counts (T-shirt/top, Trouser): {train_counts.tolist()}"
    )
    print(f"Binary validation shape: {val_x.shape}")

    learning_rates = (0.01, 0.1, 1.0)
    histories = {}
    for learning_rate in learning_rates:
        result = train(
            train_x,
            train_y,
            val_x,
            val_y,
            learning_rate=learning_rate,
        )
        history = result["history"]
        histories[f"lr={learning_rate:g}"] = history
        print(
            f"lr={learning_rate:g}: "
            f"train loss {history['train_loss'][-1]:.4f}, "
            f"validation loss {history['val_loss'][-1]:.4f}, "
            f"validation accuracy {history['val_accuracy'][-1]:.3f}"
        )

    output_path = save_loss_curves(histories, "results/logistic_loss.png")
    print(f"Loss curves saved to {output_path}")


if __name__ == "__main__":
    main()
