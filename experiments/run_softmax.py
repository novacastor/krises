"""Train softmax regression and report validation-set class metrics."""

import numpy as np

from src.data import NUM_CLASSES, load_training_validation
from src.metrics import confusion_matrix, per_class_accuracy
from src.plots import CLASS_NAMES, save_confusion_matrix, save_loss_curves
from src.softmax import accuracy, predict, train


def main() -> None:
    train_x, train_y, val_x, val_y = load_training_validation()
    result = train(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=0.1,
        epochs=20,
        batch_size=128,
        num_classes=NUM_CLASSES,
    )
    weights = result["weights"]
    bias = result["bias"]
    history = result["history"]
    predictions = predict(val_x, weights, bias)
    counts = confusion_matrix(val_y, predictions, num_classes=NUM_CLASSES)
    class_accuracy = per_class_accuracy(counts)

    print(f"Validation accuracy: {accuracy(val_y, predictions):.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print("Per-class validation accuracy:")
    for class_name, score in zip(CLASS_NAMES, class_accuracy):
        print(f"  {class_name}: {score:.4f}")
    print("Validation confusion matrix (rows=true, columns=predicted):")
    print(counts)

    matrix_path = save_confusion_matrix(counts, "results/softmax_confusion_matrix.png")
    curve_path = save_loss_curves(
        {"softmax regression": history},
        "results/softmax_loss.png",
    )
    print(f"Confusion matrix saved to {matrix_path}")
    print(f"Loss curves saved to {curve_path}")


if __name__ == "__main__":
    main()
