"""Download Fashion-MNIST and print Phase 0 split details."""

from src.data import class_counts, load_fashion_mnist
from src.plots import save_sample_grid


def main() -> None:
    train_x, train_y, val_x, val_y, test_x, test_y = load_fashion_mnist()
    print(f"train: images {train_x.shape}, labels {train_y.shape}")
    print(f"validation: images {val_x.shape}, labels {val_y.shape}")
    print(f"test: images {test_x.shape}, labels {test_y.shape}")
    print(f"train class counts: {class_counts(train_y).tolist()}")
    print(f"validation class counts: {class_counts(val_y).tolist()}")
    output_path = save_sample_grid(train_x, train_y, "results/sample_grid.png")
    print(f"sample grid saved to {output_path}")


if __name__ == "__main__":
    main()
