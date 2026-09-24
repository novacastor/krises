"""Download and prepare the raw Fashion-MNIST IDX files."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from urllib.request import urlopen

import numpy as np


# Keep the split reproducible from this one project-wide seed.
RANDOM_SEED = 42
IMAGE_SHAPE = (28, 28)
NUM_CLASSES = 10
BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion/"
)
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(filename: str, cache_dir: Path) -> Path:
    """Download one compressed IDX file if it is not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.exists():
        return destination

    url = BASE_URL + filename
    print(f"Downloading {filename}...")
    with urlopen(url, timeout=60) as response:
        contents = response.read()
    destination.write_bytes(contents)
    return destination


def load_images(path: Path) -> np.ndarray:
    """Read IDX image data and return float pixels shaped (N, 784) in [0, 1]."""
    with gzip.open(path, "rb") as compressed:
        header = compressed.read(16)
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Expected IDX image magic number 2051, got {magic}")
        if (rows, columns) != IMAGE_SHAPE:
            raise ValueError(f"Expected 28x28 images, got {rows}x{columns}")
        pixels = np.frombuffer(compressed.read(), dtype=np.uint8)

    if pixels.size != count * rows * columns:
        raise ValueError(f"Image file {path} has an unexpected data length")
    images = pixels.reshape(count, rows * columns)  # (N, 28, 28) -> (N, 784)
    return images.astype(np.float32) / 255.0  # (N, 784), values in [0, 1]


def load_labels(path: Path) -> np.ndarray:
    """Read IDX label data and return integer labels shaped (N,)."""
    with gzip.open(path, "rb") as compressed:
        header = compressed.read(8)
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Expected IDX label magic number 2049, got {magic}")
        labels = np.frombuffer(compressed.read(), dtype=np.uint8)

    if labels.size != count:
        raise ValueError(f"Label file {path} has an unexpected data length")
    if np.any(labels >= NUM_CLASSES):
        raise ValueError(f"Label file {path} contains a label outside 0–9")
    return labels.astype(np.int64)


def load_training_validation(
    cache_dir: str | Path = "data/fashion_mnist",
    validation_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the seeded train and validation split without opening test files.

    Images have shape (N, 784); labels have shape (N,). Only the official
    training data is shuffled and split.
    """
    cache_path = Path(cache_dir)
    train_images_path = download_file(FILES["train_images"], cache_path)
    train_labels_path = download_file(FILES["train_labels"], cache_path)
    all_train_images = load_images(train_images_path)
    all_train_labels = load_labels(train_labels_path)

    if len(all_train_images) != len(all_train_labels):
        raise ValueError("Training images and labels have different lengths")
    if not 0 < validation_size < len(all_train_images):
        raise ValueError("validation_size must be between 0 and training size")

    # One seeded permutation splits training data without involving the test set.
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(all_train_images))
    val_indices = order[:validation_size]
    train_indices = order[validation_size:]
    train_images = all_train_images[train_indices]  # (50,000, 784)
    train_labels = all_train_labels[train_indices]  # (50,000,)
    val_images = all_train_images[val_indices]  # (10,000, 784)
    val_labels = all_train_labels[val_indices]  # (10,000,)

    return train_images, train_labels, val_images, val_labels


def load_fashion_mnist(
    cache_dir: str | Path = "data/fashion_mnist",
    validation_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, and test arrays; opens test files explicitly."""
    train_x, train_y, val_x, val_y = load_training_validation(
        cache_dir=cache_dir,
        validation_size=validation_size,
    )
    cache_path = Path(cache_dir)
    test_images = load_images(download_file(FILES["test_images"], cache_path))
    test_labels = load_labels(download_file(FILES["test_labels"], cache_path))
    if len(test_images) != len(test_labels):
        raise ValueError("Test images and labels have different lengths")
    return train_x, train_y, val_x, val_y, test_images, test_labels


def class_counts(labels: np.ndarray) -> np.ndarray:
    """Count examples for each Fashion-MNIST class, including empty classes."""
    return np.bincount(labels, minlength=NUM_CLASSES)
