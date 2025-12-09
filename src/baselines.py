"""
Simple baseline evaluations on the test set.

Baselines:
- Majority class: always predict the most frequent action
- Random uniform: predict a random action in [0, 35]
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingDataset


def majority_class_baseline(labels: np.ndarray) -> tuple[int, float]:
    """
    Compute the majority class and its accuracy.
    """
    counts = np.bincount(labels)
    majority_class = int(np.argmax(counts))
    acc = float((labels == majority_class).mean())
    return majority_class, acc


def random_uniform_baseline(labels: np.ndarray, num_actions: int = 36) -> float:
    """
    Accuracy of a random uniform predictor on the given labels.
    """
    rng = np.random.default_rng(seed=42)
    random_preds = rng.integers(low=0, high=num_actions, size=labels.shape[0])
    acc = float((random_preds == labels).mean())
    return acc


def main() -> None:
    project_root = Path(".").resolve()
    data_root = project_root / "data"

    print("Project root:", project_root)
    print("Data root:   ", data_root)

    # Load test dataset
    test_ds = BridgeBiddingDataset(data_root, split="test")
    print("Test size:", len(test_ds))

    # Extract all labels as a NumPy array
    # We already have labels tensor stored inside the dataset
    labels = test_ds.labels.numpy()

    # Majority baseline
    majority_cls, maj_acc = majority_class_baseline(labels)
    print("\nMajority class baseline:")
    print(f"  Majority class index: {majority_cls}")
    print(f"  Accuracy:             {maj_acc:.4f}")

    # Random uniform baseline
    rand_acc = random_uniform_baseline(labels, num_actions=36)
    print("\nRandom uniform baseline:")
    print(f"  Accuracy:             {rand_acc:.4f}")


if __name__ == "__main__":
    main()
