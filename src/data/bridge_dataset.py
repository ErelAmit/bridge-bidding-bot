"""
PyTorch Dataset for Yeh and Lin bridge bidding data.

Each example:
- x: 104-dimensional feature vector (float32)
- y: integer action index in [0, 35], corresponding to the minimum-cost bid.
"""

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


SplitName = Literal["train", "validate", "test"]


class BridgeBiddingDataset(Dataset):
    """
    Dataset wrapping the Yeh and Lin .mat files.

    For a given split ("train", "validate", or "test"), this:
    - loads data_<split>.mat (104, N)
    - loads cost_<split>.mat (36, N)
    - converts them to:
        features: tensor of shape [N, 104]
        labels: tensor of shape [N], where labels[i] is the index of the
                minimum-cost action in cost[:, i]
    """

    def __init__(self, root_dir: str | Path, split: SplitName) -> None:
        super().__init__()

        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

        # Decide which files to load based on split
        if split == "train":
            data_name = "data_train.mat"
            cost_name = "cost_train.mat"
        elif split == "validate":
            data_name = "data_validate.mat"
            cost_name = "cost_validate.mat"
        elif split == "test":
            data_name = "data_test.mat"
            cost_name = "cost_test.mat"
        else:
            raise ValueError(f"Unsupported split: {split}")

        data_path = root_dir / data_name
        cost_path = root_dir / cost_name

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not cost_path.exists():
            raise FileNotFoundError(f"Cost file not found: {cost_path}")

        # Load .mat files
        data_mat = loadmat(data_path)
        cost_mat = loadmat(cost_path)

        # Extract the main arrays.
        # Different .mat files may use different key names, so we choose
        # the first non magic key that contains a NumPy array.
        def pick_main_array(mat_dict: dict, path_str: str) -> np.ndarray:
            keys = [k for k in mat_dict.keys() if not k.startswith("__")]
            if not keys:
                raise KeyError(f"No data keys found in {path_str}")
            arr = mat_dict[keys[0]]
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"Key {keys[0]} in {path_str} is not a NumPy array"
                )
            return arr

        data = pick_main_array(data_mat, str(data_path))
        cost = pick_main_array(cost_mat, str(cost_path))

        # Expected shapes: data: (104, N), cost: (36, N)
        if data.ndim != 2 or data.shape[0] != 104:
            raise ValueError(f"Unexpected data shape {data.shape}, expected (104, N)")
        if cost.ndim != 2 or cost.shape[0] != 36:
            raise ValueError(f"Unexpected cost shape {cost.shape}, expected (36, N)")

        # Transpose so we have shape [N, 104] and [N, 36]
        # Each row is one example
        data_np: np.ndarray = data.T  # [N, 104]
        cost_np: np.ndarray = cost.T  # [N, 36]

        if data_np.shape[0] != cost_np.shape[0]:
            raise ValueError(
                f"Feature and cost counts differ: {data_np.shape[0]} vs {cost_np.shape[0]}"
            )

        # Compute labels as argmin over cost for each example
        # labels_np[i] is an integer in [0, 35]
        labels_np: np.ndarray = np.argmin(cost_np, axis=1)

        # Convert to PyTorch tensors
        self.features: torch.Tensor = torch.from_numpy(data_np).float()
        self.labels: torch.Tensor = torch.from_numpy(labels_np).long()

        self.num_examples = self.features.shape[0]

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: tensor of shape [104], dtype float32
            y: tensor scalar, dtype long, in [0, 35]
        """
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

class BridgeBiddingImpsDataset(Dataset):
    """
    Dataset that returns (features, cost_vector) for IMP based training.

    - features: tensor of shape [104]
    - cost_vector: tensor of shape [36], IMP loss for each possible action
    """

    def __init__(self, data_root: str | Path, split: str = "train") -> None:
        super().__init__()
        data_root = Path(data_root)

        if split not in {"train", "validate", "test"}:
            raise ValueError(f"Unknown split: {split}")

        data_path = data_root / f"data_{split}.mat"
        cost_path = data_root / f"cost_{split}.mat"

        if not data_path.exists():
            raise FileNotFoundError(f"Missing {data_path}")
        if not cost_path.exists():
            raise FileNotFoundError(f"Missing {cost_path}")

        # Load .mat files
        data_mat = loadmat(data_path)
        cost_mat = loadmat(cost_path)

        # Pick main arrays (first non magic key with a NumPy array)
        def pick_main_array(mat_dict: dict, path_str: str) -> np.ndarray:
            keys = [k for k in mat_dict.keys() if not k.startswith("__")]
            if not keys:
                raise KeyError(f"No data keys found in {path_str}")
            arr = mat_dict[keys[0]]
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"Key {keys[0]} in {path_str} is not a NumPy array"
                )
            return arr

        data_np = pick_main_array(data_mat, str(data_path))  # shape [104, N]
        cost_np = pick_main_array(cost_mat, str(cost_path))  # shape [36, N]

        if data_np.shape[1] != cost_np.shape[1]:
            raise ValueError(
                f"Feature and cost arrays have different number of examples: "
                f"{data_np.shape[1]} vs {cost_np.shape[1]}"
            )

        # Transpose so that axis 0 is example index
        # data_np: [104, N] -> [N, 104]
        # cost_np: [36, N] -> [N, 36]
        data_np = data_np.T
        cost_np = cost_np.T

        # Convert to tensors
        self.features: torch.Tensor = torch.from_numpy(data_np).float()
        self.costs: torch.Tensor = torch.from_numpy(cost_np).float()

        self.num_examples = self.features.shape[0]

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]   # [104]
        c = self.costs[idx]      # [36], IMP losses for each action
        return x, c
