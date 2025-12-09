"""
Evaluate the saved MLP model on train and validation splits
in eval mode (no dropout, batch norm in eval).

This helps check whether it is normal that val curves look better
than train curves from the training loop.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingDataset
from src.models.mlp_bidder import MLPBidder
from src.train_supervised import evaluate


def main() -> None:
    project_root = Path(".").resolve()
    data_root = project_root / "data"
    models_dir = project_root / "models"
    ckpt_path = models_dir / "mlp_baseline.pt"

    print("Project root:", project_root)
    print("Data root:   ", data_root)
    print("Checkpoint:  ", ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run src.train_supervised first."
        )

    # Datasets and loaders
    train_ds = BridgeBiddingDataset(data_root, split="train")
    val_ds = BridgeBiddingDataset(data_root, split="validate")

    print("Train size:", len(train_ds))
    print("Val size:  ", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model must match how you trained it
    model = MLPBidder(
        input_dim=104,
        hidden_dim=256,
        num_actions=36,
        num_hidden_layers=3,
        dropout=0.2,
        use_batchnorm=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    # Evaluate in eval mode
    train_eval_loss, train_eval_acc = evaluate(
        model, train_loader, criterion, device
    )
    val_eval_loss, val_eval_acc = evaluate(
        model, val_loader, criterion, device
    )

    print("\nEval mode metrics (no dropout, BN in eval):")
    print(f"  Train loss: {train_eval_loss:.4f}, acc: {train_eval_acc:.4f}")
    print(f"  Val   loss: {val_eval_loss:.4f}, acc: {val_eval_acc:.4f}")


if __name__ == "__main__":
    main()
