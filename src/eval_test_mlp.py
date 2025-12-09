"""
Evaluate the trained MLP bidder on the test set.

Reports:
- cross entropy loss
- top 1 accuracy
- top 3 accuracy
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingDataset
from src.models.mlp_bidder import MLPBidder


def evaluate_topk(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    k: int = 3,
) -> tuple[float, float, float]:
    """
    Evaluate model on a DataLoader and compute:
    - loss
    - top 1 accuracy
    - top k accuracy
    """
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

            # Top 1
            preds_top1 = torch.argmax(logits, dim=1)
            correct_top1 += (preds_top1 == batch_y).sum().item()

            # Top k
            _, preds_topk = torch.topk(logits, k=k, dim=1)
            # preds_topk: [batch_size, k]
            match_topk = (preds_topk == batch_y.unsqueeze(1)).any(dim=1)
            correct_topk += match_topk.sum().item()

    avg_loss = running_loss / total
    acc_top1 = correct_top1 / total
    acc_topk = correct_topk / total
    return avg_loss, acc_top1, acc_topk


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

    # Test dataset and loader
    test_ds = BridgeBiddingDataset(data_root, split="test")
    print("Test size:", len(test_ds))

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model must match training config
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

    test_loss, test_acc_top1, test_acc_top3 = evaluate_topk(
        model, test_loader, criterion, device, k=3
    )

    print("\nTest set metrics (MLP, eval mode):")
    print(f"  Loss:           {test_loss:.4f}")
    print(f"  Top 1 accuracy: {test_acc_top1:.4f}")
    print(f"  Top 3 accuracy: {test_acc_top3:.4f}")


if __name__ == "__main__":
    main()
