"""
Train the hybrid attention + MLP bidder using IMP cost vectors.

Loss:
- For each example, we compute expected IMP loss under the model's
  action distribution:
    p = softmax(logits)
    loss_example = sum_a p[a] * cost[a]
- We minimize the average of loss_example across the batch.

Metrics:
- avg_expected_cost: mean of loss_example
- avg_best_cost: mean of min_a cost[a]
- avg_regret: avg_expected_cost - avg_best_cost
- acc_vs_best: accuracy of argmax p compared to argmin cost
"""

from pathlib import Path
from typing import Tuple
import csv

import torch
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingImpsDataset
from src.models.attn_rnn_bidder import AttnRnnBidder


def make_dataloaders_imps(
    data_root: str | Path,
    batch_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    data_root = Path(data_root)

    train_ds = BridgeBiddingImpsDataset(data_root, split="train")
    val_ds = BridgeBiddingImpsDataset(data_root, split="validate")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def train_one_epoch_imps(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Train for one epoch using expected IMP loss.

    Returns:
    - avg_expected_cost
    - avg_best_cost
    - avg_regret
    - acc_vs_best
    """
    model.train()
    total_examples = 0
    sum_expected_cost = 0.0
    sum_best_cost = 0.0
    correct_top1 = 0

    for batch_x, batch_cost in loader:
        batch_x = batch_x.to(device)
        batch_cost = batch_cost.to(device)  # [batch, 36]

        optimizer.zero_grad()
        logits = model(batch_x)             # [batch, 36]
        probs = torch.softmax(logits, dim=1)

        # Expected cost per example
        expected_cost_per_example = (probs * batch_cost).sum(dim=1)  # [batch]
        loss = expected_cost_per_example.mean()
        loss.backward()
        optimizer.step()

        # Metrics
        best_cost_per_example, best_action = batch_cost.min(dim=1)   # [batch]
        pred_action = probs.argmax(dim=1)

        batch_size = batch_x.size(0)
        total_examples += batch_size
        sum_expected_cost += expected_cost_per_example.detach().sum().item()
        sum_best_cost += best_cost_per_example.detach().sum().item()
        correct_top1 += (pred_action == best_action).sum().item()

    avg_expected_cost = sum_expected_cost / total_examples
    avg_best_cost = sum_best_cost / total_examples
    avg_regret = avg_expected_cost - avg_best_cost
    acc_vs_best = correct_top1 / total_examples

    return avg_expected_cost, avg_best_cost, avg_regret, acc_vs_best


def evaluate_imps(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Same metrics as train_one_epoch_imps, but without gradient updates.
    """
    model.eval()
    total_examples = 0
    sum_expected_cost = 0.0
    sum_best_cost = 0.0
    correct_top1 = 0

    with torch.no_grad():
        for batch_x, batch_cost in loader:
            batch_x = batch_x.to(device)
            batch_cost = batch_cost.to(device)

            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)

            expected_cost_per_example = (probs * batch_cost).sum(dim=1)
            best_cost_per_example, best_action = batch_cost.min(dim=1)
            pred_action = probs.argmax(dim=1)

            batch_size = batch_x.size(0)
            total_examples += batch_size
            sum_expected_cost += expected_cost_per_example.sum().item()
            sum_best_cost += best_cost_per_example.sum().item()
            correct_top1 += (pred_action == best_action).sum().item()

    avg_expected_cost = sum_expected_cost / total_examples
    avg_best_cost = sum_best_cost / total_examples
    avg_regret = avg_expected_cost - avg_best_cost
    acc_vs_best = correct_top1 / total_examples

    return avg_expected_cost, avg_best_cost, avg_regret, acc_vs_best

def save_metrics_csv(metrics, csv_path: Path) -> None:
    """
    Save a list of per epoch metrics dictionaries to a CSV file.

    Each dict should look like:
    {
        "epoch": int,
        "train_expected_cost": float,
        "train_regret": float,
        "train_acc": float,
        "val_expected_cost": float,
        "val_regret": float,
        "val_acc": float,
    }
    """
    if not metrics:
        return

    fieldnames = list(metrics[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def main() -> None:
    project_src = Path(__file__).resolve().parent
    project_root = project_src.parent
    data_root = project_root / "data"
    models_dir = project_root / "models"
    docs_dir = project_root / "docs"
    models_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    print(f"Project src:  {project_src}")
    print(f"Project root: {project_root}")
    print(f"Data root:    {data_root}")
    print(f"Models dir:   {models_dir}")
    print(f"Docs dir:     {docs_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders_imps(data_root, batch_size=512)

    # Hybrid model
    model = AttnRnnBidder(
        input_dim=104,
        num_actions=36,
        d_model=128,
        rnn_hidden=128,
        rnn_layers=1,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 50
    best_val_regret = float("inf")
    best_path = models_dir / "attn_rnn_imps.pt"

    # For logging per epoch metrics
    metrics = []
    metrics_csv_path = docs_dir / "attn_rnn_imps_metrics.csv"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_exp, train_best, train_regret, train_acc = train_one_epoch_imps(
            model, train_loader, optimizer, device
        )
        val_exp, val_best, val_regret, val_acc = evaluate_imps(
            model, val_loader, device
        )

        print("  Train expected cost: {:.4f}".format(train_exp))
        print("  Train best cost:     {:.4f}".format(train_best))
        print("  Train regret:        {:.4f}".format(train_regret))
        print("  Train acc vs best:   {:.4f}".format(train_acc))

        print("  Val   expected cost: {:.4f}".format(val_exp))
        print("  Val   best cost:     {:.4f}".format(val_best))
        print("  Val   regret:        {:.4f}".format(val_regret))
        print("  Val   acc vs best:   {:.4f}".format(val_acc))

        # Log metrics for this epoch
        metrics.append(
            {
                "epoch": epoch,
                "train_expected_cost": train_exp,
                "train_regret": train_regret,
                "train_acc": train_acc,
                "val_expected_cost": val_exp,
                "val_regret": val_regret,
                "val_acc": val_acc,
            }
        )

        if val_regret < best_val_regret:
            best_val_regret = val_regret
            torch.save(model.state_dict(), best_path)
            print(f"  New best model by regret saved to {best_path}")

    # Save metrics to CSV
    save_metrics_csv(metrics, metrics_csv_path)

    print("\nTraining complete.")
    print(f"Best validation regret: {best_val_regret:.4f}")
    print(f"Best model checkpoint: {best_path}")
    print(f"Metrics CSV written to: {metrics_csv_path}")


if __name__ == "__main__":
    main()
