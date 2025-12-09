"""
Supervised training script for the MLP bridge bidder.

This trains on:
- data_train.mat / cost_train.mat
- data_validate.mat / cost_validate.mat

and saves a model checkpoint to models/mlp_baseline.pt
"""

from pathlib import Path
from typing import Tuple
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingDataset
from src.models.mlp_bidder import MLPBidder


def make_dataloaders(
    data_root: str | Path,
    batch_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    data_root = Path(data_root)

    train_ds = BridgeBiddingDataset(data_root, split="train")
    val_ds = BridgeBiddingDataset(data_root, split="validate")

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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float | None = 1.0,
) -> tuple[float, float]:
    """
    Train for one epoch on the given DataLoader.

    Uses:
    - gradient clipping if max_grad_norm is not None
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on the given DataLoader.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def save_metrics_csv(metrics, csv_path: Path) -> None:
    """
    Save a list of per-epoch metrics dictionaries to a CSV file.

    Each dict should look like:
    {
        "epoch": int,
        "train_loss": float,
        "train_acc": float,
        "val_loss": float,
        "val_acc": float,
    }
    """
    if not metrics:
        return

    fieldnames = list(metrics[0].keys())

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def main() -> None:
    project_src = Path(__file__).resolve().parent
    project_root = project_src.parent
    data_root = project_root / "data"

    print(f"Project src:  {project_src}")
    print(f"Project root: {project_root}")
    print(f"Data root:    {data_root}")

    # Make sure models folder exists for saving checkpoints
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = make_dataloaders(data_root, batch_size=512)

    # Model with dropout and batch norm
    hidden_dim = 256
    num_layers = 3
    dropout = 0.2
    use_batchnorm = True

    model = MLPBidder(
        input_dim=104,
        hidden_dim=hidden_dim,
        num_actions=36,
        num_hidden_layers=num_layers,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay (L2 regularization)
    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    #learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    # Gradient clipping config
    max_grad_norm = 1.0

    num_epochs = 5
    best_val_acc = 0.0
    best_path = models_dir / "mlp_baseline.pt"

    # For logging metrics over epochs
    metrics = []
    metrics_csv_path = models_dir / "mlp_baseline_metrics.csv"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_grad_norm=max_grad_norm,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Current learning rate (from first param group)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Log metrics for this epoch
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved to {best_path}")

    # Save metrics to CSV
    save_metrics_csv(metrics, metrics_csv_path)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best model checkpoint: {best_path}")
    print(f"Metrics CSV written to: {metrics_csv_path}")



if __name__ == "__main__":
    main()
