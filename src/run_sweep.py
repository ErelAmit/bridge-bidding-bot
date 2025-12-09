"""
Hyperparameter and optimizer sweep script.

Runs multiple small training runs with different configs and
logs the best validation metrics to docs/sweep_results.csv.

This addresses:
- systematic hyperparameter tuning
- comparing multiple optimizers
"""

from pathlib import Path
import csv
from dataclasses import dataclass, asdict
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingDataset
from src.models.mlp_bidder import MLPBidder
from src.train_supervised import train_one_epoch, evaluate


OptimizerName = Literal["adamw", "sgd"]


@dataclass
class ExperimentConfig:
    optimizer: OptimizerName
    hidden_dim: int
    num_layers: int
    dropout: float
    use_batchnorm: bool
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    num_epochs: int
    batch_size: int


def make_optimizer(
    model: nn.Module, cfg: ExperimentConfig
) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def run_experiment(
    cfg: ExperimentConfig,
    data_root: Path,
    device: torch.device,
) -> dict:
    """
    Run one training experiment with the given config.
    Returns a dict with config and best validation metrics.
    """
    print("\n=== Experiment ===")
    print(asdict(cfg))

    # Data loaders specific to this experiment
    train_ds = BridgeBiddingDataset(data_root, split="train")
    val_ds = BridgeBiddingDataset(data_root, split="validate")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = MLPBidder(
        input_dim=104,
        hidden_dim=cfg.hidden_dim,
        num_actions=36,
        num_hidden_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_batchnorm=cfg.use_batchnorm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)

    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_grad_norm=cfg.max_grad_norm,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    result = {
        **asdict(cfg),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    print("\nBest val metrics for this experiment:")
    print(f"  loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}")
    return result


def save_sweep_results(results: list[dict], csv_path: Path) -> None:
    if not results:
        return

    fieldnames = list(results[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    project_root = Path(".").resolve()
    data_root = project_root / "data"
    docs_dir = project_root / "docs"
    csv_path = docs_dir / "sweep_results.csv"

    print("Project root:", project_root)
    print("Data root:   ", data_root)
    print("Results CSV: ", csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base = dict(
        hidden_dim=256,
        num_layers=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        num_epochs=3,    # short runs so sweep is fast
        batch_size=512,
    )

    configs: list[ExperimentConfig] = [
        ExperimentConfig(
            optimizer="adamw",
            dropout=0.0,
            use_batchnorm=False,
            **base,
        ),
        ExperimentConfig(
            optimizer="adamw",
            dropout=0.2,
            use_batchnorm=True,
            **base,
        ),
        ExperimentConfig(
            optimizer="sgd",
            dropout=0.2,
            use_batchnorm=True,
            **base,
        ),
    ]

    results: list[dict] = []
    for cfg in configs:
        result = run_experiment(cfg, data_root=data_root, device=device)
        results.append(result)

    save_sweep_results(results, csv_path)
    print("\nSaved sweep results to:", csv_path)


if __name__ == "__main__":
    main()
