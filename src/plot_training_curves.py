"""
Plot training and validation curves from metrics CSV.
CITATION: This borrows heavily from the plot_training_curves method in Prof. Fain's assinment "Recogniztion"

Reads:
    models/mlp_baseline_metrics.csv

Writes:
    docs/mlp_loss_curves.png
    docs/mlp_acc_curves.png
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


def load_metrics(csv_path: Path):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    return epochs, train_loss, train_acc, val_loss, val_acc


def plot_curves(
    epochs,
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    out_dir: Path,
) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)

    # Loss curves
    plt.figure()
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy loss")
    plt.title("MLP training vs validation loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    loss_path = out_dir / "mlp_loss_curves.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, train_acc, label="Train accuracy")
    plt.plot(epochs, val_acc, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP training vs validation accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    acc_path = out_dir / "mlp_acc_curves.png"
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote loss curves to: {loss_path}")
    print(f"Wrote accuracy curves to: {acc_path}")


def main() -> None:
    project_src = Path(__file__).resolve().parent
    project_root = project_src.parent

    models_dir = project_root / "models"
    docs_dir = project_root / "docs"
    metrics_csv_path = models_dir / "mlp_baseline_metrics.csv"

    if not metrics_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find metrics CSV at {metrics_csv_path}. "
            "Run src.train_supervised first."
        )

    epochs, train_loss, train_acc, val_loss, val_acc = load_metrics(metrics_csv_path)
    plot_curves(epochs, train_loss, val_loss, train_acc, val_acc, docs_dir)


if __name__ == "__main__":
    main()
