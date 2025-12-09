"""
Plot training and validation curves for the attention plus RNN IMP model.

Reads:
    docs/attn_rnn_imps_metrics.csv

Writes:
    docs/attn_rnn_imps_loss_curves.png
    docs/attn_rnn_imps_acc_curves.png
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


def load_metrics(csv_path: Path):
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_expected_cost"]))
            val_loss.append(float(row["val_expected_cost"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    return epochs, train_loss, val_loss, train_acc, val_acc


def plot_curves(
    epochs,
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    out_dir: Path,
) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)

    # Loss (expected cost) curves
    plt.figure()
    plt.plot(epochs, train_loss, label="Train expected cost")
    plt.plot(epochs, val_loss, label="Val expected cost")
    plt.xlabel("Epoch")
    plt.ylabel("Expected IMP cost")
    plt.title("Attention plus RNN IMP model - expected cost")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    loss_path = out_dir / "attn_rnn_imps_loss_curves.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, train_acc, label="Train acc vs best")
    plt.plot(epochs, val_acc, label="Val acc vs best")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy vs best action")
    plt.title("Attention plus RNN IMP model - accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    acc_path = out_dir / "attn_rnn_imps_acc_curves.png"
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote loss curves to: {loss_path}")
    print(f"Wrote accuracy curves to: {acc_path}")


def main() -> None:
    project_src = Path(__file__).resolve().parent
    project_root = project_src.parent

    docs_dir = project_root / "docs"
    metrics_csv_path = docs_dir / "attn_rnn_imps_metrics.csv"

    if not metrics_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find metrics CSV at {metrics_csv_path}. "
            "Run src.train_imps_hybrid first."
        )

    epochs, train_loss, val_loss, train_acc, val_acc = load_metrics(metrics_csv_path)
    plot_curves(epochs, train_loss, val_loss, train_acc, val_acc, docs_dir)


if __name__ == "__main__":
    main()
