"""
Evaluate the original MLP bidder on IMP based metrics.

We reuse the cost vectors from BridgeBiddingImpsDataset and compute:

- avg_expected_cost: E_p[cost] under model's softmax distribution
- avg_best_cost: average of min_a cost[a] (oracle best)
- avg_regret: avg_expected_cost - avg_best_cost
- acc_vs_best: how often argmax p matches argmin cost
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingImpsDataset
from src.models.mlp_bidder import MLPBidder


def evaluate_imps_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Compute IMP based metrics for a given model and data loader.
    """
    model.eval()
    total_examples = 0
    sum_expected_cost = 0.0
    sum_best_cost = 0.0
    correct_top1 = 0

    with torch.no_grad():
        for batch_x, batch_cost in loader:
            batch_x = batch_x.to(device)
            batch_cost = batch_cost.to(device)  # [batch, 36]

            logits = model(batch_x)             # [batch, 36]
            probs = torch.softmax(logits, dim=1)

            # Expected cost under the model distribution
            expected_cost_per_example = (probs * batch_cost).sum(dim=1)  # [batch]

            # Best possible cost and action from the cost vector
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

    # Use IMP cost dataset for the test split
    test_ds = BridgeBiddingImpsDataset(data_root, split="test")
    print("Test size:", len(test_ds))

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model must match how you trained the MLP
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

    avg_exp, avg_best, avg_regret, acc_vs_best = evaluate_imps_model(
        model, test_loader, device
    )

    print("\nTest set IMP metrics (MLP trained on accuracy):")
    print(f"  Avg expected cost: {avg_exp:.4f}")
    print(f"  Avg best cost:     {avg_best:.4f}")
    print(f"  Avg regret:        {avg_regret:.4f}")
    print(f"  Acc vs best:       {acc_vs_best:.4f}")


if __name__ == "__main__":
    main()
