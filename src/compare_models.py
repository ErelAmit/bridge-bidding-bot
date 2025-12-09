"""
Compare MLP vs attention plus RNN vs simple baselines on IMP based metrics.

All models and baselines are evaluated on the same IMP test set using:

- avg_expected_cost: E_p[cost] under the policy's action distribution
- avg_best_cost: average of min_a cost[a] (oracle best, same for all)
- avg_regret: avg_expected_cost - avg_best_cost
- acc_vs_best: how often the chosen action matches argmin cost

Baselines:
- Majority best action (deterministic, based on train IMP costs)
- Random uniform policy
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingImpsDataset
from src.models.mlp_bidder import MLPBidder
from src.models.attn_rnn_bidder import AttnRnnBidder
from src.eval_imps_mlp import evaluate_imps_model


def compute_imp_baselines(
    train_ds: BridgeBiddingImpsDataset,
    test_ds: BridgeBiddingImpsDataset,
) -> dict:
    """
    Compute IMP based metrics for:
    - Majority best action baseline (deterministic)
    - Random uniform baseline

    Returns a dict with two entries: "majority" and "random", each mapping to
    (avg_expected_cost, avg_best_cost, avg_regret, acc_vs_best).
    """

    # Train costs: [N_train, 36]
    train_costs = train_ds.costs  # tensor float
    # Best action per train example
    _, best_actions_train = train_costs.min(dim=1)
    # Global majority best action index
    counts = torch.bincount(best_actions_train, minlength=36)
    majority_action = int(counts.argmax().item())

    # Test costs: [N_test, 36]
    test_costs = test_ds.costs
    # Best cost and best action per test example
    best_costs_test, best_actions_test = test_costs.min(dim=1)

    avg_best_cost = best_costs_test.mean().item()

    # Majority baseline: always choose majority_action
    majority_costs = test_costs[:, majority_action]
    avg_exp_majority = majority_costs.mean().item()
    avg_regret_majority = avg_exp_majority - avg_best_cost
    acc_vs_best_majority = (best_actions_test == majority_action).float().mean().item()

    # Random uniform baseline: each action chosen with prob 1 / 36
    # Expected cost per example is the mean of the cost vector
    random_expected_costs = test_costs.mean(dim=1)
    avg_exp_random = random_expected_costs.mean().item()
    avg_regret_random = avg_exp_random - avg_best_cost

    # Probability that random uniform picks a best action:
    # For each example, count how many actions achieve the minimum cost,
    # then divide by number of actions (36) and average.
    num_actions = test_costs.size(1)
    is_best = test_costs == best_costs_test.unsqueeze(1)  # [N_test, 36]
    num_best_actions = is_best.sum(dim=1).float()         # [N_test]
    acc_vs_best_random = (num_best_actions / num_actions).mean().item()

    return {
        "majority": (
            avg_exp_majority,
            avg_best_cost,
            avg_regret_majority,
            acc_vs_best_majority,
        ),
        "random": (
            avg_exp_random,
            avg_best_cost,
            avg_regret_random,
            acc_vs_best_random,
        ),
    }


def main() -> None:
    project_root = Path(".").resolve()
    data_root = project_root / "data"
    models_dir = project_root / "models"

    mlp_ckpt = models_dir / "mlp_baseline.pt"
    attn_rnn_ckpt = models_dir / "attn_rnn_imps.pt"

    print("Project root:", project_root)
    print("Data root:   ", data_root)
    print("Models dir:  ", models_dir)
    print("MLP ckpt:    ", mlp_ckpt)
    print("AttnRNN ckpt:", attn_rnn_ckpt)

    if not mlp_ckpt.exists():
        raise FileNotFoundError(
            f"Missing MLP checkpoint at {mlp_ckpt}. "
            "Run src.train_supervised first."
        )
    if not attn_rnn_ckpt.exists():
        raise FileNotFoundError(
            f"Missing attention RNN IMP checkpoint at {attn_rnn_ckpt}. "
            "Run src.train_imps_hybrid first."
        )

    # Load IMP datasets
    train_ds = BridgeBiddingImpsDataset(data_root, split="train")
    test_ds = BridgeBiddingImpsDataset(data_root, split="test")
    print("Train size:", len(train_ds))
    print("Test size: ", len(test_ds))

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) MLP (trained on labels, evaluated on IMPs)
    mlp = MLPBidder(
        input_dim=104,
        hidden_dim=256,
        num_actions=36,
        num_hidden_layers=3,
        dropout=0.2,
        use_batchnorm=True,
    ).to(device)
    mlp_state = torch.load(mlp_ckpt, map_location=device)
    mlp.load_state_dict(mlp_state)

    mlp_exp, mlp_best, mlp_regret, mlp_acc = evaluate_imps_model(
        mlp, test_loader, device
    )

    # 2) Attention plus RNN (trained on IMPs)
    attn_rnn = AttnRnnBidder(
        input_dim=104,
        num_actions=36,
        d_model=128,
        rnn_hidden=128,
        rnn_layers=1,
        num_heads=4,
        dropout=0.1,
    ).to(device)
    attn_state = torch.load(attn_rnn_ckpt, map_location=device)
    attn_rnn.load_state_dict(attn_state)

    attn_exp, attn_best, attn_regret, attn_acc = evaluate_imps_model(
        attn_rnn, test_loader, device
    )

    # 3) IMP baselines (majority and random)
    baselines = compute_imp_baselines(train_ds, test_ds)
    maj_exp, maj_best, maj_regret, maj_acc = baselines["majority"]
    rnd_exp, rnd_best, rnd_regret, rnd_acc = baselines["random"]

    print("\n=== IMP comparison on test set ===")
    header = (
        f"{'Model':<30}"
        f"{'Avg expected cost':>20}"
        f"{'Avg best cost':>16}"
        f"{'Avg regret':>14}"
        f"{'Acc vs best':>14}"
    )
    print(header)
    print("-" * len(header))

    print(
        f"{'Random uniform':<30}"
        f"{rnd_exp:>20.4f}"
        f"{rnd_best:>16.4f}"
        f"{rnd_regret:>14.4f}"
        f"{rnd_acc:>14.4f}"
    )

    print(
        f"{'Majority best action':<30}"
        f"{maj_exp:>20.4f}"
        f"{maj_best:>16.4f}"
        f"{maj_regret:>14.4f}"
        f"{maj_acc:>14.4f}"
    )

    print(
        f"{'MLP (CE trained)':<30}"
        f"{mlp_exp:>20.4f}"
        f"{mlp_best:>16.4f}"
        f"{mlp_regret:>14.4f}"
        f"{mlp_acc:>14.4f}"
    )

    print(
        f"{'Attn+RNN (IMP trained)':<30}"
        f"{attn_exp:>20.4f}"
        f"{attn_best:>16.4f}"
        f"{attn_regret:>14.4f}"
        f"{attn_acc:>14.4f}"
    )


if __name__ == "__main__":
    main()
