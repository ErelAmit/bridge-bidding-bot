"""
Compare MLP vs attention plus RNN on IMP based metrics.

Both models are evaluated on the same IMP test set using:

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
from src.models.attn_rnn_bidder import AttnRnnBidder
from src.eval_imps_mlp import evaluate_imps_model


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    return evaluate_imps_model(model, loader, device)


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

    # Test IMP dataset and loader
    test_ds = BridgeBiddingImpsDataset(data_root, split="test")
    print("Test size:", len(test_ds))

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

    mlp_exp, mlp_best, mlp_regret, mlp_acc = evaluate_model(
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

    attn_exp, attn_best, attn_regret, attn_acc = evaluate_model(
        attn_rnn, test_loader, device
    )

    print("\n=== IMP comparison on test set ===")
    header = (
        f"{'Model':<25}"
        f"{'Avg expected cost':>20}"
        f"{'Avg best cost':>16}"
        f"{'Avg regret':>14}"
        f"{'Acc vs best':>14}"
    )
    print(header)
    print("-" * len(header))

    print(
        f"{'MLP (CE trained)':<25}"
        f"{mlp_exp:>20.4f}"
        f"{mlp_best:>16.4f}"
        f"{mlp_regret:>14.4f}"
        f"{mlp_acc:>14.4f}"
    )

    print(
        f"{'Attn+RNN (IMP trained)':<25}"
        f"{attn_exp:>20.4f}"
        f"{attn_best:>16.4f}"
        f"{attn_regret:>14.4f}"
        f"{attn_acc:>14.4f}"
    )


if __name__ == "__main__":
    main()
