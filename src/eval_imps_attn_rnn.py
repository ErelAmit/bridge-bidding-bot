"""
Evaluate the attention plus RNN bidder on the test set
using IMP based metrics.

Reports:
- avg_expected_cost
- avg_best_cost
- avg_regret
- acc_vs_best
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.bridge_dataset import BridgeBiddingImpsDataset
from src.models.attn_rnn_bidder import AttnRnnBidder
from src.train_imps_hybrid import evaluate_imps


def main() -> None:
    project_root = Path(".").resolve()
    data_root = project_root / "data"
    models_dir = project_root / "models"
    ckpt_path = models_dir / "attn_rnn_imps.pt"

    print("Project root:", project_root)
    print("Data root:   ", data_root)
    print("Checkpoint:  ", ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run src.train_imps_hybrid first."
        )

    # Test dataset and loader
    test_ds = BridgeBiddingImpsDataset(data_root, split="test")
    print("Test size:", len(test_ds))

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model must match training config
    model = AttnRnnBidder(
        input_dim=104,
        num_actions=36,
        d_model=128,
        rnn_hidden=128,
        rnn_layers=1,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    avg_exp, avg_best, avg_regret, acc_vs_best = evaluate_imps(
        model, test_loader, device
    )

    print("\nTest set IMP metrics (attention plus RNN):")
    print(f"  Avg expected cost: {avg_exp:.4f}")
    print(f"  Avg best cost:     {avg_best:.4f}")
    print(f"  Avg regret:        {avg_regret:.4f}")
    print(f"  Acc vs best:       {acc_vs_best:.4f}")


if __name__ == "__main__":
    main()
