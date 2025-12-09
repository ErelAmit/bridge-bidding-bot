"""
Simple feedforward MLP for bridge bidding.

Input:
- 104 dimensional feature vector

Output:
- 36 dimensional logits, one for each possible action
"""

from typing import List

import torch
from torch import nn


class MLPBidder(nn.Module):
    def __init__(
        self,
        input_dim: int = 104,
        hidden_dim: int = 256,
        num_actions: int = 36,
        num_hidden_layers: int = 2,
        dropout: float = 0.01,
        use_batchnorm: bool = True,
    ) -> None:
        """
        Args:
            input_dim: size of input feature vector (104 for Yeh & Lin data)
            hidden_dim: width of hidden layers
            num_actions: number of possible actions (36 in Yeh & Lin)
            num_hidden_layers: how many hidden layers to use
            dropout: dropout probability (0.0 = no dropout)
            use_batchnorm: if True, apply BatchNorm1d after each linear layer
        """
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, input_dim]
        returns: tensor of shape [batch_size, num_actions]
        """
        return self.net(x)
