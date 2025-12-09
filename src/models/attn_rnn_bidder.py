"""
Attention plus RNN bidder with suit based tokens.

Input encoding follows Yeh et al 2016 and related work:
- x: [batch, 104]
- First 52 entries: hand of player 1
- Second 52 entries: hand of player 2
- Within each 52, order of cards is:
    {♠2, ♠3, ... ♠A, ♥2, ... ♥A, ♦2, ... ♦A, ♣2, ... ♣A}

We build 8 tokens per deal:
- Player 1: spades, hearts, diamonds, clubs (each length 13)
- Player 2: spades, hearts, diamonds, clubs (each length 13)

So tokens have shape [batch, 8, 13]. We then:
- Project each token from 13 to d_model
- Apply multi head self attention across the 8 tokens
- Apply residual plus layer norm
- Run a GRU over the 8 token sequence
- Use the final GRU hidden state
- MLP head to 36 logits
"""

from typing import Tuple

import torch
from torch import nn


class AttnRnnBidder(nn.Module):
    def __init__(
        self,
        input_dim: int = 104,
        num_actions: int = 36,
        num_players: int = 2,
        suits_per_player: int = 4,
        cards_per_suit: int = 13,
        d_model: int = 128,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_players = num_players
        self.suits_per_player = suits_per_player
        self.cards_per_suit = cards_per_suit

        total_tokens = num_players * suits_per_player  # 2 * 4 = 8
        token_dim = cards_per_suit                     # 13

        expected_input_dim = num_players * 52
        assert (
            input_dim == expected_input_dim
        ), f"Expected input_dim {expected_input_dim} (two 52 card hands), got {input_dim}"

        self.num_tokens = total_tokens
        self.token_dim = token_dim

        # Project each suit token from 13 to d_model
        self.token_proj = nn.Linear(token_dim, d_model)

        # Self attention over 8 tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # GRU over the token sequence
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # MLP head from final hidden state to logits
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, num_actions),
        )

    def _split_into_suit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 104]
        Returns tokens: [batch, 8, 13] for 2 players * 4 suits * 13 cards.
        """
        batch_size, feat_dim = x.shape
        assert feat_dim == self.num_players * 52

        # Reshape into [batch, num_players, 52]
        hands = x.view(batch_size, self.num_players, 52)

        tokens = []

        for p in range(self.num_players):
            # cards_p: [batch, 52]
            cards_p = hands[:, p, :]

            # Suits in order: spades, hearts, diamonds, clubs
            spades = cards_p[:, 0:13]
            hearts = cards_p[:, 13:26]
            diamonds = cards_p[:, 26:39]
            clubs = cards_p[:, 39:52]

            tokens.extend([spades, hearts, diamonds, clubs])

        # tokens is a list of 8 tensors [batch, 13]
        # Stack into [batch, 8, 13]
        tokens_tensor = torch.stack(tokens, dim=1)
        return tokens_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 104]
        returns: [batch, num_actions] logits
        """
        # [batch, 8, 13]
        suit_tokens = self._split_into_suit_tokens(x)

        # Project tokens and apply attention
        tokens_proj = torch.relu(self.token_proj(suit_tokens))  # [batch, 8, d_model]
        attn_out, _ = self.attn(tokens_proj, tokens_proj, tokens_proj)  # [batch, 8, d_model]
        attn_out = self.attn_norm(attn_out + tokens_proj)              # residual plus norm

        # GRU over the 8 token sequence
        rnn_out, h_n = self.rnn(attn_out)  # h_n: [num_layers, batch, rnn_hidden]

        # Use last layer hidden state as summary
        final_state = h_n[-1]              # [batch, rnn_hidden]

        # MLP head
        logits = self.head(final_state)    # [batch, num_actions]
        return logits
