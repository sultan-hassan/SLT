"""
Small decoder-style transformer for modular arithmetic.
Architecture follows the grokking setup (Power et al. 2022), adapted for SLT analysis.
"""
import torch
import torch.nn as nn
import math


class ModularTransformer(nn.Module):
    """
    Predicts (a + b) mod p given tokens a, b.
    Uses a 3-position sequence [a, b, =], predicting from the = readout slot.
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.d_model = d_model

        self.embed = nn.Embedding(p + 1, d_model)          # +1 for the "=" readout token
        self.pos_embed = nn.Parameter(torch.zeros(3, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
            norm_first=True,                               # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.data, std=0.02)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b : (B,) integer tensors in [0, p)
        returns : (B, p) logits
        """
        B = a.shape[0]
        readout_id = torch.full((B,), self.p, dtype=torch.long, device=a.device)

        # Build sequence [a, b, =]
        ids = torch.stack([a, b, readout_id], dim=1)       # (B, 3)
        x = self.embed(ids) + self.pos_embed               # (B, 3, D)
        h = self.transformer(x)                            # (B, 3, D)
        return self.head(self.norm(h[:, -1, :]))           # predict from readout


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
