"""Shared test fixtures: toy self-contained sublayers (no HF dependency)
and a full-Jacobian oracle via torch.autograd.functional.jacobian.

The toy sublayers are intentionally small (d <= 8) and tightly reproducible
via explicit seeds so the tests run fast and are deterministic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Toy sublayers — same shape as real GPT-2 sublayers (h -> h + g(h)) but small
# ---------------------------------------------------------------------------

class ToyAttnSublayer(nn.Module):
    """phi(h) = h + W_O softmax_causal(Q K^T / sqrt(d)) V,  with LN in front."""

    def __init__(self, d: int, *, eps: float = 1e-5, seed: int = 0) -> None:
        super().__init__()
        self.d = d
        g = torch.Generator().manual_seed(seed)
        self.ln = nn.LayerNorm(d, eps=eps)
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        T, d = h.shape
        h_norm = self.ln(h)
        q = self.w_q(h_norm)
        k = self.w_k(h_norm)
        v = self.w_v(h_norm)
        scores = q @ k.T / (d**0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = self.w_o(attn @ v)
        return h + out


class ToyFFNSublayer(nn.Module):
    """phi(h) = h + W_2 gelu(W_1 LN(h))  (token-wise)."""

    def __init__(self, d: int, d_ffn: int = 8, *, eps: float = 1e-5, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.ln = nn.LayerNorm(d, eps=eps)
        self.w_1 = nn.Linear(d, d_ffn, bias=False)
        self.w_2 = nn.Linear(d_ffn, d, bias=False)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.ln(h)
        hidden = torch.nn.functional.gelu(self.w_1(h_norm))
        out = self.w_2(hidden)
        return h + out


# ---------------------------------------------------------------------------
# Oracle: full (T, d, T, d) Jacobian via autograd — used only in tests
# ---------------------------------------------------------------------------

def full_jacobian_oracle(phi: nn.Module, h: torch.Tensor) -> torch.Tensor:
    h = h.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(phi, h, create_graph=False, vectorize=True)
