"""Experiment 3: tiny LLaMA-2-style decoder on hyperplane-sampled inputs.

Architecture
------------
  LLaMA-2 decoder (RMSNorm + RoPE + causal MHA + GELU FFN), no token embedding.
  Width N = d_model = 20 (deliberately tiny).

Input manifold
--------------
  A fixed random orthonormal basis A ∈ R^{N×M} spans an M-dimensional linear
  subspace of R^N  (M < N).  Each token x_t = A @ z_t, where z_t ~ N(0, I_M),
  so every sequence lies on the same flat M-dim hyperplane.

Weight init
-----------
  All weight matrices: N(0, init_std²).  RMSNorm scale parameters: ones.

Depth convention (hidden-state indexing)
----------------------------------------
  depth 0          : raw hyperplane input (before any block)
  depth 2b + 1     : residual stream after block b's attention sublayer
  depth 2b + 2     : residual stream after block b's FFN sublayer
  depth 2*L + 1    : after the final RMSNorm (feeds into the head)

  For n_layers=2 there are 6 depths: 0, 1, 2, 3, 4, 5.

Latent storage (HDF5)
---------------------
  Each depth gets its own resizable dataset `/depth_NN` of shape (N, T, d_model).
  Append more samples at any time; pick depth d and token t with a simple slice:

      import h5py
      with h5py.File("latents.h5", "r") as f:
          cloud = f["depth_02"][:, 3, :]   # all samples, depth 2, token 3

Run
---
    # quick forward-pass check (no saving)
    uv run python -m experiments.exp3_llama_hyperplane

    # sample 256 sequences and save latents (appends if file exists)
    uv run python -m experiments.exp3_llama_hyperplane \\
        --n-samples 256 --batch-size 32 --save latents.h5

    # custom arch
    uv run python -m experiments.exp3_llama_hyperplane \\
        --d-model 20 --manifold-dim 8 --n-layers 4 --seq-len 16 \\
        --n-samples 512 --save latents.h5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from homeomorphism.latents import LatentStore, LatentConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    d_model: int = 20        # N — model width, also the ambient embedding dim
    manifold_dim: int = 10   # M — intrinsic dimension of the input hyperplane (M < N)
    n_heads: int = 4         # attention heads; d_model must be divisible by n_heads
    n_layers: int = 2
    d_ff: int = 80           # FFN hidden width (4 × d_model by default)
    seq_len: int = 8         # T — number of tokens per sequence
    max_seq_len: int = 512   # RoPE cache length
    init_std: float = 0.02
    seed: int = 0
    device: str = "cpu"

    @property
    def n_depths(self) -> int:
        return 2 * self.n_layers + 2  # input + 2 per block + post-norm


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


# ---------------------------------------------------------------------------
# Rotary positional embeddings (RoPE)
# ---------------------------------------------------------------------------

def _rope_freqs(d_head: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    half = d_head // 2
    theta = 1.0 / (base ** (torch.arange(0, half).float() / d_head))
    pos = torch.arange(max_seq_len).float()
    return torch.outer(pos, theta)  # (max_seq_len, half)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, d_head).  Handles odd d_head by leaving the last dim unrotated."""
    B, T, H, d = x.shape
    d_rope = (d // 2) * 2
    x_rot, x_pass = x[..., :d_rope], x[..., d_rope:]
    half = d_rope // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    cos = freqs[:T].cos().unsqueeze(0).unsqueeze(2)  # (1, T, 1, half)
    sin = freqs[:T].sin().unsqueeze(0).unsqueeze(2)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        freqs = _rope_freqs(self.d_head, cfg.max_seq_len)
        self.register_buffer("rope_freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, d = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, d)
        k = self.k_proj(x).view(B, T, H, d)
        v = self.v_proj(x).view(B, T, H, d)

        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# FFN with GELU
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


# ---------------------------------------------------------------------------
# Transformer block  (pre-norm, LLaMA-2 style)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_with_states(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (post_attn, post_ffn, output) where output == post_ffn."""
        post_attn = x + self.attn(self.attn_norm(x))
        post_ffn = post_attn + self.ffn(self.ffn_norm(post_attn))
        return post_attn, post_ffn


# ---------------------------------------------------------------------------
# Full decoder (no embedding matrix)
# ---------------------------------------------------------------------------

class LlamaHyperplane(nn.Module):
    """LLaMA-2 decoder that takes continuous R^N inputs instead of token ids."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N).  Returns (B, T, N)."""
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def forward_with_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return one tensor per depth (see module docstring for convention).

        Returns list of length n_depths = 2*n_layers + 2:
          states[0]            : raw input
          states[2b + 1]       : post-attn of block b
          states[2b + 2]       : post-ffn of block b
          states[2*n_layers+1] : after final RMSNorm
        """
        states: list[torch.Tensor] = [x]
        for block in self.blocks:
            post_attn, post_ffn = block.forward_with_states(x)
            states.append(post_attn)
            states.append(post_ffn)
            x = post_ffn
        states.append(self.norm(x))
        return states

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Hyperplane sampler
# ---------------------------------------------------------------------------

class HyperplaneSampler:
    """Samples points from the M-dimensional linear subspace of R^N.

    The subspace is fixed by a random orthonormal basis A ∈ R^{N×M}
    (columns of a QR-factored Gaussian matrix).  Each call samples
    z ~ N(0, I_M) per token and returns x = A @ z ∈ R^N.
    """

    def __init__(self, N: int, M: int, seed: int = 0, device: str = "cpu"):
        if M >= N:
            raise ValueError(f"manifold_dim M={M} must be strictly less than model width N={N}")
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        raw = torch.randn(N, M, generator=gen, device=device)
        Q, _ = torch.linalg.qr(raw)
        self.A = Q          # (N, M) orthonormal columns
        self.N = N
        self.M = M
        self.device = device

    def sample(self, B: int, T: int) -> torch.Tensor:
        """Return (B, T, N) float32 tensor; every token lies on the hyperplane."""
        z = torch.randn(B, T, self.M, device=self.device)
        return z @ self.A.T


# ---------------------------------------------------------------------------
# HDF5 latent store
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Sampling + saving
# ---------------------------------------------------------------------------

def collect_latents(
    model: LlamaHyperplane,
    sampler: HyperplaneSampler,
    n_samples: int,
    batch_size: int,
    store: LatentStore | None = None,
) -> list[torch.Tensor] | None:
    """Run the model on `n_samples` hyperplane sequences and collect hidden states.

    If `store` is given, appends each batch to the HDF5 file and returns None.
    Otherwise accumulates everything in memory and returns a list of
    (n_samples, T, d_model) tensors, one per depth.
    """
    cfg = model.cfg
    accumulated: list[list[np.ndarray]] | None = None if store else [[] for _ in range(cfg.n_depths)]
    remaining = n_samples

    while remaining > 0:
        B = min(batch_size, remaining)
        x = sampler.sample(B, cfg.seq_len)
        with torch.no_grad():
            states = model.forward_with_states(x)

        if store is not None:
            store.append(states)
        else:
            for d, h in enumerate(states):
                accumulated[d].append(h.cpu().float().numpy())

        remaining -= B
        print(f"  collected {n_samples - remaining}/{n_samples} samples", end="\r")

    print()

    if accumulated is not None:
        return [torch.from_numpy(np.concatenate(chunks, axis=0)) for chunks in accumulated]
    return None


# ---------------------------------------------------------------------------
# Demo / entry point
# ---------------------------------------------------------------------------

def run(cfg: Config, n_samples: int, batch_size: int, save_path: Path | None) -> None:
    torch.manual_seed(cfg.seed)

    sampler = HyperplaneSampler(N=cfg.d_model, M=cfg.manifold_dim, seed=cfg.seed, device=cfg.device)
    model = LlamaHyperplane(cfg).to(cfg.device)

    _x_probe = sampler.sample(B=1, T=1)
    print(f"Input vector size : {_x_probe.shape[-1]}  (must equal d_model={cfg.d_model}, NOT manifold_dim={cfg.manifold_dim})")
    assert _x_probe.shape[-1] == cfg.d_model, (
        f"BUG: sampled vector has dim {_x_probe.shape[-1]}, expected d_model={cfg.d_model}"
    )

    print(
        f"Model : {cfg.n_layers} layers, d={cfg.d_model}, heads={cfg.n_heads}, "
        f"d_ff={cfg.d_ff}, {model.n_params():,} params"
    )
    print(f"Input : M={cfg.manifold_dim}-dim hyperplane in R^{cfg.d_model}, T={cfg.seq_len}")
    print(f"Depths: {cfg.n_depths}  (0=input, 1..{cfg.n_depths - 2}=blocks, {cfg.n_depths - 1}=post-norm)")

    if save_path is not None:
        print(f"Saving: {save_path}  ({n_samples} samples, batch={batch_size})")
        latent_cfg = LatentConfig.from_exp3_config(cfg)
        with LatentStore.open(save_path, latent_cfg) as store:
            collect_latents(model, sampler, n_samples, batch_size, store=store)
            final_n = store.n_samples()
        print(f"Done.  {final_n} total samples in {save_path}")
        _print_access_hint(save_path, cfg)
    else:
        # Quick sanity run: one batch, no saving
        x = sampler.sample(B=2, T=cfg.seq_len)
        with torch.no_grad():
            states = model.forward_with_states(x)
        print(f"\nForward pass shapes (B=2):")
        for d, h in enumerate(states):
            print(f"  depth {d:2d} : {tuple(h.shape)}")


def _print_access_hint(path: Path, cfg: Config) -> None:
    print(f"""
Load with:
    import h5py
    with h5py.File("{path}", "r") as f:
        # all samples at depth 2, token 3  → shape (N, {cfg.d_model})
        cloud = f["depth_02"][:, 3, :]

        # full residual stream at depth 0  → shape (N, {cfg.seq_len}, {cfg.d_model})
        layer = f["depth_00"][:]

        # config used to generate the file
        import json; cfg = json.loads(f.attrs["config"])
""")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 3: LLaMA-2 decoder on hyperplane-sampled inputs."
    )
    p.add_argument("--d-model", type=int, default=20)
    p.add_argument("--manifold-dim", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=80)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=0,
                   help="number of sequences to collect (0 = quick demo only)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save", type=Path, default=None, metavar="PATH",
                   help="HDF5 output path; appends if file already exists")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = Config(
        d_model=args.d_model,
        manifold_dim=args.manifold_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        seed=args.seed,
        device=args.device,
    )
    n_samples = args.n_samples
    save_path = args.save

    if save_path is not None and n_samples == 0:
        n_samples = 256  # sensible default when --save is given

    run(cfg, n_samples=n_samples, batch_size=args.batch_size, save_path=save_path)


if __name__ == "__main__":
    main()
