"""Experiment 3: synthetic manifold experiments with toy and LLaMA-style models.

Generalizes exp3_llama_hyperplane.py to support multiple manifold types and models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn as nn

from homeomorphism.latents import LatentStore, LatentConfig
from homeomorphism.toy_transformer import (
    ToyTopologicalTransformer,
    HyperplaneSampler,
    sample_sphere,
    sample_hyperplane,
    sample_swiss_roll,
    sample_torus,
    sample_white_noise,
)

# ============================================================================
# Enums
# ============================================================================

class ManifoldType(Enum):
    HYPERPLANE = "hyperplane"
    SPHERE = "sphere"
    TORUS = "torus"
    SWISS_ROLL = "swiss_roll"
    WHITE_NOISE = "white_noise"

    def intrinsic_dim(self, ambient_dim: int, manifold_dim: int | None) -> int:
        """Return the ground-truth intrinsic dimension for this manifold."""
        match self:
            case ManifoldType.HYPERPLANE:
                if manifold_dim is None:
                    raise ValueError("hyperplane requires --manifold-dim")
                return manifold_dim
            case ManifoldType.SPHERE:
                return ambient_dim - 1
            case ManifoldType.TORUS:
                return 2
            case ManifoldType.SWISS_ROLL:
                return 2
            case ManifoldType.WHITE_NOISE:
                return ambient_dim

class ModelType(Enum):
    TOY = "toy"
    LLAMA = "llama"

# ============================================================================
# LLaMA model components (copied from exp3_llama_hyperplane.py)
# ============================================================================

@dataclass
class LlamaConfig:
    d_model: int = 20
    manifold_dim: int = 10
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 80
    seq_len: int = 8
    max_seq_len: int = 512
    init_std: float = 0.02
    seed: int = 0
    device: str = "cpu"

    @property
    def n_depths(self) -> int:
        return 2 * self.n_layers + 2

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

def _rope_freqs(d_head: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    half = d_head // 2
    theta = 1.0 / (base ** (torch.arange(0, half).float() / d_head))
    pos = torch.arange(max_seq_len).float()
    return torch.outer(pos, theta)

def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, T, H, d = x.shape
    d_rope = (d // 2) * 2
    x_rot, x_pass = x[..., :d_rope], x[..., d_rope:]
    half = d_rope // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    cos = freqs[:T].cos().unsqueeze(0).unsqueeze(2)
    sin = freqs[:T].sin().unsqueeze(0).unsqueeze(2)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
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
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class FFN(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
    def forward_with_states(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        post_attn = x + self.attn(self.attn_norm(x))
        post_ffn = post_attn + self.ffn(self.ffn_norm(post_attn))
        return post_attn, post_ffn

class LlamaHyperplane(nn.Module):
    def __init__(self, cfg: LlamaConfig):
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
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
    def forward_with_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        states = [x]
        for block in self.blocks:
            post_attn, post_ffn = block.forward_with_states(x)
            states.append(post_attn)
            states.append(post_ffn)
            x = post_ffn
        states.append(self.norm(x))
        return states
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Sampler wrappers for other manifolds (toy style, unbatched output)
# ============================================================================

class SphereSampler:
    def __init__(self, ambient_dim: int, radius: float, seed: int):
        self.ambient_dim = ambient_dim
        self.radius = radius
        self.seed = seed
    def sample(self, B: int, T: int) -> torch.Tensor:
        total = B * T
        g = torch.Generator().manual_seed(self.seed)
        x = torch.randn(total, self.ambient_dim, generator=g)
        x = x / x.norm(dim=1, keepdim=True) * self.radius
        return x  # (B*T, d)

class TorusSampler:
    def __init__(self, ambient_dim: int, seed: int):
        self.ambient_dim = ambient_dim
        self.seed = seed
    def sample(self, B: int, T: int) -> torch.Tensor:
        total = B * T
        g = torch.Generator().manual_seed(self.seed)
        theta = 2.0 * torch.pi * torch.rand(total, generator=g)
        phi = 2.0 * torch.pi * torch.rand(total, generator=g)
        major, minor = 2.0, 0.7
        x = (major + minor * torch.cos(phi)) * torch.cos(theta)
        y = (major + minor * torch.cos(phi)) * torch.sin(theta)
        z = minor * torch.sin(phi)
        pts = torch.stack([x, y, z], dim=1)
        embed = torch.randn(3, self.ambient_dim, generator=g)
        embed = embed / (embed.norm(dim=0, keepdim=True) + 1e-8)
        pts = pts @ embed
        pts = (pts - pts.mean(dim=0, keepdim=True)) / (pts.std(dim=0, keepdim=True) + 1e-6)
        return pts

class SwissRollSampler:
    def __init__(self, ambient_dim: int, hole: float = 0.1, seed: int = 0):
        self.ambient_dim = ambient_dim
        self.hole = hole
        self.seed = seed
    def sample(self, B: int, T: int) -> torch.Tensor:
        total = B * T
        g = torch.Generator().manual_seed(self.seed)
        t = 1.5 * torch.pi * (1 + 2 * torch.rand(total, generator=g))
        scale = 1.0 - self.hole
        x = scale * t.cos() * t
        y = scale * t.sin() * t
        z = t
        pts_r3 = torch.stack([x, y, z], dim=1)
        embed = torch.randn(3, self.ambient_dim, generator=g)
        embed = embed / (embed.norm(dim=0, keepdim=True) + 1e-8)
        pts = pts_r3 @ embed
        pts = (pts - pts.mean(dim=0, keepdim=True)) / (pts.std(dim=0, keepdim=True) + 1e-6)
        return pts

class WhiteNoiseSampler:
    def __init__(self, ambient_dim: int, seed: int):
        self.ambient_dim = ambient_dim
        self.seed = seed
    def sample(self, B: int, T: int) -> torch.Tensor:
        total = B * T
        g = torch.Generator().manual_seed(self.seed)
        return torch.randn(total, self.ambient_dim, generator=g)

# ============================================================================
# Model builders
# ============================================================================

def build_toy_model(
    d_model: int,
    n_layers: int,
    seq_len: int,
    n_heads: int | None = None,
    d_ff: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[ToyTopologicalTransformer, dict[str, Any]]:
    if n_heads is None:
        n_heads = max(4, d_model // 8)
    if d_ff is None:
        d_ff = 4 * d_model
    model = ToyTopologicalTransformer(
        d_model=d_model,
        seq_len=seq_len,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        causal=True,
    )
    model.reset_parameters_continuous(seed=seed, scale=0.05)
    model = model.to(device).eval()
    meta = {
        "d_model": d_model,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "n_depths": 2 * n_layers + 2,
        "type": "toy",
    }
    return model, meta

def build_llama_model(
    d_model: int,
    n_layers: int,
    seq_len: int,
    n_heads: int | None = None,
    d_ff: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[LlamaHyperplane, LlamaConfig]:
    if n_heads is None:
        n_heads = 4
    if d_ff is None:
        d_ff = 4 * d_model
    cfg = LlamaConfig(
        d_model=d_model,
        manifold_dim=None,  # not used in model itself
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        seq_len=seq_len,
        seed=seed,
        device=device,
    )
    model = LlamaHyperplane(cfg).to(device).eval()
    return model, cfg

# ============================================================================
# Latent collection
# ============================================================================

def collect_latents(
    model: nn.Module,
    sampler: Any,
    n_samples: int,
    batch_size: int,
    store: LatentStore,
    device: str,
) -> None:
    """Collect hidden states from model on sampler output and append to store."""
    seq_len = store._cfg.seq_len
    remaining = n_samples

    while remaining > 0:
        B = min(batch_size, remaining)
        x_unbatched = sampler.sample(B, seq_len)  # (B*T, d)
        x_unbatched = x_unbatched.to(device)

        # --- toy model path: unbatched (T, d) per call
        if hasattr(model, "batch_forward_with_states"):
            # reshape to (B, T, d) then call batch method
            x_batch = x_unbatched.view(B, seq_len, -1)
            states = model.batch_forward_with_states(x_batch)  # list[(B,T,d)]
        # --- llama model path: batched (B,T,d)
        elif hasattr(model, "forward_with_states"):
            x_batch = x_unbatched.view(B, seq_len, -1)
            with torch.no_grad():
                states = model.forward_with_states(x_batch)
        else:
            raise RuntimeError("Model lacks state-extraction method")

        store.append(states)
        remaining -= B
        print(f"  collected {n_samples - remaining}/{n_samples}", end="\r")

    print()

# ============================================================================
# Config parsing
# ============================================================================

def parse_model_config(model_config: str) -> tuple[ModelType, dict[str, Any]]:
    """Parse toy-{L}l-{D}d or llama-{L}l-{N}d-{M}m."""
    if model_config.startswith("toy-"):
        rest = model_config.removeprefix("toy-")
        parts = rest.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid toy config: {model_config}")
        L = int(parts[0].rstrip("l"))
        D = int(parts[1].rstrip("d"))
        return ModelType.TOY, {"n_layers": L, "d_model": D}
    elif model_config.startswith("llama-"):
        rest = model_config.removeprefix("llama-")
        parts = rest.split("-")
        if len(parts) != 3:
            raise ValueError(f"Invalid llama config: {model_config}")
        L = int(parts[0].rstrip("l"))
        N = int(parts[1].rstrip("d"))
        M = int(parts[2].rstrip("m"))
        return ModelType.LLAMA, {"n_layers": L, "d_model": N, "manifold_dim": M}
    else:
        raise ValueError(f"Unknown model-config: {model_config}")

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Exp 3: Synthetic manifold experiments (toy + LLaMA models)."
    )
    p.add_argument("--manifold-type", required=True,
                   choices=[e.value for e in ManifoldType])
    p.add_argument("--model-type", required=True,
                   choices=[e.value for e in ModelType])
    p.add_argument("--model-config",
                   help="Compact: toy-{L}l-{D}d or llama-{L}l-{N}d-{M}m")
    p.add_argument("--d-model", type=int, default=20,
                   help="Ambient/model dimension N")
    p.add_argument("--manifold-dim", type=int, default=None,
                   help="Intrinsic manifold dimension M (required for hyperplane; for llama implies config)")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", type=Path, default=None,
                   help="HDF5 output path (appends if exists)")
    args = p.parse_args()

    # Parse model-type and config
    model_type = ModelType(args.model_type)
    if args.model_config:
        parsed_type, cfg_dict = parse_model_config(args.model_config)
        if parsed_type != model_type:
            raise ValueError(f"model-type mismatch: {model_type} vs {parsed_type}")
        args.d_model = cfg_dict.get("d_model", args.d_model)
        args.n_layers = cfg_dict.get("n_layers", args.n_layers)
        if "manifold_dim" in cfg_dict:
            args.manifold_dim = cfg_dict["manifold_dim"]
    else:
        if model_type == ModelType.LLAMA and args.manifold_dim is None:
            raise ValueError("--manifold-dim required for llama (or use --model-config)")

    manifold = ManifoldType(args.manifold_type)
    # For hyperplane, default manifold_dim to half of d_model if not specified
    if manifold is ManifoldType.HYPERPLANE and args.manifold_dim is None:
        args.manifold_dim = args.d_model // 2
        print(f"[INFO] Defaulting --manifold-dim to {args.manifold_dim} for hyperplane")
    intrinsic_dim = manifold.intrinsic_dim(args.d_model, args.manifold_dim)

    # Seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Build model
    if model_type == ModelType.TOY:
        model, meta = build_toy_model(
            d_model=args.d_model,
            n_layers=args.n_layers,
            seq_len=args.seq_len,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            seed=args.seed,
            device=args.device,
        )
        d_model = meta["d_model"]
        n_depths = meta["n_depths"]
    else:
        model, llama_cfg = build_llama_model(
            d_model=args.d_model,
            n_layers=args.n_layers,
            seq_len=args.seq_len,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            seed=args.seed,
            device=args.device,
        )
        d_model = llama_cfg.d_model
        n_depths = llama_cfg.n_depths

    # Build sampler
    sampler: Any
    match manifold:
        case ManifoldType.HYPERPLANE:
            sampler = HyperplaneSampler(
                N=args.d_model,
                M=args.manifold_dim,
                seed=args.seed,
                device=args.device,
            )
        case ManifoldType.SPHERE:
            sampler = SphereSampler(
                ambient_dim=args.d_model,
                radius=1.0,
                seed=args.seed,
            )
        case ManifoldType.TORUS:
            sampler = TorusSampler(
                ambient_dim=args.d_model,
                seed=args.seed,
            )
        case ManifoldType.SWISS_ROLL:
            sampler = SwissRollSampler(
                ambient_dim=args.d_model,
                seed=args.seed,
            )
        case ManifoldType.WHITE_NOISE:
            sampler = WhiteNoiseSampler(
                ambient_dim=args.d_model,
                seed=args.seed,
            )

    # Print configuration
    print(f"\n=== Synthetic Manifold Experiment ===")
    print(f"  Manifold   : {manifold.value}  (intrinsic dim = {intrinsic_dim})")
    print(f"  Model      : {model_type.value}  ({args.n_layers} layers, d={d_model}, depths={n_depths})")
    print(f"  Samples    : {args.n_samples}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Device     : {args.device}")
    print()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        latent_cfg = LatentConfig.from_fields(
            d_model=d_model,
            seq_len=args.seq_len,
            n_layers=args.n_layers,
            manifold_dim=intrinsic_dim,
            manifold_type=manifold.value,
        )
        with LatentStore.open(args.save, latent_cfg) as store:
            collect_latents(model, sampler, args.n_samples, args.batch_size, store, args.device)
            final = store.n_samples()
        print(f"Saved {final} samples → {args.save}")
    else:
        # Quick demo: one batch, print shapes
        x_unbatched = sampler.sample(2, args.seq_len).to(args.device)
        x_batch = x_unbatched.view(2, args.seq_len, -1)
        with torch.no_grad():
            if hasattr(model, "forward_with_states"):
                states = model.forward_with_states(x_batch)
            else:
                states = model.forward_with_states(x_batch)  # llama also uses same
        print("Forward pass shapes (B=2):")
        for d, s in enumerate(states):
            print(f"  depth {d:2d} : {tuple(s.shape)}")

if __name__ == "__main__":
    main()
