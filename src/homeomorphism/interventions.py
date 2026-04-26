"""Intervention utilities for Section 2.1 baseline experiments.

This module centralizes baseline-specific model and input mutations so
experiments can share deterministic, padding-agnostic behavior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import torch

from . import models


BaselineName = Literal[
    "trained",
    "topological_initialisation",
    "maximum_entropy_injection",
    "syntactic_disintegration",
    "semantic_scrambling",
]

VALID_BASELINES: tuple[BaselineName, ...] = (
    "trained",
    "topological_initialisation",
    "maximum_entropy_injection",
    "syntactic_disintegration",
    "semantic_scrambling",
)


@dataclass
class PreparedInput:
    """Prepared model forward kwargs plus token IDs used for bookkeeping."""

    forward_kwargs: dict[str, torch.Tensor]
    token_ids: torch.Tensor


def derive_seed(root_seed: int, tag: str) -> int:
    payload = f"{root_seed}:{tag}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def disable_dropout(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0


def reset_norm_affine(model: torch.nn.Module) -> None:
    """Reset affine params of LayerNorm / RMSNorm-like modules only."""
    with torch.no_grad():
        for module in model.modules():
            name = module.__class__.__name__.lower()
            if "layernorm" not in name and "rmsnorm" not in name:
                continue
            w = getattr(module, "weight", None)
            b = getattr(module, "bias", None)
            if isinstance(w, torch.Tensor):
                w.fill_(1.0)
            if isinstance(b, torch.Tensor):
                b.zero_()


def load_model_for_baseline(
    *,
    model_name: str,
    weights: str,
    baseline: BaselineName,
    seed: int,
    device: str,
) -> models.Model:
    """Load and prepare a model according to baseline requirements."""
    load_weights = weights
    if baseline == "topological_initialisation" and weights == "trained":
        load_weights = "random_gaussian"
    m = models.load_model(model_name, weights=load_weights, seed=seed, device=device)  # type: ignore[arg-type]
    disable_dropout(m.model)
    if baseline == "topological_initialisation":
        reset_norm_affine(m.model)
    return m


def _apply_permutation(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    out = input_ids.clone()
    g = torch.Generator(device=input_ids.device).manual_seed(seed)
    for s in range(out.shape[0]):
        sel = attention_mask[s] == 1
        valid = out[s][sel]
        if valid.numel() <= 1:
            continue
        perm = torch.randperm(valid.numel(), generator=g, device=out.device)
        out[s][sel] = valid[perm]
    return out


def _apply_uniform_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    out = input_ids.clone()
    g = torch.Generator(device=input_ids.device).manual_seed(seed)
    for s in range(out.shape[0]):
        sel = attention_mask[s] == 1
        n = int(sel.sum().item())
        if n <= 0:
            continue
        sampled = torch.randint(1, vocab_size, (n,), generator=g, device=out.device)
        out[s][sel] = sampled
    return out


def gaussian_inputs_embeds(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate epsilon ~ N(0, (1/d)I) with shape (B, T, d)."""
    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn((batch_size, seq_len, d_model), generator=g, device=device)
    return eps / float(d_model**0.5)


def build_prepared_input(
    *,
    m: models.Model,
    text: str,
    max_tokens: int,
    baseline: BaselineName,
    root_seed: int,
    sample_index: int,
) -> PreparedInput:
    """Build baseline-mutated model input kwargs from raw text."""
    input_ids = models.tokenize(m, text, max_tokens=max_tokens)
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    token_ids = input_ids
    forward_kwargs: dict[str, torch.Tensor] = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if baseline == "trained" or baseline == "topological_initialisation":
        forward_kwargs["input_ids"] = input_ids
    elif baseline == "syntactic_disintegration":
        seed = derive_seed(root_seed, f"permute:{sample_index}")
        token_ids = _apply_permutation(input_ids, attention_mask, seed=seed)
        forward_kwargs["input_ids"] = token_ids
    elif baseline == "semantic_scrambling":
        seed = derive_seed(root_seed, f"uniform:{sample_index}")
        vocab_size = int(m.model.config.vocab_size)
        token_ids = _apply_uniform_tokens(input_ids, attention_mask, vocab_size=vocab_size, seed=seed)
        forward_kwargs["input_ids"] = token_ids
    elif baseline == "maximum_entropy_injection":
        seed = derive_seed(root_seed, f"noise:{sample_index}")
        d_model = models.hidden_size(m)
        eps = gaussian_inputs_embeds(
            batch_size=int(input_ids.shape[0]),
            seq_len=int(input_ids.shape[1]),
            d_model=d_model,
            seed=seed,
            device=input_ids.device,
        )
        forward_kwargs["inputs_embeds"] = eps
    else:
        raise ValueError(f"unknown baseline {baseline!r}")

    return PreparedInput(forward_kwargs=forward_kwargs, token_ids=token_ids)


def topological_init_calibration(
    *,
    m: models.Model,
    root_seed: int,
) -> float:
    """Return mean norm from a single noise-token calibration forward pass."""
    d_model = models.hidden_size(m)
    dev = next(m.model.parameters()).device
    eps = gaussian_inputs_embeds(
        batch_size=1,
        seq_len=1,
        d_model=d_model,
        seed=derive_seed(root_seed, "topological_init_calibration"),
        device=dev,
    )
    with torch.no_grad():
        out = m.model(inputs_embeds=eps)
    h = out.last_hidden_state
    return float(torch.linalg.vector_norm(h, dim=-1).mean().item())


def attach_metadata(base: dict[str, Any], **extra: Any) -> dict[str, Any]:
    out = dict(base)
    out.update(extra)
    return out
