"""Model loading, sublayer access, and basic inference for causal LMs.

Single module covering:
  - `Model` dataclass wrapping model + tokenizer + arch id
  - `load_model(name, weights, seed, device, dtype)` factory
  - config queries: `n_blocks`, `hidden_size`, `block_path`, `get_block`
  - `Sublayer` dataclass + `sublayer(m, block_idx, kind)` returning hook_path + phi closure
  - `tokenize`, `predict_next_token`, `generate` — thin wrappers over HF, kept for sanity checks

Architecture support via ARCH_SPECS; register new families by extending the dict
and providing phi-closure builders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .paths import hf_cache_dir


SublayerKind = Literal["attn", "ffn"]
WeightsMode = Literal["trained", "random_gaussian", "random_kaiming"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Loaded causal LM: model + tokenizer + architecture tag."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    arch: str  # family key into ARCH_SPECS, e.g., "gpt2"


@dataclass
class Sublayer:
    """One residual sublayer: enough to hook its input and re-run it as phi."""

    block_idx: int
    kind: SublayerKind
    hook_path: str  # module path whose INPUT is the residual stream at sublayer start
    phi: Callable[[torch.Tensor], torch.Tensor]  # (T, d) -> (T, d), implements h -> h + g(h)


@dataclass
class ArchSpec:
    family: str
    blocks_template: str  # "transformer.h.{n}" for GPT-2
    attn_norm: str
    attn_module: str
    ffn_norm: str
    ffn_module: str


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCH_SPECS: dict[str, ArchSpec] = {
    "gpt2": ArchSpec(
        family="gpt2",
        blocks_template="transformer.h.{n}",
        attn_norm="ln_1",
        attn_module="attn",
        ffn_norm="ln_2",
        ffn_module="mlp",
    ),
    # Future:
    # "llama": ArchSpec("llama", "model.layers.{n}", "input_layernorm", "self_attn",
    #                   "post_attention_layernorm", "mlp"),
    # "pythia": ArchSpec("pythia", "gpt_neox.layers.{n}", "input_layernorm", "attention",
    #                    "post_attention_layernorm", "mlp"),
}


def _detect_arch(model_name: str) -> str:
    n = model_name.lower()
    if "gpt2" in n or n == "distilgpt2":
        return "gpt2"
    raise ValueError(
        f"No ArchSpec registered for model_name={model_name!r}. "
        f"Extend ARCH_SPECS and _PHI_BUILDERS."
    )


# ---------------------------------------------------------------------------
# Per-architecture phi closure builders
# ---------------------------------------------------------------------------

def _make_gpt2_attn_phi(block: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """phi(h) = h + block.attn(block.ln_1(h)) for a GPT-2 block. Causal mask is
    handled inside GPT2Attention via its registered bias buffer."""

    def phi(h: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h3 = h.unsqueeze(0)  # (1, T, d)
        normed = block.ln_1(h3)
        attn_out = block.attn(normed, **kwargs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return (h3 + attn_out).squeeze(0)

    return phi


def _make_gpt2_ffn_phi(block: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """phi(h) = h + block.mlp(block.ln_2(h)) for a GPT-2 block."""

    def phi(h: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h3 = h.unsqueeze(0)
        normed = block.ln_2(h3)
        ffn_out = block.mlp(normed, **kwargs)
        return (h3 + ffn_out).squeeze(0)

    return phi


_PHI_BUILDERS: dict[tuple[str, SublayerKind], Callable[[nn.Module], Callable]] = {
    ("gpt2", "attn"): _make_gpt2_attn_phi,
    ("gpt2", "ffn"): _make_gpt2_ffn_phi,
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    name: str = "gpt2",
    *,
    weights: WeightsMode = "trained",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Model:
    """Load a HuggingFace causal LM + tokenizer as a `Model`.

    `weights`:
      - "trained": keep the pretrained weights.
      - "random_gaussian": re-init every parameter ~ N(0, 0.02^2) with `seed`.
      - "random_kaiming": re-init with Kaiming uniform (weights >= 2-D) and zeros (biases).
        WARNING: this mode is degenerate for Exp 1. Kaiming init is only defined
        for matrices, so all 1-D params go to zero — including LayerNorm scales
        (`ln_{1,2}.weight`). With LN scale = 0 and all biases = 0, every
        `ln(h)` is identically zero, so `g(h) = attn/mlp(0) = 0`, so
        phi(h) = h and the sublayer Jacobian is the identity for every input,
        layer, and token. Per-token metrics collapse to trivial constants
        (log|det| = 0, sigma_min = sigma_max = 1, kappa = 1) and the run
        carries no information about the Mityagin a.s.-invertibility claim.
        Use `random_gaussian` instead as the random-init control; keep this
        mode only as a sanity reference for "phi = id".
    """
    cache_dir = str(hf_cache_dir())
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, cache_dir=cache_dir)

    if weights == "random_gaussian":
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.02)
    elif weights == "random_kaiming":
        # NOTE: degenerate for Exp 1 — see load_model docstring. 1-D params
        # (biases AND LayerNorm scales) end up zero, which makes ln(h) == 0,
        # hence g(h) == 0, hence phi == id and J == I for every sublayer.
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    nn.init.kaiming_uniform_(p, a=5**0.5)
                else:
                    nn.init.zeros_(p)
    elif weights != "trained":
        raise ValueError(f"Unknown weights mode: {weights!r}")

    model = model.to(device).eval()
    arch = _detect_arch(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return Model(model=model, tokenizer=tokenizer, arch=arch)


# ---------------------------------------------------------------------------
# Config queries
# ---------------------------------------------------------------------------

def n_blocks(m: Model) -> int:
    cfg = m.model.config
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("cannot infer n_blocks from model config")


def n_sublayers(m: Model) -> int:
    """Pre-LN default: 2 sublayers per block (attn + ffn)."""
    return 2 * n_blocks(m)


def hidden_size(m: Model) -> int:
    cfg = m.model.config
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("cannot infer hidden_size from model config")


def block_path(m: Model, block_idx: int) -> str:
    return ARCH_SPECS[m.arch].blocks_template.format(n=block_idx)


def get_block(m: Model, block_idx: int) -> nn.Module:
    return m.model.get_submodule(block_path(m, block_idx))


# ---------------------------------------------------------------------------
# Sublayer access (the main per-experiment handle)
# ---------------------------------------------------------------------------

def sublayer(m: Model, block_idx: int, kind: SublayerKind) -> Sublayer:
    """Return a Sublayer handle for block `block_idx`, sublayer kind `kind`.

    The returned handle contains:
      - `hook_path`: module path whose INPUT is the residual stream at the
        start of this sublayer (i.e., before its norm).
      - `phi`: pure-function closure `(T, d) -> (T, d)` implementing
        h -> h + g(h) for exactly this sublayer. Safe to feed directly
        into `jacobian.build_jacobian`.
    """
    if not 0 <= block_idx < n_blocks(m):
        raise IndexError(f"block_idx={block_idx} not in [0, {n_blocks(m)})")
    spec = ARCH_SPECS[m.arch]
    block = get_block(m, block_idx)

    if kind == "attn":
        norm_name = spec.attn_norm
    elif kind == "ffn":
        norm_name = spec.ffn_norm
    else:
        raise ValueError(f"Unknown sublayer kind: {kind!r}")

    hook_path = f"{block_path(m, block_idx)}.{norm_name}"

    builder = _PHI_BUILDERS.get((spec.family, kind))
    if builder is None:
        raise NotImplementedError(f"No phi builder for ({spec.family}, {kind})")
    phi = builder(block)

    return Sublayer(block_idx=block_idx, kind=kind, hook_path=hook_path, phi=phi)


# ---------------------------------------------------------------------------
# Tokenization / prediction / generation (kept for sanity checks)
# ---------------------------------------------------------------------------

def _device_of(m: Model) -> torch.device:
    return next(m.model.parameters()).device


def tokenize(m: Model, text: str, max_tokens: int | None = None) -> torch.Tensor:
    """Return an (1, T) token-id tensor on the model device."""
    kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_tokens is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_tokens
    ids = m.tokenizer(text, **kwargs)["input_ids"]
    return ids.to(_device_of(m))


def predict_next_token(
    m: Model,
    text: str,
    *,
    top_k: int = 5,
    max_tokens: int | None = None,
) -> dict:
    """Top-k next-token prediction at the last position."""
    input_ids = tokenize(m, text, max_tokens=max_tokens)
    with torch.no_grad():
        out = m.model(input_ids=input_ids)
    logits = out.logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k=top_k)
    return {
        "token_ids": top.indices.tolist(),
        "tokens": [m.tokenizer.decode(int(i)) for i in top.indices],
        "probs": top.values.tolist(),
        "logits": logits[top.indices].tolist(),
        "n_input_tokens": int(input_ids.shape[1]),
    }


def generate(
    m: Model,
    text: str,
    *,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Return the decoded continuation."""
    input_ids = tokenize(m, text, max_tokens=max_tokens)
    with torch.no_grad():
        out = m.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=m.tokenizer.pad_token_id,
        )
    return m.tokenizer.decode(out[0], skip_special_tokens=True)
