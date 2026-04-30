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

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPTNeoXConfig,
    LlamaConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Qwen2Config,
)

from .paths import hf_cache_dir


SublayerKind = Literal["attn", "ffn"]
WeightsMode = Literal["trained", "random_gaussian", "random_kaiming"]

# Regex patterns for custom model names:
#   tiny-gpt2-4l-256d  -> 4 layers, 256 hidden dim
#   micro-gpt2-6l-384d -> 6 layers, 384 hidden dim
#   nano-gpt2-4l-128d  -> 4 layers, 128 hidden dim
_CUSTOM_MODEL_RE = re.compile(
    r"^(?P<prefix>tiny|micro|nano)-gpt2-(?P<layers>\d+)l-(?P<hidden>\d+)d$"
)
# Custom LLaMA pattern: llama-{L}l-{N}d-{M}m  (L = layers, N = d_model, M = manifold_dim)
_CUSTOM_LLAMA_RE = re.compile(
    r"^llama-(?P<layers>\d+)l-(?P<hidden>\d+)d-(?P<manifold>\d+)m$"
)
# Custom Qwen pattern: qwen-{L}l-{N}d
_CUSTOM_QWEN_RE = re.compile(
    r"^qwen-(?P<layers>\d+)l-(?P<hidden>\d+)d$"
)
# Custom Pythia pattern: pythia-{L}l-{N}d
_CUSTOM_PYTHIA_RE = re.compile(
    r"^pythia-(?P<layers>\d+)l-(?P<hidden>\d+)d$"
)


def _select_num_heads(hidden_size: int, preferred: tuple[int, ...] = (8, 4, 2, 1)) -> int:
    """Pick a head count whose head dimension is even."""
    for num_heads in preferred:
        if hidden_size % num_heads == 0 and (hidden_size // num_heads) % 2 == 0:
            return num_heads
    for num_heads in range(min(hidden_size, 32), 0, -1):
        if hidden_size % num_heads == 0 and (hidden_size // num_heads) % 2 == 0:
            return num_heads
    raise ValueError(f"cannot choose an even head dimension for hidden_size={hidden_size}")


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
    "llama": ArchSpec(
        family="llama",
        blocks_template="model.layers.{n}",
        attn_norm="input_layernorm",
        attn_module="self_attn",
        ffn_norm="post_attention_layernorm",
        ffn_module="mlp",
    ),
    "qwen": ArchSpec(
        family="qwen",
        blocks_template="model.layers.{n}",
        attn_norm="input_layernorm",
        attn_module="self_attn",
        ffn_norm="post_attention_layernorm",
        ffn_module="mlp",
    ),
    "pythia": ArchSpec(
        family="pythia",
        blocks_template="gpt_neox.layers.{n}",
        attn_norm="input_layernorm",
        attn_module="attention",
        ffn_norm="post_attention_layernorm",
        ffn_module="mlp",
    ),
}


def _detect_arch(model_name: str) -> str:
    n = model_name.lower()
    if "gpt2" in n or n == "distilgpt2":
        return "gpt2"
    if _CUSTOM_MODEL_RE.match(n):
        return "gpt2"
    if _CUSTOM_LLAMA_RE.match(n):
        return "llama"
    if "qwen" in n:
        return "qwen"
    if _CUSTOM_PYTHIA_RE.match(n):
        return "pythia"
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


def _make_llama_attn_phi(block: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """phi(h) = h + block.self_attn(block.input_layernorm(h)) for a LLaMA decoder layer."""

    def phi(h: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h3 = h.unsqueeze(0)  # (1, T, d)
        normed = block.input_layernorm(h3)
        # Generate position_ids for the sequence length
        T = h3.shape[1]
        position_ids = torch.arange(T, device=h3.device).unsqueeze(0)  # (1, T)
        # Compute rotary position embeddings using the attached rotary_emb
        # (attached in load_model for LLaMA models)
        rotary = block.rotary_emb
        cos, sin = rotary(normed, position_ids=position_ids)
        attn_out = block.self_attn(normed, position_embeddings=(cos, sin))
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return (h3 + attn_out).squeeze(0)

    return phi


def _make_llama_ffn_phi(block: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """phi(h) = h + block.mlp(block.post_attention_layernorm(h)) for a LLaMA decoder layer."""

    def phi(h: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h3 = h.unsqueeze(0)
        normed = block.post_attention_layernorm(h3)
        ffn_out = block.mlp(normed)
        return (h3 + ffn_out).squeeze(0)

    return phi


def _make_rotary_attn_phi(block: nn.Module, *, attn_module_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """phi(h) = h + attention(norm(h)) for RoPE-based decoder layers."""

    def phi(h: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h3 = h.unsqueeze(0)
        normed = block.input_layernorm(h3)
        position_ids = torch.arange(h3.shape[1], device=h3.device).unsqueeze(0)
        cos, sin = block.rotary_emb(normed, position_ids=position_ids)
        attn_module = getattr(block, attn_module_name)
        attn_out = attn_module(
            normed,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return (h3 + attn_out).squeeze(0)

    return phi


def _make_rotary_ffn_phi(block: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """phi(h) = h + block.mlp(block.post_attention_layernorm(h)) for RoPE families."""

    def phi(h: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h3 = h.unsqueeze(0)
        normed = block.post_attention_layernorm(h3)
        ffn_out = block.mlp(normed)
        return (h3 + ffn_out).squeeze(0)

    return phi


_PHI_BUILDERS: dict[tuple[str, SublayerKind], Callable[[nn.Module], Callable]] = {
    ("gpt2", "attn"): _make_gpt2_attn_phi,
    ("gpt2", "ffn"): _make_gpt2_ffn_phi,
    ("llama", "attn"): _make_llama_attn_phi,
    ("llama", "ffn"): _make_llama_ffn_phi,
    ("qwen", "attn"): lambda block: _make_rotary_attn_phi(block, attn_module_name="self_attn"),
    ("qwen", "ffn"): _make_rotary_ffn_phi,
    ("pythia", "attn"): lambda block: _make_rotary_attn_phi(block, attn_module_name="attention"),
    ("pythia", "ffn"): _make_rotary_ffn_phi,
}

# ---------------------------------------------------------------------------
# Custom tiny GPT-2 factory
# ---------------------------------------------------------------------------


def _build_custom_gpt2_config(model_name: str) -> GPT2Config:
    """Parse tiny/micro/nano-gpt2-{L}l-{D}d names and return a GPT2Config."""
    m = _CUSTOM_MODEL_RE.match(model_name)
    if m is None:
        raise ValueError(f"custom model name does not match pattern: {model_name!r}")

    n_layer = int(m.group("layers"))
    n_embd = int(m.group("hidden"))
    n_head = max(1, n_embd // 64)  # keep head size ~64
    n_inner = n_embd * 4

    cfg = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,  # will be overridden later
    )
    return cfg


def _build_custom_llama_config(model_name: str) -> LlamaConfig:
    """Parse llama-{L}l-{N}d-{M}m names and return a LlamaConfig."""
    m = _CUSTOM_LLAMA_RE.match(model_name)
    if m is None:
        raise ValueError(f"custom llama model name does not match pattern: {model_name!r}")
    n_layers = int(m.group("layers"))
    d_model = int(m.group("hidden"))
    # manifold_dim = int(m.group("manifold"))  # not used in model architecture

    # Choose num_attention_heads to ensure even head_dim for RoPE compatibility.
    # RoPE assumes head_dim is even. Strategy: prefer 4 heads if d_model divisible by 8
    # (=> head_dim = d_model/4 is even). Otherwise use 2 heads (=> head_dim = d_model/2,
    # which is even if d_model divisible by 4). For other cases, fall back to 2 or 1.
    n_heads = None
    for h in (4, 2):
        if d_model % h == 0:
            head_dim = d_model // h
            if head_dim % 2 == 0:
                n_heads = h
                break
    if n_heads is None:
        # Last resort: use 2 heads if d_model even, else 1 head (may cause RoPE issues)
        n_heads = 2 if d_model % 2 == 0 else 1

    cfg = LlamaConfig(
        hidden_size=d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=4 * d_model,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        vocab_size=50257,  # align with GPT2 tokenizer used for custom models
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return cfg


def _build_custom_qwen_config(model_name: str) -> Qwen2Config:
    """Parse qwen-{L}l-{N}d names and return a Qwen2Config."""
    m = _CUSTOM_QWEN_RE.match(model_name)
    if m is None:
        raise ValueError(f"custom qwen model name does not match pattern: {model_name!r}")
    n_layers = int(m.group("layers"))
    d_model = int(m.group("hidden"))
    n_heads = _select_num_heads(d_model)
    n_kv_heads = max(1, n_heads // 2)
    return Qwen2Config(
        hidden_size=d_model,
        intermediate_size=4 * d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        vocab_size=50257,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def _build_custom_pythia_config(model_name: str) -> GPTNeoXConfig:
    """Parse pythia-{L}l-{N}d names and return a GPTNeoXConfig."""
    m = _CUSTOM_PYTHIA_RE.match(model_name)
    if m is None:
        raise ValueError(f"custom pythia model name does not match pattern: {model_name!r}")
    n_layers = int(m.group("layers"))
    d_model = int(m.group("hidden"))
    n_heads = _select_num_heads(d_model)
    return GPTNeoXConfig(
        hidden_size=d_model,
        intermediate_size=4 * d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        max_position_embeddings=2048,
        rotary_pct=1.0,
        vocab_size=50257,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )


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

    Custom models: names matching patterns like "tiny-gpt2-4l-256d" or
    "micro-gpt2-6l-384d" are generated from scratch and ONLY support
    random-weighted init (weights="random_gaussian" or "random_kaiming").
    """
    cache_dir = str(hf_cache_dir())
    arch = _detect_arch(name)
    is_custom = (
        (_CUSTOM_MODEL_RE.match(name) is not None)
        or (_CUSTOM_LLAMA_RE.match(name) is not None)
        or (_CUSTOM_QWEN_RE.match(name) is not None)
        or (_CUSTOM_PYTHIA_RE.match(name) is not None)
    )

    # Tokenizer: custom models reuse GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2" if is_custom else name, cache_dir=cache_dir
    )

    if is_custom:
        if weights == "trained":
            raise ValueError(
                f"custom model {name!r} does not support weights='trained'; "
                f"use 'random_gaussian' or 'random_kaiming'"
            )
        if _CUSTOM_MODEL_RE.match(name):
            cfg = _build_custom_gpt2_config(name)
        elif _CUSTOM_LLAMA_RE.match(name):
            cfg = _build_custom_llama_config(name)
        elif _CUSTOM_QWEN_RE.match(name):
            cfg = _build_custom_qwen_config(name)
        else:
            cfg = _build_custom_pythia_config(name)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, cache_dir=cache_dir
        )

    # Weight re-initialisation
    if weights == "random_gaussian":
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=dtype) * 0.02)
    elif weights == "random_kaiming":
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    nn.init.kaiming_uniform_(p, a=5**0.5)
                else:
                    nn.init.zeros_(p)
    elif weights == "trained" and is_custom:
        raise ValueError(f"custom models require random init weights; got {weights!r}")
    elif weights != "trained":
        raise ValueError(f"Unknown weights mode: {weights!r}")

    model = model.to(device).eval()

    # Ensure rotary_emb exists for RoPE-based families.
    if arch == "qwen" and not hasattr(model.model, 'rotary_emb'):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        config = model.model.config
        model.model.rotary_emb = Qwen2RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )

    # Attach rotary_emb to each decoder layer for RoPE-based families.
    if arch in {"llama", "qwen"}:
        for layer in model.model.layers:
            layer.rotary_emb = model.model.rotary_emb
    elif arch == "pythia":
        for layer in model.gpt_neox.layers:
            layer.rotary_emb = model.gpt_neox.rotary_emb

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
