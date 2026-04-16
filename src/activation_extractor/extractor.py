"""Lightweight activation extraction for transformer models.

Extracts residual-stream activations at specified layers and token positions
using plain PyTorch forward hooks. No nnsight dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration & data containers
# ---------------------------------------------------------------------------

PositionSpec = Literal["all", "last", "mean"] | list[int]
"""Which token positions to keep from each sequence.

- ``"all"``  — every position  → (T, D)
- ``"last"`` — last non-pad token  → (D,)
- ``"mean"`` — mean over non-pad tokens  → (D,)
- ``list[int]`` — explicit indices  → (len(list), D)
"""


@dataclass(frozen=True)
class ExtractionConfig:
    """What to extract and how."""

    layers: list[int] | None = None  # None → all layers
    positions: PositionSpec = "all"
    return_numpy: bool = True


@dataclass
class ActivationData:
    """Container returned by :meth:`ActivationExtractor.extract`.

    Shape conventions (when ``positions="all"``):
        activations : (batch, n_layers, seq_len, hidden_dim)
    When a reduction is applied the seq_len axis is collapsed or sliced.
    """

    activations: np.ndarray | torch.Tensor
    tokens: list[list[str]]
    token_ids: list[list[int]]
    attention_mask: np.ndarray | torch.Tensor
    layers: list[int]

    def __repr__(self) -> str:
        shape = getattr(self.activations, "shape", "N/A")
        return (
            f"ActivationData(batch={len(self.tokens)}, layers={self.layers}, "
            f"shape={shape})"
        )


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """Hook-based activation extractor for HuggingFace causal-LM models.

    Parameters
    ----------
    model : AutoModelForCausalLM
        A loaded HuggingFace causal language model.
    tokenizer : AutoTokenizer
        Matching tokenizer (must have ``pad_token`` set).
    layer_template : str
        Python-format string that resolves to the transformer block submodule
        path for a given layer index. E.g. ``"transformer.h.{layer}"`` for GPT-2,
        ``"model.layers.{layer}"`` for Llama.
    n_layers : int | None
        Total number of layers. Inferred from config if *None*.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer_template: str,
        n_layers: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_template = layer_template

        # Infer n_layers from common config attributes if not provided
        if n_layers is not None:
            self.n_layers = n_layers
        else:
            cfg = model.config
            for attr in ("n_layer", "num_hidden_layers", "num_layers"):
                if hasattr(cfg, attr):
                    self.n_layers = getattr(cfg, attr)
                    break
            else:
                raise ValueError(
                    "Cannot infer n_layers from model config; pass it explicitly."
                )

        # Ensure tokenizer can pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_layer_module(self, layer: int) -> nn.Module:
        """Return the nn.Module for a given layer index.

        Useful for downstream Jacobian computation.
        """
        return self.model.get_submodule(self.layer_template.format(layer=layer))

    def extract(
        self,
        input_text: str | list[str],
        config: ExtractionConfig | None = None,
    ) -> ActivationData:
        """Run a forward pass and return activations.

        Parameters
        ----------
        input_text : str or list[str]
            One or more input sequences.
        config : ExtractionConfig, optional
            Controls which layers / positions to capture and output format.
        """
        if config is None:
            config = ExtractionConfig()

        if isinstance(input_text, str):
            input_text = [input_text]

        layers = config.layers if config.layers is not None else list(range(self.n_layers))

        # Tokenize
        encoded = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True,
        )
        input_ids: torch.Tensor = encoded["input_ids"].to(self.model.device)
        attn_mask: torch.Tensor = encoded["attention_mask"].to(self.model.device)

        # --- hook & forward ---
        cache: dict[int, torch.Tensor] = {}
        handles = self._attach_hooks(layers, cache)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attn_mask)
        finally:
            for h in handles:
                h.remove()

        # Stack: (batch, n_layers, seq_len, hidden_dim)
        acts = torch.stack([cache[l] for l in layers], dim=1).cpu().float()

        # Position selection
        acts = self._select_positions(acts, attn_mask.cpu(), config.positions)

        # Token info
        tokens_list, ids_list = self._decode_tokens(input_ids.cpu())

        out_mask = attn_mask.cpu()
        if config.return_numpy:
            acts = acts.numpy()
            out_mask = out_mask.numpy()

        return ActivationData(
            activations=acts,
            tokens=tokens_list,
            token_ids=ids_list,
            attention_mask=out_mask,
            layers=layers,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attach_hooks(
        self, layers: list[int], cache: dict[int, torch.Tensor],
    ) -> list[torch.utils.hooks.RemovableHook]:
        handles = []
        for layer_idx in layers:
            module = self.get_layer_module(layer_idx)

            def hook(
                _mod: nn.Module,
                _inp: tuple,
                output,
                idx: int = layer_idx,
            ):
                value = output[0] if isinstance(output, tuple) else output
                cache[idx] = value.detach()

            handles.append(module.register_forward_hook(hook))
        return handles

    @staticmethod
    def _select_positions(
        acts: torch.Tensor,
        attn_mask: torch.Tensor,
        positions: PositionSpec,
    ) -> torch.Tensor:
        """Slice or reduce the sequence dimension.

        acts shape: (B, L, T, D)
        """
        if positions == "all":
            return acts

        B, L, T, D = acts.shape

        if positions == "last":
            # Index of last non-pad token per sequence
            lengths = attn_mask.sum(dim=1).long()  # (B,)
            last_idx = (lengths - 1).clamp(min=0)
            # Gather: (B, L, D)
            idx = last_idx[:, None, None, None].expand(B, L, 1, D)
            return acts.gather(2, idx).squeeze(2)

        if positions == "mean":
            # Masked mean over token dimension
            mask = attn_mask[:, None, :, None].float()  # (B, 1, T, 1)
            return (acts * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        if isinstance(positions, list):
            return acts[:, :, positions, :]

        raise ValueError(f"Unknown positions spec: {positions!r}")

    def _decode_tokens(
        self, input_ids: torch.Tensor,
    ) -> tuple[list[list[str]], list[list[int]]]:
        tokens_list = []
        ids_list = []
        for row in input_ids:
            ids = row.tolist()
            tokens = [self.tokenizer.decode(tid) for tid in ids]
            tokens_list.append(tokens)
            ids_list.append(ids)
        return tokens_list, ids_list


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

# Common layer template strings
LAYER_TEMPLATES: dict[str, str] = {
    "gpt2": "transformer.h.{layer}",
    "llama": "model.layers.{layer}",
    "pythia": "gpt_neox.layers.{layer}",
}


def make_extractor(
    model_name: str,
    *,
    layer_template: str | None = None,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
) -> ActivationExtractor:
    """Load a model + tokenizer and return a ready-to-use extractor.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. ``"gpt2"`` or ``"EleutherAI/pythia-160m"``.
    layer_template : str, optional
        Override the layer module path template. If *None*, guessed from
        ``model_name`` using :data:`LAYER_TEMPLATES`.
    device : str, optional
        Device string (e.g. ``"cuda"``). Defaults to ``"cpu"``.
    torch_dtype : torch.dtype, optional
        Model weight dtype.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs: dict = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if device is not None:
        model = model.to(device)

    # Guess template
    if layer_template is None:
        key = model_name.split("/")[-1].lower()
        for prefix, tmpl in LAYER_TEMPLATES.items():
            if prefix in key:
                layer_template = tmpl
                break
        else:
            raise ValueError(
                f"Cannot guess layer_template for '{model_name}'. "
                f"Pass it explicitly. Known prefixes: {list(LAYER_TEMPLATES)}"
            )

    return ActivationExtractor(model, tokenizer, layer_template)
