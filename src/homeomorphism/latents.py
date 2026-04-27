"""Shared utilities for latent-state HDF5 storage.

This module provides a reusable LatentStore for saving and appending hidden
states from transformer models, along with a generic LatentConfig dataclass
that describes the storage layout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


_Tensor = torch.Tensor


@dataclass
class LatentConfig:
    """Generic configuration describing a latent store.

    Attributes
    ----------
    d_model : int
        Model width / residual dimension.
    seq_len : int
        Sequence length (number of tokens).
    n_layers : int
        Number of transformer layers.
    n_depths : int
        Total number of depth checkpoints (input + per-sub-block + post-norm).
        Typically 2 * n_layers + 2 for pre-norm LLaMA-style models.
    manifold_dim : int | None
        Intrinsic dimension of the input manifold (if applicable).
    manifold_type : str | None
        Name of the manifold (e.g. "hyperplane", "sphere", "torus").
    """

    d_model: int
    seq_len: int
    n_layers: int
    n_depths: int
    manifold_dim: int | None = None
    manifold_type: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        d_model: int,
        seq_len: int,
        n_layers: int,
        manifold_dim: int | None = None,
        manifold_type: str | None = None,
    ) -> LatentConfig:
        n_depths = 2 * n_layers + 2
        return cls(
            d_model=d_model,
            seq_len=seq_len,
            n_layers=n_layers,
            n_depths=n_depths,
            manifold_dim=manifold_dim,
            manifold_type=manifold_type,
        )

    @classmethod
    def from_exp3_config(cls, cfg: Any) -> LatentConfig:
        """Convert an experiment-specific config object to LatentConfig.

        The input object must have attributes: d_model, seq_len, n_layers.
        Optional: manifold_dim, manifold_type.
        n_depths is computed as 2 * n_layers + 2.
        """
        required = ("d_model", "seq_len", "n_layers")
        missing = [k for k in required if not hasattr(cfg, k)]
        if missing:
            raise TypeError(f"Config object missing required attributes: {missing}")

        manifold_dim = getattr(cfg, "manifold_dim", None)
        manifold_type = getattr(cfg, "manifold_type", None)

        return cls(
            d_model=cfg.d_model,
            seq_len=cfg.seq_len,
            n_layers=cfg.n_layers,
            n_depths=2 * cfg.n_layers + 2,
            manifold_dim=manifold_dim,
            manifold_type=manifold_type,
        )


_DEPTH_KEY = "depth_{:02d}"


def _check_compatible(stored: dict, cfg: LatentConfig) -> None:
    """Raise if the on-disk config is incompatible with the current cfg.

    Checks
    ------
    - d_model  (must match exactly)
    - seq_len  (must match exactly)
    - n_layers (must match exactly)
    - n_depths (optional; warn if mismatched)

    Mismatches indicate the file was created with a different model
    architecture; appending would produce misaligned datasets.
    """
    for key in ("d_model", "seq_len", "n_layers"):
        s = stored.get(key)
        c = getattr(cfg, key)
        if s is not None and s != c:
            raise ValueError(
                f"File was created with {key}={s}; current config has {key}={c}. "
                "Use a different output file or match the config."
            )


class LatentStore:
    """Append-friendly HDF5 store for hidden states across model depths.

    Layout
    ------
      /depth_00   float32 (N, T, d_model) — resizable along axis 0 (samples)
      /depth_01   float32 (N, T, d_model)
      ...
      /depth_{n_depths-1} float32 (N, T, d_model)

    File-level attributes store the config as JSON so the file is self-describing.

    Usage
    -----
      cfg = LatentConfig.from_fields(d_model=20, seq_len=8, n_layers=2)
      with LatentStore.open("latents.h5", cfg) as store:
          store.append(states)   # states: list[Tensor] of length n_depths, each (B, T, d)

    The store can be re-opened in append mode to extend an existing dataset.
    """

    CHUNK_SAMPLES = 64  # HDF5 chunk size along sample axis

    def __init__(self, h5file: h5py.File, cfg: LatentConfig):
        self._f = h5file
        self._cfg = cfg

    @classmethod
    def open(cls, path: str | Path, cfg: LatentConfig) -> LatentStore:
        path = Path(path)
        if path.exists():
            f = h5py.File(path, "a")
            stored_cfg = json.loads(f.attrs.get("config", "{}"))
            _check_compatible(stored_cfg, cfg)
        else:
            f = h5py.File(path, "w")
            f.attrs["config"] = json.dumps(asdict(cfg))
            T, d = cfg.seq_len, cfg.d_model
            chunk = (cls.CHUNK_SAMPLES, T, d)
            for depth in range(cfg.n_depths):
                f.create_dataset(
                    _DEPTH_KEY.format(depth),
                    shape=(0, T, d),
                    maxshape=(None, T, d),
                    dtype="float32",
                    chunks=chunk,
                    compression="lzf",
                )
        return cls(f, cfg)

    def append(self, states: list[_Tensor]) -> None:
        """Append one batch of hidden states across all depths.

        Parameters
        ----------
        states : list[torch.Tensor]
            List of length n_depths; each tensor has shape (B, T, d_model).
        """
        if len(states) != self._cfg.n_depths:
            raise ValueError(
                f"Expected {self._cfg.n_depths} depth tensors, got {len(states)}"
            )
        for depth, h in enumerate(states):
            arr = h.detach().cpu().float().numpy()  # (B, T, d)
            ds = self._f[_DEPTH_KEY.format(depth)]
            old_n = ds.shape[0]
            ds.resize(old_n + arr.shape[0], axis=0)
            ds[old_n:] = arr

    def n_samples(self) -> int:
        return self._f[_DEPTH_KEY.format(0)].shape[0]

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> LatentStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()
