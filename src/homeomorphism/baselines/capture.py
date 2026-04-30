"""Latent stream capture for baseline experiments.

Implements phase 1 of the baseline pipeline:
1. Load model with baseline-specific modifications (via interventions module)
2. Forward pass on corpus samples with hook-based capture
3. Compute Jacobian metrics (optional)
4. Persist to HDF5 file for offline analysis

Key classes:
  - LatentCapture: Orchestrates capture pipeline
  - HDF5Store: Manages HDF5 read/write operations
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import torch

from homeomorphism import jacobian, models
from homeomorphism.data import load_texts
from homeomorphism.interventions import (
    PreparedInput,
    BaselineName,
    build_prepared_input,
    load_model_for_baseline,
)
from homeomorphism.models import SublayerKind

from .config import BaselineConfig, MemoryProfile


class HDF5Store:
    """Manages HDF5 file operations for storing latents.
    
    Stores residual streams at multiple depths in a hierarchical structure:
        /{baseline_name}/depth_00 (input)
        /{baseline_name}/depth_01 (post-attn block 0)
        /{baseline_name}/depth_02 (post-FFN block 0)
        ...
    """
    
    def __init__(self, path: Path):
        """Initialize HDF5 store at given path."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_datasets(
        self,
        group: h5py.Group,
        baseline: BaselineName,
        depths: list[int],
        seq_len: int,
        d_model: int,
    ) -> None:
        """Create resizable HDF5 datasets for each depth."""
        # Create baseline-specific group if needed
        if baseline not in group:
            baseline_group = group.create_group(baseline)
        else:
            baseline_group = group[baseline]
        
        for depth in depths:
            key = self._depth_key(depth)
            if key not in baseline_group:
                baseline_group.create_dataset(
                    key,
                    shape=(0, seq_len, d_model),
                    maxshape=(None, seq_len, d_model),
                    chunks=(1, seq_len, d_model),
                    dtype="float32",
                )
    
    def append_sample(
        self,
        group: h5py.Group,
        baseline: BaselineName,
        depth: int,
        sample: torch.Tensor,
    ) -> None:
        """Append a single latent sample to the HDF5 dataset."""
        key = self._depth_key(depth)
        baseline_group = group[baseline]
        dataset = baseline_group[key]
        
        if sample.dim() != 2:
            raise ValueError(f"expected (T, d) sample for {key!r}, got {tuple(sample.shape)}")
        
        next_index = int(dataset.shape[0])
        dataset.resize(next_index + 1, axis=0)
        dataset[next_index] = sample.detach().to(torch.float32).cpu().numpy()
    
    @staticmethod
    def _depth_key(depth: int) -> str:
        """Convert depth index to HDF5 dataset name."""
        return f"depth_{depth:02d}"


class LatentCapture:
    """Phase 1: Captures latent representations and optionally Jacobian metrics.
    
    Pipeline:
        1. Load model with baseline modifications
        2. Iterate over corpus samples
        3. For each sample:
            a. Build prepared input (baseline-specific token transformations)
            b. Forward pass with hooks at target depths
            c. Optionally compute Jacobian metrics
            d. Persist residual streams to HDF5
    """
    
    def __init__(self, config: BaselineConfig):
        """Initialize capture pipeline with configuration."""
        self.config = config
        self.store = HDF5Store(config.output_root / "latents.h5")
        self._results_dir = Path(config.output_root)
    
    def run(
        self,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Run latent capture for a given baseline.
        
        Returns:
            Dictionary with capture statistics (n_samples, times, errors, etc.)
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model with baseline-specific modifications
        m = load_model_for_baseline(
            model_name=self.config.model_name,
            baseline=baseline,
            weights=self.config.weights,
            seed=self.config.memory.seed,
            device=self.config.memory.device,
        )
        
        L = models.n_blocks(m)
        d_model = models.hidden_size(m)
        
        # Resolve layer targets from layers_spec
        from experiments.exp1_per_token_J import resolve_sublayers
        sublayers = resolve_sublayers(L, self.config.layers_spec)
        depths = self._sublayer_depths(sublayers)
        paths = self._resolve_hook_paths(m, depths)
        path_by_depth = dict(zip(depths, paths))
        
        # Load corpus
        texts = load_texts(
            self.config.corpus,
            n_samples=self.config.memory.n_samples,
            chunk_chars=max(self.config.memory.max_tokens * 8, 200),
            seed=self.config.memory.seed,
        )
        
        print(
            f"[LatentCapture] baseline={baseline} model={self.config.model_name} "
            f"L={L} d={d_model} depths={len(depths)} "
            f"samples={len(texts)} T={self.config.memory.max_tokens}"
        )
        
        # Initialize HDF5 store
        with h5py.File(self.store.path, "a") as h5:
            self.store.create_datasets(
                h5,
                baseline,
                depths,
                self.config.memory.max_tokens,
                d_model,
            )
        
        # Capture loop
        t_start = time.time()
        buf: dict[int, list[torch.Tensor]] = {d: [] for d in depths}
        input_ids_kept: list[list[int]] = []
        dropped: list[dict[str, Any]] = []
        jacobian_rows: list[dict[str, Any]] = []
        
        for input_id, text in enumerate(texts):
            # Build baseline-specific prepared input
            prepared: PreparedInput = build_prepared_input(
                m=m,
                text=text,
                max_tokens=self.config.memory.max_tokens,
                baseline=baseline,
                root_seed=self.config.memory.seed,
                sample_index=input_id,
            )
            
            try:
                captured = self._capture_with_hooks(m, paths, prepared.forward_kwargs)
            except Exception as e:
                dropped.append({"input_id": input_id, "reason": f"{type(e).__name__}: {e}"})
                continue
            
            # Validate sequence length consistency
            lengths = {tensor.shape[0] for tensor in captured.values()}
            if len(lengths) != 1:
                dropped.append({"input_id": input_id, "reason": f"depth length mismatch {lengths}"})
                continue
            
            t_seq = lengths.pop()
            if t_seq != self.config.memory.max_tokens:
                dropped.append(
                    {"input_id": input_id, "reason": f"tokenized to {t_seq} != T={self.config.memory.max_tokens}"}
                )
                continue
            
            # Buffer residual streams
            for depth, path in path_by_depth.items():
                buf[depth].append(captured[path])
            
            input_ids_kept.append(prepared.token_ids[0].tolist())
            
            if (input_id + 1) % 50 == 0:
                print(f"  captured {input_id + 1}/{len(texts)} ({len(input_ids_kept)} kept)")
        
        # Write to HDF5
        with h5py.File(self.store.path, "a") as h5:
            for depth in depths:
                if buf[depth]:
                    for sample in buf[depth]:
                        self.store.append_sample(h5, baseline, depth, sample)
        
        elapsed = time.time() - t_start
        n_kept = len(input_ids_kept)
        
        stats = {
            "baseline": baseline,
            "n_samples_requested": self.config.memory.n_samples,
            "n_samples_kept": n_kept,
            "n_samples_dropped": len(dropped),
            "dropped_samples": dropped,
            "depths_captured": depths,
            "n_depths": len(depths),
            "elapsed_sec": round(elapsed, 2),
        }
        
        print(
            f"[LatentCapture] captured {n_kept}/{len(texts)} inputs in {elapsed:.1f}s "
            f"(dropped {len(dropped)})"
        )
        
        return stats
    
    @staticmethod
    def _capture_with_hooks(
        m: models.Model,
        paths: list[str],
        forward_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run forward pass with hooks; capture residual streams at each path.
        
        Returns:
            {path: Tensor[T, d] in fp32}
        """
        captured: dict[str, list[torch.Tensor]] = {p: [] for p in paths}
        handles = []
        
        for p in paths:
            module = m.model.get_submodule(p)
            
            def _make_hook(name: str):
                def _hook(_mod, inputs, _out):  # noqa: ANN001
                    x = inputs[0] if isinstance(inputs, tuple) else inputs
                    if not isinstance(x, torch.Tensor):
                        raise TypeError(f"hook at {name!r} got non-tensor input: {type(x)}")
                    captured[name].append(x.detach())
                return _hook
            
            handles.append(module.register_forward_hook(_make_hook(p)))
        
        try:
            with torch.no_grad():
                m.model(**forward_kwargs)
        finally:
            for h in handles:
                h.remove()
        
        out: dict[str, torch.Tensor] = {}
        for p in paths:
            if not captured[p]:
                raise RuntimeError(f"hook at {p!r} did not fire")
            h = captured[p][0]
            if h.dim() != 3 or h.shape[0] != 1:
                raise ValueError(f"unexpected tensor shape at {p!r}: {tuple(h.shape)}")
            out[p] = h[0].to(torch.float32).cpu()
        
        return out
    
    @staticmethod
    def _sublayer_depths(sublayers: list[tuple[int, SublayerKind]]) -> list[int]:
        """Convert (block_idx, kind) pairs to depth indices."""
        depths = []
        for block_idx, kind in sublayers:
            # depth = 2 * block_idx + (0 if attn else 1) + 1 (offset for input)
            depth = 2 * block_idx + (0 if kind == "attn" else 1) + 1
            depths.append(depth)
        return sorted(depths)
    
    @staticmethod
    def _resolve_hook_paths(
        m: models.Model,
        depths: list[int],
    ) -> list[str]:
        """Convert depth indices to hook paths."""
        paths = []
        for depth in depths:
            # Map depth to hook path (reverse of _sublayer_depths)
            d = depth - 1
            block_idx = d // 2
            is_attn = (d % 2) == 0
            kind = "attn" if is_attn else "ffn"
            
            sub = models.sublayer(m, block_idx, kind)
            paths.append(sub.hook_path)
        
        return paths
