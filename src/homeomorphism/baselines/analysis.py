"""Baseline analysis framework for phase 2 processing.

Phase 2 is offline analysis: reads HDF5 latents and computes various metrics:
  - Intrinsic dimension (ID) estimates using multiple estimators
  - k-NN neighborhood overlap between consecutive depths
  - (Future) Statistical comparisons, anomalies, visualizations

Key classes:
  - BaselineAnalyzer: Interface for all analyzers
  - IDAnalyzer: Computes intrinsic dimension estimates
  - OverlapAnalyzer: Computes neighborhood overlap metrics
  - AnalyzerPipeline: Chains multiple analyzers
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from homeomorphism.id_est import EstimatorName, estimate_id as estimate_id_torch
from homeomorphism.interventions import BaselineName

from .config import BaselineConfig, Granularity


class BaselineAnalyzer(ABC):
    """Interface for baseline analysis components.
    
    Analyzers read HDF5 latents and write JSONL results. Implementations
    should be composable (multiple analyzers can process the same HDF5).
    """
    
    @abstractmethod
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Process HDF5 file and write results to JSONL.
        
        Args:
            h5_path: Path to HDF5 file with latents
            output_path: Path to write JSONL results
            baseline: Baseline name being analyzed
        
        Returns:
            Statistics about the analysis run (n_rows, time, errors, etc.)
        """
        pass


class IDAnalyzer(BaselineAnalyzer):
    """Computes intrinsic dimension (ID) estimates from latent clouds.
    
    For each (baseline, depth, granularity, token_idx, estimator) combination,
    reads the latent cloud from HDF5 and estimates its intrinsic dimension.
    
    Output JSONL schema:
        {
            "baseline": str,
            "depth": int,
            "hook_path": str | null,
            "granularity": "full_stream" | "per_token" | "last_token",
            "token_idx": int | null,
            "estimator": "twonn" | "ess" | "participation_ratio",
            "n_points": int,
            "ambient_dim": int,
            "id_estimate": float | null,
            "error": str | null,
        }
    """
    
    def __init__(
        self,
        granularities: list[Granularity],
        estimators: list[EstimatorName],
    ):
        """Initialize ID analyzer with configuration.
        
        Args:
            granularities: Data slicing strategies ('full_stream', 'per_token', 'last_token')
            estimators: ID estimators to compute ('twonn', 'ess', 'participation_ratio')
        """
        self.granularities = granularities
        self.estimators = estimators
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Compute ID estimates and write to JSONL."""
        n_rows = 0
        errors = []
        t_start = time.time()
        
        with h5py.File(h5_path, "r") as h5:
            if baseline not in h5:
                return {
                    "analyzer": "IDAnalyzer",
                    "baseline": baseline,
                    "n_rows": 0,
                    "error": f"baseline {baseline!r} not found in HDF5",
                }
            
            baseline_group = h5[baseline]
            depths = self._extract_depths(baseline_group)
            
            if not depths:
                return {
                    "analyzer": "IDAnalyzer",
                    "baseline": baseline,
                    "n_rows": 0,
                    "error": "no depth datasets found",
                }
            
            seq_len = baseline_group[f"depth_{depths[0]:02d}"].shape[1]
            
            with output_path.open("a") as fo:
                for depth in depths:
                    key = f"depth_{depth:02d}"
                    reps = torch.from_numpy(baseline_group[key][:])  # (N, T, d)
                    N = int(reps.shape[0])
                    
                    if N < 2:
                        continue
                    
                    for granularity in self.granularities:
                        token_indices: list[int | None]
                        if granularity == "per_token":
                            token_indices = list(range(seq_len))
                        else:
                            token_indices = [None]
                        
                        for tok_idx in token_indices:
                            cloud = self._cloud(reps, granularity, tok_idx)
                            
                            for estimator in self.estimators:
                                err: str | None = None
                                try:
                                    val = float(estimate_id_torch(cloud, estimator))
                                except Exception as e:
                                    val = float("nan")
                                    err = f"{type(e).__name__}: {e}"
                                    errors.append(err)
                                
                                # Convert NaN/Inf to None for JSON
                                id_json: float | None
                                if val != val or val in (float("inf"), float("-inf")):
                                    id_json = None
                                else:
                                    id_json = val
                                
                                row = {
                                    "baseline": baseline,
                                    "depth": depth - 1,  # User-facing depth (0-indexed)
                                    "hook_path": None,
                                    "granularity": granularity,
                                    "token_idx": tok_idx,
                                    "estimator": estimator,
                                    "n_points": int(cloud.shape[0]),
                                    "ambient_dim": int(cloud.shape[1]),
                                    "id_estimate": id_json,
                                    "error": err,
                                }
                                fo.write(json.dumps(row) + "\n")
                                n_rows += 1
        
        elapsed = time.time() - t_start
        return {
            "analyzer": "IDAnalyzer",
            "baseline": baseline,
            "n_rows": n_rows,
            "elapsed_sec": round(elapsed, 2),
            "n_errors": len(errors),
        }
    
    @staticmethod
    def _cloud(
        reps: torch.Tensor,
        granularity: Granularity,
        token_idx: int | None,
    ) -> torch.Tensor:
        """Slice latent cloud according to granularity.
        
        Args:
            reps: Shape (N, T, d)
            granularity: Slicing strategy
            token_idx: Token position (for per_token), or None
        
        Returns:
            Cloud of shape (N, d') where d' depends on granularity
        """
        if granularity == "full_stream":
            N, T, d = reps.shape
            return reps.reshape(N, T * d)
        if granularity == "per_token":
            if token_idx is None:
                raise ValueError("per_token granularity requires token_idx")
            return reps[:, token_idx, :]
        if granularity == "last_token":
            return reps[:, -1, :]
        raise ValueError(f"unknown granularity {granularity!r}")
    
    @staticmethod
    def _extract_depths(group: h5py.Group) -> list[int]:
        """Extract depth indices from group keys (e.g., depth_00, depth_01, ...)."""
        depths = []
        for key in group.keys():
            if key.startswith("depth_"):
                try:
                    depth = int(key.split("_")[1])
                    depths.append(depth)
                except (IndexError, ValueError):
                    pass
        return sorted(depths)


class OverlapAnalyzer(BaselineAnalyzer):
    """Computes k-NN neighborhood overlap between consecutive depths.
    
    Measures how well the k-nearest neighbors are preserved when transforming
    from one depth to the next. Useful for understanding information flow.
    
    Output JSONL schema:
        {
            "baseline": str,
            "depth_from": int,
            "depth_to": int,
            "k": int,
            "granularity": "full_stream" | "last_token",
            "neighbour_overlap": float,
        }
    """
    
    def __init__(self, k: int = 5, granularities: list[Granularity] | None = None):
        """Initialize overlap analyzer.
        
        Args:
            k: Number of neighbors to check
            granularities: Data slicing strategies (default: full_stream, last_token)
        """
        self.k = k
        self.granularities = granularities or ["full_stream", "last_token"]
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Compute neighborhood overlap and write to JSONL."""
        n_rows = 0
        t_start = time.time()
        
        with h5py.File(h5_path, "r") as h5:
            if baseline not in h5:
                return {
                    "analyzer": "OverlapAnalyzer",
                    "baseline": baseline,
                    "n_rows": 0,
                    "error": f"baseline {baseline!r} not found in HDF5",
                }
            
            baseline_group = h5[baseline]
            depths = IDAnalyzer._extract_depths(baseline_group)
            
            with output_path.open("a") as fo:
                # Compute overlap for consecutive depth pairs
                for i in range(len(depths) - 1):
                    d_from, d_to = depths[i], depths[i + 1]
                    key_from = f"depth_{d_from:02d}"
                    key_to = f"depth_{d_to:02d}"
                    
                    reps_from = torch.from_numpy(baseline_group[key_from][:])
                    reps_to = torch.from_numpy(baseline_group[key_to][:])
                    
                    for granularity in self.granularities:
                        cloud_from = IDAnalyzer._cloud(reps_from, granularity, None)
                        cloud_to = IDAnalyzer._cloud(reps_to, granularity, None)
                        
                        overlap = self._compute_overlap(cloud_from, cloud_to, self.k)
                        
                        row = {
                            "baseline": baseline,
                            "depth_from": d_from - 1,
                            "depth_to": d_to - 1,
                            "k": self.k,
                            "granularity": granularity,
                            "neighbour_overlap": overlap,
                        }
                        fo.write(json.dumps(row) + "\n")
                        n_rows += 1
        
        elapsed = time.time() - t_start
        return {
            "analyzer": "OverlapAnalyzer",
            "baseline": baseline,
            "n_rows": n_rows,
            "elapsed_sec": round(elapsed, 2),
        }
    
    @staticmethod
    def _compute_overlap(
        cloud_a: torch.Tensor,
        cloud_b: torch.Tensor,
        k: int,
    ) -> float:
        """Compute k-NN neighborhood overlap between two clouds.
        
        Measures: fraction of points whose k-nearest neighbors in cloud_a
        are preserved in cloud_b's k-nearest neighbors.
        """
        n_points = min(int(cloud_a.shape[0]), int(cloud_b.shape[0]))
        if n_points < 2:
            return float("nan")

        effective_k = min(k, n_points - 1)

        # Compute pairwise distances
        dists_a = torch.cdist(cloud_a, cloud_a)
        dists_b = torch.cdist(cloud_b, cloud_b)
        
        # Get k nearest neighbors (excluding self)
        _, nn_a = torch.topk(dists_a, effective_k + 1, dim=1, largest=False)
        nn_a = nn_a[:, 1:]  # Skip self
        
        _, nn_b = torch.topk(dists_b, effective_k + 1, dim=1, largest=False)
        nn_b = nn_b[:, 1:]  # Skip self
        
        # Compute overlap
        overlaps = []
        for i in range(len(cloud_a)):
            neighbors_a_set = set(nn_a[i].tolist())
            neighbors_b_set = set(nn_b[i].tolist())
            overlap_count = len(neighbors_a_set & neighbors_b_set)
            overlaps.append(overlap_count / effective_k)
        
        return float(np.mean(overlaps))


class AnalyzerPipeline:
    """Chains multiple analyzers for sequential processing of baseline results.
    
    Allows composing multiple analyses on the same HDF5 file, with shared
    I/O operations for efficiency.
    """
    
    def __init__(self, analyzers: list[BaselineAnalyzer] | None = None):
        """Initialize pipeline.
        
        Args:
            analyzers: List of analyzers to chain (can add more with add_analyzer)
        """
        self.analyzers = analyzers or []
    
    def add_analyzer(self, analyzer: BaselineAnalyzer) -> AnalyzerPipeline:
        """Add analyzer to pipeline; return self for chaining."""
        self.analyzers.append(analyzer)
        return self
    
    def run(
        self,
        h5_path: Path,
        output_dir: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Run all analyzers in sequence; each writes to separate JSONL file.
        
        Returns:
            Dictionary mapping analyzer name to stats
        """
        results = {}
        
        for analyzer in self.analyzers:
            analyzer_name = analyzer.__class__.__name__
            output_path = output_dir / f"{analyzer_name.lower()}.jsonl"
            
            try:
                stats = analyzer.run(h5_path, output_path, baseline)
                results[analyzer_name] = stats
            except Exception as e:
                results[analyzer_name] = {
                    "error": f"{type(e).__name__}: {e}",
                }
        
        return results
