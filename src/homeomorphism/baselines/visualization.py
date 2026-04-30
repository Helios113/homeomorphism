"""Visualization analyzers for baseline results.

Provides extensible visualization infrastructure:
  - DepthTrajectoryVisualizer: ID vs depth line plots
  - BaselineComparisonVisualizer: Bar charts across baselines
  - OverlapTrajectoryVisualizer: k-NN overlap vs depth line plots
  - EstimatorCorrelationVisualizer: Scatter correlation plots between estimators

Can be used standalone or as part of AnalyzerPipeline.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from homeomorphism.interventions import BaselineName

from .analysis import BaselineAnalyzer

# Use non-interactive backend for server environments
matplotlib.use("Agg")

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass


class DepthTrajectoryVisualizer(BaselineAnalyzer):
    """Creates line plots of ID vs depth for each baseline.
    
    Reads idanalyzer.jsonl and produces one comprehensive plot per baseline,
    showing how intrinsic dimension (mean +/- std) evolves through network depth.
    """
    
    def __init__(self, granularity: str = "last_token", output_dir: str = "plots"):
        """Initialize visualizer.
        
        Args:
            granularity: Data granularity to visualize ('full_stream', 'per_token', 'last_token')
            output_dir: Subdirectory for plots (relative to output_path parent)
        """
        self.granularity = granularity
        self.output_dir = output_dir
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Generate depth-trajectory plots from id.jsonl.
        
        Looks for id.jsonl in output_path.parent and creates plots.
        
        Returns:
            Statistics about plots created.
        """
        plots_dir = output_path.parent / self.output_dir
        plots_dir.mkdir(exist_ok=True)
        
        # Load ID results from JSONL
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        if not id_jsonl.exists():
            return {
                "analyzer": "DepthTrajectoryVisualizer",
                "baseline": baseline,
                "n_plots": 0,
                "error": f"idanalyzer.jsonl not found at {id_jsonl}",
            }
        
        # Aggregate data by depth and estimator
        data = self._aggregate_id_data(id_jsonl, baseline)
        
        if not data:
            return {
                "analyzer": "DepthTrajectoryVisualizer",
                "baseline": baseline,
                "n_plots": 0,
            }
        
        # Create plots
        n_plots = 0
        fig = self._plot_trajectory(baseline, data)
        if fig is not None:
            plot_file = plots_dir / f"id_vs_depth_{baseline}.png"
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_plots += 1
        
        return {
            "analyzer": "DepthTrajectoryVisualizer",
            "baseline": baseline,
            "n_plots": n_plots,
            "plot_directory": str(plots_dir),
        }
    
    def _aggregate_id_data(
        self,
        id_jsonl: Path,
        baseline: BaselineName,
    ) -> dict[str, dict[int, dict[str, float]]]:
        """Load JSONL and aggregate by (estimator, depth).
        
        Returns:
            {estimator: {depth: {"mean": val, "std": val}}}
        """
        data = defaultdict(lambda: defaultdict(list))
        
        with id_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                
                if row.get("baseline") != baseline:
                    continue
                if row.get("granularity") != self.granularity:
                    continue
                
                est = row.get("estimator")
                depth = row.get("depth")
                id_est = row.get("id_estimate")
                
                if est and depth is not None and id_est is not None:
                    data[est][depth].append(id_est)
        
        # Average across samples
        result = {}
        for est, depth_dict in data.items():
            result[est] = {
                d: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for d, vals in depth_dict.items()
                if vals
            }
        
        return result
    
    def _plot_trajectory(
        self,
        baseline: str,
        data: dict[str, dict[int, dict[str, float]]],
    ) -> matplotlib.figure.Figure | None:
        """Create a single comprehensive trajectory plot with all estimators."""
        if not data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for est, depth_values in data.items():
            depths = sorted(depth_values.keys())
            if not depths:
                continue
            
            means = np.array([depth_values[d]["mean"] for d in depths])
            stds = np.array([depth_values[d]["std"] for d in depths])
            
            line = ax.plot(depths, means, marker="o", linewidth=2, markersize=6, label=est.upper())
            ax.fill_between(depths, means - stds, means + stds, alpha=0.2, color=line[0].get_color())
        
        ax.set_xlabel("Depth", fontsize=11)
        ax.set_ylabel("Intrinsic Dimension", fontsize=11)
        ax.set_title(f"ID vs Depth — {baseline} ({self.granularity})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig


class BaselineComparisonVisualizer(BaselineAnalyzer):
    """Creates bar charts comparing baselines at specific depths.
    
    Useful for visualizing how different baselines affect intrinsic dimension
    at key depths (e.g., input, middle, output).
    """
    
    def __init__(
        self,
        depth: int = 0,
        estimator: str = "twonn",
        granularity: str = "last_token",
        output_dir: str = "plots",
    ):
        """Initialize comparator.
        
        Args:
            depth: Depth to visualize (default: 0 = input)
            estimator: Estimator to use (default: twonn)
            granularity: Data granularity
            output_dir: Subdirectory for plots
        """
        self.depth = depth
        self.estimator = estimator
        self.granularity = granularity
        self.output_dir = output_dir
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Generate comparison plots across multiple baselines.
        
        Note: This analyzer assumes multiple baseline JSONL files are available
        in the output directory. It aggregates across them.
        """
        plots_dir = output_path.parent / self.output_dir
        plots_dir.mkdir(exist_ok=True)
        
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        if not id_jsonl.exists():
            return {
                "analyzer": "BaselineComparisonVisualizer",
                "baseline": baseline,
                "n_plots": 0,
            }
        
        # Aggregate data across baselines
        baseline_data = self._aggregate_baselines(id_jsonl)
        
        if not baseline_data:
            return {
                "analyzer": "BaselineComparisonVisualizer",
                "baseline": baseline,
                "n_plots": 0,
            }
        
        # Create comparison plot
        fig = self._plot_comparison(baseline_data)
        if fig is not None:
            plot_file = plots_dir / f"baseline_comparison_depth{self.depth}.png"
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            return {
                "analyzer": "BaselineComparisonVisualizer",
                "baseline": baseline,
                "n_plots": 1,
                "plot_file": str(plot_file),
            }
        
        return {
            "analyzer": "BaselineComparisonVisualizer",
            "baseline": baseline,
            "n_plots": 0,
        }
    
    def _aggregate_baselines(self, id_jsonl: Path) -> dict[str, dict[str, float]]:
        """Load all baselines' ID at target depth.
        
        Returns:
            {baseline_name: {"mean": float, "std": float}}
        """
        data = defaultdict(list)
        
        with id_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                
                if (row.get("depth") != self.depth
                    or row.get("estimator") != self.estimator
                    or row.get("granularity") != self.granularity
                ):
                    continue
                
                baseline = row.get("baseline")
                id_est = row.get("id_estimate")
                
                if baseline and id_est is not None:
                    data[baseline].append(id_est)
        
        result = {}
        for b, vals in data.items():
            result[b] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        
        return result
    
    def _plot_comparison(self, baseline_data: dict[str, dict[str, float]]) -> matplotlib.figure.Figure | None:
        """Create comparison bar chart."""
        if not baseline_data:
            return None
        
        baselines = sorted(baseline_data.keys())
        means = [baseline_data[b]["mean"] for b in baselines]
        stds = [baseline_data[b]["std"] for b in baselines]
        
        fig, ax = plt.subplots(figsize=(max(8, len(baselines) * 1.2), 5))
        ax.bar(baselines, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7)
        
        ax.set_ylabel("Intrinsic Dimension", fontsize=11)
        ax.set_xlabel("Baseline", fontsize=11)
        ax.set_title(
            f"Baseline Comparison at Depth {self.depth} "
            f"({self.estimator.upper()}, {self.granularity})",
            fontsize=12,
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        return fig


class OverlapTrajectoryVisualizer(BaselineAnalyzer):
    """Creates line plots of k-NN overlap vs depth.
    
    Reads overlapanalyzer.jsonl and shows how neighborhood structures are preserved
    between consecutive depths for each baseline.
    """
    
    def __init__(self, granularity: str = "last_token", output_dir: str = "plots"):
        self.granularity = granularity
        self.output_dir = output_dir
        
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        plots_dir = output_path.parent / self.output_dir
        plots_dir.mkdir(exist_ok=True)
        
        overlap_jsonl = output_path.parent / "overlapanalyzer.jsonl"
        if not overlap_jsonl.exists():
            return {
                "analyzer": "OverlapTrajectoryVisualizer",
                "baseline": baseline,
                "n_plots": 0,
                "error": f"overlapanalyzer.jsonl not found",
            }
            
        depth_values = self._aggregate_overlap(overlap_jsonl, baseline)
        if not depth_values:
            return {"analyzer": "OverlapTrajectoryVisualizer", "baseline": baseline, "n_plots": 0}
            
        fig = self._plot_overlap(baseline, depth_values)
        if fig is not None:
            plot_file = plots_dir / f"overlap_vs_depth_{baseline}.png"
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"analyzer": "OverlapTrajectoryVisualizer", "baseline": baseline, "n_plots": 1}
            
        return {"analyzer": "OverlapTrajectoryVisualizer", "baseline": baseline, "n_plots": 0}
        
    def _aggregate_overlap(self, jsonl: Path, baseline: str) -> dict[int, dict[str, float]]:
        data = defaultdict(list)
        with jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("baseline") != baseline or row.get("granularity") != self.granularity:
                    continue
                
                # Using depth_to as the x-axis alignment point
                depth = row.get("depth_to")
                overlap = row.get("neighbour_overlap")
                if depth is not None and overlap is not None:
                    data[depth].append(overlap)
                    
        return {d: {"mean": float(np.mean(v)), "std": float(np.std(v))} for d, v in data.items() if v}
        
    def _plot_overlap(self, baseline: str, depth_values: dict[int, dict[str, float]]) -> matplotlib.figure.Figure | None:
        if not depth_values:
            return None
            
        depths = sorted(depth_values.keys())
        means = np.array([depth_values[d]["mean"] for d in depths])
        stds = np.array([depth_values[d]["std"] for d in depths])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        line = ax.plot(depths, means, marker="s", linewidth=2, markersize=6, color="forestgreen", label="k-NN Overlap")
        ax.fill_between(depths, means - stds, means + stds, alpha=0.2, color="forestgreen")
        
        ax.set_xlabel("Depth Transition (to)", fontsize=11)
        ax.set_ylabel("Neighborhood Overlap Fraction", fontsize=11)
        ax.set_title(f"Information Preservation vs Depth — {baseline} ({self.granularity})", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig


class EstimatorCorrelationVisualizer(BaselineAnalyzer):
    """Creates scatter plots to correlate ID estimators against each other.
    
    Reveals the structural relationship between estimators (e.g., TwoNN vs ESS).
    """
    
    def __init__(self, granularity: str = "last_token", output_dir: str = "plots"):
        self.granularity = granularity
        self.output_dir = output_dir
        
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        plots_dir = output_path.parent / self.output_dir
        plots_dir.mkdir(exist_ok=True)
        
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        if not id_jsonl.exists():
            return {"analyzer": "EstimatorCorrelationVisualizer", "baseline": baseline, "n_plots": 0}
            
        scatter_data = self._aggregate_for_scatter(id_jsonl, baseline)
        
        n_plots = 0
        estimators = list(scatter_data.keys())
        for i in range(len(estimators)):
            for j in range(i + 1, len(estimators)):
                est_x, est_y = estimators[i], estimators[j]
                fig = self._plot_scatter(baseline, est_x, est_y, scatter_data[est_x], scatter_data[est_y])
                if fig is not None:
                    plot_file = plots_dir / f"correlation_{est_x}_vs_{est_y}_{baseline}.png"
                    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    n_plots += 1
                    
        return {"analyzer": "EstimatorCorrelationVisualizer", "baseline": baseline, "n_plots": n_plots}
        
    def _aggregate_for_scatter(self, jsonl: Path, baseline: str) -> dict[str, list[float]]:
        # Gather paired estimates by (depth, token_idx) to ensure alignment
        aligned_data = defaultdict(dict)
        estimators = set()
        
        with jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("baseline") != baseline or row.get("granularity") != self.granularity:
                    continue
                
                est = row.get("estimator")
                depth = row.get("depth")
                tok_idx = row.get("token_idx", 0)
                val = row.get("id_estimate")
                
                if est and depth is not None and val is not None:
                    aligned_data[(depth, tok_idx)][est] = val
                    estimators.add(est)
                    
        # Extract aligned vectors
        results = {est: [] for est in estimators}
        for key, est_dict in aligned_data.items():
            if len(est_dict) == len(estimators):  # Ensure complete pairs
                for est in estimators:
                    results[est].append(est_dict[est])
                    
        return results
        
    def _plot_scatter(self, baseline: str, est_x: str, est_y: str, vals_x: list[float], vals_y: list[float]) -> matplotlib.figure.Figure | None:
        if not vals_x or not vals_y:
            return None
            
        fig, ax = plt.subplots(figsize=(7, 7))
        
        ax.scatter(vals_x, vals_y, alpha=0.6, edgecolors="w", s=40)
        
        # Perfect agreement line
        max_val = max(max(vals_x), max(vals_y))
        min_val = min(min(vals_x), min(vals_y))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="Perfect Agreement")
        
        ax.set_xlabel(f"ID ({est_x.upper()})", fontsize=11)
        ax.set_ylabel(f"ID ({est_y.upper()})", fontsize=11)
        ax.set_title(f"Estimator Correlation — {baseline} ({self.granularity})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig


# ============================================================================
# Standalone utilities (for use outside of analyzer pipeline)
# ============================================================================

def plot_all_baselines_comprehensive(
    results_dir: Path,
    output_dir: Path | None = None,
) -> None:
    """Generate comprehensive plots from a baseline results directory.
    
    Convenience function that runs multiple visualizers on existing results.
    
    Args:
        results_dir: Directory containing id.jsonl and overlap.jsonl
        output_dir: Where to save plots (default: results_dir/plots)
    """
    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    id_jsonl = results_dir / "idanalyzer.jsonl"
    
    baselines = set()
    if id_jsonl.exists():
        with id_jsonl.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "baseline" in data:
                        baselines.add(data["baseline"])
                        
    # Create dummy output path just to resolve parent directory properly in visualizers
    dummy_output_path = results_dir / "dummy.jsonl"
    
    # Run all visualizers for all baselines
    visualizers = [
        DepthTrajectoryVisualizer(output_dir=output_dir.name),
        BaselineComparisonVisualizer(output_dir=output_dir.name),
        OverlapTrajectoryVisualizer(output_dir=output_dir.name),
        EstimatorCorrelationVisualizer(output_dir=output_dir.name),
    ]
    
    for baseline in baselines:
        for viz in visualizers:
            viz.run(Path("dummy.h5"), dummy_output_path, baseline)  # type: ignore
    
    print(f"Plots saved to {output_dir}")
