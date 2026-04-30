"""Template and base classes for future analyzer implementations.

This module provides guidelines and base classes for extending the baseline
analysis framework with new capabilities like visualization and statistical tests.

Analysts can subclass BaselineAnalyzer to implement custom metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from homeomorphism.interventions import BaselineName

from .analysis import BaselineAnalyzer


class VisualizationAnalyzer(BaselineAnalyzer):
    """Base class for visualization-based analyzers.
    
    Subclasses should implement plot generation from HDF5 + JSONL results.
    Can write to PNG, SVG, interactive HTML, etc.
    
    Example subclasses:
        - DepthTrajectoryPlotter: ID vs depth line plots
        - BaselineComparisonPlotter: Bar charts across baselines
        - HeatmapVisualizer: 2D heatmaps of ID landscapes
        - InteractiveDashboard: Plotly-based interactive exploration
    """
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Generate visualizations and save to output_path."""
        raise NotImplementedError("Subclasses must implement visualization logic")


class StatisticalTestAnalyzer(BaselineAnalyzer):
    """Base class for statistical test analyzers.
    
    Subclasses compute statistical comparisons between baselines or
    perform hypothesis testing on baseline properties.
    
    Example subclasses:
        - BaselineComparisonTest: T-tests or Mann-Whitney U tests between baselines
        - AnomalyDetector: Identify outliers in ID estimates
        - TrendAnalyzer: Detect statistically significant patterns in depth-ID curves
        - CorrelationAnalyzer: Correlate baseline ID with model properties
    """
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Perform statistical tests; write results to JSONL."""
        raise NotImplementedError("Subclasses must implement statistical testing logic")


# ============================================================================
# Example: Skeleton for a custom visualization analyzer
# ============================================================================

class CustomVisualizationExample(VisualizationAnalyzer):
    """Example skeleton showing how to implement a custom visualization analyzer.
    
    To implement:
        1. Load JSONL results (id.jsonl, overlap.jsonl, etc.)
        2. Parse and aggregate data
        3. Create plots using matplotlib/plotly/seaborn
        4. Save to output_path / plots/
        5. Return metadata (n_plots, files_written, etc.)
    """
    
    def __init__(self, plot_dir: str = "plots"):
        self.plot_dir = plot_dir
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Example implementation structure."""
        # Create output directory
        plot_output = output_path.parent / self.plot_dir
        plot_output.mkdir(exist_ok=True)
        
        # TODO: Load JSONL files (id.jsonl, overlap.jsonl, etc.)
        # TODO: Parse and aggregate data by depth, estimator, etc.
        # TODO: Create plots
        # TODO: Save plots to plot_output/
        
        return {
            "analyzer": "CustomVisualizationExample",
            "baseline": baseline,
            "n_plots": 0,  # Update with actual count
            "plot_directory": str(plot_output),
        }


# ============================================================================
# Example: Skeleton for a custom statistical test analyzer
# ============================================================================

class CustomStatisticalTestExample(StatisticalTestAnalyzer):
    """Example skeleton showing how to implement a custom statistical test.
    
    To implement:
        1. Load ID estimates from multiple baselines or samples
        2. Prepare data (handle NaN, normalize if needed)
        3. Run statistical test (t-test, Mann-Whitney U, KS test, etc.)
        4. Compute p-values, effect sizes, confidence intervals
        5. Write results to JSONL
    """
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Example implementation structure."""
        # TODO: Load ID estimates from output_path (or from H5 if needed)
        # TODO: Prepare data (filter, normalize, aggregate)
        # TODO: Run statistical test
        # TODO: Write results to output_path
        
        results = {
            "analyzer": "CustomStatisticalTestExample",
            "baseline": baseline,
            "n_tests": 0,  # Update with actual count
            "tests": [],  # List of test results
        }
        
        with output_path.open("a") as f:
            f.write(json.dumps(results) + "\n")
        
        return results
