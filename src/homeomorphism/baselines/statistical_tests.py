"""Statistical test analyzers for baseline comparisons.

Provides framework for hypothesis testing, anomaly detection, and pattern
analysis on baseline results.

Example use cases:
  - Compare ID distributions between baselines (t-test, Mann-Whitney U)
  - Detect anomalous depth trajectories
  - Correlate baseline properties with model architecture
  - Trend analysis (e.g., is ID increasing/decreasing with depth?)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from homeomorphism.interventions import BaselineName

from .analysis import BaselineAnalyzer


def _load_id_data(jsonl_path: Path, target_baseline: str) -> dict[str, dict[int, list[float]]]:
    """Load ID estimates from JSONL for a specific baseline."""
    data = defaultdict(lambda: defaultdict(list))
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("baseline") == target_baseline:
                est = row.get("estimator")
                depth = row.get("depth")
                val = row.get("id_estimate")
                if est and depth is not None and val is not None:
                    data[est][depth].append(val)
    return data


class StatisticalComparisonAnalyzer(BaselineAnalyzer):
    """Compares ID distributions across multiple baselines.
    
    Runs statistical tests (t-test, Mann-Whitney U, etc.) to determine if
    differences in ID estimates between baselines are statistically significant.
    
    Output JSONL schema:
        {
            "baseline": str,
            "comparison_type": "vs_trained" | "depth_trend",
            "test_name": "ttest" | "mannwhitney" | "ks" | "wasserstein",
            "statistic": float,
            "p_value": float,
            "effect_size": float,
            "result": "significant" | "not_significant",
        }
    """
    
    def __init__(
        self,
        reference_baseline: str = "trained",
        test_type: str = "mannwhitney",
        alpha: float = 0.05,
    ):
        """Initialize statistical test analyzer.
        
        Args:
            reference_baseline: The baseline to compare against (e.g., 'trained')
            test_type: Statistical test to use ('ttest', 'mannwhitney', 'ks', 'wasserstein')
            alpha: Significance level (default: 0.05)
        """
        self.reference_baseline = reference_baseline
        if test_type not in ("ttest", "mannwhitney", "ks", "wasserstein"):
            raise ValueError(f"unknown test_type: {test_type!r}")
        self.test_type = test_type
        self.alpha = alpha
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Run statistical tests; write results to JSONL."""
        n_tests = 0
        
        # Load ID data from JSONL
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        if not id_jsonl.exists():
            return {
                "analyzer": "StatisticalComparisonAnalyzer",
                "baseline": baseline,
                "n_tests": 0,
                "error": f"idanalyzer.jsonl not found",
            }
        
        if baseline == self.reference_baseline:
            return {
                "analyzer": "StatisticalComparisonAnalyzer",
                "baseline": baseline,
                "n_tests": 0,
                "status": "skipped (is reference)",
            }

        base_data = _load_id_data(id_jsonl, baseline)
        ref_data = _load_id_data(id_jsonl, self.reference_baseline)

        with output_path.open("a") as f:
            for est in base_data:
                if est not in ref_data:
                    continue
                for depth in base_data[est]:
                    if depth not in ref_data[est]:
                        continue
                    
                    vals_b = np.array(base_data[est][depth])
                    vals_r = np.array(ref_data[est][depth])
                    
                    if len(vals_b) < 2 or len(vals_r) < 2:
                        continue
                        
                    comp = compare_distributions(vals_b, vals_r, self.test_type, self.alpha)
                    row = {
                        "baseline": baseline,
                        "reference_baseline": self.reference_baseline,
                        "depth": depth,
                        "estimator": est,
                        "test_name": self.test_type,
                        "statistic": float(comp["statistic"]) if np.isfinite(comp["statistic"]) else None,
                        "p_value": float(comp["p_value"]) if np.isfinite(comp["p_value"]) else None,
                        "significant": comp["significant"]
                    }
                    f.write(json.dumps(row) + "\n")
                    n_tests += 1
        
        return {
            "analyzer": "StatisticalComparisonAnalyzer",
            "baseline": baseline,
            "n_tests": n_tests,
            "test_type": self.test_type,
            "alpha": self.alpha,
        }


class AnomalyDetectionAnalyzer(BaselineAnalyzer):
    """Detects anomalous ID estimates using statistical methods.
    
    Uses IQR, Z-score, or MAD (Median Absolute Deviation) to identify outliers in
    ID estimates that may indicate:
      - Unstable optimization
      - Numerical issues
      - Genuine model phenomena
    
    Output JSONL schema:
        {
            "baseline": str,
            "depth": int,
            "estimator": str,
            "n_total": int,
            "n_anomalies": int,
            "anomaly_rate": float,
            "anomaly_indices": [int, ...],
        }
    """
    
    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
    ):
        """Initialize anomaly detector.
        
        Args:
            method: Detection method ('iqr', 'zscore', or 'mad')
            threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3)
        """
        if method not in ("iqr", "zscore", "mad"):
            raise ValueError(f"unknown method: {method!r}")
        self.method = method
        self.threshold = threshold
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Detect anomalies in ID estimates."""
        n_anomalies_found = 0
        n_groups_checked = 0
        
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        if not id_jsonl.exists():
            return {"analyzer": "AnomalyDetectionAnalyzer", "baseline": baseline, "n_anomalies_found": 0}
            
        base_data = _load_id_data(id_jsonl, baseline)
        
        with output_path.open("a") as f:
            for est, depths_dict in base_data.items():
                for depth, vals in depths_dict.items():
                    arr = np.array(vals)
                    if len(arr) < 4:
                        continue
                        
                    if self.method == "iqr":
                        anomalies = detect_outliers_iqr(arr, self.threshold)
                    elif self.method == "mad":
                        anomalies = detect_outliers_mad(arr, self.threshold)
                    else:
                        anomalies = detect_outliers_zscore(arr, self.threshold)
                        
                    n_anom = int(np.sum(anomalies))
                    n_anomalies_found += n_anom
                    n_groups_checked += 1
                    
                    row = {
                        "baseline": baseline,
                        "depth": depth,
                        "estimator": est,
                        "n_total": len(arr),
                        "n_anomalies": n_anom,
                        "anomaly_rate": float(n_anom / len(arr))
                    }
                    f.write(json.dumps(row) + "\n")
        
        return {
            "analyzer": "AnomalyDetectionAnalyzer",
            "baseline": baseline,
            "n_anomalies_found": n_anomalies_found,
            "n_groups_checked": n_groups_checked,
            "method": self.method,
            "threshold": self.threshold,
        }


class TrendAnalysisAnalyzer(BaselineAnalyzer):
    """Analyzes trends in ID vs depth.
    
    Fits trends to ID vs depth curves and tests for statistically significant patterns:
      - Is ID increasing with depth? (Spearman correlation, Kendall's Tau)
      - What is the rate of change? (linear regression slope)
    
    Output JSONL schema:
        {
            "baseline": str,
            "estimator": str,
            "trend": "increasing" | "decreasing" | "no_trend",
            "slope": float,
            "r_squared": float,
            "p_value": float,
            "correlation": float,
            "kendall_tau": float,
            "kendall_p": float,
        }
    """
    
    def __init__(self):
        """Initialize trend analyzer."""
        pass
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Analyze depth-ID trends."""
        n_trends_analyzed = 0
        
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        if not id_jsonl.exists():
            return {"analyzer": "TrendAnalysisAnalyzer", "baseline": baseline, "n_trends_analyzed": 0}
            
        base_data = _load_id_data(id_jsonl, baseline)
        
        with output_path.open("a") as f:
            for est, depths_dict in base_data.items():
                depths = sorted(depths_dict.keys())
                if len(depths) < 3:
                    continue
                    
                means = [np.mean(depths_dict[d]) for d in depths]
                
                res_lin = stats.linregress(depths, means)
                rho, p_rho = stats.spearmanr(depths, means)
                tau, p_tau = stats.kendalltau(depths, means)
                
                if res_lin.pvalue < 0.05:
                    trend = "increasing" if res_lin.slope > 0 else "decreasing"
                else:
                    trend = "no_trend"
                    
                row = {
                    "baseline": baseline,
                    "estimator": est,
                    "trend": trend,
                    "linear_slope": float(res_lin.slope) if np.isfinite(res_lin.slope) else None,
                    "r_squared": float(res_lin.rvalue**2) if np.isfinite(res_lin.rvalue) else None,
                    "p_value_linear": float(res_lin.pvalue) if np.isfinite(res_lin.pvalue) else None,
                    "spearman_rho": float(rho) if np.isfinite(rho) else None,
                    "spearman_p": float(p_rho) if np.isfinite(p_rho) else None,
                    "kendall_tau": float(tau) if np.isfinite(tau) else None,
                    "kendall_p": float(p_tau) if np.isfinite(p_tau) else None
                }
                f.write(json.dumps(row) + "\n")
                n_trends_analyzed += 1
        
        return {
            "analyzer": "TrendAnalysisAnalyzer",
            "baseline": baseline,
            "n_trends_analyzed": n_trends_analyzed,
        }


class CorrelationAnalyzer(BaselineAnalyzer):
    """Correlates baseline properties with model architecture/performance.
    
    Useful for understanding if baseline ID patterns correlate with:
      - Model size (# parameters)
      - Model depth (# layers)
      - Model width (hidden dimension)
      - Actual model accuracy on tasks
    """
    
    def run(
        self,
        h5_path: Path,
        output_path: Path,
        baseline: BaselineName,
    ) -> dict[str, Any]:
        """Compute correlations between baseline metrics and model properties."""
        n_correlations = 0
        
        id_jsonl = output_path.parent / "idanalyzer.jsonl"
        overlap_jsonl = output_path.parent / "overlapanalyzer.jsonl"
        
        if not id_jsonl.exists():
            return {"analyzer": "CorrelationAnalyzer", "baseline": baseline, "n_correlations": 0}
            
        # Load ID means
        id_data = _load_id_data(id_jsonl, baseline)
        id_means = {est: {d: np.mean(v) for d, v in depth_dict.items()} for est, depth_dict in id_data.items()}
        
        # Load Overlap means
        overlap_means = {}
        if overlap_jsonl.exists():
            overlap_raw = defaultdict(list)
            with overlap_jsonl.open() as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    if row.get("baseline") == baseline:
                        d_to = row.get("depth_to")
                        val = row.get("neighbour_overlap")
                        if d_to is not None and val is not None:
                            overlap_raw[d_to].append(val)
            overlap_means = {d: np.mean(v) for d, v in overlap_raw.items()}
            
        with output_path.open("a") as f:
            estimators = list(id_means.keys())
            
            # 1. Correlate ID with Neighborhood Overlap (Structural properties)
            if overlap_means:
                for est in estimators:
                    common_depths = sorted(set(id_means[est].keys()) & set(overlap_means.keys()))
                    if len(common_depths) > 2:
                        v_id = [id_means[est][d] for d in common_depths]
                        v_ov = [overlap_means[d] for d in common_depths]
                        
                        rho, p = stats.spearmanr(v_id, v_ov)
                        row = {
                            "baseline": baseline,
                            "metric_x": f"id_{est}",
                            "metric_y": "neighbour_overlap",
                            "correlation_type": "spearman",
                            "correlation": float(rho) if np.isfinite(rho) else None,
                            "p_value": float(p) if np.isfinite(p) else None
                        }
                        f.write(json.dumps(row) + "\n")
                        n_correlations += 1
        
        return {
            "analyzer": "CorrelationAnalyzer",
            "baseline": baseline,
            "n_correlations": n_correlations,
        }


# ============================================================================
# Utility functions for common statistical operations
# ============================================================================

def compare_distributions(
    data_a: np.ndarray,
    data_b: np.ndarray,
    test: str = "mannwhitney",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare two distributions using specified statistical test.
    
    Args:
        data_a, data_b: Arrays of values to compare
        test: Test type ('ttest', 'mannwhitney', 'ks', 'wasserstein')
        alpha: Significance level
    
    Returns:
        Dictionary with test statistic, p-value, and result interpretation
    """
    if test == "ttest":
        stat, pval = stats.ttest_ind(data_a, data_b)
    elif test == "mannwhitney":
        stat, pval = stats.mannwhitneyu(data_a, data_b)
    elif test == "ks":
        stat, pval = stats.ks_2samp(data_a, data_b)
    elif test == "wasserstein":
        stat = stats.wasserstein_distance(data_a, data_b)
        pval = float('nan') # Wasserstein is a distance metric, no explicit p-value in scipy
    else:
        raise ValueError(f"unknown test: {test!r}")
    
    return {
        "test": test,
        "statistic": float(stat),
        "p_value": float(pval) if np.isfinite(pval) else None,
        "significant": pval < alpha if np.isfinite(pval) else None,
        "alpha": alpha,
    }


def detect_outliers_iqr(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """Detect outliers using interquartile range method.
    
    Args:
        data: Array of values
        multiplier: IQR multiplier (default 1.5 for standard outliers)
    
    Returns:
        Boolean array indicating outliers
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method.
    
    Args:
        data: Array of values
        threshold: Z-score threshold (default 3.0)
    
    Returns:
        Boolean array indicating outliers
    """
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold


def detect_outliers_mad(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Detect outliers using the Median Absolute Deviation (MAD) method.
    Highly robust for small samples sizes with extreme outliers.
    
    Args:
        data: Array of values
        threshold: Modified Z-score threshold (default 3.5)
    
    Returns:
        Boolean array indicating outliers
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros_like(data, dtype=bool)
    modified_z_score = 0.6745 * (data - median) / mad
    return np.abs(modified_z_score) > threshold
