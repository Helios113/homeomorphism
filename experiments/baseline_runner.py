"""Unified baseline runner orchestrating capture + analysis pipeline.

This is the main entry point for baseline experiments, replacing the three
separate orchestrators (run_baseline_configs_gpu.py, run_tiny_model_baselines.py,
run_comprehensive_experiments.py baseline section).

Features:
  - Single CLI with all arguments from previous 3 runners
  - Modular analyzer pipeline (ID, overlap, future: stats, visualization)
  - Unified configuration from baseline_configs.py
  - Memory-aware parameter selection
  - Comprehensive logging and result tracking

Usage:
    python experiments/baseline_runner.py \
        --model gpt2 \
        --baseline-group groupA \
        --n-samples 32 \
        --max-tokens 16 \
        --output-root results/baselines

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from homeomorphism.baselines import (
    BaselineConfig,
    BaselineGroup,
    MemoryProfile,
    LatentCapture,
    AnalyzerPipeline,
    IDAnalyzer,
    OverlapAnalyzer,
    BASELINE_GROUPS,
)
from homeomorphism.baselines.statistical_tests import (
    StatisticalComparisonAnalyzer,
    AnomalyDetectionAnalyzer,
    TrendAnalysisAnalyzer,
    CorrelationAnalyzer,
)
from homeomorphism.baselines.visualization import (
    DepthTrajectoryVisualizer,
    BaselineComparisonVisualizer,
    OverlapTrajectoryVisualizer,
    EstimatorCorrelationVisualizer,
)
from homeomorphism.interventions import VALID_BASELINES, BaselineName


class BaselineRunner:
    """Orchestrates complete baseline pipeline: capture + analysis.
    
    Two-phase workflow:
        1. Capture: Load model + corpus, forward pass, hook capture, HDF5 persistence
        2. Analysis: Read HDF5, compute ID, overlap, write JSONL results
    """
    
    def __init__(self, config: BaselineConfig):
        """Initialize runner with configuration."""
        self.config = config
        self.run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = (
            Path(config.output_root)
            / config.baseline_group.name
            / config.model_name
            / self.run_timestamp
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> dict[str, dict]:
        """Execute full pipeline (capture + analysis) for all baselines in group.
        
        Returns:
            Dictionary with results for each baseline.
        """
        print(f"\n{'='*70}")
        print(f"BASELINE RUNNER")
        print(f"{'='*70}")
        print(f"Model          : {self.config.model_name}")
        print(f"Group          : {self.config.baseline_group.name}")
        print(f"Baselines      : {', '.join(self.config.baseline_group.baselines)}")
        print(f"Run directory  : {self.run_dir}")
        print(f"Corpus         : {self.config.corpus}")
        print(f"Samples        : {self.config.memory.n_samples}")
        print(f"Max tokens     : {self.config.memory.max_tokens}")
        print(f"Device         : {self.config.memory.device}")
        print(f"{'='*70}\n")
        
        # Save config
        self._save_config()
        
        results = {}
        
        # Phase 1: Capture for each baseline
        print("\n[PHASE 1] CAPTURING LATENTS")
        print("-" * 70)
        capture = LatentCapture(self.config)
        
        for baseline in self.config.baseline_group.baselines:
            print(f"\n► Capturing {baseline}...")
            try:
                capture_stats = capture.run(baseline)
                results[baseline] = {"phase": "capture", **capture_stats}
                print(f"  ✓ Complete: {capture_stats['n_samples_kept']} samples captured")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[baseline] = {"phase": "capture", "error": str(e)}
                continue
        
        # Phase 2: Analyze
        print(f"\n[PHASE 2] ANALYZING RESULTS")
        print("-" * 70)
        
        # Build analyzer pipeline
        pipeline = self._build_analyzer_pipeline()
        
        h5_path = Path(self.config.output_root) / "latents.h5"
        
        for baseline in self.config.baseline_group.baselines:
            if baseline not in results or "error" in results[baseline]:
                print(f"\n► Skipping analysis for {baseline} (capture failed)")
                continue
            
            print(f"\n► Analyzing {baseline}...")
            try:
                analysis_results = pipeline.run(
                    h5_path=h5_path,
                    output_dir=self.run_dir,
                    baseline=baseline,
                )
                
                # Merge analysis results into baseline results
                if "phase" in results[baseline]:
                    results[baseline].pop("phase")
                results[baseline]["analysis"] = analysis_results
                
                # Summary
                for analyzer_name, stats in analysis_results.items():
                    if "error" not in stats:
                        print(f"  ✓ {analyzer_name}: {stats.get('n_rows', '?')} rows")
                    else:
                        print(f"  ✗ {analyzer_name}: {stats['error']}")
            
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[baseline]["analysis"] = {"error": str(e)}
                
        # Phase 3: Advanced Analysis & Visualization
        print(f"\n[PHASE 3] ADVANCED ANALYSIS & VISUALIZATION")
        print("-" * 70)
        
        advanced_pipeline = AnalyzerPipeline([
            AnomalyDetectionAnalyzer(),
            TrendAnalysisAnalyzer(),
            CorrelationAnalyzer(),
            StatisticalComparisonAnalyzer(reference_baseline="trained"),
            DepthTrajectoryVisualizer(output_dir="plots"),
            BaselineComparisonVisualizer(output_dir="plots"),
            OverlapTrajectoryVisualizer(output_dir="plots"),
            EstimatorCorrelationVisualizer(output_dir="plots"),
        ])
        
        for baseline in self.config.baseline_group.baselines:
            if baseline not in results or "error" in results[baseline]:
                continue
                
            print(f"\n► Advanced Analysis for {baseline}...")
            try:
                adv_results = advanced_pipeline.run(
                    h5_path=h5_path,
                    output_dir=self.run_dir,
                    baseline=baseline,
                )
                
                results[baseline]["analysis"].update(adv_results)
                
                for analyzer_name, stats in adv_results.items():
                    if "error" not in stats:
                        count = stats.get('n_plots', stats.get('n_tests', stats.get('n_anomalies_found', stats.get('n_trends_analyzed', stats.get('n_correlations', '?')))))
                        print(f"  ✓ {analyzer_name}: {count} items")
                    else:
                        print(f"  ✗ {analyzer_name}: {stats['error']}")
            
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[baseline]["analysis"]["advanced_error"] = str(e)
        
        # Save summary
        self._save_summary(results)
        
        print(f"\n[COMPLETE] Results saved to {self.run_dir}")
        return results
    
    def _build_analyzer_pipeline(self) -> AnalyzerPipeline:
        """Build analyzer pipeline from config."""
        pipeline = AnalyzerPipeline()
        
        # ID analyzer
        pipeline.add_analyzer(
            IDAnalyzer(
                granularities=self.config.baseline_group.granularities,
                estimators=self.config.baseline_group.estimators,
            )
        )
        
        # Overlap analyzer
        pipeline.add_analyzer(
            OverlapAnalyzer(
                k=self.config.baseline_group.overlap_k,
                granularities=["full_stream", "last_token"],
            )
        )
        
        return pipeline
    
    def _save_config(self) -> None:
        """Save configuration to JSON."""
        config_dict = {
            "model_name": self.config.model_name,
            "baseline_group": self.config.baseline_group.name,
            "baselines": self.config.baseline_group.baselines,
            "corpus": self.config.corpus,
            "n_samples": self.config.memory.n_samples,
            "max_tokens": self.config.memory.max_tokens,
            "batch_size": self.config.memory.batch_size,
            "device": self.config.memory.device,
            "weights": self.config.weights,
            "layers_spec": self.config.layers_spec,
            "granularities": self.config.baseline_group.granularities,
            "estimators": self.config.baseline_group.estimators,
            "overlap_k": self.config.baseline_group.overlap_k,
        }
        
        config_file = self.run_dir / "config.json"
        with config_file.open("w") as f:
            json.dump(config_dict, f, indent=2)
    
    def _save_summary(self, results: dict) -> None:
        """Save summary of results."""
        summary = {
            "run_timestamp": self.run_timestamp,
            "run_directory": str(self.run_dir),
            "config": {
                "model_name": self.config.model_name,
                "baseline_group": self.config.baseline_group.name,
            },
            "results": results,
        }
        
        summary_file = self.run_dir / "summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Unified baseline experiment runner (phase 1: capture + phase 2: analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run groupA on GPT-2
  python experiments/baseline_runner.py --model gpt2 --baseline-group groupA
  
  # Run groupB with custom parameters
  python experiments/baseline_runner.py \\
    --model tiny-gpt2-4l-256d \\
    --baseline-group groupB \\
    --n-samples 64 \\
    --max-tokens 32 \\
    --device cuda
  
  # Quick test
  python experiments/baseline_runner.py --model gpt2 --quick
        """,
    )
    
    # Model & experiment
    p.add_argument(
        "--model",
        default="gpt2",
        help="Model name (default: gpt2)",
    )
    p.add_argument(
        "--baseline-group",
        default="groupA",
        choices=list(BASELINE_GROUPS.keys()),
        help="Baseline group to run (default: groupA)",
    )
    p.add_argument(
        "--weights",
        default="trained",
        help="Weight mode: trained, random_gaussian (default: trained)",
    )
    
    # Data & corpus
    p.add_argument(
        "--corpus",
        default="shakespeare",
        help="Data source (default: shakespeare)",
    )
    p.add_argument(
        "--layers",
        dest="layers_spec",
        default="all",
        help="Layer selection: all, 0.attn, 0,1, etc. (default: all)",
    )
    
    # Memory & compute
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples (default: from memory profile)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per sequence (default: from memory profile)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from memory profile)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    # Analysis
    p.add_argument(
        "--granularity",
        nargs="+",
        default=None,
        help="Data granularities: full_stream per_token last_token (default: group-specific)",
    )
    p.add_argument(
        "--estimator",
        nargs="+",
        default=None,
        help="ID estimators: twonn ess participation_ratio (default: group-specific)",
    )
    p.add_argument(
        "--overlap-k",
        type=int,
        default=None,
        help="k for k-NN overlap (default: group-specific)",
    )
    
    # Output
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/baselines"),
        help="Output root directory (default: results/baselines)",
    )
    
    # Convenience
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: reduced samples/tokens",
    )
    
    return p


def main() -> None:
    """Main entry point."""
    args = _build_parser().parse_args()
    
    # Get baseline group
    baseline_group = BASELINE_GROUPS[args.baseline_group]
    
    # Build memory profile
    memory = MemoryProfile(
        n_samples=args.n_samples or (4 if args.quick else 32),
        max_tokens=args.max_tokens or (8 if args.quick else 16),
        batch_size=args.batch_size or 2,
        device=args.device,
        seed=args.seed,
    )
    
    # Override group settings if provided via CLI
    if args.granularity:
        baseline_group.granularities = args.granularity
    if args.estimator:
        baseline_group.estimators = args.estimator
    if args.overlap_k:
        baseline_group.overlap_k = args.overlap_k
    
    # Build config
    config = BaselineConfig(
        model_name=args.model,
        baseline_group=baseline_group,
        corpus=args.corpus,
        memory=memory,
        output_root=args.output_root,
        weights=args.weights,
        layers_spec=args.layers_spec,
    )
    
    # Run
    runner = BaselineRunner(config)
    results = runner.run()
    
    # Summary
    n_success = sum(1 for r in results.values() if "error" not in r)
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_success}/{len(results)} baselines completed successfully")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
