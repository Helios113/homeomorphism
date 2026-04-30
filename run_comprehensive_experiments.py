#!/usr/bin/env python3
"""Comprehensive orchestrator for all baseline and synthetic experiments within 18GB GPU memory.

This script runs all combinations of:
- Models: GPT-2 + tiny variants + LLaMA-style models
- Baselines: groupA (trained, topological, max_entropy) + groupB (syntactic, semantic)
- Synthetic: All manifolds × model configurations
- Parameters: Conservative settings that fit in 18GB GPU memory

Memory allocation strategy:
- Tiny models: Very low memory (<500MB each run)
- GPT-2: Medium memory (~2GB each run)
- LLaMA models: Higher memory (up to 8GB for largest)
- Batch sizes adjusted to stay under 18GB total
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ============================================================================
# MODEL CONFIGURATIONS (All fit in 18GB GPU memory)
# ============================================================================

# Tiny GPT-2 variants (very low memory)
TINY_MODELS = [
    "nano-gpt2-4l-128d",   # ~2M params, <100MB
    "tiny-gpt2-4l-256d",   # ~8M params, <200MB
    "tiny-gpt2-6l-256d",   # ~12M params, <300MB
    "micro-gpt2-4l-384d",  # ~18M params, <500MB
    "micro-gpt2-6l-384d",  # ~26M params, <700MB
    "tiny-gpt2-8l-256d",   # ~16M params, <400MB
]

# Standard GPT-2 (medium memory)
STANDARD_MODELS = [
    "gpt2",                # ~124M params, ~2GB
    "distilgpt2",          # ~82M params, ~1.5GB
]

# LLaMA-style models (higher memory but still fits)
LLAMA_MODELS = [
    "llama-2l-20d-10m",    # Custom LLaMA-2L, ~1GB
    "llama-4l-32d-16m",    # Custom LLaMA-4L, ~2GB
    "pythia-2l-20d",       # Custom Pythia-2L, ~1GB
    "qwen-2l-20d",         # Custom Qwen-2L, ~1GB
]

ALL_MODELS = TINY_MODELS + STANDARD_MODELS + LLAMA_MODELS

# ============================================================================
# BASELINE CONFIGURATIONS
# ============================================================================

BASELINE_CONFIGS = {
    "groupA": {
        "baselines": "trained,topological_initialisation,maximum_entropy_injection",
        "n_samples": 16,       # Balanced sample size for reliable ID estimation
        "max_tokens": 16,      # Increased to 16 for better manifold sampling
        "granularity": "full_stream",  # Use all tokens for maximum statistical power
        "estimator": "twonn,ess",
        "overlap_k": 5,
    },
    "groupB": {
        "baselines": "syntactic_disintegration,semantic_scrambling",
        "n_samples": 16,       # Balanced sample size (was 64, reduced to 16 for consistency)
        "max_tokens": 16,      # Consistent token count across all baselines
        "granularity": "full_stream",  # Use all tokens for maximum statistical power
        "estimator": "twonn,ess,participation_ratio",
        "overlap_k": 5,
    },
}

# ============================================================================
# SYNTHETIC CONFIGURATIONS
# ============================================================================

SYNTHETIC_MANIFOLDS = [
    "hyperplane",
    "sphere",
    "torus",
    "swiss_roll",
    "white_noise"
]

SYNTHETIC_MODELS = [
    "toy-2l-32d",          # Very small toy model
    "toy-4l-64d",          # Small toy model
    "llama-2l-20d-10m",    # Small LLaMA model
    "pythia-2l-20d",       # Small Pythia model
    "qwen-2l-20d",         # Small Qwen model
]

# ============================================================================
# MEMORY-AWARE PARAMETER TUNING
# ============================================================================

def get_memory_aware_params(model_name: str, experiment_type: str, memory_safe: bool = False, force_cpu: bool = False) -> dict[str, Any]:
    """Return memory-safe parameters based on model and experiment type."""

    # Base parameters
    params = {
        "device": "cpu" if force_cpu else "cuda",
        "seed": 42,
    }

    # Model-specific adjustments (ultra-conservative for reliable execution)
    if model_name in TINY_MODELS:
        # Tiny models: balanced 16 samples × 16 tokens for consistent ID estimation
        base_samples = 16 if memory_safe else 32
        base_tokens = 16 if memory_safe else 32
        base_batch = 4 if memory_safe else 8
        params.update({
            "n_samples": base_samples,
            "max_tokens": base_tokens,
            "batch_size": base_batch,
        })
    elif model_name in STANDARD_MODELS:
        # Standard models: balanced for reliable ID estimation (16 samples × 16 tokens)
        base_samples = 16 if memory_safe else 32
        base_tokens = 16 if memory_safe else 32
        base_batch = 2 if memory_safe else 4
        params.update({
            "n_samples": base_samples,
            "max_tokens": base_tokens,
            "batch_size": base_batch,
        })
    elif model_name in LLAMA_MODELS:
        # LLaMA models: balanced for reliable ID estimation (16 samples × 16 tokens)
        base_samples = 16 if memory_safe else 32
        base_tokens = 16 if memory_safe else 32
        base_batch = 1 if memory_safe else 2
        params.update({
            "n_samples": base_samples,
            "max_tokens": base_tokens,
            "batch_size": base_batch,
        })
    else:
        # Unknown model: balanced parameters
        if experiment_type == "synthetic":
            base_samples = 16 if memory_safe else 32
            base_tokens = 16 if memory_safe else 32
        else:
            base_samples = 2 if memory_safe else 4
            base_tokens = 2 if memory_safe else 4
        params.update({
            "n_samples": base_samples,
            "max_tokens": base_tokens,
            "batch_size": 1,
        })

    # Experiment-specific overrides
    if experiment_type == "baseline":
        # Baselines need input processing, more conservative
        params["n_samples"] = min(params["n_samples"], 32 if memory_safe else 64)
    elif experiment_type == "synthetic":
        # Synthetics can be larger
        params["n_samples"] = min(params["n_samples"], 128 if memory_safe else 256)

    return params

# ============================================================================
# EXECUTION HELPERS
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

def _get_python_executable():
    """Get the correct Python executable from the virtual environment."""
    # Try to find uv virtual environment
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    # Fall back to current python
    return sys.executable

def clear_gpu_memory():
    """Attempt to clear GPU memory between runs."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 Cleared GPU memory cache")
    except Exception as e:
        print(f"⚠️  Could not clear GPU memory: {e}")

def run_command(cmd: list[str], description: str, *, allow_memory_errors: bool = True) -> bool:
    """Run a command with logging and error handling."""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=False,  # Show output live
            text=True
        )

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            # Check if it's a CUDA out of memory error
            if allow_memory_errors and "CUDA out of memory" in str(result.stderr):
                print(f"💾 MEMORY ERROR: {description} - CUDA out of memory")
                print("   This is expected with large models. Try reducing parameters or using CPU.")
                clear_gpu_memory()
                return False
            else:
                print(f"❌ FAILED: {description} (exit code: {result.returncode})")
                return False

    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        clear_gpu_memory()
        return False

    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False

# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

def run_baseline_experiments(output_root: Path, args) -> dict[str, list[bool]]:
    """Run all baseline experiments for all compatible models."""
    results = {}

    # GroupA can only run on GPT-2 (requires trained weights)
    groupA_models = [m for m in ALL_MODELS if m in ["gpt2", "distilgpt2"]]

    # GroupB can run on all models
    groupB_models = ALL_MODELS

    print(f"BASELINE EXPERIMENTS:")
    print(f"  GroupA models: {groupA_models}")
    print(f"  GroupB models: {groupB_models}")
    print(f"  Output root: {output_root}")

    # Filter models if requested
    model_filter = args.models
    group_filter = args.groups
    if model_filter:
        groupA_models = [m for m in groupA_models if m in model_filter]
        groupB_models = [m for m in groupB_models if m in model_filter]

    # Filter groups if requested
    run_groupA = not group_filter or "groupA" in group_filter
    run_groupB = not group_filter or "groupB" in group_filter

    # Run GroupA (trained models only)
    results["groupA"] = []
    if run_groupA:
        for model in groupA_models:
            params = get_memory_aware_params(model, "baseline", args.memory_safe, args.force_cpu)
            config = BASELINE_CONFIGS["groupA"]

            cmd = [
                _get_python_executable(), "-m", "experiments.exp3_section2_baselines",
                "--model", model,
                "--weights", "trained",
                "--corpus", "shakespeare",
                "--n-samples", str(params["n_samples"]),
                "--max-tokens", str(params["max_tokens"]),
                "--baselines", config["baselines"],
                "--granularity", config["granularity"],
                "--estimator", config["estimator"],
                "--overlap-k", str(config["overlap_k"]),
                "--seed", str(params["seed"]),
                "--device", params["device"],
                "--output-root", str(output_root / "baselines/groupA" / model),
            ]

            success = run_command(cmd, f"GroupA baseline for {model}")
            results["groupA"].append(success)

    # Run GroupB (all models)
    results["groupB"] = []
    if run_groupB:
        for model in groupB_models:
            params = get_memory_aware_params(model, "baseline", args.memory_safe, args.force_cpu)

            # Determine weight mode
            weight_mode = "trained" if model in ["gpt2", "distilgpt2"] else "random_gaussian"
            config = BASELINE_CONFIGS["groupB"]

            cmd = [
                _get_python_executable(), "-m", "experiments.exp3_section2_baselines",
                "--model", model,
                "--weights", weight_mode,
                "--corpus", "shakespeare",
                "--n-samples", str(params["n_samples"]),
                "--max-tokens", str(params["max_tokens"]),
                "--baselines", config["baselines"],
                "--granularity", config["granularity"],
                "--estimator", config["estimator"],
                "--overlap-k", str(config["overlap_k"]),
                "--seed", str(params["seed"]),
                "--device", params["device"],
                "--output-root", str(output_root / "baselines/groupB" / model),
            ]

            success = run_command(cmd, f"GroupB baseline for {model}")
            results["groupB"].append(success)

    return results

# ============================================================================
# SYNTHETIC EXPERIMENTS
# ============================================================================

def run_synthetic_experiments(output_root: Path, args) -> dict[str, list[bool]]:
    """Run all synthetic manifold experiments for all model configurations."""
    results = {}

    print(f"SYNTHETIC EXPERIMENTS:")
    print(f"  Manifolds: {SYNTHETIC_MANIFOLDS}")
    print(f"  Models: {SYNTHETIC_MODELS}")
    print(f"  Output root: {output_root}")

    # Apply filters
    manifolds_to_run = args.manifolds if args.manifolds else SYNTHETIC_MANIFOLDS
    model_configs_to_run = args.model_configs if args.model_configs else SYNTHETIC_MODELS

    results["synthetic"] = []
    for manifold in manifolds_to_run:
        for model_config in model_configs_to_run:
            params = get_memory_aware_params(model_config, "synthetic", args.memory_safe, args.force_cpu)

            # Determine manifold dimension for hyperplane
            manifold_dim = None
            if manifold == "hyperplane":
                # Extract from model config or set reasonable default
                if "32d" in model_config:
                    manifold_dim = 16  # Half of 32
                elif "64d" in model_config:
                    manifold_dim = 32  # Half of 64
                elif "20d" in model_config:
                    manifold_dim = 10  # Reasonable for small model
                else:
                    manifold_dim = 8   # Safe default

            # Determine model type
            model_type = "toy" if model_config.startswith("toy-") else "llama"

            cmd = [
                _get_python_executable(), "-m", "experiments.exp3_synthetic_manifolds",
                "--manifold-type", manifold,
                "--model-type", model_type,
                "--model-config", model_config,
                "--n-samples", str(params["n_samples"]),
                "--batch-size", str(params["batch_size"]),
                "--seq-len", str(params["max_tokens"]),
                "--seed", str(params["seed"]),
                "--device", params["device"],
                "--save", str(output_root / "synthetic" / manifold / model_config / "42" / "latents.h5"),
            ]

            if manifold_dim is not None:
                cmd.extend(["--manifold-dim", str(manifold_dim)])

            success = run_command(cmd, f"Synthetic {manifold} × {model_config}")
            results["synthetic"].append(success)

            # If exp3 succeeded, run exp4 to generate ID estimates
            if success:
                latents_path = output_root / "synthetic" / manifold / model_config / "42" / "latents.h5"
                exp4_cmd = [
                    _get_python_executable(), "-m", "experiments.exp4_id_estimation",
                    "--latents", str(latents_path),
                    "--token", "0",  # default token
                    "--depth", "0", "1", "2", "3", "4", "5", "6", "7", "8",  # all depths for toy models
                    "--estimator", "twonn", "ess", "participation_ratio",
                    "--save", str(latents_path.parent / "exp4_id_estimates.jsonl"),
                ]

                exp4_success = run_command(exp4_cmd, f"ID estimation for {manifold} × {model_config}")
                # Note: We don't add exp4 success to results since exp3 success is what matters

    return results

# ============================================================================
# ANALYSIS
# ============================================================================

def run_analysis(output_root: Path) -> dict[str, bool]:
    """Run comprehensive analysis on all generated results."""
    results = {}

    # Analyze baselines
    if (output_root / "baselines").exists():
        print(f"\n{'='*60}")
        print("ANALYZING BASELINE RESULTS")
        print(f"{'='*60}")

        # Run baseline analysis
        cmd = [
            sys.executable, "-c",
            f"""
import sys
sys.path.append('.')
from experiments.plot_baseline_id import main as plot_main
import sys
sys.argv = ['plot_baseline_id', '--results-root', '{output_root}/baselines', '--estimator', 'twonn', '--output-dir', '{output_root}/baselines/plots']
plot_main()
"""
        ]
        results["baseline_analysis"] = run_command(cmd, "Baseline analysis and plotting")

    # Analyze synthetics
    if (output_root / "synthetic").exists():
        print(f"\n{'='*60}")
        print("ANALYZING SYNTHETIC RESULTS")
        print(f"{'='*60}")

        # Run synthetic analysis
        cmd = [
            sys.executable, "-c",
            f"""
import sys
sys.path.append('.')
from experiments.visualize_synthetic_results import main as viz_main
import sys
sys.argv = ['visualize_synthetic_results', '--results-root', '{output_root}/synthetic']
viz_main()
"""
        ]
        results["synthetic_analysis"] = run_command(cmd, "Synthetic analysis and plotting")

    return results

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive baseline and synthetic experiments within 18GB GPU memory"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/comprehensive_experiments"),
        help="Root directory for all experiment outputs"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baselines", "synthetic", "analysis", "all"],
        default=["all"],
        help="Which experiments to run (default: all)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to run (default: all available)"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["groupA", "groupB"],
        default=None,
        help="Specific baseline groups to run (default: all compatible)"
    )
    parser.add_argument(
        "--manifolds",
        nargs="+",
        default=None,
        help="Specific manifolds to run (default: all)"
    )
    parser.add_argument(
        "--model-configs",
        nargs="+",
        default=None,
        help="Specific model configs for synthetics (default: all)"
    )
    parser.add_argument(
        "--n-samples-override",
        type=int,
        help="Override n_samples for all experiments (for testing)"
    )
    parser.add_argument(
        "--max-tokens-override",
        type=int,
        help="Override max_tokens for all experiments (for testing)"
    )
    parser.add_argument(
        "--device-override",
        choices=["cpu", "cuda"],
        help="Override device for all experiments"
    )
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Use ultra-conservative memory settings (recommended for first runs)"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution for all experiments"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis phase (just run experiments)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"COMPREHENSIVE EXPERIMENT SUITE")
    print(f"{'='*50}")
    print(f"Timestamp: {timestamp}")
    print(f"Output root: {args.output_root}")
    print(f"Experiments: {args.experiments}")
    print(f"Memory safe mode: {args.memory_safe}")
    print(f"Force CPU: {args.force_cpu}")

    # GPU memory check
    if not args.force_cpu:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                print(f"GPU memory available: {gpu_memory:.1f}GB")
                if gpu_memory < 8:
                    print("⚠️  WARNING: GPU has less than 8GB memory. Consider using --force-cpu or --memory-safe")
            else:
                print("⚠️  No CUDA GPU available, will use CPU")
                args.force_cpu = True
        except Exception as e:
            print(f"⚠️  Could not check GPU memory: {e}")
    else:
        print("CPU mode enabled")

    print()

    # Track all results
    all_results = {}

    # Run baselines
    if "baselines" in args.experiments or "all" in args.experiments:
        print(f"{'='*60}")
        print("PHASE 1: BASELINE EXPERIMENTS")
        print(f"{'='*60}")
        baseline_results = run_baseline_experiments(args.output_root, args)
        all_results.update(baseline_results)

    # Run synthetics
    if "synthetic" in args.experiments or "all" in args.experiments:
        print(f"\n{'='*60}")
        print("PHASE 2: SYNTHETIC EXPERIMENTS")
        print(f"{'='*60}")
        synthetic_results = run_synthetic_experiments(args.output_root, args)
        all_results.update(synthetic_results)

    # Run analysis
    if not args.skip_analysis and ("analysis" in args.experiments or "all" in args.experiments):
        print(f"\n{'='*60}")
        print("PHASE 3: ANALYSIS AND VISUALIZATION")
        print(f"{'='*60}")
        analysis_results = run_analysis(args.output_root)
        all_results.update(analysis_results)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_root}")
    print(f"Total experiments attempted: {sum(len(v) if isinstance(v, list) else 1 for v in all_results.values())}")

    for category, results in all_results.items():
        if isinstance(results, list):
            success_count = sum(results)
            total_count = len(results)
            print(f"{category}: {success_count}/{total_count} successful")
        else:
            status = "✅" if results else "❌"
            print(f"{category}: {status}")

    print(f"\n{'='*60}")
    print("EXPERIMENT SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"Check {args.output_root} for all results and plots")

if __name__ == "__main__":
    main()