"""Orchestrate synthetic manifold experiments across manifold types and model sizes.

This script mirrors run_baseline_configs_gpu.py but for exp3_synthetic_manifolds.py.
It iterates over manifold types and model configurations, calls the experiment
script via subprocess, and optionally runs exp4_id_estimation on each result.

Usage
-----
  # Quick smoke test (1 manifold, 1 model, minimal samples)
  uv run python -m experiments.run_synthetic_experiments --quick

  # Full sweep
  uv run python -m experiments.run_synthetic_experiments \\
      --manifolds hyperplane,sphere,torus,swiss_roll,white_noise \\
      --model-configs toy-2l-32d,toy-4l-64d,llama-2l-20d-10m \\
      --n-samples 256 --seed 0 --device cuda --run-exp4

Output organization
-------------------
  {output-root}/
    {manifold_type}/
      {model_config}/
        {seed}/
          latents.h5
          exp4_id_estimates.jsonl   [if --run-exp4]
          manifest.json             [from exp3 run]

Each (manifold, model_config, seed) triple gets a unique subdirectory, so
re-running with same seed appends to existing latents.h5 safely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Default configuration sweeps
# ============================================================================

DEFAULT_MANIFOLDS = [
    "hyperplane",
    "sphere",
    "torus",
    "swiss_roll",
    "white_noise",
]

DEFAULT_TOY_CONFIGS = [
    "toy-2l-32d",
    "toy-4l-64d",
    "toy-6l-128d",
]

DEFAULT_LLAMA_CONFIGS = [
    "llama-2l-20d-10m",
    "llama-4l-32d-16m",
    "llama-6l-48d-24m",
]

QUICK_MANIFOLDS = ["hyperplane"]
QUICK_MODEL_CONFIGS = ["toy-2l-32d"]

# ============================================================================
# Subprocess helper
# ============================================================================

def _python_executable(repo_root: Path) -> str:
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable

def _run(cmd: list[str], cwd: Path, *, quiet: bool = False) -> None:
    if not quiet:
        print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        print(f"[ERROR] Command failed with exit code {completed.returncode}")
        raise SystemExit(completed.returncode)

# ============================================================================
# Orchestration
# ============================================================================

def run_synthetic_experiments(
    *,
    manifolds: list[str],
    model_configs: list[str],
    seed: int,
    device: str,
    n_samples: int,
    batch_size: int,
    output_root: Path,
    run_exp4: bool,
    quick: bool = False,
    strict: bool = False,
) -> None:
    repo_root = EXPERIMENT_ROOT.parent

    # Verify CUDA if requested
    if device == "cuda":
        _run(
            [
                _python_executable(repo_root),
                "-c",
                "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; print('CUDA OK:', torch.cuda.get_device_name(0))",
            ],
            cwd=repo_root,
            quiet=True,
        )

    py = _python_executable(repo_root)
    exp3_script = "experiments/exp3_synthetic_manifolds.py"

    total_runs = len(manifolds) * len(model_configs)
    run_idx = 0

    for manifold in manifolds:
        for model_cfg in model_configs:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"Run {run_idx}/{total_runs}: {manifold} × {model_cfg}")
            print(f"{'='*60}")

            # Output directory: output_root/{manifold}/{model_cfg}/{seed}/
            out_dir = output_root / manifold / model_cfg / str(seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            latents_path = out_dir / "latents.h5"

            # Determine model type from config string
            if model_cfg.startswith("toy-"):
                model_type_flag = "toy"
            elif model_cfg.startswith("llama-"):
                model_type_flag = "llama"
            else:
                raise ValueError(f"Unknown model config prefix: {model_cfg}")

            # Build command
            cmd = [
                py,
                exp3_script,
                "--manifold-type", manifold,
                "--model-type", model_type_flag,
                "--model-config", model_cfg,
                "--seed", str(seed),
                "--device", device,
                "--n-samples", str(n_samples),
                "--batch-size", str(batch_size),
                "--save", str(latents_path),
            ]

            # Run exp3
            try:
                _run(cmd, cwd=repo_root)
            except SystemExit as e:
                msg = f"exp3 failed for {manifold}/{model_cfg}/{seed}"
                if strict:
                    raise RuntimeError(msg) from e
                print(f"[WARN] {msg}, skipping...")
                continue

            # Write manifest
            manifest = {
                "manifold": manifold,
                "model_config": model_cfg,
                "seed": seed,
                "n_samples": n_samples,
                "latents": str(latents_path),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            manifest_path = out_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            # Optionally run exp4
            if run_exp4:
                print(f"\n  → Running ID estimation on {latents_path.name}")
                exp4_cmd = [
                    py,
                    "experiments/exp4_id_estimation.py",
                    "--latents", str(latents_path),
                    "--token", "0",  # default token
                    "--depth", "0", "1", "2", "3",
                    "--estimator", "twonn", "ess",
                    "--save", str(out_dir / "exp4_id_estimates.jsonl"),
                ]
                try:
                    _run(exp4_cmd, cwd=repo_root, quiet=True)
                    print(f"  ✓ ID estimates saved to {out_dir/'exp4_id_estimates.jsonl'}")
                except SystemExit as e:
                    print(f"[WARN] exp4 failed, exit={e}")
                    if strict:
                        raise

    print("\n" + "="*60)
    print(f"All runs completed. Results in: {output_root}")

# ============================================================================
# CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orchestrate synthetic manifold experiments (exp3 + optional exp4)."
    )
    p.add_argument("--manifolds", default=None,
                   help="Comma-separated manifold types (default: all)")
    p.add_argument("--model-configs", default=None,
                   help="Comma-separated model configs (e.g. toy-2l-32d,llama-2l-20d-10m)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for all runs")
    p.add_argument("--device", default="cpu",
                   help="Torch device (cpu/cuda)")
    p.add_argument("--n-samples", type=int, default=256,
                   help="Number of sequences per run")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for forward passes")
    p.add_argument("--output-root", type=Path,
                   default=Path("results/synthetic"),
                   help="Base output directory")
    p.add_argument("--run-exp4", action="store_true",
                   help="Run exp4_id_estimation after each exp3 run")
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke test (1 manifold, 1 toy model, 4 samples)")
    p.add_argument("--strict", action="store_true",
                   help="Fail on first error instead of continuing")
    return p

def main() -> None:
    args = _build_parser().parse_args()

    # Resolve lists
    if args.quick:
        manifolds = QUICK_MANIFOLDS
        model_configs = QUICK_MODEL_CONFIGS
        n_samples = 4
        print("[QUICK MODE] Smoke test: hyperplane + toy-2l-32d, n_samples=4")
    else:
        manifolds = args.manifolds.split(",") if args.manifolds else DEFAULT_MANIFOLDS
        model_configs = args.model_configs.split(",") if args.model_configs else (
            DEFAULT_TOY_CONFIGS + DEFAULT_LLAMA_CONFIGS
        )
        n_samples = args.n_samples

    run_synthetic_experiments(
        manifolds=manifolds,
        model_configs=model_configs,
        seed=args.seed,
        device=args.device,
        n_samples=n_samples,
        batch_size=args.batch_size,
        output_root=args.output_root,
        run_exp4=args.run_exp4,
        quick=args.quick,
        strict=args.strict,
    )

if __name__ == "__main__":
    main()
