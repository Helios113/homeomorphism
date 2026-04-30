"""Run multiple baseline experiment configurations on GPU.

This script orchestrates representative baseline runs for the new Section 2 framework.
It intentionally uses CUDA and fails fast if no GPU is available.

Usage:
  .venv/bin/python experiments/run_baseline_configs_gpu.py
  .venv/bin/python experiments/run_baseline_configs_gpu.py --quick
"""

.. deprecated:: 2025-04
    This script is deprecated. Use :mod:`experiments.baseline_runner` instead:
        python experiments/baseline_runner.py --model gpt2 --baseline-group groupA --device cuda
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _python_executable(repo_root: Path) -> str:
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments on GPU")
    parser.add_argument("--quick", action="store_true", help="Run a small quick suite")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="model name; 'gpt2' or custom small models like 'tiny-gpt2-4l-256d'",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = _python_executable(repo_root)

    # Fail fast if CUDA is unavailable.
    _run(
        [
            py,
            "-c",
            "import torch; assert torch.cuda.is_available(), 'CUDA GPU is required for this runner'; print('CUDA OK:', torch.cuda.get_device_name(0))",
        ],
        cwd=repo_root,
    )

    common = [
        py,
        "experiments/exp3_section2_baselines.py",
        "--model",
        args.model,
        "--weights",
        "trained",
        "--corpus",
        "shakespeare",
        "--layers",
        "0.attn",
        "--device",
        "cuda",
        "--seed",
        "0",
    ]

    if args.quick:
        runs = [
            common
            + [
                "--baselines",
                "trained,maximum_entropy_injection",
                "--n-samples",
                "4",
                "--max-tokens",
                "16",
                "--granularity",
                "last_token",
                "--estimator",
                "twonn",
                "--overlap-k",
                "5",
            ]
        ]
    else:
        runs = [
            common
            + [
                "--baselines",
                "trained,topological_initialisation,maximum_entropy_injection",
                "--n-samples",
                "8",
                "--max-tokens",
                "32",
                "--granularity",
                "full_stream,last_token",
                "--estimator",
                "twonn,ess",
                "--overlap-k",
                "10",
            ],
            common
            + [
                "--baselines",
                "syntactic_disintegration,semantic_scrambling",
                "--n-samples",
                "8",
                "--max-tokens",
                "32",
                "--granularity",
                "last_token",
                "--estimator",
                "twonn,participation_ratio",
                "--overlap-k",
                "10",
            ],
        ]

    for cmd in runs:
        _run(cmd, cwd=repo_root)

    print("\nAll configured GPU baseline runs completed.")


if __name__ == "__main__":
    main()
