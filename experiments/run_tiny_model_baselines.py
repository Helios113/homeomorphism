"""Orchestrate baseline experiments across standard and tiny GPT-2 models.

Runs exp3_section2_baselines.py for each model configuration with appropriate
baseline groups and weight modes.

Models
-------
- Standard GPT-2 ("gpt2"): uses weights="trained"
- Custom tiny models (tiny-gpt2-*, micro-gpt2-*, nano-gpt2-*): weights="random_gaussian"

Baseline Groups
---------------
  groupA: trained, topological_initialisation, maximum_entropy_injection
      (requires trained weights → only runs for gpt2)
  groupB: syntactic_disintegration, semantic_scrambling
      (works for any model; uses same weight mode as model default)

Output
-------
  results/baselines/{group}/{model_name}/{timestamp}/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# All model names from SMALL_MODELS.md
ALL_MODELS = [
    "gpt2",
    "nano-gpt2-4l-128d",
    "tiny-gpt2-4l-256d",
    "tiny-gpt2-6l-256d",
    "micro-gpt2-4l-384d",
    "micro-gpt2-6l-384d",
    "tiny-gpt2-8l-256d",
]

# Baseline group definitions
BASELINE_GROUPS = {
    "groupA": {
        "baselines": "trained,topological_initialisation,maximum_entropy_injection",
        "n_samples": 256,
        "max_tokens": 64,
        "granularity": "full_stream,last_token",
        "estimator": "twonn,ess",
        "overlap_k": 10,
    },
    "groupB": {
        "baselines": "syntactic_disintegration,semantic_scrambling",
        "n_samples": 256,
        "max_tokens": 64,
        "granularity": "last_token",
        "estimator": "twonn,ess,participation_ratio",
        "overlap_k": 10,
    },
}

DEFAULT_CORPUS = "shakespeare"
DEFAULT_SEED = 0
DEVICE = "cuda"  # requires CUDA

EXPERIMENT_SCRIPT = "experiments/exp3_section2_baselines.py"
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_executable() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable

def _is_custom(model_name: str) -> bool:
    return model_name != "gpt2"

def _weight_mode_for_model(model_name: str) -> str:
    """Return default weight mode for a given model.

    Standard GPT-2 uses pretrained weights ('trained'). Custom tiny models
    are randomly initialized ('random_gaussian'). The baseline groups are
    chosen to be compatible with these modes.
    """
    return "trained" if model_name == "gpt2" else "random_gaussian"

def _run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print("\n$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        print(f"[ERROR] Exit code {completed.returncode}")
        raise SystemExit(completed.returncode)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline_for_model(
    model_name: str,
    group_key: str,
    output_root: Path,
    seed: int,
    device: str,
    corpus: str,
    quick: bool = False,
) -> None:
    group_cfg = BASELINE_GROUPS[group_key]
    weight_mode = _weight_mode_for_model(model_name)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"{group_key}/{model_name}/{ts}"

    print(f"\n{'='*70}")
    print(f"Model: {model_name}  Group: {group_key}  Weights: {weight_mode}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}")

    py = _python_executable()
    cmd = [
        py, EXPERIMENT_SCRIPT,
        "--model", model_name,
        "--weights", weight_mode,
        "--corpus", corpus,
        "--n-samples", str(group_cfg["n_samples"] if not quick else 4),
        "--max-tokens", str(group_cfg["max_tokens"] if not quick else 16),
        "--layers", "all",
        "--baselines", group_cfg["baselines"],
        "--granularity", group_cfg["granularity"],
        "--estimator", group_cfg["estimator"],
        "--overlap-k", str(group_cfg["overlap_k"] if not quick else 5),
        "--seed", str(seed),
        "--device", device,
        "--output-root", str(out_dir.parent),
    ]

    _run(cmd)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run baseline sweeps across all tiny GPT-2 models (exp3_section2)."
    )
    p.add_argument("--models", default=None,
                   help="Comma-separated model names (default: all)")
    p.add_argument("--groups", default="groupA,groupB",
                   help="Baseline groups to run: groupA,groupB (default: both)")
    p.add_argument("--output-root", type=Path, default=Path("results/baselines"),
                   help="Base output directory (default: results/baselines)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke test (reduced samples/tokens)")
    args = p.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    models = args.models.split(",") if args.models else ALL_MODELS
    groups = args.groups.split(",") if args.groups else ["groupA", "groupB"]

    # Validate models
    for m in models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m!r}. Known: {ALL_MODELS}")

    # Validate groups
    for g in groups:
        if g not in BASELINE_GROUPS:
            raise ValueError(f"Unknown group: {g!r}. Known: {list(BASELINE_GROUPS)}")

    print(f"[INFO] Models    : {models}")
    print(f"[INFO] Groups    : {groups}")
    print(f"[INFO] Output-   : {args.output_root}")
    print(f"[INFO] Device    : {args.device}")
    print(f"[INFO] Corpus    : {args.corpus}")
    print(f"[INFO] Quick     : {args.quick}")

    total = len(models) * len(groups)
    run_idx = 0
    failures = 0

    for model_name in models:
        for group_key in groups:
            run_idx += 1
            # Skip groupA for custom models (it includes 'trained' baseline)
            if group_key == "groupA" and _is_custom(model_name):
                print(f"\n[SKIP] {model_name} × {group_key}: groupA requires trained weights, "
                      f"but {model_name} is custom (random init only)")
                continue

            try:
                run_baseline_for_model(
                    model_name=model_name,
                    group_key=group_key,
                    output_root=args.output_root,
                    seed=args.seed,
                    device=args.device,
                    corpus=args.corpus,
                    quick=args.quick,
                )
            except SystemExit as e:
                failures += 1
                if e.code != 0:
                    print(f"[FAIL] model={model_name} group={group_key} exit={e.code}")
                continue

    print(f"\n{'='*70}")
    print(f"Completed. Total attempted: {run_idx}, failures: {failures}")

if __name__ == "__main__":
    main()
