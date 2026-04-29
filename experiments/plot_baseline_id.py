"""Plot intrinsic dimension estimates from baseline experiments (exp3_section2).

Scans a baseline run directory (or multiple) for id.jsonl files and produces:
  - Depth-trajectory plots per model + baseline group (ID vs depth)
  - Summary bar chart comparing input-depth (depth 0) across baselines

Usage
-----
  # Single run
  uv run python -m experiments.plot_baseline_id \
      --results-dir results/baselines/groupA/gpt2/20260427_123456

  # Multiple runs (across models)
  uv run python -m experiments.plot_baseline_id \
      --results-dirs results/baselines/groupA/gpt2/* results/baselines/groupB/gpt2/*

  # All baseline results under results/baselines
  uv run python -m experiments.plot_baseline_id \
      --results-root results/baselines

Output is written to each run directory's plots/ subfolder, and a combined
summary plot is written to the provided --output-dir (or cwd).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _depth_label(depth: int) -> str:
    """Convert numeric depth to human label (b0.attn, b0.ffn, …)."""
    block, rem = divmod(depth, 2)
    return f"b{block}.{'attn' if rem == 0 else 'ffn'}"


def load_id_jsonl(path: Path) -> list[dict]:
    """Read id.jsonl and return list of rows."""
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def discover_run_dirs(root: Path) -> list[Path]:
    """Find all directories containing an id.jsonl file."""
    if root.is_file() and root.name == "id.jsonl":
        return [root.parent]
    jsonl_files = list(root.rglob("id.jsonl"))
    return sorted({p.parent for p in jsonl_files})


def get_metadata(run_dir: Path) -> dict:
    """Read manifest to extract model/run metadata."""
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            return {
                "model_name": m.get("config", {}).get("model_name", "unknown"),
                "weights": m.get("config", {}).get("weights", "?"),
                "n_layers": m.get("n_blocks", "?"),
                "d_model": m.get("hidden_size", "?"),
            }
        except Exception:
            pass
    # Fallback: read latents.h5 attrs if manifest missing
    latents = run_dir / "latents.h5"
    if latents.exists():
        try:
            import h5py
            with h5py.File(latents, "r") as f:
                cfg = json.loads(f.attrs.get("config", "{}"))
                return {
                    "model_name": cfg.get("model_name", "unknown"),
                    "weights": cfg.get("weights", "?"),
                    "n_layers": cfg.get("n_layers", "?"),
                    "d_model": cfg.get("d_model", "?"),
                }
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def aggregate_by_baseline(run_dir: Path) -> dict:
    """Load id.jsonl and group rows by (baseline, depth, estimator).

    Returns nested dict: baseline -> depth -> estimator -> list of values
    (list across token positions or repeats).
    """
    id_path = run_dir / "id.jsonl"
    if not id_path.exists():
        print(f"[WARN] No id.jsonl in {run_dir}, skipping")
        return {}

    rows = load_id_jsonl(id_path)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in rows:
        # Only full_stream or per-token/last-token rows; ignore others
        gran = r.get("granularity", "")
        if gran not in ("full_stream", "per_token", "last_token"):
            continue
        baseline = r["baseline"]
        depth = r["depth"]
        est = r["estimator"]
        val = r.get("id_estimate")
        if val is not None and np.isfinite(val):
            data[baseline][depth][est].append(float(val))

    # Average across tokens / samples within each (baseline, depth, estimator)
    averaged = defaultdict(dict)
    for baseline, depth_dict in data.items():
        for depth, est_dict in depth_dict.items():
            averaged[baseline][depth] = {
                est: (np.mean(vals) if vals else None)
                for est, vals in est_dict.items()
            }
    return averaged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_depth_trajectories_per_baseline(
    run_dir: Path,
    aggregated: dict,
    estimators: list[str],
    *,
    title_suffix: str = "",
) -> None:
    """Create one multi-line plot per baseline: ID vs depth (one line per estimator)."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Determine depth range
    all_depths = sorted({d for bg in aggregated.values() for d in bg.keys()})
    if not all_depths:
        return
    x = np.array(all_depths)

    for baseline, depth_data in aggregated.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for est in estimators:
            y = [depth_data.get(d, {}).get(est) for d in all_depths]
            ax.plot(x, y, marker="o", label=est.upper())
        ax.set_xlabel("Depth")
        ax.set_ylabel("Intrinsic dimension")
        ax.set_title(f"ID vs Depth — baseline: {baseline}{title_suffix}")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        fname = plots_dir / f"id_vs_depth_baseline_{baseline}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)


def plot_baseline_comparison_at_input(
    run_dirs: list[Path],
    output_path: Path,
    estimator: str = "twonn",
) -> None:
    """Bar chart: compare all baselines' ID at depth 0 (input) across runs."""
    # Collect: run_label -> baseline -> ID
    run_labels = []
    baseline_names = set()
    bar_data = defaultdict(dict)  # baseline -> list of (run_label, value)

    for run_dir in run_dirs:
        md = get_metadata(run_dir)
        label = f"{md.get('model_name','?')}:{md.get('weights','?')}"
        run_labels.append(label)
        aggregated = aggregate_by_baseline(run_dir)
        for baseline, depth_data in aggregated.items():
            val = depth_data.get(0, {}).get(estimator)
            if val is not None:
                baseline_names.add(baseline)
                bar_data[baseline][label] = val

    if not bar_data:
        print("[WARN] No data for input-depth comparison plot")
        return

    baseline_names = sorted(baseline_names)
    run_labels_sorted = sorted(run_labels)  # consistent ordering
    n_baselines = len(baseline_names)
    n_runs = len(run_labels_sorted)

    fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.2), 5))
    x = np.arange(n_runs)
    bar_w = 0.8 / n_baselines

    for i, baseline in enumerate(baseline_names):
        values = [bar_data[baseline].get(rl, np.nan) for rl in run_labels_sorted]
        ax.bar(x + i * bar_w - 0.4 + bar_w/2, values, bar_w, label=baseline)

    ax.set_xlabel("Run (model:weights)")
    ax.set_ylabel(f"Intrinsic dimension ({estimator.upper()}) at depth 0 (input)")
    ax.set_title("Baseline comparison — input depth ID")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels_sorted, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved baseline comparison → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot baseline ID estimates from exp3_section2 results."
    )
    p.add_argument(
        "--results-dirs", nargs="+", default=[],
        help="One or more run directories containing id.jsonl",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/baselines"),
        help="Root directory under which all baseline runs live",
    )
    p.add_argument(
        "--estimator",
        default="twonn",
        choices=["twonn", "ess", "participation_ratio"],
        help="Estimator to use for summary bar chart (default: twonn)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write summary plot (default: cwd)",
    )
    args = p.parse_args()

    # Discover run directories
    run_dirs: list[Path] = []
    for rd in args.results_dirs:
        run_dirs.extend(discover_run_dirs(Path(rd)))
    if not run_dirs and args.results_root:
        run_dirs.extend(discover_run_dirs(args.results_root))
    if not run_dirs:
        raise SystemExit("No run directories found; check --results-dirs / --results-root")

    print(f"[INFO] Found {len(run_dirs)} run(s) with id.jsonl")

    # Per-run per-baseline depth trajectory plots
    successful_runs = 0
    for run_dir in run_dirs:
        print(f"[PLOT] {run_dir}")
        aggregated = aggregate_by_baseline(run_dir)
        if not aggregated:
            print(f"  [SKIP] no ID data in {run_dir} (check if run completed)")
            continue

        # Determine estimators present
        estimators_set = set()
        for dd in aggregated.values():
            for d_est in dd.values():
                estimators_set.update(d_est.keys())
        estimators = sorted(estimators_set) if estimators_set else ["twonn", "ess"]

        try:
            plot_depth_trajectories_per_baseline(run_dir, aggregated, estimators)
            successful_runs += 1
        except Exception as e:
            print(f"  [ERROR] plotting failed: {e}")
            continue

    print(f"[INFO] Successfully plotted {successful_runs}/{len(run_dirs)} runs")

    # Combined summary bar chart
    out_dir = args.output_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "baseline_input_depth_comparison.png"
    plot_baseline_comparison_at_input(run_dirs, summary_path, estimator=args.estimator)

    print("\nDone.")


if __name__ == "__main__":
    main()
