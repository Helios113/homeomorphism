"""Visualization script for synthetic manifold ID estimation results (exp4).

Scans results/synthetic/ for exp4_id_estimates.jsonl files, reads corresponding
latents.h5 for ground-truth config, and produces:
  1) Per-run depth trajectory plots (TwoNN & ESS vs depth)
  2) Summary bar chart at input depth across all runs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Optional: use seaborn styling if available
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def get_ground_truth(manifold_type: str, d_model: int, manifold_dim: int | None) -> int:
    """Return intrinsic dimension for a manifold."""
    m = manifold_type.lower()
    if m == "hyperplane":
        if manifold_dim is None:
            raise ValueError("hyperplane requires manifold_dim in config")
        return manifold_dim
    if m == "sphere":
        return d_model - 1
    if m == "torus":
        return 2
    if m == "swiss_roll":
        return 2
    if m == "white_noise":
        return d_model
    raise ValueError(f"Unknown manifold type: {manifold_type}")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def gather_records(root: Path) -> list[dict]:
    """Walk root for exp4_id_estimates.jsonl and return a flat list of records."""
    records = []
    jsonl_files = list(root.rglob("exp4_id_estimates.jsonl"))
    if not jsonl_files:
        print(f"[WARN] No exp4_id_estimates.jsonl files found under {root}")

    for jsonl_path in jsonl_files:
        # Expected directory layout: root/{manifold}/{model_config}/{seed}/exp4_id_estimates.jsonl
        try:
            rel = jsonl_path.relative_to(root)
            parts = rel.parts
            if len(parts) < 4:
                print(f"[WARN] Unexpected path structure: {jsonl_path}, skipping")
                continue
            manifold, model_config, seed = parts[0], parts[1], parts[2]
        except Exception as e:
            print(f"[WARN] Cannot parse path {jsonl_path}: {e}")
            continue

        # Locate latents.h5
        latents_path = jsonl_path.parent / "latents.h5"
        if not latents_path.exists():
            print(f"[WARN] Missing latents.h5 for {jsonl_path}, skipping")
            continue

        # Read config from HDF5
        try:
            with h5py.File(latents_path, "r") as f:
                if "config" not in f.attrs:
                    print(f"[WARN] No 'config' attribute in {latents_path}, skipping")
                    continue
                cfg = json.loads(f.attrs["config"])
                d_model = int(cfg["d_model"])
                manifold_dim = cfg.get("manifold_dim")
                # n_layers = cfg["n_layers"]  # not needed
        except Exception as e:
            print(f"[WARN] Failed to read config from {latents_path}: {e}")
            continue

        # Compute ground truth
        try:
            gt = get_ground_truth(manifold, d_model, manifold_dim)
        except Exception as e:
            print(f"[WARN] Cannot compute ground truth for {manifold}: {e}")
            continue

        # Parse JSONL rows (token 0 only)
        try:
            with jsonl_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("token_id") != 0:
                        continue
                    records.append({
                        "manifold": manifold,
                        "model_config": model_config,
                        "seed": int(seed),
                        "depth": row["depth"],
                        "depth_label": row.get("depth_label", f"depth_{row['depth']}"),
                        "d_model": d_model,
                        "id_twonn": row.get("id_twonn"),
                        "id_ess": row.get("id_ess"),
                        "id_part_ratio": row.get("id_participation_ratio"),
                        "gt": gt,
                    })
        except Exception as e:
            print(f"[WARN] Failed to read {jsonl_path}: {e}")
            continue

    return records


# ---------------------------------------------------------------------------
# Individual depth-trajectory plots
# ---------------------------------------------------------------------------

def plot_depth_trajectories(records: list[dict], output_dir: Path) -> None:
    """Create one plot per (manifold, model_config, seed) showing ID vs depth.

    If a (manifold, model_config) has only one seed, save to
    `{manifold}_{model_config}.png`.  If multiple seeds exist, include the
    seed in the filename to avoid collisions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary grouping: (manifold, model_config, seed) -> rows
    primary_groups = defaultdict(list)
    for r in records:
        key = (r["manifold"], r["model_config"], r["seed"])
        primary_groups[key].append(r)

    # Secondary grouping to discover multiplicity: (manifold, model_config) -> list of seeds
    mc_to_seeds = defaultdict(set)
    for (manifold, model_config, seed) in primary_groups.keys():
        mc_to_seeds[(manifold, model_config)].add(seed)

    for (manifold, model_config, seed), rows in primary_groups.items():
        mc_key = (manifold, model_config)
        seeds_for_mc = mc_to_seeds[mc_key]
        multiple_seeds = len(seeds_for_mc) > 1

        rows_sorted = sorted(rows, key=lambda x: x["depth"])
        depths = [r["depth"] for r in rows_sorted]
        depth_labels = [r["depth_label"] for r in rows_sorted]
        twonn_vals = [r["id_twonn"] for r in rows_sorted]
        ess_vals = [r["id_ess"] for r in rows_sorted]
        gt_val = rows_sorted[0]["gt"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(depths, twonn_vals, marker='o', label='TwoNN', linewidth=2, markersize=8)
        ax.plot(depths, ess_vals, marker='s', label='ESS', linewidth=2, markersize=8)
        ax.axhline(y=gt_val, color='gray', linestyle='--', linewidth=2, label=f'Ground truth ({gt_val})')

        ax.set_xlabel('Depth')
        ax.set_ylabel('Estimated Intrinsic Dimension')
        title = f'{manifold} — {model_config}'
        if multiple_seeds:
            title += f' (seed {seed})'
        ax.set_title(title)
        ax.set_xticks(depths)
        ax.set_xticklabels(depth_labels, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()

        # Build filename
        if multiple_seeds:
            fname = f"{manifold}_{model_config}_seed{seed}.png"
        else:
            fname = f"{manifold}_{model_config}.png"
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Summary bar chart at input depth (depth 0)
# ---------------------------------------------------------------------------

def plot_summary(records: list[dict], output_dir: Path) -> None:
    """Grouped bar chart comparing ID estimates at depth 0 across all (manifold, model_config)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect depth 0 records
    depth0 = [r for r in records if r["depth"] == 0]
    if not depth0:
        print("[WARN] No depth 0 records found for summary plot")
        return

    # Group by (manifold, model_config); may have multiple seeds, aggregate to mean
    groups_data = defaultdict(lambda: {"twonn": [], "ess": [], "gt": None})
    for r in depth0:
        key = (r["manifold"], r["model_config"])
        groups_data[key]["twonn"].append(r["id_twonn"])
        groups_data[key]["ess"].append(r["id_ess"])
        groups_data[key]["gt"] = r["gt"]  # same across seeds

    # Compute means and stds
    summary = {}
    for key, vals in groups_data.items():
        twonn_arr = [v for v in vals["twonn"] if v is not None]
        ess_arr = [v for v in vals["ess"] if v is not None]
        twonn_mean = np.mean(twonn_arr) if twonn_arr else np.nan
        ess_mean = np.mean(ess_arr) if ess_arr else np.nan
        twonn_std = np.std(twonn_arr) if len(twonn_arr) > 1 else 0.0
        ess_std = np.std(ess_arr) if len(ess_arr) > 1 else 0.0
        summary[key] = {
            "twonn_mean": twonn_mean,
            "ess_mean": ess_mean,
            "twonn_std": twonn_std,
            "ess_std": ess_std,
            "gt": vals["gt"],
        }

    # Plot
    keys = list(summary.keys())
    x = np.arange(len(keys))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(keys)*1.5), 6))

    # Bars for TwoNN and ESS
    twonn_bars = ax.bar(x - bar_width/2,
                        [summary[k]["twonn_mean"] for k in keys],
                        bar_width,
                        yerr=[summary[k]["twonn_std"] for k in keys],
                        label='TwoNN',
                        capsize=5,
                        color='C0')
    ess_bars = ax.bar(x + bar_width/2,
                      [summary[k]["ess_mean"] for k in keys],
                       bar_width,
                       yerr=[summary[k]["ess_std"] for k in keys],
                       label='ESS',
                       capsize=5,
                       color='C1')

    # Ground truth lines: one per group across the width of both bars
    for i, key in enumerate(keys):
        gt_val = summary[key]["gt"]
        left = i - bar_width
        right = i + bar_width
        ax.hlines(y=gt_val, xmin=left, xmax=right,
                  colors='red', linestyles='--', linewidth=2,
                  label='Ground truth' if i == 0 else "")

    ax.set_xlabel('Manifold (model configuration)')
    ax.set_ylabel('Estimated Intrinsic Dimension at Input Depth')
    ax.set_title('ID Estimates at Input Depth Across Manifolds')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({cfg})" for m, cfg in keys], rotation=45, ha='right')
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(output_dir / "summary_input_depth.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize synthetic manifold ID estimation results (exp4)."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/synthetic"),
                        help="Root directory containing synthetic experiment results")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save plots (default: results-root/plots)")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"Results root does not exist: {args.results_root}")

    records = gather_records(args.results_root)
    if not records:
        raise SystemExit("No data records collected; aborting.")

    out_dir = args.output_dir or args.results_root / "plots"
    plot_depth_trajectories(records, out_dir)
    plot_summary(records, out_dir)
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
