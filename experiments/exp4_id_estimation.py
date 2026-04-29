"""Experiment 4: intrinsic-dimension estimation on latent clouds from exp3.

Depth convention (matches exp3's forward_with_states, offset by 1)
------------------------------------------------------------------
  user depth 0   = post-attention  of block 0   (HDF5 depth_01)
  user depth 1   = post-FFN        of block 0   (HDF5 depth_02)
  user depth 2   = post-attention  of block 1   (HDF5 depth_03)
  user depth 2k  = post-attention  of block k
  user depth 2k+1= post-FFN        of block k

  The raw input (HDF5 depth_00) and the post-norm output (last HDF5 depth)
  are excluded so that depth indexing is purely over transformer sublayers.

Point cloud
-----------
  For a given (token_id, depth), the cloud is the N×d_model matrix
      f["depth_{d+1:02d}"][:, token_id, :]
  i.e. one R^{d_model} point per sampled sequence, all at the same
  token position and residual-stream depth.

Estimators
----------
  TwoNN  — Facco et al. (2017).  Fast, parameter-free.
  ESS    — Johnsson et al. (2015).  Slower, estimates local structure.

Run
---
    # all depths, token 0
    uv run python -m experiments.exp4_id_estimation --latents latents.h5 --token 0

    # specific depths and tokens, save results
    uv run python -m experiments.exp4_id_estimation \\
        --latents latents.h5 --token 0 3 7 --depth 0 1 2 3 \\
        --save results/exp4/id_estimates.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from homeomorphism.id_est import EstimatorName, estimate_id as estimate_id_torch

# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def _n_all_depths(h5: h5py.File) -> int:
    """Total user-visible depths = 2 * n_layers + 2 (input + blocks + post-norm)."""
    cfg = json.loads(h5.attrs["config"])
    # Try config JSON first (synthetic experiments), then individual attrs (baseline experiments)
    n_layers = cfg.get("n_layers") or h5.attrs.get("n_layers")
    if n_layers is None:
        raise KeyError("n_layers not found in config or attributes")
    return 2 * int(n_layers) + 2


def _hdf5_key(user_depth: int) -> str:
    """User depth maps 1-to-1 to HDF5 dataset index."""
    return f"depth_{user_depth:02d}"


def _depth_label(user_depth: int, n_layers: int) -> str:
    n_block_depths = 2 * n_layers
    if user_depth == 0:
        return "input"
    if user_depth == n_block_depths + 1:
        return "post-norm"
    block, sub = divmod(user_depth - 1, 2)
    return f"block{block}.{'attn' if sub == 0 else 'ffn'}"


# ---------------------------------------------------------------------------
# ID estimation (shared codebase estimators)
# ---------------------------------------------------------------------------

_ESTIMATORS: tuple[EstimatorName, ...] = ("twonn", "ess", "participation_ratio")


def estimate_id(cloud: np.ndarray, estimator_name: EstimatorName) -> float:
    """Run shared project estimator on cloud (N, d) and return the ID estimate."""
    try:
        pts = torch.from_numpy(cloud).to(torch.float32)
        return float(estimate_id_torch(pts, estimator_name))
    except Exception as e:
        print(f"    [{estimator_name}] failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run(
    latents_path: Path,
    token_ids: list[int] | None,
    depths: list[int] | None,
    estimator_names: list[EstimatorName],
    save_path: Path | None,
) -> None:
    rows: list[dict] = []

    with h5py.File(latents_path, "r") as f:
        cfg = json.loads(f.attrs["config"])
        n_all = _n_all_depths(f)
        n_samples = f[_hdf5_key(0)].shape[0]
        seq_len = cfg["seq_len"]
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]

        if depths is None:
            depths = list(range(n_all))
        if token_ids is None:
            token_ids = list(range(seq_len))

        _probe = f[_hdf5_key(0)][0, 0, :]
        print(f"Vector dim in file: {_probe.shape[0]}  (d_model={d_model}, manifold_dim={cfg['manifold_dim']})")
        assert _probe.shape[0] == d_model, (
            f"BUG: stored vectors have dim {_probe.shape[0]}, expected d_model={d_model}"
        )

        print(
            f"File  : {latents_path}  ({n_samples} samples, "
            f"T={seq_len}, d={d_model})"
        )
        print(f"Depths: {[_depth_label(d, n_layers) for d in depths]}")
        print(f"Tokens: {token_ids}")
        print(f"Estimators: {estimator_names}")
        print()

        for token_id in token_ids:
            if not 0 <= token_id < seq_len:
                print(f"[WARN] token_id={token_id} out of range [0, {seq_len}), skipping")
                continue

            print(f"=== token {token_id} ===")
            for depth in depths:
                key = _hdf5_key(depth)
                if key not in f:
                    print(f"  [WARN] {key} not in file, skipping depth {depth}")
                    continue

                cloud = f[key][:, token_id, :]        # (N, d_model) ndarray
                label = _depth_label(depth, n_layers)

                estimates: dict[str, float] = {}
                for name in estimator_names:
                    t0 = time.time()
                    val = estimate_id(cloud, name)
                    elapsed = time.time() - t0
                    estimates[name] = val
                    print(f"  depth {depth:2d} ({label:12s})  {name:6s}: {val:.3f}  ({elapsed:.2f}s)")

                rows.append({
                    "token_id": token_id,
                    "depth": depth,
                    "depth_label": label,
                    "n_samples": int(cloud.shape[0]),
                    "d_model": d_model,
                    **{f"id_{name}": (v if v == v else None) for name, v in estimates.items()},
                })
            print()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} rows → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 4: intrinsic-dim estimation on exp3 latent clouds."
    )
    p.add_argument("--latents", type=Path, required=True,
                   help="HDF5 file produced by exp3 (latents.h5)")
    p.add_argument("--token", type=int, nargs="+", default=None, metavar="T",
                   help="token position(s) to analyse (default: all tokens in file)")
    p.add_argument("--depth", type=int, nargs="+", default=None, metavar="D",
                   help="user depth(s) to analyse (default: all block depths)")
    p.add_argument("--estimator", nargs="+", default=["twonn", "ess"],
                   choices=list(_ESTIMATORS), metavar="E",
                   help="estimators to run (default: twonn ess)")
    p.add_argument("--save", type=Path, default=None, metavar="PATH",
                   help="optional JSONL output path for results")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    estimator_names: list[EstimatorName] = [e for e in args.estimator]
    run(
        latents_path=args.latents,
        token_ids=args.token,   # None → all tokens resolved inside run()
        depths=args.depth,
        estimator_names=estimator_names,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
