"""Experiment 2: intrinsic-dim estimation of residual-stream manifolds across depth.

Scope
-----
For a causal LM, capture the residual stream `h^n(s) in R^(T x d)` at every
depth n of interest and every input s in a corpus, then estimate the intrinsic
dimension of the resulting point cloud at three granularities:

  1. full_stream   — each input contributes one point in R^(T*d);
                     the point cloud has N = n_inputs points.
                     Tests the direct topological-invariance claim.

  2. per_token     — for each token position i in [0, T), the cloud
                     `{ h^n(s)_{i,:} : s in corpus } subset R^d`,
                     with N points per (depth, token_idx).
                     Tests the homeomorphism claim + the (unstated)
                     transversality assumption.

  2b. last_token   — per-token at i = T-1. The "decoder latent" manifold
                     that inference uses.

Representations are kept uncompressed as (N, T, d) until ID-estimation time so
the structure is preserved for later re-analysis — no early concatenation.

Depth indexing (Pre-LN convention):
  depth 2b   = block b attn-sublayer INPUT  (= ln_1 input)
  depth 2b+1 = block b ffn-sublayer INPUT   (= ln_2 input) = post block b attn
  depth 2L   = input to ln_f                = post last ffn

For each `(block, kind)` sublayer in `--layers`, we capture its "pre" depth and
its "post" depth, and dedupe across the full set so every forward pass yields
every requested depth in one go.

Run layout:
  results/exp2/<run_id>/
    config.json            - CLI flags
    manifest.json          - metadata, captured depths, input_ids, git sha
    results.jsonl          - one row per (depth, granularity, [token_idx], estimator)
    plots/                 - created for later plotting scripts
    representations.npz    - ONLY written if --save-reps is passed (opt-in).
                             Per-depth (N, T, d) fp32 arrays, can be multi-GB at
                             realistic N/T/d; don't save by default.

CLI
---
    uv run python -m experiments.exp2_intrinsic_dim \\
        --model gpt2 --weights trained \\
        --n-samples 500 --max-tokens 64 \\
        --layers all \\
        --granularity full_stream,per_token,last_token \\
        --estimator twonn

NOTE: id_est methods are placeholders returning NaN. The pipeline runs end-to-end
and writes all rows, but id_estimate values will be NaN until twonn/ess/etc.
are implemented in src/homeomorphism/id_est.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from homeomorphism import hooks, models
from homeomorphism.data import load_texts
from homeomorphism.id_est import EstimatorName, estimate_id
from homeomorphism.models import SublayerKind

from experiments.exp1_per_token_J import resolve_sublayers


Granularity = Literal["full_stream", "per_token", "last_token"]

_VALID_GRANULARITIES: tuple[Granularity, ...] = ("full_stream", "per_token", "last_token")
_VALID_ESTIMATORS: tuple[EstimatorName, ...] = ("twonn", "ess", "participation_ratio")


# ---------------------------------------------------------------------------
# Depth <-> hook path
# ---------------------------------------------------------------------------

def sublayer_depths(sublayers: list[tuple[int, SublayerKind]]) -> list[int]:
    """Map (block, kind) pairs to the set of depth indices (pre + post), sorted.

    depth 2b   = pre block b attn
    depth 2b+1 = pre block b ffn  (= post block b attn)
    depth 2L   = post last ffn    (= input to ln_f)
    """
    depths: set[int] = set()
    for b, k in sublayers:
        pre = 2 * b + (0 if k == "attn" else 1)
        depths.add(pre)
        depths.add(pre + 1)  # post
    return sorted(depths)


def _final_norm_path(m: models.Model) -> str:
    if m.arch == "gpt2":
        return "transformer.ln_f"
    raise NotImplementedError(f"final norm path not registered for arch {m.arch!r}")


def depth_to_hook_path(m: models.Model, depth: int) -> str:
    """Return the module whose INPUT is the residual stream at this depth."""
    L = models.n_blocks(m)
    if depth == 2 * L:
        return _final_norm_path(m)
    if not 0 <= depth < 2 * L:
        raise IndexError(f"depth {depth} out of [0, {2 * L}]")
    b, rem = divmod(depth, 2)
    kind: SublayerKind = "attn" if rem == 0 else "ffn"
    return models.sublayer(m, b, kind).hook_path


# ---------------------------------------------------------------------------
# Multi-path capture (one forward pass, many hooks)
# ---------------------------------------------------------------------------

def capture_multi(
    m: models.Model,
    paths: list[str],
    text: str,
    max_tokens: int,
) -> dict[str, torch.Tensor]:
    """Run one forward pass on `text`; capture INPUT at each module in `paths`.

    Returns {path: Tensor[T, d] in fp32}. Raises if any hook doesn't fire.
    """
    captured: dict[str, list[torch.Tensor]] = {p: [] for p in paths}
    handles = []

    for p in paths:
        module = m.model.get_submodule(p)

        def _make_hook(name: str):
            def _hook(_mod, inputs, _out):  # noqa: ANN001
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"hook at {name!r} got non-tensor input: {type(x)}")
                captured[name].append(x.detach())

            return _hook

        handles.append(module.register_forward_hook(_make_hook(p)))

    try:
        input_ids = models.tokenize(m, text, max_tokens=max_tokens)
        with torch.no_grad():
            m.model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    out: dict[str, torch.Tensor] = {}
    for p in paths:
        if not captured[p]:
            raise RuntimeError(f"hook at {p!r} did not fire")
        h = captured[p][0]
        if h.dim() != 3 or h.shape[0] != 1:
            raise ValueError(f"unexpected tensor shape at {p!r}: {tuple(h.shape)}")
        out[p] = h[0].to(torch.float32).cpu()
    return out


# ---------------------------------------------------------------------------
# Granularity -> point-cloud view of (N, T, d) representation tensor
# ---------------------------------------------------------------------------

def _cloud(reps: torch.Tensor, granularity: Granularity, token_idx: int | None) -> torch.Tensor:
    """reps: (N, T, d). Return a 2-D (N', m) point cloud for ID estimation."""
    if granularity == "full_stream":
        N, T, d = reps.shape
        return reps.reshape(N, T * d)
    if granularity == "per_token":
        if token_idx is None:
            raise ValueError("per_token granularity requires token_idx")
        return reps[:, token_idx, :]
    if granularity == "last_token":
        return reps[:, -1, :]
    raise ValueError(f"unknown granularity {granularity!r}")


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------

def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run_exp2(
    *,
    model_name: str,
    weights: str,
    corpus: str,
    n_samples: int,
    max_tokens: int,
    layers_spec: str,
    granularities: list[Granularity],
    estimators: list[EstimatorName],
    save_reps: bool,
    seed: int,
    device: str,
    output_root: Path,
) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    config: dict[str, Any] = {
        "model_name": model_name,
        "weights": weights,
        "corpus": corpus,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "layers_spec": layers_spec,
        "granularities": granularities,
        "estimators": estimators,
        "save_reps": save_reps,
        "seed": seed,
        "device": device,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[exp2] run_dir = {run_dir}")

    # --- load ---
    m = models.load_model(model_name, weights=weights, seed=seed, device=device)  # type: ignore[arg-type]
    L = models.n_blocks(m)
    d_model = models.hidden_size(m)
    sublayers = resolve_sublayers(L, layers_spec)
    depths = sublayer_depths(sublayers)
    paths = [depth_to_hook_path(m, d) for d in depths]
    path_by_depth = dict(zip(depths, paths))

    texts = load_texts(
        corpus,  # type: ignore[arg-type]
        n_samples=n_samples,
        chunk_chars=max(max_tokens * 8, 200),
        seed=seed,
    )

    print(
        f"[exp2] model={model_name} weights={weights} L={L} d={d_model} "
        f"depths={len(depths)} (indices {depths[0]}..{depths[-1]}) "
        f"inputs={len(texts)} T={max_tokens}"
    )

    # --- capture reps (one forward pass per input, all depths at once) ---
    T = max_tokens
    buf: dict[int, list[torch.Tensor]] = {d: [] for d in depths}
    input_ids_kept: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    t_cap = time.time()

    for input_id, text in enumerate(texts):
        try:
            captured = capture_multi(m, paths, text, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            dropped.append({"input_id": input_id, "reason": f"{type(e).__name__}: {e}"})
            continue

        lengths = {tensor.shape[0] for tensor in captured.values()}
        if len(lengths) != 1:
            dropped.append({"input_id": input_id, "reason": f"depth length mismatch {lengths}"})
            continue
        t_seq = lengths.pop()
        if t_seq != T:
            dropped.append(
                {"input_id": input_id, "reason": f"tokenized to {t_seq} != T={T}"}
            )
            continue

        for depth_, path in path_by_depth.items():
            buf[depth_].append(captured[path])

        ids = models.tokenize(m, text, max_tokens=max_tokens)[0].tolist()
        input_ids_kept.append(ids)

        if (input_id + 1) % 50 == 0:
            print(f"  captured {input_id + 1}/{len(texts)} ({len(input_ids_kept)} kept)")

    N = len(input_ids_kept)
    if N < 2:
        raise RuntimeError(f"only {N} inputs survived capture; need >= 2 for ID estimation")

    reps_by_depth: dict[int, torch.Tensor] = {
        d: torch.stack(buf[d], dim=0) for d in depths
    }
    print(
        f"[exp2] captured {N}/{len(texts)} inputs in {time.time() - t_cap:.1f}s "
        f"(dropped {len(dropped)})"
    )

    # --- persist raw reps ---
    if save_reps:
        to_save = {f"depth_{d:03d}": reps_by_depth[d].numpy() for d in depths}
        reps_path = run_dir / "representations.npz"
        np.savez_compressed(reps_path, **to_save)
        size_mb = reps_path.stat().st_size / (1024 * 1024)
        print(f"[exp2] saved representations.npz ({size_mb:.1f} MB)")

    # --- ID estimation ---
    results_path = run_dir / "results.jsonl"
    n_rows = 0
    t_est = time.time()
    with results_path.open("w") as f:
        for depth in depths:
            reps = reps_by_depth[depth]  # (N, T, d)
            for gran in granularities:
                token_indices: list[int | None]
                if gran == "per_token":
                    token_indices = list(range(T))
                else:
                    token_indices = [None]
                for tok in token_indices:
                    cloud = _cloud(reps, gran, tok)
                    for est in estimators:
                        err: str | None = None
                        try:
                            val = float(estimate_id(cloud, est))
                        except Exception as e:  # noqa: BLE001
                            val = float("nan")
                            err = f"{type(e).__name__}: {e}"
                        # Strict JSON: serialize NaN/Inf as null so consumers
                        # (jq, pandas, duckdb) don't trip on non-standard tokens.
                        id_json: float | None
                        if val != val or val in (float("inf"), float("-inf")):
                            id_json = None
                        else:
                            id_json = val
                        row = {
                            "depth": depth,
                            "hook_path": path_by_depth[depth],
                            "granularity": gran,
                            "token_idx": tok,
                            "estimator": est,
                            "n_points": int(cloud.shape[0]),
                            "ambient_dim": int(cloud.shape[1]),
                            "id_estimate": id_json,
                            "error": err,
                        }
                        f.write(json.dumps(row) + "\n")
                        n_rows += 1
    print(f"[exp2] wrote {n_rows} id rows in {time.time() - t_est:.1f}s")

    # --- manifest ---
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config,
        "n_blocks": L,
        "n_sublayers_total": models.n_sublayers(m),
        "hidden_size": d_model,
        "max_tokens": T,
        "depths_captured": [
            {"depth": d, "hook_path": path_by_depth[d]} for d in depths
        ],
        "sublayers_requested": [[b, k] for (b, k) in sublayers],
        "n_inputs_requested": len(texts),
        "n_inputs_kept": N,
        "n_inputs_dropped": len(dropped),
        "dropped_inputs": dropped,
        "input_token_ids": input_ids_kept,
        "n_id_rows_written": n_rows,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[exp2] done; outputs in {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 2: intrinsic-dim estimation per depth / granularity."
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--weights",
        default="trained",
        choices=["trained", "random_gaussian", "random_kaiming"],
    )
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--layers",
        default="all",
        help="'all', or comma-separated 'block.kind' like '0.attn,5.ffn,11.attn'.",
    )
    p.add_argument(
        "--granularity",
        default="full_stream,per_token,last_token",
        help=f"comma-list from {_VALID_GRANULARITIES}",
    )
    p.add_argument(
        "--estimator",
        default="twonn",
        help=f"comma-list from {_VALID_ESTIMATORS}",
    )
    p.add_argument(
        "--save-reps",
        action="store_true",
        help=(
            "opt-in: write representations.npz with per-depth (N, T, d) fp32 arrays. "
            "Off by default because the file can balloon to multi-GB at realistic "
            "N/T/d (e.g. 500 x 64 x 768 x 25 depths = ~2 GB). Enable only when you "
            "actually need to re-estimate IDs from saved reps."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", type=Path, default=Path("results/exp2"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    granularities = _parse_csv(args.granularity)
    estimators = _parse_csv(args.estimator)
    for g in granularities:
        if g not in _VALID_GRANULARITIES:
            raise SystemExit(f"unknown granularity {g!r}; pick from {_VALID_GRANULARITIES}")
    for e in estimators:
        if e not in _VALID_ESTIMATORS:
            raise SystemExit(f"unknown estimator {e!r}; pick from {_VALID_ESTIMATORS}")
    run_exp2(
        model_name=args.model,
        weights=args.weights,
        corpus=args.corpus,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        layers_spec=args.layers,
        granularities=granularities,  # type: ignore[arg-type]
        estimators=estimators,  # type: ignore[arg-type]
        save_reps=args.save_reps,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
