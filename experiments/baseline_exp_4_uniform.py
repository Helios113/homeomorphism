"""Baseline Experiment 4: Semantic Scrambling (Uniform Tokens).

Measures per-sublayer Jacobian properties AND multi-depth ID estimation when the
model uses semantic scrambling: tokens are replaced with uniform random token IDs.

Output schema (results.jsonl):
  Two types of rows:
  1. Jacobian row (per-sublayer):
     {
       "input_id": int,
       "baseline": "semantic_scrambling",
       "measurement_type": "jacobian",
       "block_idx": int,
       "sublayer_kind": "attn" | "ffn",
       ... (same as baseline_exp_1_topological.py fields)
     }
  2. ID estimation row (multi-depth):
     {
       "input_id": int,
       "baseline": "semantic_scrambling",
       "measurement_type": "id_estimation",
       "depth": int,
       ... (same as baseline_exp_2_maximum_entropy.py fields)
     }

Run layout:
  results/baseline_uniform/<run_id>/
    config.json
    manifest.json
    results.jsonl

CLI:
  python experiments/baseline_exp_4_uniform.py \\
    --model gpt2 --n-samples 10 --max-tokens 32 \\
    --layers all --depths all --estimator twonn --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from homeomorphism import jacobian, models, interventions
from homeomorphism.data import load_texts
from homeomorphism.id_est import EstimatorName, estimate_id
from homeomorphism.models import SublayerKind

from experiments.exp2_intrinsic_dim import (
    sublayer_depths,
    depth_to_hook_path,
    _cloud,
    _git_sha,
)
from experiments.exp1_per_token_J import resolve_sublayers

Granularity = Literal["full_stream", "per_token", "last_token"]
_KAPPA_ALERT_THRESHOLD = 1e8


def measure_sublayer(
    m: models.Model,
    text: str,
    block_idx: int,
    kind: SublayerKind,
    max_tokens: int,
    baseline: str,
    root_seed: int,
    sample_index: int,
) -> dict[str, Any]:
    """Measure per-token Jacobian for a sublayer with baseline-prepared input."""
    t0 = time.time()
    sub = models.sublayer(m, block_idx, kind)
    
    # Get the baseline-prepared input (uniform tokens)
    prepared = interventions.build_prepared_input(
        m=m,
        text=text,
        max_tokens=max_tokens,
        baseline=baseline,  # type: ignore
        root_seed=root_seed,
        sample_index=sample_index,
    )

    # Capture activation with uniform tokens from prepared forward kwargs.
    captured = capture_multi_with_prepared_input(
        m,
        [sub.hook_path],
        text,
        max_tokens=max_tokens,
        baseline=baseline,
        root_seed=root_seed,
        sample_index=sample_index,
    )
    h = captured[sub.hook_path].to(torch.float32)
    T, _d = h.shape

    bj, per_diag = jacobian.build_jacobian(
        sub.phi, h, scope="diagonal", evaluate="per_diagonal_slogdet"
    )
    assert bj.T == T

    per_token_log: list[float] = []
    per_token_sign: list[int] = []
    per_token_sigma_min: list[float] = []
    per_token_sigma_max: list[float] = []
    per_token_cond: list[float] = []

    for i in range(T):
        sign_i, log_i = per_diag[i]
        sv = bj.svdvals(i, i)
        sig_max = float(sv[0].item())
        sig_min = float(sv[-1].item())
        cond = float("inf") if sig_min == 0.0 else sig_max / sig_min
        per_token_log.append(float(log_i.item()))
        per_token_sign.append(int(sign_i.item()))
        per_token_sigma_min.append(sig_min)
        per_token_sigma_max.append(sig_max)
        per_token_cond.append(cond)

    n_kappa_alert = sum(1 for c in per_token_cond if c > _KAPPA_ALERT_THRESHOLD)
    n_invalid_cond = sum(1 for c in per_token_cond if not (c == c) or c in (float("inf"), float("-inf")))

    full_log = float(sum(per_token_log))
    full_sign = 1
    for s in per_token_sign:
        full_sign *= s

    return {
        "block_idx": block_idx,
        "sublayer_kind": kind,
        "n_tokens": T,
        "baseline": "semantic_scrambling",
        "measurement_type": "jacobian",
        "input_token_ids": prepared.token_ids[0].tolist(),
        "sign": full_sign,
        "log_abs_det": full_log,
        "per_token_log_abs_det": per_token_log,
        "per_token_sign": per_token_sign,
        "per_token_sigma_min": per_token_sigma_min,
        "per_token_sigma_max": per_token_sigma_max,
        "per_token_condition_number": per_token_cond,
        "kappa_alert_threshold": _KAPPA_ALERT_THRESHOLD,
        "n_kappa_alert": n_kappa_alert,
        "kappa_alert_fraction": float(n_kappa_alert / T if T > 0 else 0.0),
        "n_invalid_condition_number": n_invalid_cond,
        "elapsed_sec": round(time.time() - t0, 3),
    }


def capture_multi_with_prepared_input(
    m: models.Model,
    paths: list[str],
    text: str,
    max_tokens: int,
    baseline: str,
    root_seed: int,
    sample_index: int,
) -> dict[str, torch.Tensor]:
    """Run one forward pass with baseline-prepared input; capture at each path."""
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
        prepared = interventions.build_prepared_input(
            m=m,
            text=text,
            max_tokens=max_tokens,
            baseline=baseline,  # type: ignore
            root_seed=root_seed,
            sample_index=sample_index,
        )
        with torch.no_grad():
            m.model(**prepared.forward_kwargs)
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


def run(
    *,
    model_name: str,
    corpus: str,
    n_samples: int,
    max_tokens: int,
    layers_spec: str,
    depths_spec: str,
    granularities: list[Granularity],
    estimators: list[EstimatorName],
    seed: int,
    device: str,
    output_root: Path,
) -> Path:
    """Run semantic scrambling baseline; return path to the run directory."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    config: dict[str, Any] = {
        "model_name": model_name,
        "baseline": "semantic_scrambling",
        "corpus": corpus,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "layers_spec": layers_spec,
        "depths_spec": depths_spec,
        "granularities": granularities,
        "estimators": estimators,
        "seed": seed,
        "device": device,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[baseline_uniform] run_dir = {run_dir}")

    # Load model normally (trained weights, no topological init)
    m = models.load_model(model_name, weights="trained", seed=seed, device=device)
    L = models.n_blocks(m)
    d_model = models.hidden_size(m)
    
    # Parse layers and depths
    sublayers = resolve_sublayers(L, layers_spec)
    if depths_spec.lower() == "all":
        depths = list(range(2 * L + 1))
    else:
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
        f"[baseline_uniform] model={model_name} L={L} d={d_model} "
        f"sublayers={len(sublayers)} depths={len(depths)} inputs={len(texts)} T={max_tokens}"
    )

    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"

    # Capture representations and measure Jacobians
    T = max_tokens
    buf: dict[int, list[torch.Tensor]] = {d: [] for d in depths}
    input_ids_kept: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    n_jacobian_rows = 0
    total_kappa_alerts = 0
    total_invalid_conds = 0
    t_run = time.time()

    with results_path.open("w") as f:
        for input_id, text in enumerate(texts):
            print(f"\n=== input {input_id} === preview: {text[:60]!r}")

            # Measure Jacobians
            for block_idx, kind in sublayers:
                try:
                    row = measure_sublayer(
                        m,
                        text=text,
                        block_idx=block_idx,
                        kind=kind,
                        max_tokens=max_tokens,
                        baseline="semantic_scrambling",
                        root_seed=seed,
                        sample_index=input_id,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  [ERR] Jacobian {block_idx}.{kind}: {type(e).__name__}: {e}")
                    continue
                ordered = {
                    "input_id": input_id,
                    "input_preview": text[:50],
                    **row,
                }
                f.write(json.dumps(ordered) + "\n")
                f.flush()
                n_jacobian_rows += 1
                total_kappa_alerts += int(row["n_kappa_alert"])
                total_invalid_conds += int(row["n_invalid_condition_number"])

            # Capture residual streams for ID estimation
            try:
                captured = capture_multi_with_prepared_input(
                    m,
                    paths,
                    text,
                    max_tokens=max_tokens,
                    baseline="semantic_scrambling",
                    root_seed=seed,
                    sample_index=input_id,
                )
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

        # ID estimation
        N = len(input_ids_kept)
        if N >= 2:
            reps_by_depth: dict[int, torch.Tensor] = {
                d: torch.stack(buf[d], dim=0) for d in depths
            }
            n_id_rows = 0
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
                            id_json: float | None
                            if val != val or val in (float("inf"), float("-inf")):
                                id_json = None
                            else:
                                id_json = val
                            row = {
                                "baseline": "semantic_scrambling",
                                "measurement_type": "id_estimation",
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
                            f.flush()
                            n_id_rows += 1
        else:
            n_id_rows = 0
            print(f"[baseline_uniform] insufficient inputs for ID estimation (N={N} < 2)")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config,
        "sublayers_resolved": [[b, k] for (b, k) in sublayers],
        "depths_resolved": depths,
        "n_blocks": L,
        "hidden_size": d_model,
        "n_inputs_processed": len(texts),
        "n_inputs_kept_for_id": N if N >= 2 else 0,
        "n_inputs_dropped": len(dropped),
        "n_jacobian_rows": n_jacobian_rows,
        "n_id_rows": n_id_rows if N >= 2 else 0,
        "kappa_alert_threshold": _KAPPA_ALERT_THRESHOLD,
        "total_kappa_alerts": total_kappa_alerts,
        "total_invalid_condition_numbers": total_invalid_conds,
        "duration_sec": round(time.time() - t_run, 1),
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"\n[baseline_uniform] done: {n_jacobian_rows} jacobian rows, "
        f"{n_id_rows if N >= 2 else 0} id rows, {manifest['duration_sec']}s total"
    )
    print(f"[baseline_uniform] outputs in {run_dir}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline Exp 4: Semantic Scrambling (Uniform Tokens)")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--layers",
        default="0.attn",
        help="'all', or comma-separated 'block.kind' like '0.attn,5.ffn,11.attn'.",
    )
    p.add_argument(
        "--depths",
        default="all",
        help="'all', or comma-separated depths.",
    )
    p.add_argument(
        "--granularity",
        default="full_stream",
        help="Comma-separated list of 'full_stream', 'per_token', 'last_token'.",
    )
    p.add_argument(
        "--estimator",
        default="twonn",
        help="Comma-separated list of 'twonn', 'ess', 'participation_ratio'.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", type=Path, default=Path("results/baseline_uniform"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    granularities: list[Granularity] = [g.strip() for g in args.granularity.split(",")]  # type: ignore
    estimators: list[EstimatorName] = [e.strip() for e in args.estimator.split(",")]  # type: ignore
    run(
        model_name=args.model,
        corpus=args.corpus,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        layers_spec=args.layers,
        depths_spec=args.depths,
        granularities=granularities,
        estimators=estimators,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
