"""Baseline Experiment 2: Maximum-Entropy Injection.

Measures intrinsic dimension of residual streams when the model uses maximum-entropy
injection: Gaussian noise (covariance (1/d)I) on inputs_embeds instead of actual token
embeddings.

Output schema (results.jsonl):
  {
    "baseline": "maximum_entropy_injection",
    "depth": int,
    "hook_path": str,
    "granularity": "full_stream" | "per_token" | "last_token",
    "token_idx": int | null,
    "estimator": "twonn" | "ess" | "participation_ratio",
    "n_points": int,
    "ambient_dim": int,
    "id_estimate": float | null,
    "error": str | null,
  }

Run layout:
  results/baseline_maximum_entropy/<run_id>/
    config.json
    manifest.json
    results.jsonl

CLI:
  python experiments/baseline_exp_2_maximum_entropy.py \\
    --model gpt2 --n-samples 10 --max-tokens 32 \\
    --depths all --estimator twonn --seed 42
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

from homeomorphism import models, interventions
from homeomorphism.data import load_texts
from homeomorphism.id_est import EstimatorName, estimate_id
from homeomorphism.interventions import PreparedInput

from experiments.exp2_intrinsic_dim import (
    sublayer_depths,
    depth_to_hook_path,
    _cloud,
    _git_sha,
)
from experiments.exp1_per_token_J import resolve_sublayers

Granularity = Literal["full_stream", "per_token", "last_token"]


def capture_multi_with_prepared_input(
    m: models.Model,
    paths: list[str],
    *,
    forward_kwargs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run one forward pass from prepared kwargs; capture at each path.

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
        with torch.no_grad():
            m.model(**forward_kwargs)
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
    granularities: list[Granularity],
    estimators: list[EstimatorName],
    seed: int,
    device: str,
    output_root: Path,
) -> Path:
    """Run maximum-entropy injection baseline; return path to the run directory."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    config: dict[str, Any] = {
        "model_name": model_name,
        "baseline": "maximum_entropy_injection",
        "corpus": corpus,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "layers_spec": layers_spec,
        "granularities": granularities,
        "estimators": estimators,
        "seed": seed,
        "device": device,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[baseline_maximum_entropy] run_dir = {run_dir}")

    # Load model normally (trained weights, no topological init)
    m = models.load_model(model_name, weights="trained", seed=seed, device=device)
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
        f"[baseline_maximum_entropy] model={model_name} L={L} d={d_model} "
        f"depths={len(depths)} (indices {depths[0]}..{depths[-1]}) "
        f"inputs={len(texts)} T={max_tokens}"
    )

    # Capture representations using baseline-prepared inputs
    T = max_tokens
    buf: dict[int, list[torch.Tensor]] = {d: [] for d in depths}
    input_ids_kept: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    t_cap = time.time()

    for input_id, text in enumerate(texts):
        prepared: PreparedInput = interventions.build_prepared_input(
            m=m,
            text=text,
            max_tokens=max_tokens,
            baseline="maximum_entropy_injection",
            root_seed=seed,
            sample_index=input_id,
        )
        try:
            captured = capture_multi_with_prepared_input(
                m,
                paths,
                forward_kwargs=prepared.forward_kwargs,
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

        input_ids_kept.append(prepared.token_ids[0].tolist())

        if (input_id + 1) % 50 == 0:
            print(f"  captured {input_id + 1}/{len(texts)} ({len(input_ids_kept)} kept)")

    N = len(input_ids_kept)
    if N < 2:
        print(f"[baseline_maximum_entropy] insufficient inputs for ID estimation (N={N} < 2)")

    reps_by_depth: dict[int, torch.Tensor] = {}
    if N >= 1:
        reps_by_depth = {d: torch.stack(buf[d], dim=0) for d in depths}
    print(
        f"[baseline_maximum_entropy] captured {N}/{len(texts)} inputs in {time.time() - t_cap:.1f}s "
        f"(dropped {len(dropped)})"
    )

    # ID estimation
    results_path = run_dir / "results.jsonl"
    n_rows = 0
    t_est = time.time()
    with results_path.open("w") as f:
        if N < 2:
            pass
        for depth in depths:
            if N < 2:
                break
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
                            "baseline": "maximum_entropy_injection",
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
                        n_rows += 1

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config,
        "sublayers_resolved": [[b, k] for (b, k) in sublayers],
        "depths_resolved": depths,
        "n_blocks": L,
        "hidden_size": d_model,
        "n_inputs_kept": N,
        "n_inputs_dropped": len(dropped),
        "n_rows_written": n_rows,
        "capture_time_sec": round(time.time() - t_cap, 1),
        "estimate_time_sec": round(time.time() - t_est, 1),
        "total_time_sec": round(time.time() - t_cap, 1),
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\n[baseline_maximum_entropy] done: {n_rows} rows in {manifest['total_time_sec']}s"
    )
    print(f"[baseline_maximum_entropy] outputs in {run_dir}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline Exp 2: Maximum-Entropy Injection")
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
    p.add_argument("--output-root", type=Path, default=Path("results/baseline_maximum_entropy"))
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
        granularities=granularities,
        estimators=estimators,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
