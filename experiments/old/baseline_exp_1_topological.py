"""Baseline Experiment 1: Topological Initialisation.

Measures per-sublayer Jacobian properties (log|det J|, condition number) when the
model uses topological initialisation: random_gaussian weight re-init with norm
affine reset (weight→1, bias→0).

Output schema (results.jsonl):
  {
    "input_id": int,
    "input_preview": str,
    "n_tokens": int,
    "baseline": "topological_initialisation",
    "block_idx": int,
    "sublayer_kind": "attn" | "ffn",
    "sign": int,
    "log_abs_det": float,
    "per_token_log_abs_det": [float, ...],
    "per_token_sign": [int, ...],
    "per_token_sigma_min": [float, ...],
    "per_token_sigma_max": [float, ...],
    "per_token_condition_number": [float, ...],
    "kappa_alert_threshold": float,
    "n_kappa_alert": int,
    "kappa_alert_fraction": float,
    "n_invalid_condition_number": int,
    "elapsed_sec": float,
  }

Run layout:
  results/baseline_topological/<run_id>/
    config.json
    manifest.json
    results.jsonl

CLI:
  python experiments/baseline_exp_1_topological.py \\
    --model gpt2 --n-samples 10 --max-tokens 32 \\
    --layers all --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from homeomorphism import hooks, jacobian, models, interventions
from homeomorphism.data import load_texts
from homeomorphism.models import SublayerKind

from experiments.exp1_per_token_J import resolve_sublayers

_KAPPA_ALERT_THRESHOLD = 1e8


def measure_sublayer(
    m: models.Model,
    text: str,
    block_idx: int,
    kind: SublayerKind,
    max_tokens: int,
) -> dict[str, Any]:
    """Capture h^n at (block_idx, kind); return per-token + full slogdet + SVD summary."""
    t0 = time.time()
    sub = models.sublayer(m, block_idx, kind)
    input_ids = models.tokenize(m, text, max_tokens=max_tokens)
    h = hooks.capture_activation(m, sub.hook_path, text, max_tokens=max_tokens).to(torch.float32)
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
        "baseline": "topological_initialisation",
        "input_token_ids": input_ids[0].tolist(),
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


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run(
    *,
    model_name: str,
    n_samples: int,
    max_tokens: int,
    layers_spec: str,
    seed: int,
    device: str,
    output_root: Path,
    corpus: str,
) -> Path:
    """Run topological initialisation baseline; return path to the run directory."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"

    config = {
        "model_name": model_name,
        "baseline": "topological_initialisation",
        "corpus": corpus,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "layers_spec": layers_spec,
        "seed": seed,
        "device": device,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"[baseline_topological] run_dir = {run_dir}")

    # Load model with topological_initialisation baseline
    m = interventions.load_model_for_baseline(
        model_name=model_name,
        weights="trained",
        baseline="topological_initialisation",
        seed=seed,
        device=device,
    )
    sublayers = resolve_sublayers(models.n_blocks(m), layers_spec)
    texts = load_texts(
        corpus,  # type: ignore[arg-type]
        n_samples=n_samples,
        chunk_chars=max(max_tokens * 8, 200),
        seed=seed,
    )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config,
        "sublayers_resolved": [[b, k] for (b, k) in sublayers],
        "n_blocks": models.n_blocks(m),
        "n_sublayers_total": models.n_sublayers(m),
        "hidden_size": models.hidden_size(m),
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "start_time_epoch": time.time(),
    }

    n_rows = 0
    n_errors = 0
    total_tokens_measured = 0
    total_kappa_alerts = 0
    total_invalid_conds = 0
    t_run = time.time()
    with results_path.open("w") as f:
        for input_id, text in enumerate(texts):
            print(f"\n=== input {input_id} === preview: {text[:60]!r}")
            for block_idx, kind in sublayers:
                try:
                    row = measure_sublayer(
                        m, text=text, block_idx=block_idx, kind=kind, max_tokens=max_tokens
                    )
                except Exception as e:  # noqa: BLE001
                    n_errors += 1
                    print(f"  [ERR] input {input_id}, {block_idx}.{kind}: {type(e).__name__}: {e}")
                    continue
                ordered = {
                    "input_id": input_id,
                    "input_preview": text[:50],
                    **row,
                }
                f.write(json.dumps(ordered) + "\n")
                f.flush()
                n_rows += 1
                total_tokens_measured += int(row["n_tokens"])
                total_kappa_alerts += int(row["n_kappa_alert"])
                total_invalid_conds += int(row["n_invalid_condition_number"])
                print(
                    f"  {block_idx}.{kind}: "
                    f"T={row['n_tokens']:<3d} sign={row['sign']:+d} "
                    f"log|det|={row['log_abs_det']:+10.4f} "
                    f"sigma_min={min(row['per_token_sigma_min']):.2e} "
                    f"({row['elapsed_sec']:.1f}s)"
                )

    manifest["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["end_time_epoch"] = time.time()
    manifest["duration_sec"] = round(time.time() - t_run, 1)
    manifest["n_rows_written"] = n_rows
    manifest["n_errors"] = n_errors
    manifest["kappa_alert_threshold"] = _KAPPA_ALERT_THRESHOLD
    manifest["total_tokens_measured"] = total_tokens_measured
    manifest["total_kappa_alerts"] = total_kappa_alerts
    manifest["kappa_alert_fraction"] = float(
        total_kappa_alerts / total_tokens_measured if total_tokens_measured > 0 else 0.0
    )
    manifest["total_invalid_condition_numbers"] = total_invalid_conds
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[baseline_topological] done: {n_rows} rows, {n_errors} errors, {manifest['duration_sec']}s total")
    print(f"[baseline_topological] outputs in {run_dir}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline Exp 1: Topological Initialisation")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--layers",
        default="0.attn",
        help="'all', or comma-separated 'block.kind' like '0.attn,5.ffn,11.attn'.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", type=Path, default=Path("results/baseline_topological"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run(
        model_name=args.model,
        corpus=args.corpus,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        layers_spec=args.layers,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
