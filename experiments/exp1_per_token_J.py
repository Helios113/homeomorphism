"""Experiment 1: per-sublayer full |det J| (and singular-value summary) across
inputs and selected sublayers, for Part A of the homeomorphism claim.

Per-row output (results.jsonl):
  {
    "input_id": int,                 # index into the corpus
    "input_preview": str,            # first 50 chars of the input
    "n_tokens": int,                 # tokenized length T
    "block_idx": int,                # transformer block index
    "sublayer_kind": "attn" | "ffn",
    "sign": int,                     # sign of det(full Jacobian)
    "log_abs_det": float,            # log|det(full J)|  (= sum_i log|det J^(i)|)
    "per_token_log_abs_det": [float, ...],  # one per token i
    "per_token_sign": [int, ...],           # one per token i
    "per_token_sigma_min": [float, ...],    # smallest singular value of each J^(i)
    "per_token_sigma_max": [float, ...],    # largest singular value of each J^(i)
    "per_token_condition_number": [float, ...],  # sigma_max / sigma_min
    "elapsed_sec": float,
  }

Run layout (folder per run):
  results/exp1/<run_id>/
    config.json      - CLI args that produced this run
    manifest.json    - metadata + counts + git sha + timings
    results.jsonl    - per-row measurements (as above)
    plots/           - created on demand by the plotter script

CLI
---
    uv run python -m experiments.exp1_per_token_J \\
        --model gpt2 --weights trained \\
        --corpus shakespeare --n-samples 4 --max-tokens 32 \\
        --layers 0.attn

    # several sublayers:
    --layers 0.attn,0.ffn,5.attn,11.ffn

    # the whole model:
    --layers all

    # random-init control (Mityagin at init):
    --weights random_gaussian --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from homeomorphism import hooks, jacobian, models
from homeomorphism.data import load_texts
from homeomorphism.models import SublayerKind


# ---------------------------------------------------------------------------
# Sublayer-spec parsing
# ---------------------------------------------------------------------------

def resolve_sublayers(n_blocks: int, spec: str) -> list[tuple[int, SublayerKind]]:
    """Parse --layers into [(block_idx, kind), ...].

    Accepted forms:
      "all"                         - every (block, kind) pair
      "0.attn"                      - single sublayer
      "0.attn,5.ffn,11.attn"        - explicit comma-list
    """
    spec = spec.strip()
    if spec == "all":
        return [(b, k) for b in range(n_blocks) for k in ("attn", "ffn")]

    result: list[tuple[int, SublayerKind]] = []
    for item in spec.split(","):
        item = item.strip()
        if "." not in item:
            raise ValueError(f"malformed spec {item!r}; expected 'block.kind'")
        block_str, kind_str = item.split(".", 1)
        try:
            block_idx = int(block_str)
        except ValueError as e:
            raise ValueError(f"block index not int: {block_str!r}") from e
        if not 0 <= block_idx < n_blocks:
            raise ValueError(f"block_idx {block_idx} out of range [0, {n_blocks})")
        if kind_str not in ("attn", "ffn"):
            raise ValueError(f"sublayer kind must be 'attn' or 'ffn'; got {kind_str!r}")
        result.append((block_idx, kind_str))  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

def measure_sublayer(
    m: models.Model,
    text: str,
    block_idx: int,
    kind: SublayerKind,
    max_tokens: int,
) -> dict[str, Any]:
    """Capture h^n at (block_idx, kind); return per-token + full slogdet + SVD summary.

    The returned row starts with the identifying fields (input_id must be
    filled by the caller) so that differences across samples are visible
    at a glance when rows are grouped by (block_idx, sublayer_kind).
    """
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

    full_log = float(sum(per_token_log))
    full_sign = 1
    for s in per_token_sign:
        full_sign *= s

    # Identification fields come first so different-sample rows stand out visually.
    return {
        # -- identification --
        "block_idx": block_idx,
        "sublayer_kind": kind,
        "n_tokens": T,
        "input_token_ids": input_ids[0].tolist(),  # full fingerprint of what went in
        # -- full-sublayer summary --
        "sign": full_sign,
        "log_abs_det": full_log,
        # -- per-token detail --
        "per_token_log_abs_det": per_token_log,
        "per_token_sign": per_token_sign,
        "per_token_sigma_min": per_token_sigma_min,
        "per_token_sigma_max": per_token_sigma_max,
        "per_token_condition_number": per_token_cond,
        # -- runtime --
        "elapsed_sec": round(time.time() - t0, 3),
    }


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


def run_exp1(
    *,
    model_name: str,
    weights: str,
    corpus: str,
    n_samples: int,
    max_tokens: int,
    layers_spec: str,
    seed: int,
    device: str,
    output_root: Path,
) -> Path:
    """Run Exp 1; return path to the run directory."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"

    # Save config (exactly what was passed in)
    config = {
        "model_name": model_name,
        "weights": weights,
        "corpus": corpus,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "layers_spec": layers_spec,
        "seed": seed,
        "device": device,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"[exp1] run_dir = {run_dir}")

    # --- Load ---
    m = models.load_model(model_name, weights=weights, seed=seed, device=device)  # type: ignore[arg-type]
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

    # --- Measure ---
    n_rows = 0
    n_errors = 0
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
                # Prepend sample identity so it leads the row on disk and at a glance.
                ordered = {
                    "input_id": input_id,
                    "input_preview": text[:50],
                    **row,
                }
                f.write(json.dumps(ordered) + "\n")
                f.flush()
                n_rows += 1
                print(
                    f"  {block_idx}.{kind}: "
                    f"T={row['n_tokens']:<3d} sign={row['sign']:+d} "
                    f"log|det|={row['log_abs_det']:+10.4f} "
                    f"sigma_min={min(row['per_token_sigma_min']):.2e} "
                    f"({row['elapsed_sec']:.1f}s)"
                )

    # --- Finalize ---
    manifest["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["end_time_epoch"] = time.time()
    manifest["duration_sec"] = round(time.time() - t_run, 1)
    manifest["n_rows_written"] = n_rows
    manifest["n_errors"] = n_errors
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[exp1] done: {n_rows} rows, {n_errors} errors, {manifest['duration_sec']}s total")
    print(f"[exp1] outputs in {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 1: per-sublayer log|det J| on a causal LM.")
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--weights",
        default="trained",
        choices=["trained", "random_gaussian", "random_kaiming"],
    )
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
    p.add_argument("--output-root", type=Path, default=Path("results/exp1"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_exp1(
        model_name=args.model,
        weights=args.weights,
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
