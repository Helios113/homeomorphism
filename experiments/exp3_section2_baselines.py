"""Experiment 3: unified Section 2.1 baseline interventions.

This experiment runs the Section 2.1 intervention baselines with shared hooks
and deterministic seeds while keeping existing Exp1/Exp2 scripts unchanged.

Outputs (project-local):
  results/exp3_section2/<run_id>/
    config.json
    manifest.json
    jacobian.jsonl
    id.jsonl
    overlap.jsonl
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

import h5py
import torch

from homeomorphism import jacobian, models
from homeomorphism.data import load_texts
from homeomorphism.id_est import EstimatorName, estimate_id
from homeomorphism.interventions import (
    VALID_BASELINES,
    BaselineName,
    build_prepared_input,
    load_model_for_baseline,
    topological_init_calibration,
)
from homeomorphism.models import SublayerKind

from experiments.exp1_per_token_J import resolve_sublayers


Granularity = Literal["full_stream", "per_token", "last_token"]
_VALID_GRANULARITIES: tuple[Granularity, ...] = ("full_stream", "per_token", "last_token")
_VALID_ESTIMATORS: tuple[EstimatorName, ...] = ("twonn", "ess", "participation_ratio")
_KAPPA_ALERT_THRESHOLD = 1e8

_BASELINE_GROUP_NAMES: dict[BaselineName, str] = {
    "trained": "trained",
    "topological_initialisation": "topological_init",
    "maximum_entropy_injection": "max_entropy",
    "syntactic_disintegration": "permuted",
    "semantic_scrambling": "uniform_tokens",
}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _baseline_group_name(baseline: BaselineName) -> str:
    return _BASELINE_GROUP_NAMES[baseline]


def _depth_dataset_name(depth: int) -> str:
    return f"depth_{depth:02d}"


def _make_latent_datasets(
    group: h5py.Group,
    depths: list[int],
    seq_len: int,
    d_model: int,
) -> None:
    for depth in depths:
        key = _depth_dataset_name(depth)
        if key in group:
            continue
        group.create_dataset(
            key,
            shape=(0, seq_len, d_model),
            maxshape=(None, seq_len, d_model),
            chunks=(1, seq_len, d_model),
            dtype="float32",
        )


def _append_latent_sample(group: h5py.Group, depth: int, sample: torch.Tensor) -> None:
    key = _depth_dataset_name(depth)
    dataset = group[key]
    if sample.dim() != 2:
        raise ValueError(f"expected (T, d) sample for {key!r}, got {tuple(sample.shape)}")
    next_index = int(dataset.shape[0])
    dataset.resize(next_index + 1, axis=0)
    dataset[next_index] = sample.detach().to(torch.float32).cpu().numpy()


def _model_input_stream(m: models.Model, forward_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the residual stream entering block 0, shape (T, d)."""
    if m.arch == "gpt2":
        if "inputs_embeds" in forward_kwargs:
            x = forward_kwargs["inputs_embeds"]
        else:
            input_ids = forward_kwargs["input_ids"]
            x = m.model.transformer.wte(input_ids)

        position_ids = forward_kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("prepared input is missing position_ids")

        pos = m.model.transformer.wpe(position_ids)
        x = x + pos
        x = m.model.transformer.drop(x)
    elif m.arch == "llama":
        if "inputs_embeds" in forward_kwargs:
            x = forward_kwargs["inputs_embeds"]
        else:
            input_ids = forward_kwargs["input_ids"]
            # LLaMAForCausalLM stores the base model in .model; embed_tokens there
            x = m.model.model.embed_tokens(input_ids)
        # No separate position embeddings; RoPE applied inside attention
        # No additional dropout needed (model.eval() disables dropout)
    else:
        raise NotImplementedError(f"input-stream capture not registered for arch {m.arch!r}")

    if x.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got {x.shape[0]}")
    return x[0].detach().to(torch.float32).cpu()


def _cloud(reps: torch.Tensor, granularity: Granularity, token_idx: int | None) -> torch.Tensor:
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


def _estimate_id_rows(
    *,
    baseline: BaselineName,
    group: h5py.Group,
    depths: list[int],
    seq_len: int,
    granularities: list[Granularity],
    estimators: list[EstimatorName],
    id_path: Path,
    overlap_path: Path,
    overlap_k: int,
) -> tuple[int, int]:
    """Offline stage: read HDF5 latents and write ID/overlap JSONL rows."""
    n_id_rows = 0
    n_overlap_rows = 0

    with id_path.open("a") as fi:
        if not depths:
            return 0, 0

        for depth in depths:
            key = _depth_dataset_name(depth)
            if key not in group:
                continue
            reps = group[key][:]  # (N, T, d)
            N = int(reps.shape[0])
            T = int(reps.shape[1])
            user_depth = depth - 1

            if N < 2:
                continue

            for gran in granularities:
                token_indices: list[int | None]
                if gran == "per_token":
                    token_indices = list(range(T))
                else:
                    token_indices = [None]

                for tok in token_indices:
                    cloud = _cloud(torch.from_numpy(reps), gran, tok)
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
                            "baseline": baseline,
                            "depth": user_depth,
                            "hook_path": group.attrs.get(f"hook_path_{user_depth}", None),
                            "granularity": gran,
                            "token_idx": tok,
                            "estimator": est,
                            "n_points": int(cloud.shape[0]),
                            "ambient_dim": int(cloud.shape[1]),
                            "id_estimate": id_json,
                            "error": err,
                        }
                        fi.write(json.dumps(row) + "\n")
                        n_id_rows += 1

    with overlap_path.open("a") as fo:
        consecutive = list(zip(depths[:-1], depths[1:]))
        for da, db in consecutive:
            key_a = _depth_dataset_name(da)
            key_b = _depth_dataset_name(db)
            if key_a not in group or key_b not in group:
                continue
            reps_a = torch.from_numpy(group[key_a][:])
            reps_b = torch.from_numpy(group[key_b][:])
            cloud_a = _cloud(reps_a, "full_stream", None)
            cloud_b = _cloud(reps_b, "full_stream", None)
            ov = _neighbor_overlap(cloud_a, cloud_b, overlap_k)
            fo.write(
                json.dumps(
                    {
                        "baseline": baseline,
                        "depth_from": da - 1,
                        "depth_to": db - 1,
                        "k": overlap_k,
                        "granularity": "full_stream",
                        "neighbour_overlap": ov,
                    }
                )
                + "\n"
            )
            n_overlap_rows += 1

            cloud_a_last = _cloud(reps_a, "last_token", None)
            cloud_b_last = _cloud(reps_b, "last_token", None)
            ov_last = _neighbor_overlap(cloud_a_last, cloud_b_last, overlap_k)
            fo.write(
                json.dumps(
                    {
                        "baseline": baseline,
                        "depth_from": da - 1,
                        "depth_to": db - 1,
                        "k": overlap_k,
                        "granularity": "last_token",
                        "neighbour_overlap": ov_last,
                    }
                )
                + "\n"
            )
            n_overlap_rows += 1

    return n_id_rows, n_overlap_rows


def sublayer_depths(sublayers: list[tuple[int, SublayerKind]]) -> list[int]:
    depths: set[int] = set()
    for b, k in sublayers:
        pre = 2 * b + (0 if k == "attn" else 1)
        depths.add(pre)
        depths.add(pre + 1)
    return sorted(depths)


def _final_norm_path(m: models.Model) -> str:
    if m.arch == "gpt2":
        return "transformer.ln_f"
    elif m.arch == "llama":
        return "model.norm"
    raise NotImplementedError(f"final norm path not registered for arch {m.arch!r}")


def depth_to_hook_path(m: models.Model, depth: int) -> str:
    L = models.n_blocks(m)
    if depth == 2 * L:
        return _final_norm_path(m)
    if not 0 <= depth < 2 * L:
        raise IndexError(f"depth {depth} out of [0, {2 * L}]")
    b, rem = divmod(depth, 2)
    kind: SublayerKind = "attn" if rem == 0 else "ffn"
    return models.sublayer(m, b, kind).hook_path


def capture_multi(
    m: models.Model,
    paths: list[str],
    *,
    forward_kwargs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    dmat = torch.cdist(points.to(torch.float32), points.to(torch.float32))
    dmat.fill_diagonal_(float("inf"))
    return torch.topk(dmat, k=min(k, points.shape[0] - 1), largest=False, dim=1).indices


def _neighbor_overlap(points_a: torch.Tensor, points_b: torch.Tensor, k: int) -> float:
    if points_a.shape[0] < 2:
        return 0.0
    k_eff = min(k, points_a.shape[0] - 1)
    ia = _knn_indices(points_a, k_eff)
    ib = _knn_indices(points_b, k_eff)
    overlap_sum = 0.0
    for i in range(points_a.shape[0]):
        sa = set(int(x) for x in ia[i].tolist())
        sb = set(int(x) for x in ib[i].tolist())
        overlap_sum += len(sa.intersection(sb)) / float(k_eff)
    return float(overlap_sum / points_a.shape[0])


def run_exp3(
    *,
    model_name: str,
    weights: str,
    baselines: list[BaselineName],
    corpus: str,
    n_samples: int,
    max_tokens: int,
    layers_spec: str,
    granularities: list[Granularity],
    estimators: list[EstimatorName],
    overlap_k: int,
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
        "baselines": baselines,
        "corpus": corpus,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "layers_spec": layers_spec,
        "granularities": granularities,
        "estimators": estimators,
        "overlap_k": overlap_k,
        "seed": seed,
        "device": device,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    probe_model = models.load_model(model_name, weights=weights, seed=seed, device=device)
    L = models.n_blocks(probe_model)
    d_model = models.hidden_size(probe_model)
    sublayers = resolve_sublayers(L, layers_spec)
    depths = sublayer_depths(sublayers)
    paths = [depth_to_hook_path(probe_model, d) for d in depths]
    path_by_depth = dict(zip(depths, paths))
    storage_depths = [0] + [d + 1 for d in depths]

    texts = load_texts(
        corpus,  # type: ignore[arg-type]
        n_samples=n_samples,
        chunk_chars=max(max_tokens * 8, 200),
        seed=seed,
    )

    latents_path = run_dir / "latents.h5"
    jac_path = run_dir / "jacobian.jsonl"
    id_path = run_dir / "id.jsonl"
    overlap_path = run_dir / "overlap.jsonl"
    jac_path.write_text("")
    id_path.write_text("")
    overlap_path.write_text("")

    baseline_summaries: list[dict[str, Any]] = []
    total_jac_rows = 0
    total_id_rows = 0
    total_overlap_rows = 0
    total_kappa_alerts = 0
    total_invalid_conds = 0

    with h5py.File(latents_path, "w") as h5, jac_path.open("a") as fj:
        h5.attrs["config"] = json.dumps(config)
        h5.attrs["model_name"] = model_name
        h5.attrs["weights"] = weights
        h5.attrs["n_layers"] = L
        h5.attrs["d_model"] = d_model
        h5.attrs["seq_len"] = max_tokens
        h5.attrs["depths_resolved"] = json.dumps(depths)

        for baseline in baselines:
            baseline_dir = _baseline_group_name(baseline)
            group = h5.create_group(baseline_dir)
            group.attrs["baseline"] = baseline
            group.attrs["model_name"] = model_name
            group.attrs["weights"] = weights
            group.attrs["seq_len"] = max_tokens
            group.attrs["d_model"] = d_model
            group.attrs["n_layers"] = L
            group.attrs["depth_offset"] = 1
            group.attrs["depths_resolved"] = json.dumps(depths)
            for depth, path in path_by_depth.items():
                group.attrs[f"hook_path_{depth}"] = path
            _make_latent_datasets(group, storage_depths, max_tokens, d_model)

            m = load_model_for_baseline(
                model_name=model_name,
                weights=weights,
                baseline=baseline,
                seed=seed,
                device=device,
            )

            calibration_norm: float | None = None
            if baseline == "topological_initialisation":
                calibration_norm = topological_init_calibration(m=m, root_seed=seed)
                # Allow a reasonable range: lower bound 0.1, upper bound scaled to model width.
                # After topological init (LayerNorm affines reset to 1/0), the residual norm
                # scales approximately like sqrt(d_model). Use 2*sqrt(d) as generous upper bound.
                upper_bound = 2.0 * (d_model ** 0.5)
                if not (0.1 <= calibration_norm <= upper_bound):
                    print(f"[WARN] Calibration norm {calibration_norm:.4f} outside [0.1, {upper_bound:.2f}]; proceeding anyway")
                    # raise RuntimeError(...)  # not fatal

            depth_to_sublayer: dict[int, tuple[int, SublayerKind]] = {
                2 * b + (0 if k == "attn" else 1): (b, k) for (b, k) in sublayers
            }

            n_jac_rows = 0
            n_inputs_kept = 0
            n_inputs_dropped = 0
            total_tokens_measured = 0
            total_kappa_alerts_baseline = 0
            total_invalid_conds_baseline = 0
            token_ids_kept: list[list[int]] = []
            dropped: list[dict[str, Any]] = []

            for input_id, text in enumerate(texts):
                prepared = build_prepared_input(
                    m=m,
                    text=text,
                    max_tokens=max_tokens,
                    baseline=baseline,
                    root_seed=seed,
                    sample_index=input_id,
                )

                try:
                    captured = capture_multi(m, paths, forward_kwargs=prepared.forward_kwargs)
                except Exception as e:  # noqa: BLE001
                    dropped.append({"input_id": input_id, "reason": f"{type(e).__name__}: {e}"})
                    n_inputs_dropped += 1
                    continue

                lengths = {tensor.shape[0] for tensor in captured.values()}
                if len(lengths) != 1:
                    dropped.append({"input_id": input_id, "reason": f"depth length mismatch {lengths}"})
                    n_inputs_dropped += 1
                    continue
                t_seq = lengths.pop()
                if t_seq != max_tokens:
                    dropped.append({"input_id": input_id, "reason": f"tokenized to {t_seq} != T={max_tokens}"})
                    n_inputs_dropped += 1
                    continue

                _append_latent_sample(group, 0, _model_input_stream(m, prepared.forward_kwargs))
                for depth, path in path_by_depth.items():
                    _append_latent_sample(group, depth + 1, captured[path])

                token_ids_kept.append(prepared.token_ids[0].tolist())
                n_inputs_kept += 1

                model_device = next(m.model.parameters()).device
                for depth, (block_idx, kind) in depth_to_sublayer.items():
                    h = captured[path_by_depth[depth]].to(device=model_device, dtype=torch.float32)
                    sub = models.sublayer(m, block_idx, kind)
                    bj, per_diag = jacobian.build_jacobian(
                        sub.phi,
                        h,
                        scope="diagonal",
                        evaluate="per_diagonal_slogdet",
                    )
                    for tok in range(h.shape[0]):
                        sign_i, log_i = per_diag[tok]
                        sv = bj.svdvals(tok, tok)
                        sigma_max = float(sv[0].item())
                        sigma_min = float(sv[-1].item())
                        cond = float("inf") if sigma_min == 0.0 else sigma_max / sigma_min
                        is_invalid = not (cond == cond) or cond in (float("inf"), float("-inf"))
                        is_alert = cond > _KAPPA_ALERT_THRESHOLD
                        row = {
                            "baseline": baseline,
                            "input_id": input_id,
                            "block_idx": block_idx,
                            "sublayer_kind": kind,
                            "sub_block_depth": depth,
                            "token_pos": tok,
                            "log_abs_det": float(log_i.item()),
                            "sign": int(sign_i.item()),
                            "sigma_min": sigma_min,
                            "sigma_max": sigma_max,
                            "condition_number": cond,
                            "kappa_alert_threshold": _KAPPA_ALERT_THRESHOLD,
                            "kappa_alert": bool(is_alert),
                            "condition_invalid": bool(is_invalid),
                        }
                        fj.write(json.dumps(row) + "\n")
                        n_jac_rows += 1
                        total_tokens_measured += 1
                        total_kappa_alerts_baseline += int(is_alert)
                        total_invalid_conds_baseline += int(is_invalid)

            n_id_rows, n_overlap_rows = _estimate_id_rows(
                baseline=baseline,
                group=group,
                depths=storage_depths[1:],
                seq_len=max_tokens,
                granularities=granularities,
                estimators=estimators,
                id_path=id_path,
                overlap_path=overlap_path,
                overlap_k=overlap_k,
            )

            baseline_summaries.append(
                {
                    "baseline": baseline,
                    "group_name": baseline_dir,
                    "n_inputs_requested": len(texts),
                    "n_inputs_kept": n_inputs_kept,
                    "n_inputs_dropped": n_inputs_dropped,
                    "dropped_inputs": dropped,
                    "input_token_ids": token_ids_kept,
                    "n_jacobian_rows": n_jac_rows,
                    "n_id_rows": n_id_rows,
                    "n_overlap_rows": n_overlap_rows,
                    "kappa_alert_threshold": _KAPPA_ALERT_THRESHOLD,
                    "total_tokens_measured": total_tokens_measured,
                    "total_kappa_alerts": total_kappa_alerts_baseline,
                    "kappa_alert_fraction": float(
                        total_kappa_alerts_baseline / total_tokens_measured if total_tokens_measured > 0 else 0.0
                    ),
                    "total_invalid_condition_numbers": total_invalid_conds_baseline,
                    "topological_init_calibration_norm": calibration_norm,
                }
            )
            total_jac_rows += n_jac_rows
            total_id_rows += n_id_rows
            total_overlap_rows += n_overlap_rows
            total_kappa_alerts += total_kappa_alerts_baseline
            total_invalid_conds += total_invalid_conds_baseline

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "config": config,
        "n_blocks": L,
        "n_sublayers_total": models.n_sublayers(probe_model),
        "hidden_size": d_model,
        "max_tokens": max_tokens,
        "depths_resolved": depths,
        "storage_depths": storage_depths,
        "sublayers_requested": [[b, k] for (b, k) in sublayers],
        "n_inputs_requested": len(texts),
        "baseline_summaries": baseline_summaries,
        "n_jacobian_rows": total_jac_rows,
        "n_id_rows": total_id_rows,
        "n_overlap_rows": total_overlap_rows,
        "kappa_alert_threshold": _KAPPA_ALERT_THRESHOLD,
        "total_kappa_alerts": total_kappa_alerts,
        "total_invalid_condition_numbers": total_invalid_conds,
        "latents_path": str(latents_path),
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 3: Section 2.1 baseline engine.")
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--weights",
        default="trained",
        choices=["trained", "random_gaussian", "random_kaiming"],
    )
    p.add_argument(
        "--baseline",
        default=None,
        choices=list(VALID_BASELINES),
        help="single-baseline shortcut; overrides --baselines when set",
    )
    p.add_argument(
        "--baselines",
        default="trained,topological_initialisation,maximum_entropy_injection,syntactic_disintegration,semantic_scrambling",
        help=f"comma-list from {list(VALID_BASELINES)}",
    )
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--layers",
        default="all",
        help="'all', or comma-separated 'block.kind' like '0.attn,5.ffn,11.attn'.",
    )
    p.add_argument(
        "--granularity",
        default="full_stream,last_token",
        help=f"comma-list from {_VALID_GRANULARITIES}",
    )
    p.add_argument(
        "--estimator",
        default="twonn,ess,participation_ratio",
        help=f"comma-list from {_VALID_ESTIMATORS}",
    )
    p.add_argument("--overlap-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", type=Path, default=Path("results/exp3_section2"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    granularities = _parse_csv(args.granularity)
    estimators = _parse_csv(args.estimator)
    baselines = [args.baseline] if args.baseline is not None else _parse_csv(args.baselines)
    for g in granularities:
        if g not in _VALID_GRANULARITIES:
            raise SystemExit(f"unknown granularity {g!r}; pick from {_VALID_GRANULARITIES}")
    for e in estimators:
        if e not in _VALID_ESTIMATORS:
            raise SystemExit(f"unknown estimator {e!r}; pick from {_VALID_ESTIMATORS}")
    for b in baselines:
        if b not in VALID_BASELINES:
            raise SystemExit(f"unknown baseline {b!r}; pick from {VALID_BASELINES}")

    run_dir = run_exp3(
        model_name=args.model,
        weights=args.weights,
        baselines=baselines,  # type: ignore[arg-type]
        corpus=args.corpus,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        layers_spec=args.layers,
        granularities=granularities,  # type: ignore[arg-type]
        estimators=estimators,  # type: ignore[arg-type]
        overlap_k=args.overlap_k,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
    )
    print(f"[exp3] done; outputs in {run_dir}")


if __name__ == "__main__":
    main()
