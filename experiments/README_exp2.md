# Experiment 2 — Intrinsic Dimension of Residual-Stream Manifolds

Empirically estimate the intrinsic dimension (ID) of the residual-stream manifold at every depth of a causal LM, to test the *geometric consequence* of the homeomorphism claim: ID is preserved across layers (up to estimator noise + transversality caveats — see `experiments_design.md` §2).

The driver is `experiments/exp2_intrinsic_dim.py`. It mirrors Exp 1's structure (load model → resolve layers → sweep corpus → write JSONL + manifest to a timestamped run dir) but outputs ID estimates instead of Jacobian slogdets.

---

## ⚠️ Status: TwoNN and ESS are NOT implemented yet

**This is the preparation implementation.** The *capture* pipeline is complete and tested; the *estimator* bodies are placeholders that validate input shape and return `NaN`.

If you run exp2 today, the pipeline will:

- load the model, tokenize, run forward passes, capture residuals at every requested depth ✓
- build point clouds for every (depth, granularity, token_idx) ✓
- call the registered estimator for each cloud ✓
- write JSONL rows with `id_estimate: null` and a one-time `UserWarning` per estimator ✓

The pipeline is wired so that **once someone writes real bodies for `twonn` / `ess` / `participation_ratio`, the `id_estimate` column fills in automatically with no other changes to exp2 or the driver.** See the next section for exactly how.

---

## How to plug in a real estimator

### 1. The only file to edit

`src/homeomorphism/id_est.py`. The public API is frozen; the bodies are the placeholders:

```python
def twonn(points: torch.Tensor) -> float:
    """Facco et al. (2017) TwoNN estimator. PLACEHOLDER: returns NaN."""
    _validate(points)
    warnings.warn("twonn is a placeholder; returning NaN", stacklevel=2)
    return float("nan")                    # ← replace this

def ess(points: torch.Tensor) -> float:
    """Effective-sample-size style ID estimator. PLACEHOLDER: returns NaN."""
    _validate(points)
    warnings.warn("ess is a placeholder; returning NaN", stacklevel=2)
    return float("nan")                    # ← replace this

def participation_ratio(points: torch.Tensor) -> float:
    ...                                    # ← same
```

### 2. The input contract — what the dispatcher hands you

Every estimator receives a **2-D `torch.Tensor`** via the call in `estimate_id`:

| Property | Value |
|---|---|
| `points.dim()` | `2` |
| `points.shape` | `(N, m)` — N points in ambient dim m |
| `points.dtype` | `torch.float32` |
| `points.device` | `cpu` |
| requires_grad | `False` (detached upstream) |
| contiguity | contiguous after `reshape`/slice in `_cloud` |

`N` and `m` depend on granularity:

| Granularity | Cloud shape | Meaning |
|---|---|---|
| `full_stream` | `(N, T·d)` | each input contributes one flattened residual stream |
| `per_token` (for each `i ∈ [0, T)`) | `(N, d)` | cloud of representations at token position `i` |
| `last_token` | `(N, d)` | cloud at position `T−1` only |

`_validate(points)` is already called for you — it raises if shape is wrong or `N < 2`. You don't need to re-check.

### 3. The output contract — what to return

A **Python `float`** (scalar) — the estimated intrinsic dimension of the point cloud. The dispatcher wraps it in `float(...)` anyway so a 0-d torch tensor also works.

Numpy/scipy/sklearn are fine — just convert at the top of the body:

```python
def twonn(points: torch.Tensor) -> float:
    _validate(points)
    pts = points.numpy()
    # ... your TwoNN implementation on numpy ...
    return float(id_estimate)
```

### 4. If you raise

The driver catches any exception from `estimate_id` and records it in the row's `error` column with `id_estimate: null`. Good for "need N ≥ 20 for stable TwoNN" style guards — just `raise ValueError("TwoNN needs N>=20, got N=...")` and the pipeline continues.

### 5. That's it

There is nothing to edit in `exp2_intrinsic_dim.py`. The driver already:
- iterates `(depth, granularity, token_idx, estimator)` exhaustively;
- calls `estimate_id(cloud, estimator_name)` for each combination;
- writes the returned float (or NaN→null) to the JSONL's `id_estimate` column;
- records ambient dim, sample count, and depth hook path alongside each estimate.

Re-run the same CLI command you would have run before, and the results will be real numbers instead of nulls.

---

## What the experiment captures

### Depth indexing (Pre-LN convention)

For an `L`-block GPT-2, depths `0 .. 2L` index the residual stream at every sublayer boundary:

```
embed → [depth 0] → ln_1 → attn → residual_add → [depth 1] → ln_2 → ffn → residual_add → [depth 2] → ln_1(next block) → ...
```

- depth `2b` = block `b` attn-sublayer input (`ln_1` input)
- depth `2b+1` = block `b` ffn-sublayer input (`ln_2` input) = post-attn
- depth `2L` = `ln_f` input = post last ffn

Each `(block, kind)` sublayer you request via `--layers` contributes its "pre" depth and its "post" depth; the union across requested sublayers is deduplicated (because "post attn of block b" literally equals "pre ffn of block b" — same hook path). A single forward pass per input captures every requested depth.

### Three granularities

| Granularity | One cloud per | ID estimates per layer | Tests |
|---|---|---|---|
| `full_stream` | (depth, estimator) | 1 | Direct topological invariance of the full residual manifold $\mathcal{M}^n$ |
| `per_token` | (depth, token_idx, estimator) | T | Per-position manifold $M^n_i$ — preservation needs extra transversality assumption |
| `last_token` | (depth, estimator) | 1 | The "decoder latent" manifold, i.e. the geometry of next-token prediction |

---

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt2` | HF model id. Only `gpt2*` registered in `ARCH_SPECS`. |
| `--weights` | `trained` | `trained` / `random_gaussian` / `random_kaiming`. See `id_est.py` note — `random_kaiming` is degenerate for Jacobian measurements but fine for ID sweeps. |
| `--corpus` | `shakespeare` | Only shakespeare wired today. |
| `--n-samples` | `500` | Inputs from the corpus. Design §2 calls for 500–1000 for stable TwoNN estimates. |
| `--max-tokens` | `64` | Truncate each input to exactly this many tokens. Inputs that don't tokenize to exactly `max_tokens` are dropped (tracked in manifest). |
| `--layers` | `all` | `all`, single `block.kind` like `5.attn`, or comma-list `0.attn,5.ffn,11.attn`. Same syntax as Exp 1. |
| `--granularity` | `full_stream,per_token,last_token` | Comma-list from `{full_stream, per_token, last_token}`. |
| `--estimator` | `twonn` | Comma-list from `{twonn, ess, participation_ratio}`. |
| `--save-reps` | off | **Opt-in.** Writes `representations.npz` with per-depth `(N, T, d)` fp32 arrays. Off by default because the file balloons at realistic N/T/d (500 × 64 × 768 × 25 depths ≈ 2 GB). Enable only when re-estimating IDs from stored reps matters. |
| `--seed` | `0` | Seeds corpus sampling and random-weight init. |
| `--device` | `cpu` | Torch device. |
| `--output-root` | `results/exp2` | Parent dir; a `<timestamp>/` subdir is created per run. |

---

## Output layout

```
results/exp2/<run_id>/
├── config.json            # the exact CLI args passed
├── manifest.json          # run metadata: git_sha, depths_captured, input_token_ids, drop reasons, ...
├── results.jsonl          # one JSON row per (depth, granularity, [token_idx], estimator)
├── representations.npz    # ONLY if --save-reps; per-depth (N, T, d) fp32 arrays
└── plots/                 # empty; reserved for future plotter output
```

One row of `results.jsonl`:

```json
{
  "depth": 12,
  "hook_path": "transformer.h.6.ln_1",
  "granularity": "per_token",
  "token_idx": 3,
  "estimator": "twonn",
  "n_points": 500,
  "ambient_dim": 768,
  "id_estimate": null,       // will be a float once twonn is implemented
  "error": null
}
```

`id_estimate` is serialized as JSON `null` for NaN/Inf so strict parsers (jq, duckdb, pandas with `lines=True`) don't choke.

---

## Interpreting the numbers (once estimators are real)

- `id_estimate` approximately flat across `depth` at fixed `granularity` → supports homeomorphism preservation.
- Sharp deviations at specific depths → flag for investigation: estimator noise, transversality failure at that projection, or a genuine topology shift.
- `per_token` vs `full_stream` divergence → transversality-induced artefact (projection collapses a dimension) rather than theorem failure. See design doc §2 on the transversality caveat.
- For small `token_idx` (especially 0), expect the ID to be bounded by the number of distinct first-tokens in the corpus.

---

## Examples

Smoke test — tiny config, no stored reps:

```bash
uv run python -m experiments.exp2_intrinsic_dim \
    --n-samples 4 --max-tokens 8 --layers 0.attn \
    --granularity last_token --estimator twonn
```

Full sweep at the design's v1 scale:

```bash
uv run python -m experiments.exp2_intrinsic_dim \
    --n-samples 500 --max-tokens 64 --layers all \
    --granularity full_stream,per_token,last_token \
    --estimator twonn,ess
```

Same sweep but also save raw representations (for offline re-estimation once a new estimator is implemented):

```bash
uv run python -m experiments.exp2_intrinsic_dim \
    --n-samples 500 --max-tokens 64 --layers all \
    --granularity full_stream,per_token,last_token \
    --estimator twonn,ess \
    --save-reps
```

Random-init control (Mityagin-at-init baseline):

```bash
uv run python -m experiments.exp2_intrinsic_dim \
    --weights random_gaussian --seed 0 \
    --n-samples 500 --max-tokens 64 --layers all
```

---

## What's NOT yet wired (future work)

- **Estimator bodies**: TwoNN, ESS, participation_ratio all return NaN. See "How to plug in a real estimator" above.
- **`--from-reps <path>`** offline re-estimation from `representations.npz` without re-running forward passes. Not implemented — currently each run captures fresh. Ask if/when this becomes useful.
- **Variable-length sequences** (needed for Exp 2c′ "last-position pooled ID across lengths"). Right now inputs that don't tokenize to exactly `max_tokens` are dropped.
- **Point-cloud preprocessing flag** (e.g. `--preprocess {none, layernorm, center, l2_normalize}`). Design §2 notes the Euclidean-vs-cosine-vs-post-LN question; not exposed yet.
- **Pythia** (and any non-GPT-2 arch). Requires an `ARCH_SPECS` entry + a `_final_norm_path` case in exp2.
- **Tests for exp2**. Model on `tests/test_integration.py`'s exp1 pattern once estimators are real.
