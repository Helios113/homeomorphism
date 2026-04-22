# Exp 2 — dev notes (temporary)

This file is a pointer for whoever picks up the estimator implementation.
Delete once TwoNN / ESS / participation_ratio bodies are real and this infra
stops being placeholder-shaped. User-facing exp2 docs live in the main `README.md`.

---

## Plugging in a real ID estimator

### The only file to edit

`src/homeomorphism/id_est.py`. The public API is frozen; the bodies are the placeholders:

```python
def twonn(points: torch.Tensor) -> float:
    _validate(points)
    warnings.warn("twonn is a placeholder; returning NaN", stacklevel=2)
    return float("nan")                    # ← replace this

def ess(points: torch.Tensor) -> float:
    _validate(points)
    warnings.warn("ess is a placeholder; returning NaN", stacklevel=2)
    return float("nan")                    # ← replace this

def participation_ratio(points: torch.Tensor) -> float:
    ...                                    # ← same
```

### Input contract — what the dispatcher hands you

| Property | Value |
|---|---|
| `points.dim()` | `2` |
| `points.shape` | `(N, m)` — N points in ambient dim m |
| `points.dtype` | `torch.float32` |
| `points.device` | `cpu` |
| requires_grad | `False` (detached upstream) |
| contiguity | contiguous after `reshape`/slice in `_cloud` |

N and m per granularity:

| Granularity | Cloud shape |
|---|---|
| `full_stream` | `(N, T·d)` |
| `per_token` (one call per token_idx) | `(N, d)` |
| `last_token` | `(N, d)` |

`_validate(points)` is already called for you (raises on wrong shape or `N < 2`); don't re-check.

### Output contract

A Python `float`. Numpy/scipy/sklearn inside is fine — convert at the top:

```python
def twonn(points: torch.Tensor) -> float:
    _validate(points)
    pts = points.numpy()
    # ... your TwoNN implementation on numpy ...
    return float(id_estimate)
```

### Raising is safe

The driver catches any exception from `estimate_id` and records it as `error: "<Type>: <msg>"` with `id_estimate: null`. Good for guards like `raise ValueError("TwoNN needs N>=20, got N=...")`.

### Nothing else to edit

`experiments/exp2_intrinsic_dim.py` already iterates every `(depth, granularity, token_idx, estimator)` combination and writes the returned float to the JSONL. Once the bodies return real numbers, a re-run of the same CLI command produces real `id_estimate` values with no other changes.

---

## Gaps still open (not blocking the plug-in)

- **`--from-reps <path>`**: offline re-estimation from `representations.npz` without re-running forward passes. Not implemented — each run currently re-captures. Add if the re-analysis workflow matters.
- **Variable-length sequences**: needed for Exp 2c′ (last-position pooled ID across lengths). Right now inputs that don't tokenize to exactly `max_tokens` are dropped.
- **Point-cloud preprocessing flag** (e.g. `--preprocess {none, layernorm, center, l2_normalize}`). Design §2 raises the Euclidean-vs-cosine-vs-post-LN question; not exposed yet.
- **Pythia / non-GPT-2 archs**: need an `ARCH_SPECS` entry + a `_final_norm_path` case in exp2.
- **Tests for exp2**: model on `tests/test_integration.py`'s exp1 pattern once estimators are real.
