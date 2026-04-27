# Baseline Revision Changelog

**Date:** Session ending Apr 2025  
**Status:** ✅ All tests passing (79 passed, 1 skipped)

## Summary

This revision enforces unified use of shared codebase APIs across baseline experiments and notebooks, adds robust logical tests, and introduces synthetic manifold infrastructure for systematic ID estimation studies.

---

## Core Baseline Experiment Fixes

### `experiments/baseline_exp_2_maximum_entropy.py`
- **Single prepared-input flow**: Each input sample now builds `PreparedInput` once and reuses it for all downstream metrics (Jacobian, intervention capture).
- **Consistent token bookkeeping**: Token IDs now sourced from `prepared.token_ids` rather than reconstructed per-stage.
- **Schema documentation**: Corrected ID row schema; removed stale `input_id` and `elapsed_sec` fields that were incorrectly documented.
- **Forward kwargs**: Capture helper now accepts `forward_kwargs` directly from `PreparedInput`.

### `experiments/baseline_exp_3_permutation.py` & `baseline_exp_4_uniform.py`
- **Prepared input reuse**: Same single-pass prepared-input flow as exp2; eliminates row drift from repeated input construction.
- **Explicit depth parser** (`_parse_depths_spec`): 
  - Validates depth specs: handles `"all"`, `"1,2,3"` ranges, deduplicates, sorts, and checks bounds.
  - Fallback depth list `[0, 1, 3, 4]` mirrors shared helper semantics.
- **Forward kwargs**: Use prepared-input's `forward_kwargs` directly.
- **Schema docs**: Fixed ID row descriptions to match actual returned fields.

### `experiments/exp4_id_estimation.py`
- **Unified estimator API**: Replaced local `skdim`-based wrapper with shared `homeomorphism.id_est.estimate_id()`.
- **Estimator choices**: Standardized to `("twonn", "ess", "participation_ratio")`.
- **Torch integration**: Added explicit `torch` import and type-safe estimator handling.

---

## Notebook Updates

### `notebooks/baseline_notebook/residual_relu_analytic_jacobian.ipynb`
- **Jacobian via shared API**: Replaced ad-hoc autograd computation with `jacobian.build_jacobian(...)`.
- **Analytic vs numerical sanity**: Computes Jacobian, extracts diagonal determinant, compares to analytical formula.
- **Status**: Executed successfully; output matches expected analytic vs computed log|det| agreement.

### `notebooks/baseline_notebook/deep_linear_condition_number.ipynb`
- **Layer-wise Jacobian composition**: Now uses `jacobian.build_jacobian(...)` per layer and composes via product.
- **Status**: Executed successfully; demonstrates condition number growth across depths.

### `notebooks/baseline_notebook/known_manifold_id.ipynb`
- **ID estimators**: Uses shared `homeomorphism.id_est.estimate_id` for TwoNN/ESS on known-manifold synthetic data.
- **Status**: Executed successfully; consistent with shared estimator implementation.

---

## New Logical Baseline Tests

**File**: `tests/baseline_test_experiments.py`

Four targeted tests for baseline drivers:

1. **`test_baseline_capture_uses_prepared_forward_kwargs`**  
   Validates that capture helper correctly uses prepared input's `forward_kwargs` dict.

2. **`test_baseline_measurement_uses_prepared_token_ids`**  
   Confirms measurement row includes token IDs from `prepared.token_ids` and matches row format.

3. **`test_baseline_depth_spec_parser` (parametrized over exp3 & exp4)**  
   - `"all"` spec yields full depth range  
   - Numeric list (deduplicated, sorted) works correctly  
   - Fallback (`""`) yields expected baseline depths  
   - Out-of-range and malformed specs raise `ValueError`

All baseline tests pass; integrated into pytest discovery via `python_files = ["test_*.py", "baseline_test_*.py"]` in `pyproject.toml`.

---

## Integration Test Fix

**File**: `tests/test_integration.py`

- **Updated schema expectations**: Added four newly added alert/metadata fields to `expected_keys`:
  - `kappa_alert_threshold`
  - `n_kappa_alert`
  - `kappa_alert_fraction`
  - `n_invalid_condition_number`
- **Reason**: Baseline `measure_sublayer` now returns these stability diagnostics alongside core Jacobian metrics.

---

## New Synthetic Manifold Infrastructure

Four new experiment files extend baseline framework to systematic manifold studies:

### `src/homeomorphism/latents.py`
- **Reusable HDF5 storage** (`LatentStore`) for collecting hidden states across model depths.
- **Self-describing format**: Stores `d_model`, `seq_len`, `n_layers`, `n_depths`, `manifold_type`, `manifold_dim` as JSON config.
- **Append-friendly**: Multiple runs with same config extend existing datasets; incompatible configs are detected and rejected.
- **Depth indexing convention**: `depth_00` = input, `depth_{2b+1}` = post-attn block `b`, `depth_{2b+2}` = post-FFN, `depth_{2L+1}` = post-final-norm.

### `src/homeomorphism/toy_transformer.py` (extended)
- **New sampling functions**: `sample_sphere()`, `sample_hyperplane()`, `sample_swiss_roll()`, `sample_white_noise()`.
- **HyperplaneSampler class**: Stateful sampler for M-dimensional linear subspace of R^N.
- **Batch-friendly forward**: Added `forward_with_states()` and `batch_forward_with_states()` returning uniform depth-indexed lists.
- **Depth counting**: New `n_depths` property (= `2 * n_layers + 2`).

### `experiments/exp3_synthetic_manifolds.py`
- **Unified entry point** for manifold experiments: supports toy and LLaMA-style models.
- **Model builders**: Configurable toy (2l-32d, 4l-64d, etc.) and LLaMA (2l-20d-10m, etc.).
- **Manifold types**: hyperplane, sphere, torus, swiss_roll, white_noise (each with ground-truth intrinsic dimension).
- **Latent collection**: Collects states to self-describing HDF5 with config.

### `experiments/run_synthetic_experiments.py`
- **Multi-config orchestrator**: Sweeps across manifold types × model configs × seeds.
- **Subprocess management**: Calls exp3, collects results, optionally runs exp4.
- **Directory structure**: `results/synthetic/{manifold}/{model_config}/{seed}/latents.h5`.
- **Smoke test mode**: `--quick` mode for rapid validation.

### `experiments/visualize_synthetic_results.py`
- **Result visualization**: Plots depth trajectories (ID vs depth per manifold) and summary bar charts (input-depth estimates across manifolds).
- **Ground-truth overlay**: Each plot includes theoretical intrinsic dimension line.

### `SYNTHETIC_EXPERIMENTS.md`
- **Comprehensive guide**: Quick start, manifold definitions, model types, HDF5 format, depth indexing, exp4 integration, recommended configs, troubleshooting.
- **User-facing examples**: CLI recipes for quick smoke tests, full sweeps, single experiments, and visualization.

---

## Configuration Updates

### `pyproject.toml`
```toml
python_files = ["test_*.py", "baseline_test_*.py"]
```
Enables pytest discovery of new baseline logical tests with prefix `baseline_test_`.

---

## Validation Summary

✅ **Full test suite**: 79 passed, 1 skipped, 5 warnings (no failures)  
✅ **Baseline tests**: All 4 new tests pass (prepared input, token IDs, depth parser)  
✅ **Integration tests**: All 7 tests pass (schema expectations updated)  
✅ **Notebook sanity**: 3 key Jacobian/ID notebooks executed successfully  

---

## Key Architectural Decisions

1. **Single prepared-input per sample**: Eliminates row drift and ensures consistent token/depth relationships across all metrics.
2. **Shared API enforcement**: All Jacobian, ID estimation, and intervention operations now use unified codebase functions.
3. **Self-describing HDF5**: Manifold experiments store full config in file attributes for reproducibility and safety.
4. **Modular manifold framework**: Samplers, model builders, and orchestration are composable for systematic exploration.

---

## Files Modified

- `experiments/baseline_exp_2_maximum_entropy.py`
- `experiments/baseline_exp_3_permutation.py`
- `experiments/baseline_exp_4_uniform.py`
- `experiments/exp4_id_estimation.py`
- `notebooks/baseline_notebook/residual_relu_analytic_jacobian.ipynb`
- `notebooks/baseline_notebook/deep_linear_condition_number.ipynb`
- `notebooks/baseline_notebook/known_manifold_id.ipynb`
- `pyproject.toml`
- `tests/test_integration.py`

## Files Created

- `tests/baseline_test_experiments.py` (new logical baseline tests)
- `src/homeomorphism/latents.py` (shared HDF5 storage utilities)
- `experiments/exp3_synthetic_manifolds.py` (unified synthetic manifold driver)
- `experiments/run_synthetic_experiments.py` (multi-config orchestrator)
- `experiments/visualize_synthetic_results.py` (result visualization)
- `SYNTHETIC_EXPERIMENTS.md` (synthetic framework guide)
- `CHANGELOG_BASELINE_REVISION.md` (this document)

---

## Next Steps (Optional)

1. Run comprehensive baseline experiment suites (`baseline_exp_2`, `3`, `4`) on GPT-2 to validate prepared-input fixes.
2. Execute synthetic manifold suite (`exp3_synthetic_manifolds.py`) on toy model for quick validation.
3. Run full `exp4_id_estimation` pipeline on generated latents.
4. Extend experiments to LLaMA-style models if desired.
