# Legacy/Archived Experiments

This folder contains unused or superseded experiment scripts and templates.

## Contents

### Legacy Baseline Scripts (Standalone)
- `baseline_exp_1_topological.py` — Jacobian measurement for topological init
- `baseline_exp_2_maximum_entropy.py` — ID measurement for max-entropy injection
- `baseline_exp_3_permutation.py` — Jacobian + ID for permuted tokens
- `baseline_exp_4_uniform.py` — Jacobian + ID for uniform random tokens

**Status**: ✗ Not used in modern pipeline. Use `exp3_section2_baselines.py` instead.

### Legacy Utilities
- `exp2_intrinsic_dim.py` — Old ID estimation template (only imported by legacy baseline_exp_*.py)
- `exp3_llama_hyperplane.py` — Old synthetic manifold script

**Status**: ✗ Replaced by unified `exp3_section2_baselines.py` and `exp3_synthetic_manifolds.py`

### Development Notes
- `exp2_dev_notes.md` — Old development iteration notes

## Modern Alternatives

**For baseline experiments**: Use `exp3_section2_baselines.py` (unified framework)
- Runs all baselines: trained, topological_init, max_entropy, permuted, uniform
- Shared HDF5 latent storage
- Deterministic seeds & consistent bookkeeping
- Orchestrated via `run_baseline_configs_gpu.py` or `run_comprehensive_experiments.py`

**For synthetic manifolds**: Use `exp3_synthetic_manifolds.py`
- Supports toy + LLaMA models
- Manifold types: hyperplane, sphere, torus, swiss_roll, white_noise
- Ground-truth intrinsic dimension tracking
- Orchestrated via `run_synthetic_experiments.py` or `run_comprehensive_experiments.py`

**For ID estimation**: Use `exp4_id_estimation.py`
- Generic postprocessor for any HDF5 latent file
- Works on both baseline and synthetic output
