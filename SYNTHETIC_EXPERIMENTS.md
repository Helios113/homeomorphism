# Synthetic Manifold Experiments

This document describes the synthetic manifold experimental framework for
studying topological properties of latent representations in transformers.

## Overview

The framework provides:

1. **Synthetic manifold generators** (sphere, torus, hyperplane, Swiss roll, white noise)
2. **Two model families**: toy topological transformer & LLaMA-style decoder
3. **Unified latent storage** via HDF5 with self-describing metadata
4. **Automated experiment orchestration** for multi-config sweeps
5. **Integration with ID estimation** (Experiment 4)

All experiments follow a consistent depth indexing convention for hidden states
and produce HDF5 files consumable by downstream analysis tools.

---

## Quick Start

### Quick smoke test

```bash
uv run python -m experiments.run_synthetic_experiments --quick
```

Runs 1 manifold (hyperplane) × 1 toy model (2 layers, 32 dimensions) with 4 samples.

### Full sweep

```bash
uv run python -m experiments.run_synthetic_experiments \
    --manifolds hyperplane,sphere,torus,swiss_roll,white_noise \
    --model-configs toy-2l-32d,toy-4l-64d,llama-2l-20d-10m \
    --n-samples 256 \
    --seed 0 \
    --device cuda \
    --run-exp4
```

Results are saved under `results/synthetic/{manifold}/{model_config}/{seed}/`.

### Single experiment

```bash
uv run python -m experiments.exp3_synthetic_manifolds \
    --manifold-type sphere \
    --model-type toy \
    --d-model 32 \
    --n-layers 4 \
    --seq-len 8 \
    --n-samples 128 \
    --save results/synthetic/sphere/toy-4l-32d/0/latents.h5
```

---

## Manifold Types

| Manifold   | Intrinsic dimension                          | Required flags          |
|------------|----------------------------------------------|-------------------------|
| hyperplane | M (user-specified)                           | `--manifold-dim M`      |
| sphere     | N − 1 (ambient − 1)                          | none                    |
| torus      | 2                                            | none                    |
| swiss_roll | 2                                            | none                    |
| white_noise| N (full ambient)                             | none                    |

All samplers are deterministic given `--seed`. Output points are centered and
standardized to avoid numerical scaling artifacts.

---

## Model Types

### Toy (`toy-{L}l-{D}d`)

Custom minimal transformer designed for exact Jacobian analysis.

```
toy-2l-32d  →  n_layers=2,  d_model=32
toy-4l-64d  →  n_layers=4,  d_model=64
```

Parameters auto-derived:
- `n_heads = max(4, d_model // 8)`
- `d_ff = 4 × d_model`

Uses continuous inputs directly (no token embeddings). Causal masking.

### LLaMA (`llama-{L}l-{N}d-{M}m`)

LLaMA-2-style decoder with RMSNorm and RoPE. Requires explicit `manifold_dim M`.

```
llama-2l-20d-10m  →  n_layers=2, d_model=20, manifold_dim=10
```

Defaults: `n_heads=4`, `d_ff=4×d_model`.

---

## Output Directory Structure

```
results/synthetic/
  hyperplane/
    toy-2l-32d/
      0/
        latents.h5                  # main latent storage
        exp4_id_estimates.jsonl     # if --run-exp4
        manifest.json               # run metadata
    llama-2l-20d-10m/
      0/
        ...
  sphere/
    ...
```

Each run is identified by `(manifold, model_config, seed)`. Re-running with the
same seed appends to the existing `latents.h5` file.

---

## HDF5 File Format

`latents.h5` contains:

- **Attributes** (JSON-serialized config):
  - `d_model`, `seq_len`, `n_layers`, `n_depths`
  - `manifold_type`, `manifold_dim`

- **Datasets** `/depth_{00..NN}`:
  - Shape `(N_samples, T, d_model)`, dtype `float32`
  - Resizable along axis 0 (samples)
  - Chunked (64 samples) and LZF-compressed

### Depth indexing convention

| Depth index | Meaning                              |
|-------------|--------------------------------------|
| 0           | Raw input (before any block)         |
| 2b + 1      | Post-attention of block `b`          |
| 2b + 2      | Post-FFN of block `b`                |
| 2L + 1      | After final RMSNorm / final output   |

Total depths: `n_depths = 2 * n_layers + 2`.

Both toy and LLaMA models follow this convention.

### Reading the HDF5 file

```python
import h5py, json

with h5py.File("latents.h5", "r") as f:
    # file metadata
    cfg = json.loads(f.attrs["config"])
    print(f"d_model={cfg['d_model']}, manifold={cfg['manifold_type']}")

    # point cloud: all samples at depth 2, token 3
    cloud = f["depth_02"][:, 3, :]   # shape (N, d_model)

    # full residual stream at input depth
    layer0 = f["depth_00"][:]         # shape (N, T, d_model)
```

---

## Running ID Estimation (Experiment 4)

After generating latents, estimate intrinsic dimension:

```bash
uv run python -m experiments.exp4_id_estimation \
    --latents results/synthetic/hyperplane/toy-2l-32d/0/latents.h5 \
    --token 0 \
    --depth 0 1 2 3 \
    --estimator twonn ess \
    --save results/synthetic/hyperplane/toy-2l-32d/0/exp4_id_estimates.jsonl
```

To run this automatically after each exp3 run, use `--run-exp4` with the
orchestrator script.

---

## Implementation Notes

### Code Organization

- `src/homeomorphism/toy_transformer.py`: toy model + manifold samplers
- `src/homeomorphism/latents.py`: shared `LatentStore` and `LatentConfig`
- `experiments/exp3_llama_hyperplane.py`: legacy script (unchanged behavior)
- `experiments/exp3_synthetic_manifolds.py`: unified experiment entry point
- `experiments/run_synthetic_experiments.py`: multi-config orchestrator
- `experiments/exp4_id_estimation.py`: ID estimation on stored latents

### Design Decisions

1. **LatentStore extraction** — The original inline class from `exp3_llama_hyperplane.py`
   was extracted to a shared module to ensure HDF5 format consistency.

2. **Model abstraction** — Both model types implement `forward_with_states()`
   returning the same depth-indexed list of tensors. The toy model additionally
   provides `batch_forward_with_states()` for efficient batch processing.

3. **Sampler interface** — Samplers expose `.sample(B, T)` returning a
   `(B×T, d_model)` tensor which is reshaped to `(B, T, d_model)` before the
   forward pass. This keeps sampler implementations simple and device-agnostic.

4. **Backward compatibility** — `exp3_llama_hyperplane.py` was only refactored
   internally to use the shared `LatentStore`; its CLI, behavior, and output
   format remain unchanged.

---

## Recommended Configurations

For rapid prototyping under 1 GB memory:

| Manifold       | Toy config  | Samples | Expected ID |
|----------------|-------------|---------|-------------|
| hyperplane     | `toy-2l-32d`| 128     | M (user set)|
| sphere         | `toy-2l-32d`| 128     | N − 1       |
| torus          | `toy-2l-32d`| 128     | 2           |
| swiss_roll     | `toy-4l-64d`| 256     | 2           |

For more rigorous runs with LLaMA-style models, use `llama-2l-20d-10m` or
`llama-4l-32d-16m` with 256–512 samples.

---

## Troubleshooting

### HDF5 compatibility errors

If you see errors about `d_model` or `seq_len` mismatches, you are trying to
append to a file created with a different model architecture. Delete the old
file or use a new output path.

### CUDA out-of-memory

Reduce `--batch-size` or use a smaller model (e.g., `toy-2l-32d` instead of
`toy-6l-128d`). The toy models are especially memory-efficient.

### Exp4 fails to read file

Ensure `exp3_synthetic_manifolds.py` completed successfully and produced a
`latents.h5` file with the expected depth datasets (`depth_00`, `depth_01`, …).
Use `h5ls` or `h5py` to inspect.
