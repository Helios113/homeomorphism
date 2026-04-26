# Section 2 Baselines README

This repository now treats the Section 2 baseline experiments as a two-stage pipeline with a shared latent store.

## What the pipeline does

The Section 2 experiments separate **online capture** from **offline estimation**:

1. **Phase 1: latent capture**
   - Load a baseline-specific model state.
   - Run a forward pass on each input.
   - Capture residual-stream tensors at the requested depths.
   - Persist the captured `(N, T, d)` tensors into `latents.h5`.
   - Log Jacobian observables immediately, while the autograd graph is still active.

2. **Phase 2: topological estimation**
   - Reopen `latents.h5`.
   - Slice the stored latent cubes by granularity.
   - Run offline intrinsic-dimension and neighborhood-overlap estimation.
   - Append rows to JSONL plotting files.

The Jacobian calculations stay online because they require the PyTorch autograd graph. ID estimation and neighborhood overlap stay offline because they only need static latent tensors.

## Baseline strategies

The engine supports the Section 2 null-model families used throughout the codebase:

- `trained`
- `topological_initialisation`
- `maximum_entropy_injection`
- `syntactic_disintegration`
- `semantic_scrambling`

### Theoretical intent

- **Trained**: reference model under the pretrained weights.
- **Topological initialisation**: random Gaussian reinitialisation with norm affine reset, used as a structural null control.
- **Maximum entropy injection**: replaces token embeddings with isotropic Gaussian inputs, scaled so the covariance is `(1/d)I`.
- **Syntactic disintegration**: permutes valid token IDs while preserving padding positions.
- **Semantic scrambling**: replaces valid token IDs with uniform random token IDs, again preserving padding positions.

## Latent storage layout

The master Section 2 script writes a single `latents.h5` file per run.

```text
latents.h5
├── /trained
├── /topological_init
├── /max_entropy
├── /permuted
└── /uniform_tokens
```

Each baseline group contains resizable datasets named `depth_XX`.
The current implementation stores a raw-input slot at `depth_00` and then stores the captured transformer depths with a one-step offset, so the HDF5 file can hold the input stream plus the measured residual-stream states in one place.

Example per-group fields:

- `depth_00`: residual stream entering block 0
- `depth_01`: first captured transformer depth
- `depth_02`: next captured depth
- and so on

The group attrs also record the model name, weight mode, depth mapping, and hook-path metadata used to reconstruct the capture.

## Granularity-aware estimation

Offline estimation follows the same slicing logic as the intrinsic-dimension tooling:

- `per_token`: slice `latents[:, t, :]` for each token position `t`
- `last_token`: slice `latents[:, -1, :]`
- `full_stream`: flatten sequence and feature dimensions with `latents.reshape(N, T * d)`

The offline pass uses the stored HDF5 tensors, not the live model state.

## Logging and run artifacts

The master runner follows the repo’s existing experiment conventions:

- `config.json`: exact CLI arguments
- `manifest.json`: git SHA, timings, counts, and baseline summaries
- `jacobian.jsonl`: online Jacobian rows
- `id.jsonl`: offline intrinsic-dimension rows
- `overlap.jsonl`: offline neighborhood-overlap rows
- `latents.h5`: persisted latent tensors
- `plots/`: reserved for downstream visualization

Rows that may produce invalid numeric values normalize them to JSON `null` so strict parsers remain happy.

## Code path reference

- [Experiment 3 master runner](experiments/exp3_section2_baselines.py)
- [Experiment 1 Jacobian template](experiments/exp1_per_token_J.py)
- [Experiment 2 intrinsic-dimension template](experiments/exp2_intrinsic_dim.py)
- [Offline HDF5 estimator reference](experiments/exp4_id_estimation.py)
- [Baseline intervention helpers](src/homeomorphism/interventions.py)

## Typical run

```bash
.venv/bin/python experiments/exp3_section2_baselines.py \
  --model gpt2 \
  --weights trained \
  --baselines trained,topological_initialisation,maximum_entropy_injection \
  --corpus shakespeare \
  --n-samples 8 \
  --max-tokens 32 \
  --layers all \
  --granularity full_stream,last_token \
  --estimator twonn,ess \
  --overlap-k 10 \
  --seed 0 \
  --device cpu
```

The run directory will contain `latents.h5` plus the JSONL logs needed for downstream analysis and plotting.
