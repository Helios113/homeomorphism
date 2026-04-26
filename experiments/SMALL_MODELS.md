# Small Custom GPT-2 Models for Baseline Experiments

This document describes the configurable small GPT-2 model variants that fit comfortably under 1GB memory, enabling baseline experiments on modest hardware.

## Available variants

All custom models share the standard GPT-2 vocabulary (50257 tokens) and architecture (pre-LN, causal attention). Only the depth and width are scaled.

Name pattern: `{size}-gpt2-{L}l-{D}d`

| Model name | Layers (L) | Hidden dim (D) | Heads (~ D/64) | FFN inner (~4D) | Parameters | FP32 Memory | FP16 Memory |
|---|---|---|---|---|---|---|---|
| `nano-gpt2-4l-128d` | 4 | 128 | 2 | 512 | ~2 M | ~8 MB | ~4 MB |
| `tiny-gpt2-4l-256d` | 4 | 256 | 4 | 1 024 | ~8 M | ~32 MB | ~16 MB |
| `tiny-gpt2-6l-256d` | 6 | 256 | 4 | 1 024 | ~12 M | ~48 MB | ~24 MB |
| `micro-gpt2-4l-384d` | 4 | 384 | 6 | 1 536 | ~18 M | ~72 MB | ~36 MB |
| `micro-gpt2-6l-384d` | 6 | 384 | 6 | 1 536 | ~26 M | ~104 MB | ~52 MB |
| `tiny-gpt2-8l-256d` | 8 | 256 | 4 | 1 024 | ~16 M | ~64 MB | ~32 MB |

All estimates assume fp32 (4 bytes/param). The largest variant above still leaves plenty of headroom under 1 GB. Even in fp32, all models comfortably fit on a modern GPU with <1 GB memory.

## Usage

Custom models ONLY support random-weight initialisation (`weights="random_gaussian"` or `"random_kaiming"`). They cannot load `weights="trained"` because no pretrained checkpoints exist for these architectures.

### Example: Run baseline experiments on a tiny model

```bash
# Quick topological baseline on the smallest model
python experiments/baseline_exp_1_topological.py \
  --model tiny-gpt2-4l-256d \
  --weights random_gaussian \
  --n-samples 8 \
  --max-tokens 32 \
  --layers 0.attn,1.attn \
  --seed 0 \
  --device cuda
```

```bash
# Section-2 unified baselines on a micro model
python experiments/exp3_section2_baselines.py \
  --model micro-gpt2-6l-384d \
  --weights random_gaussian \
  --baselines trained,topological_initialisation,maximum_entropy_injection \
  --corpus shakespeare \
  --n-samples 16 \
  --max-tokens 32 \
  --layers all \
  --granularity full_stream,last_token \
  --estimator twonn,ess \
  --overlap-k 10 \
  --seed 0 \
  --device cuda
```

The `run_baseline_configs_gpu.py` orchestrator has been updated to include small-model presets via the `--model` flag:

```bash
# Quick suite on a 4-layer, 256-d model
python experiments/run_baseline_configs_gpu.py --model tiny-gpt2-4l-256d --quick
```

## Theoretical considerations

Smaller models have fewer degrees of freedom, which may affect:
- **Condition numbers**: Fewer parameters can lead to better-conditioned Jacobians.
- **Intrinsic dimension estimates**: The ground-truth representational dimension is bounded by `n_embd`. With D=256, estimated ID cannot exceed 256, which is still plenty for probing feature geometry.
- **Sensitivity to baselines**: Random-initialised models serve as strong null baselines because they lack learned linguistic structure.

For a rigorous comparison across model scales, we recommend running the same experimental protocol on at least two sizes (e.g., `tiny-gpt2-4l-256d` and `micro-gpt2-6l-384d`) to check that trends persist as capacity increases.

## Implementation notes

Custom models are generated programmatically from `GPT2Config` via `models.load_model()`. No pretrained weights are downloaded; all parameters are randomly initialised using the seed provided via `--seed`. The same tokenizer (GPT-2 byte-pair encoding) is reused, so vocabulary statistics are held constant across scales.

All existing baseline scripts (`exp1_per_token_J.py`, `exp2_intrinsic_dim.py`, `exp3_section2_baselines.py`, and the individual baseline experiments) are model-agnostic and work with any HuggingFace causal LM name or custom pattern out of the box.
