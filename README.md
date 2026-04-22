# Homeomorphic Transformers

Experimental codebase for empirically testing the claim that each transformer residual sublayer acts as a local homeomorphism on its residual stream — concretely, that the per-token diagonal blocks of its Jacobian are non-singular almost surely (Part A of the homeomorphism claim).

Currently scoped to causal LMs (GPT-2 family is the only registered architecture). The package provides:

- A `Model` + `sublayer` abstraction producing faithful $$\phi : \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$$ closures for each residual sublayer (attn or FFN).
- An activation-capture helper that hooks the input of any sublayer and returns its residual stream at that point.
- A Jacobian toolkit that builds the per-token block grid $$[J]_{i,k}$$ of a sublayer via `torch.func.jacrev`, with evaluators for $$\log|\det|$$, singular-value spectra, and the full-Jacobian $$\log|\det|$$ via the block-triangular factorization.
- Experiment 1 (`experiments/exp1_per_token_J.py`): runs the measurements across a corpus and selected sublayers, writes per-run results + manifest to disk.

See `experiments_design.md` for the full experimental plan, proof dependencies, and meta-checklist mapping tests to functional coverage.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

This installs `torch`, `transformers`, and the dev dependency `pytest`.

---

## Project structure

```
homeomorphism/
├── src/homeomorphism/
│   ├── __init__.py
│   ├── models.py       # Model, load_model, sublayer, tokenize, predict, generate
│   ├── hooks.py        # capture_activation
│   ├── jacobian.py     # BlockJacobian, build_jacobian, sublayer_slogdet
│   ├── data.py         # load_texts (shakespeare)
│   └── id_est.py       # stub for Exp 2 (TwoNN, PR, ESS)
├── experiments/
│   └── exp1_per_token_J.py
├── tests/
│   ├── conftest.py           # toy sublayers + autograd oracle
│   ├── test_data.py
│   ├── test_hooks.py
│   ├── test_integration.py   # end-to-end on real GPT-2
│   ├── test_jacobian.py
│   └── test_models.py
├── results/               # (gitignored) experiment outputs
├── experiments_design.md  # design + claim + test meta-checklist
└── pyproject.toml
```

---

## Core API — a guided tour

Each module is one file, one concern. Import and use directly.

### `homeomorphism.models`

Load a HF causal LM, query structure, get sublayer handles, run forward passes.

```python
from homeomorphism import models

m = models.load_model("gpt2")              # trained weights (default)
m = models.load_model("gpt2", weights="random_gaussian", seed=42)  # Mityagin-at-init control

models.n_blocks(m)        # 12 for GPT-2 small
models.n_sublayers(m)     # 24 (2 per block: attn + ffn)
models.hidden_size(m)     # 768

# Get a sublayer: block 0, attention.
s = models.sublayer(m, block_idx=0, kind="attn")
s.hook_path   # e.g. "transformer.h.0.ln_1" — hook here to capture input
s.phi         # callable (T, d) -> (T, d), implements h -> h + g(h)

# Inference helpers (kept for sanity checks; not used by the experiments).
models.predict_next_token(m, "The capital of France is", top_k=5)
models.generate(m, "Once upon a time", max_new_tokens=20)
```

The phi closures are architecture-specific and produced by `_PHI_BUILDERS`. Currently `gpt2` is registered; add `llama` / `pythia` by extending `ARCH_SPECS` and implementing their phi builders.

### `homeomorphism.hooks`

One function — grab the residual stream at a sublayer's input:

```python
from homeomorphism import hooks

h = hooks.capture_activation(m, s.hook_path, "The quick brown fox", max_tokens=32)
# h has shape (T, d), detached, on the model's device
```

### `homeomorphism.jacobian`

Build the $$T \times T$$ grid of $$d \times d$$ sub-Jacobian blocks of a sublayer at a given residual stream. Access blocks by `(output_token, input_token)`; evaluate per-block or whole-matrix quantities via methods.

```python
from homeomorphism import jacobian
import torch

bj = jacobian.build_jacobian(s.phi, h.to(torch.float32))        # scope="causal" by default
bj[(2, 2)]                     # diagonal block J^(2) in R^{d x d}
bj[(3, 1)]                     # off-diagonal attention coupling from token 1 to token 3
bj.slogdet(2, 2)               # (sign, log|det|) of block (2, 2)
bj.svdvals(2, 2)               # singular values
bj.condition_number(2, 2)      # sigma_max / sigma_min

# Full-matrix log|det| via the block-triangular factorization (paper eq. 4):
sign, log_abs_det = bj.full_slogdet()

# Eager evaluation in one call — returns (BlockJacobian, eval_result).
bj, (sign, log_abs_det) = jacobian.build_jacobian(
    s.phi, h, scope="diagonal", evaluate="full_slogdet"
)

# Or the convenience wrapper for the common case:
sign, log_abs_det = jacobian.sublayer_slogdet(s.phi, h)
```

Scope options:
- `"diagonal"` — only $$(i, i)$$ blocks. Cheapest. Enough for $$\log|\det|$$.
- `"causal"` (default) — every $$(i, k)$$ with $$k \leq i$$. All nonzero blocks in a causal model.
- `"full"` — every $$(i, k)$$ including the causally-zero upper blocks (for sanity checks).

Evaluate options for the `evaluate=` flag:
- `None` — return just the `BlockJacobian`.
- `"full_slogdet"` — also return $$(sign, \log|\det|)$$ of the full Jacobian.
- `"per_diagonal_slogdet"` — also return $$\{i : (sign, \log|\det|)\}$$ per diagonal.
- `"per_block_slogdet"` — also return $$\{(i, k) : (sign, \log|\det|)\}$$ per computed block.

### `homeomorphism.data`

```python
from homeomorphism.data import load_texts

texts = load_texts("shakespeare", n_samples=8, chunk_chars=400, seed=0)
```

First call downloads Karpathy's tiny-shakespeare to `~/.cache/homeomorphism_data/` and samples text chunks at random offsets.

---

## Running Experiment 1

`experiments/exp1_per_token_J.py` loops `(input × sublayer)`, measures the full sublayer $$\log|\det|$$ plus per-token breakdown and singular-value spectrum, and writes JSONL.

### Quick smoke run

```bash
uv run python -m experiments.exp1_per_token_J \
    --model gpt2 --weights trained \
    --corpus shakespeare --n-samples 4 --max-tokens 32 \
    --layers 0.attn
```

### CLI flags — what each controls

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt2` | HF model id. Only `gpt2*` families work out of the box; add new architectures via `ARCH_SPECS` in `models.py`. |
| `--weights` | `trained` | Weight-loading mode. One of: `trained` (pretrained), `random_gaussian` (re-init every parameter $$\sim N(0, 0.02^2)$$), `random_kaiming` (Kaiming uniform). Random modes are the Mityagin-at-init controls. |
| `--corpus` | `shakespeare` | Text corpus. Only `shakespeare` is currently wired. |
| `--n-samples` | `4` | How many input texts to draw from the corpus. |
| `--max-tokens` | `32` | Truncate each input to at most this many tokens. Smaller is faster; larger gives more token positions per input. |
| `--layers` | `0.attn` | Which sublayers to measure. Three forms: `all` (every block × kind), a single `block.kind` like `5.attn`, or a comma list like `0.attn,5.ffn,11.attn`. |
| `--seed` | `0` | Seed for corpus sampling AND for random-weight modes. |
| `--device` | `cpu` | Torch device. |
| `--output-root` | `results/exp1` | Parent directory; a `<timestamp>/` subdir is created per run. |

### Output layout

Each run creates a timestamped subdirectory:

```
results/exp1/<run_id>/
├── config.json      # the exact CLI args passed
├── manifest.json    # run metadata: git_sha, timestamps, n_rows, n_errors, sublayers_resolved, ...
├── results.jsonl    # one JSON row per (input, sublayer) measurement
└── plots/           # empty; reserved for future plotter output
```

One row of `results.jsonl` (abbreviated):

```json
{
  "input_id": 0,
  "input_preview": "alters.\n\nPERDITA:\nOne of these is true...",
  "block_idx": 0,
  "sublayer_kind": "attn",
  "n_tokens": 16,
  "input_token_ids": [282, 1010, 13, ...],
  "sign": -1,
  "log_abs_det": 1586.36,
  "per_token_log_abs_det": [113.80, 94.58, 103.08, ...],
  "per_token_sign": [-1, 1, 1, ...],
  "per_token_sigma_min": [1.78e-3, 3.42e-3, ...],
  "per_token_sigma_max": [15.62, 22.72, ...],
  "per_token_condition_number": [8754.2, 6635.8, ...],
  "elapsed_sec": 2.58
}
```

Identification fields (`input_id`, `input_preview`, `block_idx`, `sublayer_kind`, `n_tokens`, `input_token_ids`) come first so rows from different samples / sublayers are distinguishable at a glance.

### Interpreting the numbers

- `log_abs_det` > 0: sublayer locally *expands* volume; < 0: *contracts*.
- `sign`: orientation-preserving (+1) vs orientation-flipping (−1) — irrelevant for the homeomorphism claim, which only needs $$|\det| > 0$$.
- `per_token_sigma_min` is the main robustness signal. $$\to 0$$ means approaching singularity in at least one direction. Practically, small values warn that fp16 / int8 quantization will produce collisions along that direction (the "Robustness to quantization" section of the paper).
- What would falsify Part A: `log_abs_det = -∞` or `sign = 0` — neither should ever appear on a well-conditioned forward pass.

See §7 of `experiments_design.md` for the full post-run verification checklist.

### Examples

Full-network sweep on GPT-2, 8 inputs:

```bash
uv run python -m experiments.exp1_per_token_J \
    --n-samples 8 --max-tokens 64 --layers all
```

Random-init control, 3 seeds:

```bash
for s in 1 2 3; do
  uv run python -m experiments.exp1_per_token_J \
      --weights random_gaussian --seed $s --layers all
done
```

Compare attn vs ffn at the middle of the network:

```bash
uv run python -m experiments.exp1_per_token_J --layers 5.attn,5.ffn,6.attn,6.ffn
```

---

## Testing

```bash
uv run pytest tests/ -v
```

51 tests, all on CPU, finishes in ~15s. Grouped by module:

- `tests/test_jacobian.py` (16) — toy sublayers + autograd oracle; covers block building, causality, per-block slogdet/svd, and paper's eq. (4) on toys.
- `tests/test_models.py` (17) — real GPT-2; covers loading, random-init determinism, config queries, phi-closure faithfulness, inference helpers.
- `tests/test_hooks.py` (6) — real GPT-2; shape/dtype/detachment + byte-equality with manual hooks.
- `tests/test_data.py` (5) — determinism + caching.
- `tests/test_integration.py` (7) — end-to-end including paper's eq. (4) verified on actual GPT-2 activations.

See §8 of `experiments_design.md` for a per-test breakdown and §9 for a meta-checklist showing that these tests together cover every functional step of Exp 1.

---

## Dependencies

- `torch >= 2.8`
- `transformers >= 5.5.4`
- `jupyter` (notebooks only)
- `pytest` (dev)

Activation capture uses plain PyTorch forward hooks; results are stored as JSONL.
