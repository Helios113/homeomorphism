# Homeomorphic Transformers

Experimental codebase for investigating whether transformer layers act as homeomorphisms (bi-Lipschitz maps) between representation manifolds at consecutive layers.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management.

```bash
uv sync
```

## Project structure

```
homeomorphism/
├── src/
│   ├── homeomorphism/          # hooks & Jacobian primitives
│   │   ├── hooks.py
│   │   └── jacobian.py
│   ├── activation_extractor/   # high-level activation extraction
│   │   ├── __init__.py
│   │   └── extractor.py
│   └── id_est/                 # intrinsic dimension estimators (planned)
│       └── id_estimators.py
├── examples/
│   └── extract_activations.py
├── notebooks/
│   └── testing.ipynb
└── pyproject.toml
```

## Modules

### `activation_extractor`

High-level activation extraction from HuggingFace causal LMs using PyTorch forward hooks.

**Classes:**

- `ActivationExtractor` — main extractor. Constructed with a model, tokenizer, and a layer template string (e.g. `"transformer.h.{layer}"` for GPT-2). Exposes:
  - `extract(input_text, config)` — run a forward pass and return activations as `ActivationData`
  - `get_layer_module(layer)` — return the `nn.Module` for a given layer (useful for Jacobian computation)
- `ExtractionConfig` — controls what to extract:
  - `layers` — list of layer indices, or `None` for all layers
  - `positions` — which token positions to keep: `"all"`, `"last"` (last non-pad), `"mean"` (masked mean), or a `list[int]` of explicit indices
  - `return_numpy` — whether to return numpy arrays (default) or torch tensors
- `ActivationData` — returned container with fields: `activations`, `tokens`, `token_ids`, `attention_mask`, `layers`

**Factory:**

- `make_extractor(model_name)` — loads a model + tokenizer and returns a ready `ActivationExtractor`. Auto-resolves layer templates for known architectures (GPT-2, Llama, Pythia). Pass `layer_template` explicitly for other models.

**Shape conventions:**

| `positions`   | `activations` shape              |
|---------------|----------------------------------|
| `"all"`       | `(batch, n_layers, seq_len, D)`  |
| `"last"`      | `(batch, n_layers, D)`           |
| `"mean"`      | `(batch, n_layers, D)`           |
| `list[int]`   | `(batch, n_layers, len(list), D)`|

### `homeomorphism.hooks`

Low-level hook utilities for attaching to arbitrary submodules.

- `list_module_names(model)` — list all named submodule paths
- `get_submodule(model, path)` / `get_submodules(model, prefix, items)` / `get_modules(model, items)` — resolve submodules by dotted path or leaf name
- `register_forward_capture_hooks(modules, cache, module_type)` — attach forward hooks to a list of modules and cache their outputs in a dict keyed by `"{module_type}_{i}"`

### `homeomorphism.jacobian`

Per-layer Jacobian computation via `torch.autograd.functional.jacobian`.

- `submodule_output_jacobian(model, module_path, model_inputs)` — runs a forward pass to capture a submodule's input, then computes the full Jacobian `d(output)/d(input)` for a single batch element. Returns a tensor of shape `(1, T, D, 1, T, D)` for the full-state Jacobian.
- `remove_hooks(handles)` — clean up hook handles

### `id_est.id_estimators`

Planned module for intrinsic dimension estimation (MLE, TwoNN). Not yet implemented.

## Example

`examples/extract_activations.py` demonstrates all extraction modes on GPT-2:

```bash
uv run python -m examples.extract_activations
```

What it does:

1. Loads GPT-2 via `make_extractor("gpt2")`
2. Extracts activations at first & last layer for a batch of two sentences (`positions="all"`)
3. Extracts last-token representations at layers 0, 5, 11 (`positions="last"`)
4. Extracts mean-pooled representations (`positions="mean"`)
5. Extracts specific token indices (`positions=[0, 1]`)
6. Extracts all layers with default config
7. Shows how to access the underlying `nn.Module` via `get_layer_module()` for Jacobian work

Sample output:

```
Model loaded — 12 layers, device=cpu

[all positions]  ActivationData(batch=2, layers=[0, 11], shape=(2, 2, 6, 768))
  tokens[0]: ['The', ' cat', ' sat', ' on', ' the', ' mat']

[last token]     ActivationData(batch=2, layers=[0, 5, 11], shape=(2, 3, 768))
[mean pooled]    ActivationData(batch=2, layers=[0, 5, 11], shape=(2, 3, 768))
[indices 0,1]    ActivationData(batch=1, layers=[0], shape=(1, 1, 2, 768))
[all layers/all] ActivationData(batch=1, layers=[0, 1, ..., 11], shape=(1, 12, 6, 768))

Layer 0 module type: GPT2Block
```

## Dependencies

- `torch >= 2.8`
- `transformers >= 5.5.4`
- `jupyter` (for notebooks)

No `nnsight` dependency — activation extraction uses plain PyTorch forward hooks.
