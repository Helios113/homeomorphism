"""Activation capture utilities — run a forward pass and grab what you need.

One function:
  `capture_activation(m, hook_path, text, max_tokens=None) -> Tensor of shape (T, d)`.

It tokenizes `text`, runs a single forward pass of `m.model`, captures the INPUT
tensor at the module located at `hook_path`, strips the batch dim, and returns
the captured residual stream without gradients. This is what the Jacobian code
expects as its `h_input`.

If you need more general hook patterns (capture multiple paths in one pass,
capture outputs instead of inputs, etc.), extend here — keep the single-purpose
function simple and add named siblings.
"""

from __future__ import annotations

import torch

from .models import Model, tokenize


def capture_activation(
    m: Model,
    hook_path: str,
    text: str,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Run a forward pass on `text` and return the INPUT residual stream at
    `hook_path` as a 2-D tensor of shape (T, d). No gradients; detached."""
    module = m.model.get_submodule(hook_path)
    captured: list[torch.Tensor] = []

    def hook(_mod, inputs, _output):  # noqa: ANN001
        h = inputs[0] if isinstance(inputs, tuple) else inputs
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"hook at {hook_path} got non-tensor input: {type(h)}")
        captured.append(h.detach())

    handle = module.register_forward_hook(hook)
    try:
        input_ids = tokenize(m, text, max_tokens=max_tokens)
        with torch.no_grad():
            m.model(input_ids=input_ids)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"hook at {hook_path} did not fire")
    h = captured[0]
    if h.dim() != 3 or h.shape[0] != 1:
        raise ValueError(
            f"expected captured tensor of shape (1, T, d); got {tuple(h.shape)}"
        )
    return h[0]  # (T, d)
