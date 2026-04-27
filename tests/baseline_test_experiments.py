from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from homeomorphism.interventions import PreparedInput

from experiments import baseline_exp_3_permutation as exp3
from experiments import baseline_exp_4_uniform as exp4


class _DummyHookHandle:
    def __init__(self, hooks: list):
        self._hooks = hooks
        self._alive = True

    def remove(self) -> None:
        if self._alive:
            self._hooks.pop()
            self._alive = False


class _DummySubmodule:
    def __init__(self) -> None:
        self.hooks: list = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _DummyHookHandle(self.hooks)


class _DummyBackbone:
    def __init__(self) -> None:
        self.sub = _DummySubmodule()

    def get_submodule(self, _path: str) -> _DummySubmodule:
        return self.sub

    def __call__(self, **kwargs):
        x = kwargs["inputs_embeds"]
        for hook in list(self.sub.hooks):
            hook(self.sub, (x,), None)


def test_baseline_capture_uses_prepared_forward_kwargs() -> None:
    m = SimpleNamespace(model=_DummyBackbone())
    x = torch.randn(1, 4, 3, dtype=torch.float32)

    out = exp4.capture_multi_with_prepared_input(
        m,
        ["any.path"],
        forward_kwargs={"inputs_embeds": x},
    )

    assert "any.path" in out
    assert out["any.path"].shape == (4, 3)
    assert torch.allclose(out["any.path"], x[0])


class _DummyBJ:
    T = 3

    def svdvals(self, _i: int, _j: int) -> torch.Tensor:
        return torch.tensor([2.0, 1.0], dtype=torch.float32)


def test_baseline_measurement_uses_prepared_token_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exp3.models,
        "sublayer",
        lambda _m, _b, _k: SimpleNamespace(hook_path="path", phi=lambda h: h),
    )
    monkeypatch.setattr(
        exp3,
        "capture_multi_with_prepared_input",
        lambda _m, _paths, *, forward_kwargs: {"path": torch.ones(3, 2, dtype=torch.float32)},
    )

    per_diag = [(torch.tensor(1), torch.tensor(0.0)) for _ in range(3)]
    monkeypatch.setattr(
        exp3.jacobian,
        "build_jacobian",
        lambda _phi, _h, scope, evaluate: (_DummyBJ(), per_diag),
    )

    prepared = PreparedInput(
        forward_kwargs={"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
        token_ids=torch.tensor([[9, 8, 7]], dtype=torch.long),
    )

    row = exp3.measure_sublayer(
        m=SimpleNamespace(),
        prepared=prepared,
        block_idx=0,
        kind="attn",
    )

    assert row["input_token_ids"] == [9, 8, 7]
    assert row["n_tokens"] == 3


@pytest.mark.parametrize(
    "module",
    [exp3, exp4],
)
def test_baseline_depth_spec_parser(module) -> None:
    sublayers = [(0, "attn"), (1, "ffn")]

    assert module._parse_depths_spec("all", 2, sublayers) == [0, 1, 2, 3, 4]
    assert module._parse_depths_spec("4,1,1", 2, sublayers) == [1, 4]
    assert module._parse_depths_spec("", 2, sublayers) == [0, 1, 3, 4]

    with pytest.raises(ValueError, match="out of range"):
        module._parse_depths_spec("5", 2, sublayers)

    with pytest.raises(ValueError, match="depth must be int"):
        module._parse_depths_spec("x", 2, sublayers)
