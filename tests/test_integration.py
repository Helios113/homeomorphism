"""End-to-end integration tests: load model -> capture -> build Jacobian -> evaluate.

These are the thin "glue" tests that guard against the composition of primitives
going wrong. Individual primitives are tested in their own test files.

Tests (all on GPT-2 small, small T to keep fast):
  - Full pipeline smoke: a single (block, sublayer) measurement returns finite log|det|.
  - Paper equation (4) on a real model: full-matrix slogdet from independent autograd
    matches the product-of-diagonals computed via sublayer_slogdet.
  - Exp 1 driver: resolve_sublayers + measure_sublayer work on a trained GPT-2.
"""

from __future__ import annotations

import pytest
import torch

from experiments.exp1_per_token_J import measure_sublayer, resolve_sublayers
from homeomorphism import hooks, jacobian, models


@pytest.fixture(scope="module")
def gpt2() -> models.Model:
    return models.load_model("gpt2", weights="trained")


# ---------------------------------------------------------------------------
# Pipeline smoke
# ---------------------------------------------------------------------------

def test_full_pipeline_smoke_layer0_attn(gpt2: models.Model) -> None:
    """Load -> capture -> build_jacobian -> sublayer_slogdet yields a finite log|det|."""
    s = models.sublayer(gpt2, 0, "attn")
    h = hooks.capture_activation(gpt2, s.hook_path, "The cat sat on the mat", max_tokens=8).to(torch.float32)
    sign, log = jacobian.sublayer_slogdet(s.phi, h)
    assert torch.isfinite(log)
    assert sign.item() in (-1, 1)


# ---------------------------------------------------------------------------
# Paper's equation (4) on a real model
# ---------------------------------------------------------------------------

def test_paper_eq4_on_real_gpt2(gpt2: models.Model) -> None:
    """On a real GPT-2 sublayer at real activations:
        sum_i log|det J^(i)| via build_jacobian
    matches
        log|det(full J)| via an independent autograd computation.

    This is the empirical check of the paper's equation (4) on the actual model.
    T must be small because we materialize the full (T*d, T*d) Jacobian here.
    """
    T = 4  # keep small: full J is (T*768, T*768)
    s = models.sublayer(gpt2, 0, "attn")
    # Pick text guaranteed to tokenize to >= T BPE tokens
    text = "The quick brown fox jumps over"
    h = hooks.capture_activation(gpt2, s.hook_path, text, max_tokens=T).to(torch.float32)
    assert h.shape[0] == T, f"captured {h.shape[0]} tokens, wanted exactly {T}"

    # Ours: product-of-diagonals via sublayer_slogdet
    sign_ours, log_ours = jacobian.sublayer_slogdet(s.phi, h)

    # Oracle: full autograd Jacobian -> reshape -> slogdet
    h_grad = h.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        J_full = torch.autograd.functional.jacobian(s.phi, h_grad, vectorize=True)
    # J_full shape: (T, d, T, d)
    d = h.shape[1]
    J_flat = J_full.reshape(T * d, T * d)
    sign_ref, log_ref = torch.linalg.slogdet(J_flat.to(torch.float32))

    assert sign_ref.item() == sign_ours.item()
    # Relative tolerance: fp32 slogdet on 3072x3072 matrix has some drift.
    rel_err = abs(log_ref.item() - log_ours.item()) / max(abs(log_ref.item()), 1.0)
    assert rel_err < 1e-4, (
        f"log|det| ref={log_ref.item():.4f} vs ours={log_ours.item():.4f} "
        f"(rel err {rel_err:.2e})"
    )


# ---------------------------------------------------------------------------
# Exp 1 driver pieces
# ---------------------------------------------------------------------------

def test_resolve_sublayers_all() -> None:
    """spec='all' yields every (block, kind) pair in forward order."""
    sl = resolve_sublayers(n_blocks=3, spec="all")
    assert sl == [(0, "attn"), (0, "ffn"), (1, "attn"), (1, "ffn"), (2, "attn"), (2, "ffn")]


def test_resolve_sublayers_list() -> None:
    sl = resolve_sublayers(n_blocks=12, spec="0.attn,5.ffn,11.attn")
    assert sl == [(0, "attn"), (5, "ffn"), (11, "attn")]


def test_resolve_sublayers_rejects_malformed() -> None:
    import pytest
    with pytest.raises(ValueError):
        resolve_sublayers(n_blocks=12, spec="not-a-spec")
    with pytest.raises(ValueError):
        resolve_sublayers(n_blocks=12, spec="0.nonsense")
    with pytest.raises(ValueError):
        resolve_sublayers(n_blocks=12, spec="99.attn")  # out of range


def test_measure_sublayer_produces_expected_row(gpt2: models.Model) -> None:
    """measure_sublayer returns all fields the JSONL schema claims, including
    the identifying fields that distinguish samples."""
    row = measure_sublayer(gpt2, text="hello world", block_idx=0, kind="attn", max_tokens=8)
    expected_keys = {
        # identification
        "block_idx",
        "sublayer_kind",
        "n_tokens",
        "input_token_ids",
        # summary
        "sign",
        "log_abs_det",
        # per-token detail
        "per_token_log_abs_det",
        "per_token_sign",
        "per_token_sigma_min",
        "per_token_sigma_max",
        "per_token_condition_number",
        # runtime
        "elapsed_sec",
    }
    assert set(row.keys()) == expected_keys
    T = row["n_tokens"]
    # All per-token lists have length T; input_token_ids has length T too
    assert len(row["input_token_ids"]) == T
    for key in [
        "per_token_log_abs_det",
        "per_token_sign",
        "per_token_sigma_min",
        "per_token_sigma_max",
        "per_token_condition_number",
    ]:
        assert len(row[key]) == T, f"{key} has wrong length"
    # Scalar log|det| equals sum of per-token
    sum_log = sum(row["per_token_log_abs_det"])
    assert abs(sum_log - row["log_abs_det"]) < 1e-6
    # sigma_min <= sigma_max per token
    for i in range(T):
        assert row["per_token_sigma_min"][i] <= row["per_token_sigma_max"][i] + 1e-9


def test_measure_sublayer_distinguishes_different_inputs(gpt2: models.Model) -> None:
    """Two different inputs produce distinct input_token_ids and DIFFERENT
    log|det| values at the same (block, sublayer). This is the contract the
    user flagged: each row's identity must be visibly tied to its input."""
    row_a = measure_sublayer(gpt2, text="The quick brown fox", block_idx=0, kind="attn", max_tokens=8)
    row_b = measure_sublayer(gpt2, text="A completely different sentence", block_idx=0, kind="attn", max_tokens=8)

    # Different inputs -> different token id sequences (at least somewhere)
    assert row_a["input_token_ids"] != row_b["input_token_ids"]
    # And different log|det| (extremely unlikely to collide on real GPT-2)
    assert row_a["log_abs_det"] != row_b["log_abs_det"]
    # Per-token details differ too
    assert row_a["per_token_log_abs_det"] != row_b["per_token_log_abs_det"]
