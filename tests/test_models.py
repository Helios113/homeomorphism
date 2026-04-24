"""Tests for homeomorphism.models — the HF-wrapping layer.

Each test validates one specific contract. Uses real GPT-2 distilled (small,
fast to load) so the checks exercise actual architecture wiring, not mocks.

Grouped:
  - load / init:          model loads and random-init mode actually changes weights.
  - config queries:       n_blocks, n_sublayers, hidden_size.
  - sublayer handle:      returned object has expected structure, hook_path resolves.
  - phi closure faith:    phi_attn(h_attn_in) reproduces the actual attn-sublayer output;
                          phi_ffn likewise; their composition equals one full block.
  - tokenize / predict:   basic sanity on the inference helpers.
"""

from __future__ import annotations

import pytest
import torch

from homeomorphism import models


# ---------------------------------------------------------------------------
# Module-scope model fixtures (load once, reuse)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_trained() -> models.Model:
    return models.load_model("gpt2", weights="trained")


@pytest.fixture(scope="module")
def gpt2_random_gaussian() -> models.Model:
    return models.load_model("gpt2", weights="random_gaussian", seed=0)


# ---------------------------------------------------------------------------
# load / init
# ---------------------------------------------------------------------------

def test_load_gpt2_returns_model_dataclass(gpt2_trained: models.Model) -> None:
    """load_model returns a Model with model + tokenizer + arch='gpt2'."""
    assert isinstance(gpt2_trained, models.Model)
    assert gpt2_trained.arch == "gpt2"
    assert gpt2_trained.model is not None
    assert gpt2_trained.tokenizer is not None


def test_load_unknown_model_raises() -> None:
    """Unknown model names produce a clear error (either from the HF hub lookup
    or from our ArchSpec detection). Both outcomes are acceptable; the contract
    is 'does not silently succeed'."""
    with pytest.raises((ValueError, OSError)):
        models.load_model("some-nonexistent-architecture-xyz")


def test_load_random_gaussian_changes_weights(gpt2_trained: models.Model, gpt2_random_gaussian: models.Model) -> None:
    """random_gaussian mode actually overwrites the pretrained weights."""
    p_trained = next(gpt2_trained.model.parameters())
    p_random = next(gpt2_random_gaussian.model.parameters())
    # Shapes match, but values shouldn't.
    assert p_trained.shape == p_random.shape
    assert not torch.allclose(p_trained, p_random)


def test_load_random_gaussian_seed_determinism() -> None:
    """Two loads with the same seed produce identical random parameters."""
    m1 = models.load_model("gpt2", weights="random_gaussian", seed=123)
    m2 = models.load_model("gpt2", weights="random_gaussian", seed=123)
    p1 = next(m1.model.parameters())
    p2 = next(m2.model.parameters())
    assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Config queries
# ---------------------------------------------------------------------------

def test_n_blocks_gpt2_small(gpt2_trained: models.Model) -> None:
    """GPT-2 small has 12 transformer blocks."""
    assert models.n_blocks(gpt2_trained) == 12


def test_n_sublayers_is_twice_n_blocks(gpt2_trained: models.Model) -> None:
    """Pre-LN convention: attn + ffn per block."""
    assert models.n_sublayers(gpt2_trained) == 2 * models.n_blocks(gpt2_trained)


def test_hidden_size_gpt2_small(gpt2_trained: models.Model) -> None:
    """GPT-2 small has d=768 hidden width."""
    assert models.hidden_size(gpt2_trained) == 768


# ---------------------------------------------------------------------------
# Sublayer handle
# ---------------------------------------------------------------------------

def test_sublayer_returns_dataclass_with_fields(gpt2_trained: models.Model) -> None:
    """sublayer(m, 0, 'attn') returns a Sublayer with block_idx, kind, hook_path, phi."""
    s = models.sublayer(gpt2_trained, 0, "attn")
    assert isinstance(s, models.Sublayer)
    assert s.block_idx == 0 and s.kind == "attn"
    assert isinstance(s.hook_path, str)
    assert callable(s.phi)


def test_sublayer_hook_path_resolves(gpt2_trained: models.Model) -> None:
    """For every (block, kind), the hook_path resolves via model.get_submodule."""
    for block_idx in [0, 5, 11]:
        for kind in ("attn", "ffn"):
            s = models.sublayer(gpt2_trained, block_idx, kind)
            # No exception => path is valid
            _ = gpt2_trained.model.get_submodule(s.hook_path)


def test_sublayer_out_of_range_raises(gpt2_trained: models.Model) -> None:
    """block_idx outside [0, n_blocks) raises IndexError."""
    with pytest.raises(IndexError):
        models.sublayer(gpt2_trained, 100, "attn")


def test_sublayer_unknown_kind_raises(gpt2_trained: models.Model) -> None:
    with pytest.raises(ValueError):
        models.sublayer(gpt2_trained, 0, "nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phi closure faithfulness — the load-bearing correctness check
# ---------------------------------------------------------------------------

def test_phi_attn_matches_actual_attn_sublayer(gpt2_trained: models.Model) -> None:
    """phi_attn(h_attn_in) must equal the residual stream after the attention sublayer
    (which is the input to the following ffn sublayer). This is what validates that
    our Jacobian is computed on the SAME map the model actually uses."""
    from homeomorphism import hooks

    text = "The quick brown fox"
    s_attn = models.sublayer(gpt2_trained, 0, "attn")
    s_ffn = models.sublayer(gpt2_trained, 0, "ffn")

    h_attn_in = hooks.capture_activation(gpt2_trained, s_attn.hook_path, text).to(torch.float32)
    h_ffn_in = hooks.capture_activation(gpt2_trained, s_ffn.hook_path, text).to(torch.float32)

    with torch.no_grad():
        out = s_attn.phi(h_attn_in)
    # Closure should match the actual model's residual stream after attn
    assert (out - h_ffn_in).abs().max().item() < 1e-3


def test_phi_ffn_matches_actual_ffn_sublayer(gpt2_trained: models.Model) -> None:
    """phi_ffn(h_ffn_in) must equal the residual stream at the start of the next block."""
    from homeomorphism import hooks

    text = "The quick brown fox"
    s_ffn = models.sublayer(gpt2_trained, 0, "ffn")
    # Next block's attn input = output of this block = output of this ffn sublayer
    s_next_attn = models.sublayer(gpt2_trained, 1, "attn")

    h_ffn_in = hooks.capture_activation(gpt2_trained, s_ffn.hook_path, text).to(torch.float32)
    h_next_in = hooks.capture_activation(gpt2_trained, s_next_attn.hook_path, text).to(torch.float32)

    with torch.no_grad():
        out = s_ffn.phi(h_ffn_in)
    assert (out - h_next_in).abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# Tokenize / predict / generate
# ---------------------------------------------------------------------------

def test_tokenize_returns_shape_1_by_T(gpt2_trained: models.Model) -> None:
    """tokenize returns a 2-D tensor with leading batch dim = 1."""
    ids = models.tokenize(gpt2_trained, "hello world")
    assert ids.dim() == 2 and ids.shape[0] == 1
    assert ids.shape[1] >= 1


def test_tokenize_max_tokens_truncates(gpt2_trained: models.Model) -> None:
    """max_tokens caps the tokenized sequence length."""
    ids = models.tokenize(gpt2_trained, "x " * 100, max_tokens=8)
    assert ids.shape[1] <= 8


def test_predict_next_token_structure(gpt2_trained: models.Model) -> None:
    """predict_next_token returns the expected keys and top_k results."""
    out = models.predict_next_token(gpt2_trained, "The capital of France is", top_k=3)
    assert set(out.keys()) >= {"token_ids", "tokens", "probs", "logits", "n_input_tokens"}
    assert len(out["token_ids"]) == 3 and len(out["tokens"]) == 3 and len(out["probs"]) == 3
    # Probs are valid
    assert all(0.0 <= p <= 1.0 for p in out["probs"])


def test_generate_returns_longer_string(gpt2_trained: models.Model) -> None:
    """generate produces output strictly longer than the input."""
    text = "Once upon a time"
    out = models.generate(gpt2_trained, text, max_new_tokens=5)
    assert isinstance(out, str)
    assert len(out) > len(text)
