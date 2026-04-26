"""Tests for interventions module (baseline utilities for Section 2.1 experiments).

Tests cover:
  - Deterministic seed derivation
  - Token mutation functions (permutation, uniform)
  - Norm affine reset
  - Dropout disabling
  - Gaussian inputs_embeds generation
  - Full baseline input preparation
"""

from __future__ import annotations

import torch
import pytest

from homeomorphism import models, interventions


# ---------------------------------------------------------------------------
# Seed derivation tests
# ---------------------------------------------------------------------------

def test_derive_seed_determinism() -> None:
    """Same (root_seed, tag) should always yield same int."""
    root_seed = 42
    tag = "test_tag"
    
    seed1 = interventions.derive_seed(root_seed, tag)
    seed2 = interventions.derive_seed(root_seed, tag)
    
    assert isinstance(seed1, int)
    assert isinstance(seed2, int)
    assert seed1 == seed2
    assert 0 <= seed1 < 2**31 - 1


def test_derive_seed_differs_by_tag() -> None:
    """Different tags should produce different seeds."""
    root_seed = 42
    
    seed_a = interventions.derive_seed(root_seed, "tag_a")
    seed_b = interventions.derive_seed(root_seed, "tag_b")
    
    assert seed_a != seed_b


def test_derive_seed_differs_by_root() -> None:
    """Different root_seed should produce different output."""
    tag = "same_tag"
    
    seed_1 = interventions.derive_seed(1, tag)
    seed_2 = interventions.derive_seed(2, tag)
    
    assert seed_1 != seed_2


# ---------------------------------------------------------------------------
# Dropout disabling tests
# ---------------------------------------------------------------------------

def test_disable_dropout() -> None:
    """Verify all Dropout layers have p set to 0.0 after disable_dropout."""
    # Create a simple model with dropout layers
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(10, 10),
        torch.nn.Dropout(p=0.3),
    )
    
    # Verify dropout is active before
    dropouts_before = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts_before) == 2
    assert any(m.p > 0.0 for m in dropouts_before)
    
    # Disable
    interventions.disable_dropout(model)
    
    # Verify all dropouts have p=0.0
    dropouts_after = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert all(m.p == 0.0 for m in dropouts_after)


# ---------------------------------------------------------------------------
# Norm affine reset tests
# ---------------------------------------------------------------------------

def test_reset_norm_affine() -> None:
    """Verify LayerNorm affine params (weight, bias) are reset to 1 and 0."""
    ln = torch.nn.LayerNorm(10)
    
    # Mutate away from defaults so the test exercises the reset logic.
    with torch.no_grad():
        ln.weight.fill_(2.5)
        ln.bias.fill_(3.0)
    
    # Reset
    interventions.reset_norm_affine(ln)
    
    # Verify reset
    assert torch.allclose(ln.weight, torch.ones(10))
    assert torch.allclose(ln.bias, torch.zeros(10))


def test_reset_norm_affine_preserves_running_stats() -> None:
    """Verify LayerNorm running stats are NOT modified."""
    ln = torch.nn.LayerNorm(10, elementwise_affine=True)
    
    # Initialize with some data to populate running stats
    x = torch.randn(5, 10)
    with torch.no_grad():
        ln(x)
    
    # Store running stats before reset
    running_mean_before = ln.running_mean.clone() if hasattr(ln, 'running_mean') else None
    running_var_before = ln.running_var.clone() if hasattr(ln, 'running_var') else None
    
    # Reset affine
    interventions.reset_norm_affine(ln)
    
    # Verify affine is reset
    assert torch.allclose(ln.weight, torch.ones(10))
    assert torch.allclose(ln.bias, torch.zeros(10))
    
    # Verify running stats are untouched (if they exist)
    if hasattr(ln, 'running_mean') and running_mean_before is not None:
        assert torch.allclose(ln.running_mean, running_mean_before)
    if hasattr(ln, 'running_var') and running_var_before is not None:
        assert torch.allclose(ln.running_var, running_var_before)


# ---------------------------------------------------------------------------
# Token mutation tests
# ---------------------------------------------------------------------------

def test_apply_permutation_valid_indices_only() -> None:
    """Verify permutation only affects non-padding tokens (where attention_mask==1)."""
    batch_size, seq_len, vocab_size = 2, 8, 50257
    
    # Create input_ids with some "padding" at the end (set to arbitrary value)
    input_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).long()
    
    # Attention mask: first 6 tokens are real, last 2 are padding
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 6:] = 0
    
    seed = interventions.derive_seed(42, "permute:0")
    permuted = interventions._apply_permutation(input_ids, attention_mask, seed=seed)
    
    # Verify padding tokens are unchanged
    assert torch.allclose(permuted[:, 6:], input_ids[:, 6:])
    
    # Verify valid tokens are permuted (for at least one sample)
    # This is probabilistic, so we just check they're different for some tokens
    valid_region = permuted[:, :6]
    orig_valid = input_ids[:, :6]
    # At least one token should be different (very high probability)
    assert not torch.allclose(valid_region, orig_valid)


def test_apply_permutation_determinism() -> None:
    """Same seed should produce same permutation."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    seed = 42
    
    perm1 = interventions._apply_permutation(input_ids, attention_mask, seed=seed)
    perm2 = interventions._apply_permutation(input_ids, attention_mask, seed=seed)
    
    assert torch.equal(perm1, perm2)


def test_apply_uniform_tokens_valid_indices_only() -> None:
    """Verify uniform token replacement only affects non-padding tokens."""
    batch_size, seq_len, vocab_size = 2, 8, 50257
    
    input_ids = torch.full((batch_size, seq_len), 50, dtype=torch.long)
    
    # Attention mask: first 6 tokens are real, last 2 are padding
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 6:] = 0
    
    seed = interventions.derive_seed(42, "uniform:0")
    uniform = interventions._apply_uniform_tokens(input_ids, attention_mask, vocab_size=vocab_size, seed=seed)
    
    # Verify padding tokens are unchanged
    assert torch.equal(uniform[:, 6:], input_ids[:, 6:])
    
    # Verify valid tokens are replaced (should not all be 50 anymore)
    valid_region = uniform[:, :6]
    assert not torch.all(valid_region == 50)


def test_apply_uniform_tokens_in_vocab_range() -> None:
    """Verify uniform tokens are in valid range [1, vocab_size)."""
    input_ids = torch.full((1, 10), 50, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    vocab_size = 1000
    
    seed = interventions.derive_seed(42, "uniform:test")
    uniform = interventions._apply_uniform_tokens(input_ids, attention_mask, vocab_size=vocab_size, seed=seed)
    
    # All values should be in [1, vocab_size)
    assert torch.all(uniform >= 1)
    assert torch.all(uniform < vocab_size)


# ---------------------------------------------------------------------------
# Gaussian inputs_embeds tests
# ---------------------------------------------------------------------------

def test_gaussian_inputs_embeds_shape() -> None:
    """Verify gaussian_inputs_embeds returns correct shape."""
    batch_size, seq_len, d_model = 4, 16, 768
    
    eps = interventions.gaussian_inputs_embeds(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        seed=42,
        device=torch.device("cpu"),
    )
    
    assert eps.shape == (batch_size, seq_len, d_model)
    assert eps.dtype == torch.float32


def test_gaussian_inputs_embeds_covariance() -> None:
    """Verify sample covariance of gaussian_inputs_embeds is approximately (1/d)I."""
    d_model = 64
    n_samples = 500
    
    # Generate many samples
    samples = []
    for i in range(n_samples):
        eps = interventions.gaussian_inputs_embeds(
            batch_size=1,
            seq_len=1,
            d_model=d_model,
            seed=42 + i,  # Different seed for each sample
            device=torch.device("cpu"),
        )
        samples.append(eps.squeeze())  # (d_model,)
    
    X = torch.stack(samples)  # (n_samples, d_model)
    
    # Compute sample covariance
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Expected covariance: (1/d) * I
    expected_diag = 1.0 / d_model
    
    # Check diagonal is close to 1/d (tolerance 10% to account for sampling variance)
    actual_diag = torch.diag(cov)
    expected_diag_tensor = torch.full_like(actual_diag, expected_diag)
    
    # Allow 20% relative tolerance for sampling variance
    rel_error = torch.abs(actual_diag - expected_diag_tensor) / expected_diag_tensor
    assert torch.all(rel_error < 0.2)


# ---------------------------------------------------------------------------
# Full baseline preparation tests
# ---------------------------------------------------------------------------

def test_build_prepared_input_trained() -> None:
    """Verify build_prepared_input works for 'trained' baseline."""
    m = models.load_model("gpt2", weights="trained", seed=42, device="cpu")
    text = "Hello world"
    
    prepared = interventions.build_prepared_input(
        m=m,
        text=text,
        max_tokens=8,
        baseline="trained",
        root_seed=42,
        sample_index=0,
    )
    
    assert isinstance(prepared, interventions.PreparedInput)
    assert "input_ids" in prepared.forward_kwargs
    assert "attention_mask" in prepared.forward_kwargs
    assert "position_ids" in prepared.forward_kwargs
    assert prepared.forward_kwargs["input_ids"].shape[0] == 1


def test_build_prepared_input_maximum_entropy_injection() -> None:
    """Verify maximum_entropy_injection uses inputs_embeds instead of input_ids."""
    m = models.load_model("gpt2", weights="trained", seed=42, device="cpu")
    text = "Hello world"
    original_ids = models.tokenize(m, text, max_tokens=8)
    
    prepared = interventions.build_prepared_input(
        m=m,
        text=text,
        max_tokens=8,
        baseline="maximum_entropy_injection",
        root_seed=42,
        sample_index=0,
    )
    
    assert isinstance(prepared, interventions.PreparedInput)
    assert "inputs_embeds" in prepared.forward_kwargs
    assert "input_ids" not in prepared.forward_kwargs
    assert "attention_mask" in prepared.forward_kwargs
    assert prepared.forward_kwargs["inputs_embeds"].shape == (1, original_ids.shape[1], 768)


def test_build_prepared_input_syntactic_disintegration() -> None:
    """Verify syntactic_disintegration permutes token IDs."""
    m = models.load_model("gpt2", weights="trained", seed=42, device="cpu")
    text = "The quick brown fox jumps over the lazy dog twice."
    
    prepared = interventions.build_prepared_input(
        m=m,
        text=text,
        max_tokens=8,
        baseline="syntactic_disintegration",
        root_seed=42,
        sample_index=0,
    )
    
    assert isinstance(prepared, interventions.PreparedInput)
    assert "input_ids" in prepared.forward_kwargs
    assert "attention_mask" in prepared.forward_kwargs
    original_ids = models.tokenize(m, text, max_tokens=8)
    permuted = prepared.forward_kwargs["input_ids"]
    valid_mask = prepared.forward_kwargs["attention_mask"].bool()
    assert permuted.shape == original_ids.shape
    assert torch.equal(permuted[~valid_mask], original_ids[~valid_mask])
    assert torch.equal(torch.sort(permuted[valid_mask]).values, torch.sort(original_ids[valid_mask]).values)


def test_build_prepared_input_semantic_scrambling() -> None:
    """Verify semantic_scrambling replaces tokens with uniform random."""
    m = models.load_model("gpt2", weights="trained", seed=42, device="cpu")
    text = "The quick brown fox jumps over the lazy dog twice."
    
    prepared = interventions.build_prepared_input(
        m=m,
        text=text,
        max_tokens=8,
        baseline="semantic_scrambling",
        root_seed=42,
        sample_index=0,
    )
    
    assert isinstance(prepared, interventions.PreparedInput)
    assert "input_ids" in prepared.forward_kwargs
    assert "attention_mask" in prepared.forward_kwargs
    original_ids = models.tokenize(m, text, max_tokens=8)
    scrambled = prepared.forward_kwargs["input_ids"]
    valid_mask = prepared.forward_kwargs["attention_mask"].bool()
    assert scrambled.shape == original_ids.shape
    assert torch.equal(scrambled[~valid_mask], original_ids[~valid_mask])
    assert torch.all(scrambled[valid_mask] >= 1)
    assert torch.all(scrambled[valid_mask] < m.model.config.vocab_size)


def test_build_prepared_input_topological_initialisation() -> None:
    """Verify topological_initialisation works (uses input_ids, weight mode handled separately)."""
    m = models.load_model("gpt2", weights="trained", seed=42, device="cpu")
    text = "Hello world"
    
    prepared = interventions.build_prepared_input(
        m=m,
        text=text,
        max_tokens=8,
        baseline="topological_initialisation",
        root_seed=42,
        sample_index=0,
    )
    
    assert isinstance(prepared, interventions.PreparedInput)
    assert "input_ids" in prepared.forward_kwargs
    assert "attention_mask" in prepared.forward_kwargs
    # For topological_initialisation, we use the original token IDs
    original_ids = models.tokenize(m, text, max_tokens=8)
    assert torch.equal(prepared.forward_kwargs["input_ids"], original_ids)


def test_load_model_for_baseline_topological_initialisation() -> None:
    """Verify load_model_for_baseline sets up topological_initialisation correctly."""
    m = interventions.load_model_for_baseline(
        model_name="gpt2",
        weights="trained",
        baseline="topological_initialisation",
        seed=42,
        device="cpu",
    )
    
    assert isinstance(m, models.Model)
    # Verify norms are reset
    for module in m.model.modules():
        name = module.__class__.__name__.lower()
        if "layernorm" in name or "rmsnorm" in name:
            if hasattr(module, "weight"):
                assert torch.allclose(module.weight, torch.ones_like(module.weight))
            if hasattr(module, "bias"):
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))
    
    # Verify dropout is disabled
    for module in m.model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.p == 0.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_all_baselines_can_prepare_input() -> None:
    """Verify all baseline modes can prepare input without crashing."""
    m = models.load_model("gpt2", weights="trained", seed=42, device="cpu")
    text = "Hello world"
    
    for baseline in interventions.VALID_BASELINES:
        prepared = interventions.build_prepared_input(
            m=m,
            text=text,
            max_tokens=8,
            baseline=baseline,  # type: ignore
            root_seed=42,
            sample_index=0,
        )
        
        # Verify structure
        assert isinstance(prepared, interventions.PreparedInput)
        assert isinstance(prepared.forward_kwargs, dict)
        assert isinstance(prepared.token_ids, torch.Tensor)
        
        # Verify all required keys
        assert "attention_mask" in prepared.forward_kwargs
        assert "position_ids" in prepared.forward_kwargs
        assert ("input_ids" in prepared.forward_kwargs or 
                "inputs_embeds" in prepared.forward_kwargs)
