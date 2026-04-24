"""Comprehensive empirical topology/Jacobian verification suite for a toy Transformer.

This file implements four requested experiments:
1) Initialization topological check (continuous vs degenerate discrete init)
2) Manifold preservation baseline with persistent homology (torus vs white noise)
3) Measure-zero stress test under adversarial quantization/noise during training
4) Causal masking Jacobian structure and determinant factorization

All experiments use the custom model in homeomorphism.toy_transformer with:
- ambient dimension d = 32
- sequence length T = 10
- attention heads = 2

The tests prioritize exact autograd Jacobians and deterministic reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn.functional as F

from homeomorphism.toy_transformer import (
    ToyTopologicalTransformer,
    apply_adversarial_intervention,
    block_condition_numbers,
    determinant_factorization_error,
    extract_diagonal_blocks,
    full_sequence_jacobian,
    is_block_strictly_lower_triangular,
    sample_torus,
    sample_white_noise,
    set_reproducible_seed,
)

D_MODEL = 32
SEQ_LEN = 10
N_HEADS = 2
N_LAYERS = 2
GLOBAL_SEED = 1337


def _make_model(*, seed: int, causal: bool) -> ToyTopologicalTransformer:
    """Build a toy model with deterministic continuous initialization."""
    set_reproducible_seed(seed)
    model = ToyTopologicalTransformer(
        d_model=D_MODEL,
        seq_len=SEQ_LEN,
        n_heads=N_HEADS,
        d_ff=4 * D_MODEL,
        n_layers=N_LAYERS,
        causal=causal,
    )
    model.reset_parameters_continuous(seed=seed, scale=0.05)
    model.eval()
    return model


def _random_sequence(*, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(SEQ_LEN, D_MODEL, generator=g, dtype=torch.float32)


def _iter_subblock_phis(model: ToyTopologicalTransformer):
    for layer_idx in range(model.n_layers):
        yield layer_idx, "attn", model.subblock_phi(layer_idx, "attn")
        yield layer_idx, "ffn", model.subblock_phi(layer_idx, "ffn")


# ---------------------------------------------------------------------------
# Experiment 1: Initialization topological check
# ---------------------------------------------------------------------------

def test_jacobian_initialization() -> None:
    """Verify non-singularity at continuous init and frequent singularity at degenerate init.

    Continuous branch:
      - 100 independent model initializations via continuous distributions
      - For each sub-block, compute exact autograd Jacobian at a random input X
      - Extract token-wise diagonal blocks and require |det| > 1e-6

    Discrete branch:
      - 100 independent degenerate initializations (binary weights)
      - Compute the same token-wise blocks
      - Require frequent singular events via determinant/rank criteria
    """
    set_reproducible_seed(GLOBAL_SEED)
    x = _random_sequence(seed=GLOBAL_SEED + 1)

    # Continuous branch: each diagonal token block should remain non-singular.
    continuous_total = 0
    continuous_non_singular = 0
    for i in range(100):
        model = _make_model(seed=1000 + i, causal=True)
        for _, _, phi in _iter_subblock_phis(model):
            jac = full_sequence_jacobian(phi, x)
            for block in extract_diagonal_blocks(jac):
                det = torch.linalg.det(block.to(torch.float32))
                continuous_total += 1
                if torch.abs(det).item() > 1e-6:
                    continuous_non_singular += 1

    assert continuous_non_singular == continuous_total, (
        f"Continuous branch produced singular token Jacobians: "
        f"{continuous_total - continuous_non_singular}/{continuous_total}"
    )

    # Discrete branch: intentionally degenerate initializations should often collapse rank.
    discrete_total = 0
    discrete_singular = 0
    for i in range(100):
        set_reproducible_seed(2000 + i)
        model = ToyTopologicalTransformer(
            d_model=D_MODEL,
            seq_len=SEQ_LEN,
            n_heads=N_HEADS,
            d_ff=4 * D_MODEL,
            n_layers=N_LAYERS,
            causal=True,
        )
        model.reset_parameters_discrete(seed=2000 + i, mode="binary")
        model.eval()

        for _, _, phi in _iter_subblock_phis(model):
            jac = full_sequence_jacobian(phi, x)
            for block in extract_diagonal_blocks(jac):
                det = torch.linalg.det(block.to(torch.float32))
                rank = torch.linalg.matrix_rank(block.to(torch.float32), tol=1e-6)
                is_singular = (torch.abs(det).item() <= 1e-6) or (int(rank.item()) < D_MODEL)
                discrete_total += 1
                if is_singular:
                    discrete_singular += 1

    singular_rate = discrete_singular / max(discrete_total, 1)
    assert singular_rate >= 0.25, (
        f"Expected frequent singular matrices in degenerate branch, got rate={singular_rate:.3f}"
    )


# ---------------------------------------------------------------------------
# Experiment 2: Manifold preservation (persistent homology)
# ---------------------------------------------------------------------------

def _collect_layerwise_clouds(
    model: ToyTopologicalTransformer,
    sequences: torch.Tensor,
) -> list[torch.Tensor]:
    """Collect point clouds per depth from batch sequences.

    Depth indexing:
      0: input cloud
      1: layer0 after attention
      2: layer0 after FFN
      3: layer1 after attention
      4: layer1 after FFN
      ...

    Each depth cloud has shape (B*T, d).
    """
    if sequences.dim() != 3:
        raise ValueError(f"Expected shape (B, T, d), got {tuple(sequences.shape)}")

    b, t, d = sequences.shape
    n_depths = 1 + 2 * model.n_layers

    clouds_per_depth: list[list[torch.Tensor]] = [[] for _ in range(n_depths)]
    for i in range(b):
        x = sequences[i]
        _, states = model.forward(x, return_intermediates=True)
        all_depths = [x] + states
        for depth_idx, rep in enumerate(all_depths):
            clouds_per_depth[depth_idx].append(rep)

    clouds: list[torch.Tensor] = []
    for depth_idx in range(n_depths):
        stacked = torch.stack(clouds_per_depth[depth_idx], dim=0)  # (B, T, d)
        clouds.append(stacked.reshape(b * t, d))
    return clouds


def _betti_numbers(points: torch.Tensor, *, lifetime_threshold: float = 0.15) -> tuple[int, int, int]:
    """Estimate robust Betti numbers (b0, b1, b2) from ripser diagrams.

    We count persistent features with lifetime > threshold, and treat one
    infinite H0 component as the connected manifold component.
    """
    ripser_mod = pytest.importorskip("ripser", reason="ripser/scikit-tda not installed in uv env")
    result = ripser_mod.ripser(points.detach().cpu().numpy(), maxdim=2)
    dgms = result["dgms"]

    betti: list[int] = []
    for dim in range(3):
        dgm = dgms[dim]
        if dgm.size == 0:
            betti.append(0)
            continue

        births = torch.from_numpy(dgm[:, 0]).to(torch.float32)
        deaths = torch.from_numpy(dgm[:, 1]).to(torch.float32)
        finite = torch.isfinite(deaths)
        lifetimes = torch.zeros_like(births)
        lifetimes[finite] = deaths[finite] - births[finite]
        count = int((lifetimes > lifetime_threshold).sum().item())

        if dim == 0 and (~finite).any():
            # Preserve exactly one connected component for H0 when infinity bar exists.
            count = max(count, 1)

        betti.append(count)

    return (betti[0], betti[1], betti[2])


@pytest.mark.slow
def test_manifold_preservation() -> None:
    """Structured torus should preserve core topology layer-by-layer vs white-noise baseline."""
    set_reproducible_seed(GLOBAL_SEED + 10)

    # Build non-causal model for cleaner manifold transport without positional truncation.
    model = _make_model(seed=3100, causal=False)

    n_points = 200  # divisible by T=10
    structured = sample_torus(n_samples=n_points, ambient_dim=D_MODEL, seed=3101)
    baseline = sample_white_noise(n_samples=n_points, ambient_dim=D_MODEL, seed=3102)

    structured_seq = structured.view(n_points // SEQ_LEN, SEQ_LEN, D_MODEL)
    baseline_seq = baseline.view(n_points // SEQ_LEN, SEQ_LEN, D_MODEL)

    structured_clouds = _collect_layerwise_clouds(model, structured_seq)
    baseline_clouds = _collect_layerwise_clouds(model, baseline_seq)

    structured_betti = [_betti_numbers(c) for c in structured_clouds]
    baseline_betti = [_betti_numbers(c) for c in baseline_clouds]

    # Log for debug visibility in test output.
    print("Structured Betti by depth:", structured_betti)
    print("Noise Betti by depth:", baseline_betti)

    # Foundational torus signature expectation: one connected component, two loops, one void.
    # We allow >= to account for finite-sample artifacts while guarding against topological collapse.
    for b0, b1, b2 in structured_betti:
        assert b0 >= 1
        assert b1 >= 2
        assert b2 >= 1

    # White-noise baseline should not consistently preserve torus-like high-order structure.
    # We require at least one depth where H1 or H2 falls below torus foundation.
    assert any((b1 < 2) or (b2 < 1) for _, b1, b2 in baseline_betti)


# ---------------------------------------------------------------------------
# Experiment 3: Measure-zero stress test under adversarial quantization
# ---------------------------------------------------------------------------

def run_stress_test_training(
    *,
    steps: int,
    output_dir: Path,
    seed: int = 0,
) -> tuple[list[dict[str, float]], Path, Path]:
    """Run stress training and record Jacobian condition-number trajectories.

    Returns:
      - list of per-step logs
      - jsonl log path
      - plot path
    """
    set_reproducible_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _make_model(seed=seed + 1, causal=True)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    g_data = torch.Generator().manual_seed(seed + 2)
    probe = torch.randn(SEQ_LEN, D_MODEL, generator=g_data, dtype=torch.float32)

    logs: list[dict[str, float]] = []
    for step in range(steps):
        batch = torch.randn(8, SEQ_LEN, D_MODEL, generator=g_data, dtype=torch.float32)
        target = batch.clone()

        optim.zero_grad(set_to_none=True)
        pred = model.batch_forward(batch)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optim.step()

        meta = apply_adversarial_intervention(model, step=step, total_steps=steps, seed=seed + 100)

        # Probe Jacobian conditioning on the first attention sub-block.
        phi = model.subblock_phi(0, "attn")
        jac = full_sequence_jacobian(phi, probe)
        conds = block_condition_numbers(jac)

        row = {
            "step": float(step),
            "loss": float(loss.item()),
            "mean_cond": float(sum(conds) / len(conds)),
            "max_cond": float(max(conds)),
            "min_cond": float(min(conds)),
            "n_levels": float(meta["n_levels"]),
            "noise_std": float(meta["noise_std"]),
        }
        logs.append(row)

    log_path = output_dir / "stress_condition_numbers.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")

    plot_path = output_dir / "stress_condition_numbers.png"
    xs = [r["step"] for r in logs]
    ys = [r["max_cond"] for r in logs]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, linewidth=2.0)
    plt.xlabel("Training step")
    plt.ylabel("Max token-wise condition number")
    plt.title("Jacobian conditioning under adversarial quantization/noise")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return logs, log_path, plot_path


@pytest.mark.slow
def test_measure_zero_stress_training(tmp_path: Path) -> None:
    """Condition numbers should blow up as quantization/noise becomes aggressive."""
    logs, log_path, plot_path = run_stress_test_training(steps=24, output_dir=tmp_path, seed=4400)

    assert log_path.exists()
    assert plot_path.exists()
    assert len(logs) == 24

    first_quarter = logs[:6]
    last_quarter = logs[-6:]

    first_mean = sum(x["max_cond"] for x in first_quarter) / len(first_quarter)
    last_mean = sum(x["max_cond"] for x in last_quarter) / len(last_quarter)

    # The main stress-test claim: stronger intervention drives near-singularity,
    # reflected by exploding condition numbers.
    assert last_mean > first_mean * 2.0, (
        f"Expected condition numbers to increase substantially; first={first_mean:.3e}, "
        f"last={last_mean:.3e}"
    )


# ---------------------------------------------------------------------------
# Experiment 4: Causal masking permutations and determinant factorization
# ---------------------------------------------------------------------------

def _assert_upper_blocks_zero(jac: torch.Tensor, *, atol: float = 1e-7) -> None:
    """Assert strict lower-triangular block structure in a full Jacobian."""
    t = jac.shape[0]
    for i in range(t):
        for j in range(i + 1, t):
            assert torch.allclose(
                jac[i, :, j, :],
                torch.zeros_like(jac[i, :, j, :]),
                atol=atol,
            ), f"Expected zero block for d h_i / d h_j with j>i at (i={i}, j={j})"


def test_causal_jacobian_structure() -> None:
    """Verify lower-triangularity and determinant factorization on X and permuted X."""
    set_reproducible_seed(GLOBAL_SEED + 40)
    model = _make_model(seed=5500, causal=True)
    phi = model.subblock_phi(0, "attn")

    x = _random_sequence(seed=5501)
    perm = torch.randperm(SEQ_LEN, generator=torch.Generator().manual_seed(5502))
    x_perm = x[perm]

    jac = full_sequence_jacobian(phi, x)
    jac_perm = full_sequence_jacobian(phi, x_perm)

    # Verification 1: strict lower-triangular block structure.
    assert is_block_strictly_lower_triangular(jac, atol=1e-7)
    assert is_block_strictly_lower_triangular(jac_perm, atol=1e-7)
    _assert_upper_blocks_zero(jac, atol=1e-7)
    _assert_upper_blocks_zero(jac_perm, atol=1e-7)

    # Verification 2: global determinant equals product of diagonal token blocks.
    full_det, diag_prod, abs_err = determinant_factorization_error(jac)
    full_det_perm, diag_prod_perm, abs_err_perm = determinant_factorization_error(jac_perm)

    assert torch.allclose(full_det, diag_prod, atol=1e-4, rtol=1e-4), (
        f"Determinant factorization failed on X; abs_err={abs_err.item():.3e}"
    )
    assert torch.allclose(full_det_perm, diag_prod_perm, atol=1e-4, rtol=1e-4), (
        f"Determinant factorization failed on permuted X; abs_err={abs_err_perm.item():.3e}"
    )
