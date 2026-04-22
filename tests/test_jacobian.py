"""Tests for homeomorphism.jacobian on toy self-contained sublayers.

Each test is tightly scoped: it tests ONE specific property of the Jacobian
machinery. See the module docstring of each test for the exact claim.

The hierarchy:
  - Build/access:      build_jacobian constructs correct (i, k) blocks.
  - Correctness:       each computed block matches an autograd oracle.
  - Causality:         above-diagonal blocks are numerically zero.
  - Per-block methods: slogdet, svdvals, condition_number behave correctly.
  - Full-matrix id:    the paper's equation (4) holds on toy sublayers.
  - evaluate= flag:    eager computations match explicit method calls.
  - Convenience:       sublayer_slogdet wraps the common path.
"""

from __future__ import annotations

import torch

from conftest import ToyAttnSublayer, ToyFFNSublayer, full_jacobian_oracle
from homeomorphism.jacobian import (
    BlockJacobian,
    build_jacobian,
    sublayer_slogdet,
)


# ---------------------------------------------------------------------------
# Build / access
# ---------------------------------------------------------------------------

def test_build_causal_returns_lower_triangular_key_set() -> None:
    """scope='causal' populates exactly the set {(i, k) : 0 <= k <= i < T}."""
    T, d = 4, 6
    phi = ToyAttnSublayer(d, seed=0).eval()
    torch.manual_seed(42)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="causal")
    assert isinstance(bj, BlockJacobian)
    assert bj.T == T and bj.d == d
    assert set(bj.keys()) == {(i, k) for i in range(T) for k in range(i + 1)}
    for i, k in bj.keys():
        assert bj[(i, k)].shape == (d, d)


def test_build_diagonal_populates_only_diagonal() -> None:
    """scope='diagonal' populates exactly the set {(i, i) : 0 <= i < T}."""
    T, d = 4, 6
    phi = ToyFFNSublayer(d, d_ffn=8, seed=1).eval()
    torch.manual_seed(43)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="diagonal")
    assert set(bj.keys()) == {(i, i) for i in range(T)}


def test_build_full_populates_all_and_upper_blocks_numerically_zero() -> None:
    """scope='full' builds the whole T*T grid; upper blocks come out ~0 by causality."""
    T, d = 4, 5
    phi = ToyAttnSublayer(d, seed=2).eval()
    torch.manual_seed(44)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="full")
    assert set(bj.keys()) == {(i, k) for i in range(T) for k in range(T)}
    for i in range(T):
        for k in range(i + 1, T):
            assert bj[(i, k)].abs().max().item() < 1e-6


def test_get_returns_zero_for_non_stored_upper_blocks() -> None:
    """bj.get(i, k) for k > i returns a zero tensor even when not stored (e.g., causal scope)."""
    T, d = 4, 5
    phi = ToyFFNSublayer(d, seed=3).eval()
    torch.manual_seed(45)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="causal")
    Z = bj.get(0, 3)  # k=3 > i=0, not stored under causal scope
    assert Z.shape == (d, d) and Z.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Correctness: blocks match the autograd oracle
# ---------------------------------------------------------------------------

def test_causal_blocks_match_autograd_oracle_attn() -> None:
    """For an attention sublayer: every (i, k) with k <= i matches the (i, :, k, :) slice
    of the full (T, d, T, d) autograd Jacobian."""
    T, d = 4, 6
    phi = ToyAttnSublayer(d, seed=4).eval()
    torch.manual_seed(46)
    h = torch.randn(T, d)

    J_full = full_jacobian_oracle(phi, h)
    bj = build_jacobian(phi, h, scope="causal")
    for i in range(T):
        for k in range(i + 1):
            diff = (bj[(i, k)] - J_full[i, :, k, :]).abs().max().item()
            assert diff < 1e-5, f"(i={i}, k={k}) diff={diff:.2e}"


def test_ffn_sublayer_is_token_wise_in_jacobian() -> None:
    """FFN is token-wise; autograd oracle confirms (i, k) = 0 for k != i.
    Our build_jacobian with scope='full' agrees bit-for-bit."""
    T, d = 4, 5
    phi = ToyFFNSublayer(d, d_ffn=8, seed=5).eval()
    torch.manual_seed(47)
    h = torch.randn(T, d)

    J_full = full_jacobian_oracle(phi, h)
    bj = build_jacobian(phi, h, scope="full")
    for i in range(T):
        for k in range(T):
            diff = (bj[(i, k)] - J_full[i, :, k, :]).abs().max().item()
            assert diff < 1e-5, f"FFN (i={i}, k={k}) diff={diff:.2e}"
            if k != i:
                # Oracle says FFN is token-wise — off-diagonal is zero
                assert J_full[i, :, k, :].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Per-block method tests
# ---------------------------------------------------------------------------

def test_slogdet_consistent_with_det() -> None:
    """bj.slogdet(i, i) agrees with torch.linalg.det on the same block."""
    T, d = 3, 4
    phi = ToyFFNSublayer(d, d_ffn=6, seed=6).eval()
    torch.manual_seed(48)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="diagonal")
    for i in range(T):
        sign, logabsdet = bj.slogdet(i, i)
        det = torch.linalg.det(bj[(i, i)].to(torch.float32))
        assert torch.allclose(logabsdet, torch.log(det.abs()), atol=1e-5)
        assert sign.item() == torch.sign(det).item()


def test_svdvals_sorted_and_nonnegative() -> None:
    """bj.svdvals returns monotonically non-increasing non-negative entries."""
    T, d = 3, 5
    phi = ToyAttnSublayer(d, seed=7).eval()
    torch.manual_seed(49)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="diagonal")
    for i in range(T):
        sv = bj.svdvals(i, i)
        assert sv.shape == (d,)
        assert (sv >= 0).all()
        assert ((sv[:-1] - sv[1:]) >= -1e-7).all()


def test_condition_number_geq_one() -> None:
    """For any square matrix, sigma_max / sigma_min >= 1."""
    T, d = 3, 4
    phi = ToyFFNSublayer(d, d_ffn=6, seed=8).eval()
    torch.manual_seed(50)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="diagonal")
    for i in range(T):
        assert bj.condition_number(i, i).item() >= 1.0 - 1e-7


def test_slogdet_equals_sum_log_svdvals() -> None:
    """Matrix identity: log|det J| = sum_j log sigma_j(J)."""
    T, d = 3, 4
    phi = ToyAttnSublayer(d, seed=9).eval()
    torch.manual_seed(51)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="diagonal")
    for i in range(T):
        _, logabsdet = bj.slogdet(i, i)
        sv = bj.svdvals(i, i)
        diff = abs(logabsdet.item() - float(torch.log(sv).sum()))
        assert diff < 1e-4


# ---------------------------------------------------------------------------
# Paper's equation (4): the block-triangular determinant identity
# ---------------------------------------------------------------------------

def test_full_slogdet_via_diagonals_matches_direct_slogdet() -> None:
    """Paper eq. (4) as a numerical check on a toy sublayer:
        slogdet(full J) == (prod_i sign(J^(i)), sum_i log|det J^(i)|).

    If this fails, the entire block-triangular reduction is wrong for us."""
    T, d = 3, 4
    phi = ToyAttnSublayer(d, seed=10).eval()
    torch.manual_seed(52)
    h = torch.randn(T, d)

    # Direct: full matrix slogdet
    J_full = full_jacobian_oracle(phi, h).reshape(T * d, T * d)
    sign_ref, log_ref = torch.linalg.slogdet(J_full.to(torch.float32))

    # Via diagonals using our helper
    bj = build_jacobian(phi, h, scope="diagonal")
    sign_ours, log_ours = bj.full_slogdet()

    assert sign_ref.item() == sign_ours.item()
    assert abs(log_ref.item() - log_ours.item()) < 1e-4


def test_full_slogdet_requires_all_diagonals() -> None:
    """full_slogdet raises if a diagonal block is missing (no silent wrong answer)."""
    T, d = 3, 4
    phi = ToyFFNSublayer(d, seed=11).eval()
    torch.manual_seed(53)
    h = torch.randn(T, d)

    bj = build_jacobian(phi, h, scope="diagonal")
    del bj._blocks[(1, 1)]
    raised = False
    try:
        bj.full_slogdet()
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# evaluate= flag on build_jacobian
# ---------------------------------------------------------------------------

def test_evaluate_full_slogdet_matches_method_call() -> None:
    """build_jacobian(..., evaluate='full_slogdet') returns the same (sign, log) as bj.full_slogdet()."""
    T, d = 3, 4
    phi = ToyFFNSublayer(d, seed=12).eval()
    torch.manual_seed(54)
    h = torch.randn(T, d)

    bj_a = build_jacobian(phi, h, scope="diagonal")
    ref_sign, ref_log = bj_a.full_slogdet()

    bj_b, (sign, log) = build_jacobian(phi, h, scope="diagonal", evaluate="full_slogdet")
    assert sign.item() == ref_sign.item()
    assert abs(log.item() - ref_log.item()) < 1e-7


def test_evaluate_per_diagonal_slogdet_keys_and_values() -> None:
    """evaluate='per_diagonal_slogdet' gives (sign, log) for every diagonal block."""
    T, d = 3, 4
    phi = ToyAttnSublayer(d, seed=13).eval()
    torch.manual_seed(55)
    h = torch.randn(T, d)

    bj, per_diag = build_jacobian(
        phi, h, scope="diagonal", evaluate="per_diagonal_slogdet"
    )
    assert set(per_diag.keys()) == set(range(T))
    for i in range(T):
        sign_ref, log_ref = bj.slogdet(i, i)
        sign_got, log_got = per_diag[i]
        assert sign_got.item() == sign_ref.item()
        assert abs(log_got.item() - log_ref.item()) < 1e-7


def test_evaluate_per_block_slogdet_spans_causal_keys() -> None:
    """evaluate='per_block_slogdet' with scope='causal' hands back (sign, log) for every (i, k) with k <= i."""
    T, d = 3, 4
    phi = ToyAttnSublayer(d, seed=14).eval()
    torch.manual_seed(56)
    h = torch.randn(T, d)

    bj, per_block = build_jacobian(phi, h, scope="causal", evaluate="per_block_slogdet")
    for i in range(T):
        for k in range(i + 1):
            assert (i, k) in per_block


# ---------------------------------------------------------------------------
# Convenience: sublayer_slogdet
# ---------------------------------------------------------------------------

def test_sublayer_slogdet_matches_full_slogdet_via_build() -> None:
    """sublayer_slogdet(phi, h) == build_jacobian(phi, h, scope='diagonal', evaluate='full_slogdet')[1]."""
    T, d = 3, 4
    phi = ToyFFNSublayer(d, seed=15).eval()
    torch.manual_seed(57)
    h = torch.randn(T, d)

    direct_sign, direct_log = sublayer_slogdet(phi, h)
    _, (ref_sign, ref_log) = build_jacobian(phi, h, scope="diagonal", evaluate="full_slogdet")
    assert direct_sign.item() == ref_sign.item()
    assert abs(direct_log.item() - ref_log.item()) < 1e-7
