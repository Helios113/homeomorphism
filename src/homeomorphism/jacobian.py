"""T x T grid of sub-Jacobians of a transformer sublayer at a residual stream.

Given a sublayer closure phi implementing h -> h + g(h), `build_jacobian`
constructs the block-lower-triangular Jacobian as a grid of d x d blocks
[J]_{i, k} = d phi(h)_{i, :} / d h_{k, :}. The result is a `BlockJacobian`
object with:
  - indexed access to individual blocks         bj[(i, k)]
  - per-block evaluators                        bj.slogdet(i, k), bj.svdvals(i, k), ...
  - full-matrix evaluators via diagonals        bj.full_slogdet()

`build_jacobian` also takes an optional `evaluate=` flag for eager computation
of common reductions in a single call.

Conventions:
  - Above-diagonal blocks (k > i) are zero by causality; we don't compute them
    by default. Use `scope="full"` to compute them anyway (sanity-check path).
  - All evaluators cast to fp32 internally for numerical stability.
"""

from collections.abc import Callable
from typing import Any, Literal

import torch


# ---------------------------------------------------------------------------
# BlockJacobian: grid + evaluators
# ---------------------------------------------------------------------------

class BlockJacobian:
    """T x T grid of d x d sub-Jacobian blocks for one sublayer at one h."""

    def __init__(
        self,
        blocks: dict[tuple[int, int], torch.Tensor],
        T: int,
        d: int,
    ) -> None:
        self._blocks = blocks
        self.T = T
        self.d = d

    # -- access -----------------------------------------------------------

    def __getitem__(self, key: tuple[int, int]) -> torch.Tensor:
        """Block [J]_{i, k} as d x d tensor. Raises KeyError if not computed."""
        return self._blocks[key]

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._blocks

    def get(self, i: int, k: int) -> torch.Tensor:
        """Block [J]_{i, k}; returns a d x d zero tensor if (k > i) and not stored.
        Raises KeyError for missing lower/diagonal blocks."""
        if (i, k) in self._blocks:
            return self._blocks[(i, k)]
        if k > i:
            dtype = next(iter(self._blocks.values())).dtype
            return torch.zeros(self.d, self.d, dtype=dtype)
        raise KeyError(f"Block ({i}, {k}) not computed and not causally zero")

    def keys(self) -> list[tuple[int, int]]:
        return list(self._blocks.keys())

    def diagonal(self) -> dict[int, torch.Tensor]:
        """{i: J^(i)} for present diagonal blocks."""
        return {i: self._blocks[(i, i)] for i in range(self.T) if (i, i) in self._blocks}

    # -- per-block evaluators ---------------------------------------------

    def slogdet(self, i: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(sign, log|det|) of block (i, k). Requires square block (as all ours are)."""
        J = self._blocks[(i, k)]
        if J.shape[0] != J.shape[1]:
            raise ValueError(f"slogdet requires square block; got shape {tuple(J.shape)}")
        return torch.linalg.slogdet(J.to(torch.float32))

    def svdvals(self, i: int, k: int) -> torch.Tensor:
        """Singular values of block (i, k), sorted descending, fp32."""
        return torch.linalg.svdvals(self._blocks[(i, k)].to(torch.float32))

    def condition_number(self, i: int, k: int) -> torch.Tensor:
        sigmas = self.svdvals(i, k)
        sig_min = sigmas[-1]
        if sig_min.item() == 0.0:
            return torch.tensor(float("inf"), dtype=torch.float32)
        return sigmas[0] / sig_min

    # -- full-matrix evaluators via block-triangular factorization -------

    def full_slogdet(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(sign, log|det|) of the full Td x Td Jacobian using diagonals only.
        Applies: det(full J) = prod_i det(J^(i))."""
        missing = [i for i in range(self.T) if (i, i) not in self._blocks]
        if missing:
            raise ValueError(
                f"full_slogdet requires all T={self.T} diagonal blocks; missing {missing}"
            )
        sign = torch.tensor(1.0, dtype=torch.float32)
        logabsdet = torch.tensor(0.0, dtype=torch.float32)
        for i in range(self.T):
            s, l = self.slogdet(i, i)
            sign = sign * s
            logabsdet = logabsdet + l
        return sign, logabsdet

    # -- repr -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BlockJacobian(T={self.T}, d={self.d}, "
            f"n_blocks={len(self._blocks)})"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _compute_block(
    phi: Callable[[torch.Tensor], torch.Tensor],
    h_input: torch.Tensor,
    output_token: int,
    input_token: int,
) -> torch.Tensor:
    """Compute block [J]_{output_token, input_token} at h_input via jacrev.

    All rows of h other than input_token are treated as detached constants.
    """
    T, _d = h_input.shape
    h_base = h_input.detach()

    def phi_at(h_k: torch.Tensor) -> torch.Tensor:
        rows = [h_base[t] if t != input_token else h_k for t in range(T)]
        h = torch.stack(rows, dim=0)
        return phi(h)[output_token]

    h_k_val = h_base[input_token].clone()
    return torch.func.jacrev(phi_at)(h_k_val)


def build_jacobian(
    phi: Callable[[torch.Tensor], torch.Tensor],
    h_input: torch.Tensor,
    *,
    scope: Literal["diagonal", "causal", "full"] = "causal",
    evaluate: Literal[None, "full_slogdet", "per_block_slogdet", "per_diagonal_slogdet"] = None,
) -> BlockJacobian | tuple[BlockJacobian, Any]:
    """Build the T x T grid of sub-Jacobian blocks of phi at h_input.

    Parameters
    ----------
    phi
        Sublayer closure (T, d) -> (T, d) implementing h -> h + g(h).
    h_input
        Residual stream, shape (T, d).
    scope
        Which blocks to compute:
          "diagonal" - only (i, i); cheapest, sufficient for det(full J).
          "causal"   - all (i, k) with k <= i; every nonzero block in a causal model.
          "full"     - all (i, k), including the causally-zero upper blocks
                       (for sanity-checking causality empirically).
    evaluate
        Optional eager evaluation returned alongside the BlockJacobian:
          None                     - just the BlockJacobian.
          "full_slogdet"           - + (sign, log|det|) of the full Td x Td Jacobian,
                                     requires `scope` covers diagonals.
          "per_diagonal_slogdet"   - + {i: (sign, log|det|)} for each diagonal block.
          "per_block_slogdet"      - + {(i, k): (sign, log|det|)} for every computed block.

    Returns
    -------
    If `evaluate is None`: BlockJacobian.
    Otherwise:             (BlockJacobian, evaluation_result).
    """
    if h_input.dim() != 2:
        raise ValueError(f"h_input must be 2-D (T, d); got shape {tuple(h_input.shape)}")
    T, d = h_input.shape

    blocks: dict[tuple[int, int], torch.Tensor] = {}
    for i in range(T):
        if scope == "diagonal":
            ks: list[int] = [i]
        elif scope == "causal":
            ks = list(range(i + 1))
        elif scope == "full":
            ks = list(range(T))
        else:
            raise ValueError(f"Unknown scope: {scope!r}")
        for k in ks:
            blocks[(i, k)] = _compute_block(phi, h_input, i, k)

    bj = BlockJacobian(blocks, T=T, d=d)

    if evaluate is None:
        return bj
    if evaluate == "full_slogdet":
        return bj, bj.full_slogdet()
    if evaluate == "per_diagonal_slogdet":
        return bj, {i: bj.slogdet(i, i) for i in range(T) if (i, i) in bj}
    if evaluate == "per_block_slogdet":
        return bj, {key: bj.slogdet(*key) for key in bj.keys()}
    raise ValueError(f"Unknown evaluate: {evaluate!r}")


# ---------------------------------------------------------------------------
# Convenience: the primary project quantity — (sign, log|det|) of a sublayer's
# full Jacobian at one residual stream, via the block-triangular factorization.
# ---------------------------------------------------------------------------

def sublayer_slogdet(
    phi: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(sign, log|det|) of the full Td x Td Jacobian of a sublayer phi at h.

    Computes only the T diagonal blocks (the minimum needed) and applies
    det(full J) = prod_i det(J^(i)). This is the headline number recorded
    by the homeomorphism experiments.
    """
    _, result = build_jacobian(phi, h, scope="diagonal", evaluate="full_slogdet")
    return result
