"""Intrinsic-dimension estimators for Exp 2.

Public API:
  estimate_id(points, method) -> float  — dispatcher
  twonn(points)                         — Facco et al. 2017 max-likelihood ID
    ess(points)                           — entropy effective-rank style ID
  participation_ratio(points)           — (sum lam)^2 / sum lam^2 from sample covariance

Each function takes a 2-D tensor of shape (N, m) (N points in ambient dim m)
and returns a single float — the estimated intrinsic dimension.
"""

from __future__ import annotations

from typing import Literal

import torch

EstimatorName = Literal["twonn", "ess", "participation_ratio"]


def _validate(pts: torch.Tensor) -> tuple[int, int]:
    """Check (N, m) shape with N >= 2. Return (N, m)."""
    if pts.dim() != 2:
        raise ValueError(f"point cloud must be 2-D (N, m); got shape {tuple(pts.shape)}")
    N, m = pts.shape
    if N < 2:
        raise ValueError(f"need >= 2 points for ID estimation; got N={N}")
    return int(N), int(m)


def _cov_eigvals(points: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of sample covariance of centered points (fp64 for stability)."""
    x = points.to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least 2 points to compute covariance")
    cov = (x.T @ x) / float(n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.clamp_min(0.0)
    return eigvals


def twonn(points: torch.Tensor) -> float:
    """Facco et al. (2017) TwoNN MLE estimator.

    Uses pairwise Euclidean distances and nearest/second-nearest ratios.
    """
    N, _m = _validate(points)
    if N < 3:
        raise ValueError(f"TwoNN needs at least 3 points; got N={N}")

    x = points.to(torch.float64)
    dmat = torch.cdist(x, x, p=2)
    dmat.fill_diagonal_(float("inf"))
    nn2 = torch.topk(dmat, k=2, largest=False, dim=1).values
    r1 = nn2[:, 0]
    r2 = nn2[:, 1]

    mask = torch.isfinite(r1) & torch.isfinite(r2) & (r1 > 0) & (r2 > r1)
    if int(mask.sum().item()) < 3:
        raise ValueError("TwoNN has insufficient valid neighbor ratios")

    mu = (r2[mask] / r1[mask]).clamp_min(1.0 + 1e-12)
    logs = torch.log(mu)
    denom = logs.sum()
    if denom <= 0:
        raise ValueError("TwoNN invalid ratio statistics (non-positive log-sum)")
    est = float((logs.numel() - 1) / denom)
    if not (est == est) or est <= 0:
        raise ValueError("TwoNN produced a non-positive or NaN estimate")
    return est


def ess(points: torch.Tensor) -> float:
    """Entropy effective-rank intrinsic dimension from covariance spectrum.

    Defined as exp(H(p)) where p are normalized covariance eigenvalues.
    """
    _validate(points)
    eigvals = _cov_eigvals(points)
    total = eigvals.sum()
    if total <= 0:
        return 0.0
    p = (eigvals / total).clamp_min(1e-18)
    h = -(p * torch.log(p)).sum()
    return float(torch.exp(h).item())


def participation_ratio(points: torch.Tensor) -> float:
    """Participation ratio (sum lam)^2 / sum lam^2 of covariance eigenvalues."""
    _validate(points)
    eigvals = _cov_eigvals(points)
    s1 = eigvals.sum()
    s2 = (eigvals**2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1 / s2).item())


_ESTIMATORS: dict[str, callable] = {
    "twonn": twonn,
    "ess": ess,
    "participation_ratio": participation_ratio,
}


def estimate_id(points: torch.Tensor, method: EstimatorName) -> float:
    """Dispatch to one of the registered estimators."""
    if method not in _ESTIMATORS:
        raise ValueError(
            f"unknown estimator {method!r}; choose from {sorted(_ESTIMATORS)}"
        )
    return _ESTIMATORS[method](points)


__all__ = [
    "EstimatorName",
    "estimate_id",
    "twonn",
    "ess",
    "participation_ratio",
]
