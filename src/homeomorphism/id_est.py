"""Intrinsic-dimension estimators — placeholder implementations for Exp 2.

The public surface is stable; the bodies are stubs that validate inputs and
return NaN (plus a one-time UserWarning) until real estimators land.

Public API:
  estimate_id(points, method) -> float  — dispatcher
  twonn(points)                         — Facco et al. 2017 max-likelihood ID
  ess(points)                           — effective-sample-size / participation-style ID
  participation_ratio(points)           — (sum lam)^2 / sum lam^2 from sample covariance

Each function takes a 2-D tensor of shape (N, m) (N points in ambient dim m)
and returns a single float — the estimated intrinsic dimension.
"""

from __future__ import annotations

import warnings
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


def twonn(points: torch.Tensor) -> float:
    """Facco et al. (2017) TwoNN estimator. PLACEHOLDER: returns NaN."""
    _validate(points)
    warnings.warn("twonn is a placeholder; returning NaN", stacklevel=2)
    return float("nan")


def ess(points: torch.Tensor) -> float:
    """Effective-sample-size style ID estimator. PLACEHOLDER: returns NaN."""
    _validate(points)
    warnings.warn("ess is a placeholder; returning NaN", stacklevel=2)
    return float("nan")


def participation_ratio(points: torch.Tensor) -> float:
    """Participation ratio (sum lam)^2 / sum lam^2 of the sample covariance.
    PLACEHOLDER: returns NaN."""
    _validate(points)
    warnings.warn("participation_ratio is a placeholder; returning NaN", stacklevel=2)
    return float("nan")


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
