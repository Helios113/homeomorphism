"""Project-local path helpers for cache/temp/artifact storage.

All runtime-generated files should stay inside the repository tree.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root inferred from this package location."""
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    p = project_root() / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_cache_dir() -> Path:
    p = cache_dir() / "homeomorphism_data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_cache_dir() -> Path:
    p = cache_dir() / "huggingface"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tmp_dir() -> Path:
    p = project_root() / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p
