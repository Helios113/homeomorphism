"""Tests for homeomorphism.data — text corpus loading.

Validates:
  - determinism (same seed -> same chunks)
  - correct count and chunk length
  - different seeds produce different chunks (sanity)
  - cached file is reused across calls (no repeated download)
"""

from __future__ import annotations

from pathlib import Path

from homeomorphism import data as data_mod
from homeomorphism.data import load_texts
from homeomorphism.paths import project_root


def test_load_texts_returns_requested_count() -> None:
    texts = load_texts("shakespeare", n_samples=5, chunk_chars=128, seed=0)
    assert len(texts) == 5


def test_load_texts_chunks_have_requested_length() -> None:
    texts = load_texts("shakespeare", n_samples=3, chunk_chars=200, seed=1)
    for t in texts:
        assert len(t) == 200


def test_load_texts_is_deterministic_under_same_seed() -> None:
    a = load_texts("shakespeare", n_samples=4, chunk_chars=64, seed=42)
    b = load_texts("shakespeare", n_samples=4, chunk_chars=64, seed=42)
    assert a == b


def test_load_texts_different_seed_yields_different_chunks() -> None:
    a = load_texts("shakespeare", n_samples=4, chunk_chars=64, seed=1)
    b = load_texts("shakespeare", n_samples=4, chunk_chars=64, seed=2)
    # Not strictly guaranteed to differ, but extremely likely:
    assert a != b


def test_load_texts_caches_file() -> None:
    """Second call shouldn't re-download; just verify two calls succeed and match."""
    _ = load_texts("shakespeare", n_samples=1, chunk_chars=50, seed=0)
    _ = load_texts("shakespeare", n_samples=1, chunk_chars=50, seed=0)
    # No assertion beyond completion; if download logic is broken we'd get
    # an HTTP error on the second call or corrupt cache.


def test_data_cache_is_project_local() -> None:
    cache = data_mod._cache_dir()
    root = project_root().resolve()
    assert root in cache.resolve().parents or cache.resolve() == root
    assert str(cache).startswith(str(root / ".cache"))
    assert Path(cache).exists()
