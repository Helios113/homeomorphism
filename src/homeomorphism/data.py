"""Text corpus loading.

Currently supports:
  - "shakespeare": Karpathy's tiny-shakespeare; downloaded to ~/.cache on first use.

Add new corpora by extending `_CORPORA` with a URL and handling the fetch.
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path
from typing import Literal

CorpusName = Literal["shakespeare"]

_CORPORA: dict[CorpusName, str] = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    ),
}


def _cache_dir() -> Path:
    cache = Path.home() / ".cache" / "homeomorphism_data"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download(name: CorpusName) -> str:
    url = _CORPORA[name]
    target = _cache_dir() / f"{name}.txt"
    if not target.exists():
        print(f"[data] downloading {name} from {url}")
        urllib.request.urlretrieve(url, target)
    return target.read_text()


def load_texts(
    name: CorpusName = "shakespeare",
    n_samples: int = 8,
    chunk_chars: int = 512,
    seed: int = 0,
) -> list[str]:
    """Return `n_samples` text chunks of length `chunk_chars`, sampled at
    random character offsets from the named corpus."""
    full = _download(name)
    if len(full) < chunk_chars:
        raise ValueError(
            f"corpus {name} length {len(full)} shorter than chunk_chars={chunk_chars}"
        )
    rng = random.Random(seed)
    max_start = len(full) - chunk_chars
    return [full[s : s + chunk_chars] for s in (rng.randint(0, max_start) for _ in range(n_samples))]
