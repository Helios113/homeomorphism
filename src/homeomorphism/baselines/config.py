"""Baseline configuration framework.

Unified configuration for baseline experiments, replacing scattered dict configs
across multiple orchestrators. Provides:
  - BaselineConfig: main experiment parameters
  - BaselineGroup: group-level settings (groupA, groupB, etc.)
  - MemoryProfile: device-aware compute/memory settings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from homeomorphism.interventions import BaselineName

Granularity = Literal["full_stream", "per_token", "last_token"]
EstimatorName = Literal["twonn", "ess", "participation_ratio"]


@dataclass(frozen=True)
class ModelSpec:
    """Model registry entry for refactored baseline experiments."""

    name: str
    family: str
    arch: str
    layers: int
    hidden_size: int
    default_weights: str = "random_gaussian"
    categories: tuple[str, ...] = ()
    description: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen-2l-20d": ModelSpec(
        name="qwen-2l-20d",
        family="qwen",
        arch="qwen",
        layers=2,
        hidden_size=20,
        default_weights="random_gaussian",
        categories=("tiny", "qwen", "custom"),
        description="Tiny Qwen-style custom model for smoke tests",
    ),
    "qwen-4l-32d": ModelSpec(
        name="qwen-4l-32d",
        family="qwen",
        arch="qwen",
        layers=4,
        hidden_size=32,
        default_weights="random_gaussian",
        categories=("tiny", "qwen", "custom"),
        description="Small Qwen-style custom model",
    ),
    "pythia-2l-20d": ModelSpec(
        name="pythia-2l-20d",
        family="pythia",
        arch="pythia",
        layers=2,
        hidden_size=20,
        default_weights="random_gaussian",
        categories=("tiny", "pythia", "custom"),
        description="Tiny Pythia-style custom model for smoke tests",
    ),
    "pythia-4l-32d": ModelSpec(
        name="pythia-4l-32d",
        family="pythia",
        arch="pythia",
        layers=4,
        hidden_size=32,
        default_weights="random_gaussian",
        categories=("tiny", "pythia", "custom"),
        description="Small Pythia-style custom model",
    ),
}


@dataclass
class MemoryProfile:
    """Device-aware memory and compute settings for a model/experiment type.
    
    Attributes:
        n_samples: Number of input samples to process.
        max_tokens: Maximum sequence length per sample.
        batch_size: Batch size for processing.
        device: Compute device ('cuda' or 'cpu').
        seed: Random seed for reproducibility.
    """
    n_samples: int = 32
    max_tokens: int = 16
    batch_size: int = 4
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self) -> None:
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class BaselineGroup:
    """Configuration for a baseline group (groupA, groupB, etc.).
    
    Attributes:
        name: Group name (e.g., 'groupA', 'groupB').
        baselines: Comma-separated baseline names to run.
        granularities: List of data granularities to analyze.
        estimators: ID estimators to use.
        overlap_k: k for k-NN neighborhood overlap computation.
        description: Human-readable description of this group.
    """
    name: str
    baselines: list[BaselineName]
    granularities: list[Granularity] = field(default_factory=lambda: ["last_token"])
    estimators: list[EstimatorName] = field(default_factory=lambda: ["twonn", "ess"])
    overlap_k: int = 5
    description: str = ""
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("group name cannot be empty")
        if not self.baselines:
            raise ValueError("baselines list cannot be empty")
        if self.overlap_k < 1:
            raise ValueError(f"overlap_k must be >= 1, got {self.overlap_k}")
        
        # Validate granularities
        valid_grans = {"full_stream", "per_token", "last_token"}
        for g in self.granularities:
            if g not in valid_grans:
                raise ValueError(f"invalid granularity {g!r}, expected one of {valid_grans}")
        
        # Validate estimators
        valid_ests = {"twonn", "ess", "participation_ratio"}
        for e in self.estimators:
            if e not in valid_ests:
                raise ValueError(f"invalid estimator {e!r}, expected one of {valid_ests}")


@dataclass
class BaselineConfig:
    """Complete configuration for a baseline experiment.
    
    Combines model info, baseline group, memory profile, and I/O settings.
    Suitable for passing to BaselineRunner or LatentCapture.
    
    Attributes:
        model_name: Name of the model (e.g., 'gpt2', 'tiny-gpt2-4l-256d').
        baseline_group: BaselineGroup defining which baselines/estimators to use.
        corpus: Data source ('shakespeare', 'openwebtext', etc.).
        memory: MemoryProfile with device, batch size, and sample count.
        output_root: Root directory for results.
        weights: Weight mode ('trained', 'random_gaussian', etc.).
        layers_spec: Layer selection string ('all', '0.attn', '0,1', etc.).
    """
    model_name: str
    baseline_group: BaselineGroup
    corpus: str = "shakespeare"
    memory: MemoryProfile = field(default_factory=MemoryProfile)
    output_root: Path | str = Path("results/baselines")
    weights: str = "trained"
    layers_spec: str = "all"
    
    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.corpus:
            raise ValueError("corpus cannot be empty")
        if not self.weights:
            raise ValueError("weights cannot be empty")
        
        # Convert output_root to Path if needed
        if isinstance(self.output_root, str):
            self.output_root = Path(self.output_root)
    
    def run_tag(self) -> str:
        """Generate a unique tag for this configuration for logging/tracking."""
        return (
            f"{self.model_name}/"
            f"{self.baseline_group.name}/"
            f"{self.corpus}_{self.memory.n_samples}s_{self.memory.max_tokens}t"
        )


# ============================================================================
# Preset Baseline Groups
# ============================================================================

BASELINE_GROUP_A = BaselineGroup(
    name="groupA",
    baselines=["trained", "topological_initialisation", "maximum_entropy_injection"],
    granularities=["full_stream", "last_token"],
    estimators=["twonn", "ess"],
    overlap_k=10,
    description="Group A: structural & semantic nulls (requires trained weights)",
)

BASELINE_GROUP_B = BaselineGroup(
    name="groupB",
    baselines=["syntactic_disintegration", "semantic_scrambling"],
    granularities=["last_token"],
    estimators=["twonn", "ess", "participation_ratio"],
    overlap_k=10,
    description="Group B: token corruption baselines (works with any weight mode)",
)

# Registry for quick lookup
BASELINE_GROUPS = {
    "groupA": BASELINE_GROUP_A,
    "groupB": BASELINE_GROUP_B,
}
