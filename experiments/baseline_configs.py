"""Centralized registry for baseline experiment configurations.

Single source of truth for:
  - Model definitions (name → architecture, parameters, memory)
  - Baseline group presets (groupA, groupB, custom groups)
  - Memory profiles (ultra-conservative, moderate, aggressive)

This replaces scattered config in run_baseline_configs_gpu.py,
run_tiny_model_baselines.py, run_comprehensive_experiments.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from homeomorphism.baselines import BaselineGroup, MemoryProfile, BASELINE_GROUPS


# ============================================================================
# Model Registry
# ============================================================================

@dataclass
class ModelInfo:
    """Metadata about a model.
    
    Attributes:
        name: Model identifier (e.g., 'gpt2', 'tiny-gpt2-4l-256d')
        arch: Architecture type ('gpt2', 'llama', 'custom_toy')
        n_params: Approximate parameter count in millions
        default_weights: Default weight mode ('trained', 'random_gaussian')
        categories: Tags for filtering (e.g., 'standard', 'tiny', 'llama')
    """
    name: str
    arch: str
    n_params: int
    default_weights: str = "trained"
    categories: list[str] | None = None


class ModelRegistry:
    """Registry of available models with metadata."""
    
    # Standard models
    GPT2 = ModelInfo(
        name="gpt2",
        arch="gpt2",
        n_params=124,
        default_weights="trained",
        categories=["standard", "gpt2"],
    )
    DISTILGPT2 = ModelInfo(
        name="distilgpt2",
        arch="gpt2",
        n_params=82,
        default_weights="trained",
        categories=["standard", "gpt2"],
    )
    
    # Tiny GPT-2 variants
    NANO_GPT2_4L_128D = ModelInfo(
        name="nano-gpt2-4l-128d",
        arch="gpt2",
        n_params=2,
        default_weights="random_gaussian",
        categories=["tiny", "gpt2"],
    )
    TINY_GPT2_4L_256D = ModelInfo(
        name="tiny-gpt2-4l-256d",
        arch="gpt2",
        n_params=8,
        default_weights="random_gaussian",
        categories=["tiny", "gpt2"],
    )
    TINY_GPT2_6L_256D = ModelInfo(
        name="tiny-gpt2-6l-256d",
        arch="gpt2",
        n_params=12,
        default_weights="random_gaussian",
        categories=["tiny", "gpt2"],
    )
    TINY_GPT2_8L_256D = ModelInfo(
        name="tiny-gpt2-8l-256d",
        arch="gpt2",
        n_params=16,
        default_weights="random_gaussian",
        categories=["tiny", "gpt2"],
    )
    MICRO_GPT2_4L_384D = ModelInfo(
        name="micro-gpt2-4l-384d",
        arch="gpt2",
        n_params=18,
        default_weights="random_gaussian",
        categories=["tiny", "gpt2"],
    )
    MICRO_GPT2_6L_384D = ModelInfo(
        name="micro-gpt2-6l-384d",
        arch="gpt2",
        n_params=26,
        default_weights="random_gaussian",
        categories=["tiny", "gpt2"],
    )
    
    # LLaMA-style models
    LLAMA_2L_20D_10M = ModelInfo(
        name="llama-2l-20d-10m",
        arch="llama",
        n_params=1,
        default_weights="trained",
        categories=["llama", "tiny"],
    )
    LLAMA_4L_32D_16M = ModelInfo(
        name="llama-4l-32d-16m",
        arch="llama",
        n_params=2,
        default_weights="trained",
        categories=["llama", "tiny"],
    )

    # Qwen-style custom models
    QWEN_2L_20D = ModelInfo(
        name="qwen-2l-20d",
        arch="qwen",
        n_params=0.01,
        default_weights="random_gaussian",
        categories=["qwen", "tiny"],
    )
    QWEN_4L_32D = ModelInfo(
        name="qwen-4l-32d",
        arch="qwen",
        n_params=0.02,
        default_weights="random_gaussian",
        categories=["qwen", "tiny"],
    )

    # Pythia-style custom models
    PYTHIA_2L_20D = ModelInfo(
        name="pythia-2l-20d",
        arch="pythia",
        n_params=0.01,
        default_weights="random_gaussian",
        categories=["pythia", "tiny"],
    )
    PYTHIA_4L_32D = ModelInfo(
        name="pythia-4l-32d",
        arch="pythia",
        n_params=0.02,
        default_weights="random_gaussian",
        categories=["pythia", "tiny"],
    )
    
    # Toy models for testing
    TOY_2L_32D = ModelInfo(
        name="toy-2l-32d",
        arch="gpt2",
        n_params=0.01,
        default_weights="random_gaussian",
        categories=["toy"],
    )
    TOY_4L_64D = ModelInfo(
        name="toy-4l-64d",
        arch="gpt2",
        n_params=0.05,
        default_weights="random_gaussian",
        categories=["toy"],
    )
    
    # Registry
    _MODELS = [
        GPT2,
        DISTILGPT2,
        NANO_GPT2_4L_128D,
        TINY_GPT2_4L_256D,
        TINY_GPT2_6L_256D,
        TINY_GPT2_8L_256D,
        MICRO_GPT2_4L_384D,
        MICRO_GPT2_6L_384D,
        LLAMA_2L_20D_10M,
        LLAMA_4L_32D_16M,
        QWEN_2L_20D,
        QWEN_4L_32D,
        PYTHIA_2L_20D,
        PYTHIA_4L_32D,
        TOY_2L_32D,
        TOY_4L_64D,
    ]
    
    _BY_NAME = {m.name: m for m in _MODELS}
    _BY_CATEGORY = {}
    
    def __init_subclass__(cls):
        """Build category index."""
        for model in cls._MODELS:
            if model.categories:
                for cat in model.categories:
                    if cat not in cls._BY_CATEGORY:
                        cls._BY_CATEGORY[cat] = []
                    cls._BY_CATEGORY[cat].append(model.name)
    
    @classmethod
    def get(cls, name: str) -> ModelInfo | None:
        """Get model by name."""
        return cls._BY_NAME.get(name)
    
    @classmethod
    def all(cls) -> list[ModelInfo]:
        """Get all registered models."""
        return list(cls._MODELS)
    
    @classmethod
    def by_category(cls, category: str) -> list[str]:
        """Get model names in a category."""
        return cls._BY_CATEGORY.get(category, [])


# Initialize category index
ModelRegistry.__init_subclass__()


# ============================================================================
# Memory Profiles (Pre-tuned for different scenarios)
# ============================================================================

class MemoryProfiles:
    """Pre-tuned memory profiles for different scenarios."""
    
    # Ultra-conservative (maximum compatibility)
    CONSERVATIVE = MemoryProfile(
        n_samples=4,
        max_tokens=4,
        batch_size=1,
        device="cuda",
    )
    
    # Moderate (balanced)
    MODERATE = MemoryProfile(
        n_samples=16,
        max_tokens=16,
        batch_size=2,
        device="cuda",
    )
    
    # Aggressive (for memory-rich setups)
    AGGRESSIVE = MemoryProfile(
        n_samples=64,
        max_tokens=64,
        batch_size=8,
        device="cuda",
    )
    
    # For testing/debugging
    QUICK = MemoryProfile(
        n_samples=2,
        max_tokens=4,
        batch_size=1,
        device="cuda",
    )
    
    @staticmethod
    def for_model_and_experiment(
        model_name: str,
        experiment_type: Literal["baseline", "synthetic"],
        profile: Literal["conservative", "moderate", "aggressive", "quick"] = "moderate",
    ) -> MemoryProfile:
        """Select memory profile based on model and experiment type.
        
        Args:
            model_name: Model to profile
            experiment_type: 'baseline' or 'synthetic'
            profile: Pre-configured profile level
        
        Returns:
            MemoryProfile with tuned parameters
        """
        model = ModelRegistry.get(model_name)
        if not model:
            raise ValueError(f"Unknown model: {model_name!r}")
        
        # Base profile
        if profile == "quick":
            base = MemoryProfiles.QUICK
        elif profile == "conservative":
            base = MemoryProfiles.CONSERVATIVE
        elif profile == "aggressive":
            base = MemoryProfiles.AGGRESSIVE
        else:
            base = MemoryProfiles.MODERATE
        
        # Adjust for model size
        if model.n_params < 1:
            # Toy/micro model: can handle more data
            factor = 4 if profile == "aggressive" else 1
        elif model.n_params < 20:
            # Tiny model
            factor = 2 if profile == "aggressive" else 1
        elif model.n_params < 100:
            # Standard model
            factor = 1
        else:
            # Large model: be more conservative
            factor = 0.5
        
        # Adjust for experiment type
        if experiment_type == "synthetic":
            # Synthetic experiments are lighter (no tokenization)
            factor *= 1.5
        elif experiment_type == "baseline":
            # Baselines need input processing
            factor *= 0.8
        
        return MemoryProfile(
            n_samples=max(1, int(base.n_samples * factor)),
            max_tokens=int(base.max_tokens),
            batch_size=max(1, int(base.batch_size)),
            device=base.device,
            seed=base.seed,
        )


# ============================================================================
# Baseline Group Presets (Custom variants)
# ============================================================================

class BaselineGroupRegistry:
    """Registry of baseline group configurations."""
    
    # Standard groups (already defined in baselines.config)
    GROUP_A = BASELINE_GROUPS["groupA"]
    GROUP_B = BASELINE_GROUPS["groupB"]
    
    # Custom groups for specific research questions
    MINIMAL = BaselineGroup(
        name="minimal",
        baselines=["trained", "maximum_entropy_injection"],
        granularities=["last_token"],
        estimators=["twonn"],
        overlap_k=5,
        description="Minimal baseline group for quick testing",
    )
    
    COMPREHENSIVE = BaselineGroup(
        name="comprehensive",
        baselines=["trained", "topological_initialisation", "maximum_entropy_injection", 
                   "syntactic_disintegration", "semantic_scrambling"],
        granularities=["full_stream", "per_token", "last_token"],
        estimators=["twonn", "ess", "participation_ratio"],
        overlap_k=10,
        description="Comprehensive baseline group (all baselines + granularities)",
    )
    
    _GROUPS = {
        "groupA": GROUP_A,
        "groupB": GROUP_B,
        "minimal": MINIMAL,
        "comprehensive": COMPREHENSIVE,
    }
    
    @classmethod
    def get(cls, name: str) -> BaselineGroup | None:
        """Get baseline group by name."""
        return cls._GROUPS.get(name)
    
    @classmethod
    def all_names(cls) -> list[str]:
        """Get all registered group names."""
        return list(cls._GROUPS.keys())
    
    @classmethod
    def register(cls, name: str, group: BaselineGroup) -> None:
        """Register a new baseline group."""
        cls._GROUPS[name] = group


# ============================================================================
# Convenient Presets for Common Scenarios
# ============================================================================

QUICK_TEST_CONFIG = {
    "models": ["gpt2"],
    "groups": ["groupA"],
    "memory_profile": "quick",
}

STANDARD_BASELINE_CONFIG = {
    "models": ["gpt2", "distilgpt2"],
    "groups": ["groupA", "groupB"],
    "memory_profile": "moderate",
}

COMPREHENSIVE_BASELINE_CONFIG = {
    "models": ModelRegistry.by_category("standard") + ModelRegistry.by_category("tiny"),
    "groups": ["groupA", "groupB"],
    "memory_profile": "moderate",
}

TINY_MODEL_CONFIG = {
    "models": ModelRegistry.by_category("tiny"),
    "groups": ["groupA", "groupB"],
    "memory_profile": "aggressive",
}
