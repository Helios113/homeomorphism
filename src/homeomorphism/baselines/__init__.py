"""Baseline interventions framework.

Modular, extensible system for baseline experiments on transformer models.

Key components:
  - config: Unified configuration for baseline experiments
  - capture: Phase 1 - latent capture and Jacobian computation
  - analysis: Phase 2 - offline analysis (ID, overlap, stats)

Usage:
    from homeomorphism.baselines import BaselineConfig, BaselineGroup, LatentCapture, AnalyzerPipeline
    from homeomorphism.baselines import IDAnalyzer, OverlapAnalyzer
    
    config = BaselineConfig(
        model_name="gpt2",
        baseline_group=BaselineGroup.A,
        memory=MemoryProfile(n_samples=32, max_tokens=16),
    )
    
    # Phase 1: Capture latents
    capture = LatentCapture(config)
    stats = capture.run(baseline="trained")
    
    # Phase 2: Analyze
    pipeline = AnalyzerPipeline([
        IDAnalyzer(
            granularities=["full_stream", "last_token"],
            estimators=["twonn", "ess"],
        ),
        OverlapAnalyzer(k=5),
    ])
    results = pipeline.run(
        h5_path=config.output_root / "latents.h5",
        output_dir=config.output_root,
        baseline="trained",
    )
"""

from __future__ import annotations

from .config import (
    BaselineConfig,
    BaselineGroup,
    MemoryProfile,
    ModelSpec,
    MODEL_SPECS,
    Granularity,
    EstimatorName,
    BASELINE_GROUP_A,
    BASELINE_GROUP_B,
    BASELINE_GROUPS,
)
from .capture import LatentCapture, HDF5Store
from .analysis import (
    BaselineAnalyzer,
    IDAnalyzer,
    OverlapAnalyzer,
    AnalyzerPipeline,
)

__all__ = [
    # Config
    "BaselineConfig",
    "BaselineGroup",
    "MemoryProfile",
    "ModelSpec",
    "MODEL_SPECS",
    "Granularity",
    "EstimatorName",
    "BASELINE_GROUP_A",
    "BASELINE_GROUP_B",
    "BASELINE_GROUPS",
    # Capture
    "LatentCapture",
    "HDF5Store",
    # Analysis
    "BaselineAnalyzer",
    "IDAnalyzer",
    "OverlapAnalyzer",
    "AnalyzerPipeline",
]
