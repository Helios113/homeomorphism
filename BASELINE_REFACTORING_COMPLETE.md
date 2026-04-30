# Baseline Framework Modernization & Refactoring

**Date**: April 29, 2026  
**Status**: ✅ Complete - All phases implemented and tested

## Summary

The baseline experiment framework has been comprehensively refactored from procedural scripts into a modular, extensible architecture. This enables:

- **Cleaner code**: Class-based design with clear separation of concerns
- **Reusability**: Modular capture, analysis, and visualization components
- **Extensibility**: Plugin-based analyzer system for future features (stats, visualization, etc.)
- **Single source of truth**: Unified configuration replaces scattered config across 3 runners
- **Backward compatibility**: Old scripts marked deprecated but still functional

## Architecture Overview

### Modular Structure (`src/homeomorphism/baselines/`)

```
baselines/
├── config.py              # Unified configuration (BaselineConfig, MemoryProfile, etc.)
├── capture.py             # Phase 1: LatentCapture, HDF5Store (capture pipeline)
├── analysis.py            # Phase 2: Analyzer interface, IDAnalyzer, OverlapAnalyzer
├── visualization.py       # VisualizationAnalyzer, plot generators (extensible)
├── statistical_tests.py   # StatisticalAnalyzer framework + utility functions
├── template_analyzers.py  # Templates showing how to extend the framework
└── __init__.py            # Public API exports
```

### Orchestrators

- **New**: [experiments/baseline_runner.py](experiments/baseline_runner.py) - Unified entry point (replaces 3 old runners)
- **New**: [experiments/baseline_configs.py](experiments/baseline_configs.py) - Centralized registry (models, groups, memory profiles)

### Tests

- **New**: [tests/test_baselines_integration.py](tests/test_baselines_integration.py) - Integration tests (11 tests, all passing)

## Migration Guide

### ❌ Old Way (Deprecated)

```bash
# Three separate runners, duplicated config
python experiments/run_baseline_configs_gpu.py --quick
python experiments/run_tiny_model_baselines.py --models gpt2 tiny-gpt2-4l-256d
python experiments/exp3_section2_baselines.py --model gpt2 --baselines trained,topological_initialisation
```

### ✅ New Way (Recommended)

```bash
# Single unified runner, centralized config
python experiments/baseline_runner.py \
    --model gpt2 \
    --baseline-group groupA \
    --n-samples 32 \
    --max-tokens 16 \
    --device cuda

# With custom parameters
python experiments/baseline_runner.py \
    --model tiny-gpt2-4l-256d \
    --baseline-group groupB \
    --granularity last_token \
    --estimator twonn ess participation_ratio \
    --quick  # Reduced samples/tokens for testing
```

## Key Components

### 1. Configuration System

**Before**: Config scattered across multiple files
```python
# Was spread across run_baseline_configs_gpu.py, run_tiny_model_baselines.py, etc.
TINY_MODELS = [...]  # duplicated
BASELINE_GROUPS = {...}  # duplicated
```

**After**: Single source of truth
```python
from homeomorphism.baselines import BaselineConfig, MemoryProfile, BASELINE_GROUPS

config = BaselineConfig(
    model_name="gpt2",
    baseline_group=BASELINE_GROUPS["groupA"],
    memory=MemoryProfile(n_samples=32, max_tokens=16),
)
```

### 2. Capture Pipeline (Phase 1)

**Before**: Mixed into exp3_section2_baselines.py (470 lines)
```python
# Code was procedural, hard to reuse
```

**After**: Modular LatentCapture class
```python
from homeomorphism.baselines import LatentCapture

capture = LatentCapture(config)
stats = capture.run(baseline="trained")
# → Writes HDF5 with latent streams
```

### 3. Analysis Pipeline (Phase 2)

**Before**: Offline analysis mixed with capture logic

**After**: Composable analyzer plugins
```python
from homeomorphism.baselines import AnalyzerPipeline, IDAnalyzer, OverlapAnalyzer

pipeline = AnalyzerPipeline([
    IDAnalyzer(
        granularities=["full_stream", "last_token"],
        estimators=["twonn", "ess"],
    ),
    OverlapAnalyzer(k=5),
    # Future: StatisticalTestAnalyzer, VisualizationAnalyzer, etc.
])

results = pipeline.run(h5_path, output_dir, baseline="trained")
```

### 4. Extensibility

**Template for custom analyzers** (in `template_analyzers.py`):
```python
class MyCustomAnalyzer(BaselineAnalyzer):
    def run(self, h5_path, output_path, baseline):
        # 1. Load data from h5_path or existing JSONL
        # 2. Run analysis
        # 3. Write results to output_path (JSONL or plots)
        return {"analyzer": "MyCustomAnalyzer", ...}
```

**Future analyzer types** (framework ready, implementations TBD):
- `StatisticalComparisonAnalyzer` - t-tests, Mann-Whitney U between baselines
- `AnomalyDetectionAnalyzer` - Outlier detection in ID estimates
- `TrendAnalysisAnalyzer` - Polynomial fitting, trend significance
- `VisualizationAnalyzer` - Heatmaps, interactive dashboards

## File Mapping

| Old File | New Location | Status |
|----------|--------------|--------|
| `experiments/exp3_section2_baselines.py` | `experiments/baseline_runner.py` | ⚠️ Deprecated (backward compat via wrapper) |
| `experiments/run_baseline_configs_gpu.py` | `experiments/baseline_runner.py` | ⚠️ Deprecated |
| `experiments/run_tiny_model_baselines.py` | `experiments/baseline_runner.py` | ⚠️ Deprecated |
| (config scattered) | `experiments/baseline_configs.py` | ✅ New (single source of truth) |
| (capture logic) | `src/homeomorphism/baselines/capture.py` | ✅ New (modular) |
| (analysis logic) | `src/homeomorphism/baselines/analysis.py` | ✅ New (extensible) |
| `experiments/plot_baseline_id.py` | `src/homeomorphism/baselines/visualization.py` | ✅ New (refactored + extensible) |

## Implementation Phases Completed

### ✅ Phase 1: Core Architecture
- Created `baselines/` package with modular design
- Extracted capture logic → `LatentCapture` class
- Extracted analysis logic → `BaselineAnalyzer` interface
- Unified configuration → `BaselineConfig`, `MemoryProfile`

### ✅ Phase 2: Unified Orchestrator
- Created `baseline_runner.py` (single entry point)
- Consolidates all 3 old runners' CLI arguments
- Coordinates capture + analysis phases

### ✅ Phase 3: Extensible Analysis
- Created `IDAnalyzer`, `OverlapAnalyzer` (concrete implementations)
- Created `AnalyzerPipeline` (composition framework)
- Created `visualization.py` (refactored plot_baseline_id.py)
- Created `statistical_tests.py` (framework for future expansions)

### ✅ Phase 4: Centralized Configuration
- Created `baseline_configs.py` with:
  - `ModelRegistry` (all models with metadata)
  - `MemoryProfiles` (pre-tuned settings)
  - `BaselineGroupRegistry` (group presets)

### ✅ Phase 5: Testing & Validation
- Created `test_baselines_integration.py` (11 integration tests)
- Tests verify: config validation, capture initialization, analyzer pipeline, JSONL output
- **All tests passing** ✅

### ✅ Phase 6: Deprecation & Cleanup
- Marked old scripts with deprecation notices
- Added guidance to new `baseline_runner.py`

## Quality Metrics

| Metric | Value |
|--------|-------|
| New lines of modular code | ~1500 |
| Integration tests | 11 (all passing) |
| Test coverage target | ≥80% on baselines/ package |
| Backward compatibility | ✅ Maintained (with deprecation warnings) |
| CLI arguments preserved | ✅ Yes (in baseline_runner.py) |
| Code duplication eliminated | ~300 lines removed |

## Next Steps & Future Expansion

The framework is ready for:

1. **Statistical Tests**: Implement concrete subclasses of `StatisticalComparisonAnalyzer`, `AnomalyDetectionAnalyzer`, etc.
2. **Visualization**: Add heatmaps, interactive Plotly dashboards via `VisualizationAnalyzer` subclasses
3. **New Baselines**: Add new baseline types by extending `build_prepared_input()` in interventions.py
4. **Performance Analysis**: Create custom analyzers for correlation with model properties
5. **Notebook Integration**: Use modular components in Jupyter for interactive analysis

## Usage Examples

### Quick Test
```bash
python experiments/baseline_runner.py --model gpt2 --quick
```

### Standard Run (groupA)
```bash
python experiments/baseline_runner.py \
    --model gpt2 \
    --baseline-group groupA \
    --n-samples 32 \
    --max-tokens 16
```

### Multiple Models (Tiny)
```bash
python experiments/baseline_runner.py \
    --model tiny-gpt2-4l-256d \
    --baseline-group groupB \
    --estimator twonn ess participation_ratio
```

### With Custom Output
```bash
python experiments/baseline_runner.py \
    --model gpt2 \
    --baseline-group groupA \
    --output-root results/custom_baseline_run
```

## Breaking Changes

⚠️ **Minimal breaking changes** - Old scripts still work but are deprecated:

1. Old scripts write to different directories (migrate via path update if needed)
2. Old CLI arguments still accepted via `baseline_runner.py` (transparent)
3. Output format unchanged (same JSONL schemas)

## API Reference

### Import Examples

```python
# Configuration
from homeomorphism.baselines import (
    BaselineConfig, BaselineGroup, MemoryProfile, BASELINE_GROUPS
)

# Pipeline components
from homeomorphism.baselines import LatentCapture, AnalyzerPipeline

# Analyzers
from homeomorphism.baselines import (
    IDAnalyzer, OverlapAnalyzer, BaselineAnalyzer
)

# Visualization (future)
from homeomorphism.baselines.visualization import DepthTrajectoryVisualizer

# Statistical tests (framework ready)
from homeomorphism.baselines.statistical_tests import (
    StatisticalComparisonAnalyzer, compare_distributions
)

# Configs registry
from experiments.baseline_configs import (
    ModelRegistry, MemoryProfiles, BaselineGroupRegistry
)
```

## Documentation

- **API Docstrings**: All classes and functions have comprehensive docstrings
- **Example Code**: See `test_baselines_integration.py` for usage patterns
- **Extension Guide**: See `template_analyzers.py` for how to create custom analyzers

---

**For questions or contributions**, refer to docstrings in:
- `src/homeomorphism/baselines/`
- `experiments/baseline_runner.py`
