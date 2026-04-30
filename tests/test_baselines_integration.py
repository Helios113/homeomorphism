"""Integration tests for the baseline framework.

Tests the complete pipeline:
  1. Configuration loading and validation
  2. LatentCapture with minimal model/data
  3. Analyzer pipeline (ID, overlap)
  4. Output validation (HDF5, JSONL)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import pytest
import torch

from homeomorphism import models
from homeomorphism.baselines import (
    BaselineConfig,
    BaselineGroup,
    MemoryProfile,
    LatentCapture,
    IDAnalyzer,
    OverlapAnalyzer,
    AnalyzerPipeline,
)
from experiments.baseline_configs import ModelRegistry


class TestBaselineConfig:
    """Test configuration loading and validation."""
    
    def test_baseline_config_valid(self):
        """Valid config should load without errors."""
        group = BaselineGroup(
            name="test",
            baselines=["trained"],
        )
        config = BaselineConfig(
            model_name="toy-2l-32d",
            baseline_group=group,
        )
        assert config.model_name == "toy-2l-32d"
        assert config.baseline_group.name == "test"
    
    def test_baseline_config_invalid_model_name(self):
        """Empty model name should raise ValueError."""
        group = BaselineGroup(name="test", baselines=["trained"])
        with pytest.raises(ValueError, match="model_name"):
            BaselineConfig(model_name="", baseline_group=group)
    
    def test_memory_profile_invalid_device(self):
        """Invalid device should raise ValueError."""
        with pytest.raises(ValueError, match="device"):
            MemoryProfile(device="tpu")
    
    def test_memory_profile_invalid_samples(self):
        """Invalid n_samples should raise ValueError."""
        with pytest.raises(ValueError, match="n_samples"):
            MemoryProfile(n_samples=0)


class TestLatentCaptureMock:
    """Test LatentCapture with minimal mocking."""
    
    def test_latent_capture_initialization(self):
        """LatentCapture should initialize with valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = BaselineGroup(name="test", baselines=["trained"])
            config = BaselineConfig(
                model_name="toy-2l-32d",
                baseline_group=group,
                output_root=Path(tmpdir),
                memory=MemoryProfile(n_samples=2, max_tokens=4),
            )
            capture = LatentCapture(config)
            assert capture.config == config
            assert capture.store.path == Path(tmpdir) / "latents.h5"


class TestAnalyzerPipeline:
    """Test analyzer pipeline with mock HDF5 data."""
    
    @pytest.fixture
    def mock_h5_file(self):
        """Create a minimal mock HDF5 file with latent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test_latents.h5"
            
            # Create mock HDF5 with dummy latent data
            with h5py.File(h5_path, "w") as f:
                # Add a baseline group with depth datasets
                baseline_group = f.create_group("trained")
                
                # depth_00, depth_01, depth_02, etc. with shape (N, T, d)
                for depth in [0, 1, 2, 3]:
                    key = f"depth_{depth:02d}"
                    data = torch.randn(10, 4, 32).numpy().astype("float32")
                    baseline_group.create_dataset(key, data=data)
                
                # Store config as attribute
                config = json.dumps({"n_layers": 2, "seq_len": 4, "d_model": 32})
                f.attrs["config"] = config
            
            yield h5_path
    
    def test_id_analyzer_runs(self, mock_h5_file):
        """IDAnalyzer should process HDF5 and produce JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "id_estimates.jsonl"
            
            analyzer = IDAnalyzer(
                granularities=["last_token"],
                estimators=["twonn"],
            )
            
            result = analyzer.run(
                h5_path=mock_h5_file,
                output_path=output_path,
                baseline="trained",
            )
            
            # Check result dict
            assert result["analyzer"] == "IDAnalyzer"
            assert result["baseline"] == "trained"
            assert result["n_rows"] > 0
            
            # Check output file
            assert output_path.exists()
            with output_path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
            assert len(rows) > 0
            assert "id_estimate" in rows[0]
            assert "estimator" in rows[0]
    
    def test_overlap_analyzer_runs(self, mock_h5_file):
        """OverlapAnalyzer should compute neighborhood overlap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overlap.jsonl"
            
            analyzer = OverlapAnalyzer(k=3, granularities=["full_stream"])
            
            result = analyzer.run(
                h5_path=mock_h5_file,
                output_path=output_path,
                baseline="trained",
            )
            
            # Check result dict
            assert result["analyzer"] == "OverlapAnalyzer"
            assert result["n_rows"] > 0
            
            # Check output file
            assert output_path.exists()
            with output_path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
            assert len(rows) > 0
            assert "neighbour_overlap" in rows[0]
            assert 0 <= rows[0]["neighbour_overlap"] <= 1
    
    def test_analyzer_pipeline_chains_analyzers(self, mock_h5_file):
        """Pipeline should run multiple analyzers in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            pipeline = AnalyzerPipeline([
                IDAnalyzer(granularities=["last_token"], estimators=["twonn"]),
                OverlapAnalyzer(k=3),
            ])
            
            results = pipeline.run(
                h5_path=mock_h5_file,
                output_dir=output_dir,
                baseline="trained",
            )
            
            # Check that both analyzers ran
            assert "IDAnalyzer" in results
            assert "OverlapAnalyzer" in results
            assert results["IDAnalyzer"]["n_rows"] > 0
            assert results["OverlapAnalyzer"]["n_rows"] > 0
            
            # Check output files
            assert (output_dir / "idanalyzer.jsonl").exists()
            assert (output_dir / "overlapanalyzer.jsonl").exists()

    def test_advanced_analyzers_smoke(self, mock_h5_file):
        """Advanced analyzers should run on mock data without throwing exceptions."""
        from homeomorphism.baselines.statistical_tests import TrendAnalysisAnalyzer, StatisticalComparisonAnalyzer
        from homeomorphism.baselines.visualization import DepthTrajectoryVisualizer, BaselineComparisonVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create a fake idanalyzer.jsonl to act as phase 2 output
            id_jsonl = output_dir / "idanalyzer.jsonl"
            with id_jsonl.open("w") as f:
                f.write(json.dumps({"baseline": "trained", "depth": 0, "estimator": "twonn", "granularity": "last_token", "id_estimate": 10.5}) + "\n")
                f.write(json.dumps({"baseline": "trained", "depth": 1, "estimator": "twonn", "granularity": "last_token", "id_estimate": 8.2}) + "\n")
                f.write(json.dumps({"baseline": "trained", "depth": 2, "estimator": "twonn", "granularity": "last_token", "id_estimate": 5.1}) + "\n")
            
            pipeline = AnalyzerPipeline([
                TrendAnalysisAnalyzer(),
                StatisticalComparisonAnalyzer(reference_baseline="trained"),
                DepthTrajectoryVisualizer(output_dir="plots"),
                BaselineComparisonVisualizer(output_dir="plots")
            ])
            
            results = pipeline.run(h5_path=mock_h5_file, output_dir=output_dir, baseline="trained")
            
            assert "TrendAnalysisAnalyzer" in results
            assert "error" not in results["TrendAnalysisAnalyzer"]
            assert "DepthTrajectoryVisualizer" in results
            assert "error" not in results["DepthTrajectoryVisualizer"]

class TestBaselineGroupValidation:
    """Test BaselineGroup validation."""
    
    def test_invalid_granularity(self):
        """Invalid granularity should raise ValueError."""
        with pytest.raises(ValueError, match="invalid granularity"):
            BaselineGroup(
                name="test",
                baselines=["trained"],
                granularities=["invalid"],
            )
    
    def test_invalid_estimator(self):
        """Invalid estimator should raise ValueError."""
        with pytest.raises(ValueError, match="invalid estimator"):
            BaselineGroup(
                name="test",
                baselines=["trained"],
                estimators=["invalid"],
            )
    
    def test_valid_granularities_and_estimators(self):
        """Valid granularities/estimators should load."""
        group = BaselineGroup(
            name="test",
            baselines=["trained"],
            granularities=["full_stream", "per_token", "last_token"],
            estimators=["twonn", "ess", "participation_ratio"],
        )
        assert len(group.granularities) == 3
        assert len(group.estimators) == 3


class TestCustomModelFamilies:
    """Test Qwen and Pythia support in the loader and registry."""

    def test_qwen_and_pythia_are_registered(self):
        assert ModelRegistry.get("qwen-2l-20d") is not None
        assert ModelRegistry.get("qwen-4l-32d") is not None
        assert ModelRegistry.get("pythia-2l-20d") is not None
        assert ModelRegistry.get("pythia-4l-32d") is not None

    @pytest.mark.parametrize(
        ("model_name", "expected_arch"),
        [
            ("qwen-2l-20d", "qwen"),
            ("pythia-2l-20d", "pythia"),
        ],
    )
    def test_custom_model_families_load(self, model_name: str, expected_arch: str):
        m = models.load_model(model_name, weights="random_gaussian", device="cpu")
        assert m.arch == expected_arch
        assert models.n_blocks(m) == 2
        assert models.hidden_size(m) == 20
        sub_attn = models.sublayer(m, 0, "attn")
        sub_ffn = models.sublayer(m, 0, "ffn")
        assert sub_attn.hook_path
        assert sub_ffn.hook_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
