#!/usr/bin/env python3
"""Test the improved baseline analysis functions."""

import json
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
RESULTS_ROOT = Path("results/baselines")
ESTIMATORS = ["twonn", "ess", "participation_ratio"]

# ── Data loading functions ──────────────────────────────────────────────
def load_baseline_results(results_root: Path):
    """Load all id.jsonl files from baseline experiment runs with robust metadata extraction."""
    all_rows = []
    run_info = []

    # Find all id.jsonl files
    jsonl_files = list(results_root.rglob("id.jsonl"))
    print(f"Found {len(jsonl_files)} id.jsonl files to process")

    for jsonl_path in jsonl_files:
        try:
            # Check if file has content
            if jsonl_path.stat().st_size == 0:
                print(f"  Skipping empty file: {jsonl_path.relative_to(results_root)}")
                continue

            with open(jsonl_path, 'r') as f:
                rows = [json.loads(line) for line in f if line.strip()]

            if not rows:
                print(f"  Skipping file with no valid rows: {jsonl_path.relative_to(results_root)}")
                continue

            # Extract metadata from path and manifest
            rel_path = jsonl_path.relative_to(results_root)
            parts = rel_path.parts

            # Default metadata
            metadata = {
                'run_group': 'unknown',
                'model': 'unknown',
                'run_id': 'unknown',
                'weights': 'unknown',
                'n_layers': 'unknown',
                'd_model': 'unknown'
            }

            # Try to parse from path
            if len(parts) >= 3:
                metadata.update({
                    'run_group': parts[0],
                    'model': parts[1],
                    'run_id': parts[2]
                })

            # Try to get additional metadata from manifest
            manifest_path = jsonl_path.parent / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        config = manifest.get('config', {})
                        metadata.update({
                            'weights': config.get('weights', metadata['weights']),
                            'n_layers': config.get('n_layers', metadata['n_layers']),
                            'd_model': config.get('d_model', metadata['d_model'])
                        })
                except Exception as e:
                    print(f"  Warning: Could not read manifest {manifest_path}: {e}")
            else:
                print(f"  Warning: No manifest found at {manifest_path}")

            # Add metadata to all rows
            for row in rows:
                row.update(metadata)

            all_rows.extend(rows)
            run_info.append({
                'path': str(rel_path),
                'n_rows': len(rows),
                'metadata': metadata
            })

            print(f"  ✓ Loaded {len(rows)} rows from {rel_path}")
            print(f"    Model: {metadata['model']}, Weights: {metadata['weights']}")

        except Exception as e:
            print(f"  ✗ Error loading {jsonl_path.relative_to(results_root)}: {e}")

    df = pd.DataFrame(all_rows)

    # Summary of loaded runs
    print(f"\nLoaded {len(run_info)} successful runs:")
    for info in run_info:
        meta = info['metadata']
        print(f"  - {info['path']}: {info['n_rows']} rows ({meta['model']}, {meta['weights']})")

    return df

# Load data
print("Testing improved baseline analysis...")
print("=" * 50)

try:
    df = load_baseline_results(RESULTS_ROOT)
    print(f"\n✅ Data loading successful!")
    print(f"Total rows loaded: {len(df)}")

    if len(df) > 0:
        # Safe column access with fallbacks
        if 'model' in df.columns:
            print(f"Unique models: {sorted(df['model'].unique())}")
        else:
            print("⚠️  Warning: 'model' column not found")

        if 'baseline' in df.columns:
            print(f"Unique baselines: {sorted(df['baseline'].unique())}")
        else:
            print("⚠️  Warning: 'baseline' column not found")

        if 'depth' in df.columns:
            print(f"Depth range: {df['depth'].min()} to {df['depth'].max()}")
        else:
            print("⚠️  Warning: 'depth' column not found")

        if 'granularity' in df.columns:
            print(f"Granularities: {df['granularity'].unique()}")
        else:
            print("⚠️  Warning: 'granularity' column not found")

        if 'estimator' in df.columns:
            print(f"Estimators: {sorted(df['estimator'].unique())}")
        else:
            print("⚠️  Warning: 'estimator' column not found")
    else:
        print("❌ No data loaded! Check results directory and file formats.")

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Test basic statistics
    if not df.empty and 'id_estimate' in df.columns:
        valid_data = df.dropna(subset=['id_estimate'])
        print(f"\nValid measurements: {len(valid_data)}")
        print(f"ID range: {valid_data['id_estimate'].min():.2f} - {valid_data['id_estimate'].max():.2f}")
        print(f"Mean ID: {valid_data['id_estimate'].mean():.2f} ± {valid_data['id_estimate'].std():.2f}")

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")