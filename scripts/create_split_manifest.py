#!/usr/bin/env python3
"""Create a reviewed, immutable ALQAC split manifest (does not label test untouched)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ExperimentConfig, new_split_manifest
from src.data_loader import load_vilqa_csv

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
config = ExperimentConfig.from_yaml(args.config)
cases = load_vilqa_csv(config.dataset.path)
manifest = new_split_manifest(config.dataset.path, [case.case_id for case in cases], config.dataset.split)
output = args.output or config.dataset.split_manifest
manifest.save(output)
print(f"Wrote {output} ({manifest.counts}). This manifest is frozen/reviewed, not an untouched-test claim.")
