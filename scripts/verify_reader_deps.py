#!/usr/bin/env python3
"""Verify Hugging Face reader fine-tuning dependencies on the current machine."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from src.reader.finetune_reader import (
            READER_TRAINING_INSTALL_HINT,
            check_reader_training_dependencies,
        )
    except ImportError as exc:
        print(f"ERROR: cannot import project modules: {exc}", file=sys.stderr)
        print("Run from repo root: python scripts/verify_reader_deps.py", file=sys.stderr)
        return 1

    try:
        versions = check_reader_training_dependencies()
    except ImportError as exc:
        print("FAIL: reader training dependencies not ready.", file=sys.stderr)
        print(exc, file=sys.stderr)
        print(f"\nInstall:\n  {READER_TRAINING_INSTALL_HINT}", file=sys.stderr)
        return 1

    print("OK: reader training dependencies")
    for name, version in versions.items():
        print(f"  {name}: {version}")

    try:
        import torch

        print(f"  cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda_device: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
