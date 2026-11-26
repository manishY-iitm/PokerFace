"""Shared pytest fixtures for the Track 4 Red-Team Evaluation Suite."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.gen_chaos_data import generate_chaos_dataset


@pytest.fixture(scope="session")
def chaos_run(tmp_path_factory):
    """Generate 500-patient chaos dataset and run main.py once for the session."""
    base = tmp_path_factory.mktemp("raksha_chaos")
    inp, manifest_path = generate_chaos_dataset(base, num_patients=500, seed=42)
    out_alpha = base / "out_alpha"
    out_alpha.mkdir()

    env = dict(os.environ, RAKSHA_SECRET="redteam-secret-alpha", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--input", str(inp), "--output", str(out_alpha), "--config", str(ROOT / "config" / "policy.yaml")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.monotonic() - t0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "base": base,
        "input": inp,
        "output": out_alpha,
        "manifest": manifest,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed": elapsed,
    }
