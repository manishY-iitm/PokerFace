"""Track 4 Determinism, Secret Sensitivity & Resource Limits Verifier (Category 6)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _output_digests(out_dir: Path) -> dict[str, str]:
    digs = {}
    for sub in ("sanitized", "records.jsonl"):
        p = out_dir / sub
        if p.is_file():
            digs[sub] = hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file():
                    digs[fp.relative_to(out_dir).as_posix()] = hashlib.sha256(fp.read_bytes()).hexdigest()
    return digs


def _run_with_secret(inp: Path, out: Path, secret: str) -> subprocess.CompletedProcess[str]:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RAKSHA_SECRET=secret, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--input", str(inp), "--output", str(out), "--config", str(ROOT / "config" / "policy.yaml")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_determinism_secret_sensitivity_and_resource_limits(chaos_run):
    inp = Path(chaos_run["input"])
    out1 = Path(chaos_run["output"])
    base = Path(chaos_run["base"])

    # Run 2: SAME secret ('redteam-secret-alpha') -> byte-for-byte identical outputs!
    out2 = base / "out_alpha_2"
    _run_with_secret(inp, out2, "redteam-secret-alpha")
    assert _output_digests(out1) == _output_digests(out2)

    # Run 3: DIFFERENT secret ('redteam-secret-beta') -> 100% of PSNs & clinical dates changed!
    out3 = base / "out_beta"
    _run_with_secret(inp, out3, "redteam-secret-beta")

    recs1 = [json.loads(ln) for ln in (out1 / "records.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs3 = [json.loads(ln) for ln in (out3 / "records.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    psns1 = {r["patient"] for r in recs1 if r["patient"]}
    psns3 = {r["patient"] for r in recs3 if r["patient"]}
    assert psns1 and psns3
    assert len(psns1 & psns3) == 0, "Pseudonyms collided across different RAKSHA_SECRET!"

    # Verify clinical dates also changed between secret alpha and beta
    csv1 = (out1 / "sanitized" / "csv" / "reg_english.csv").read_text(encoding="utf-8")
    csv3 = (out3 / "sanitized" / "csv" / "reg_english.csv").read_text(encoding="utf-8")
    assert csv1 != csv3

    # Verify runtime (<60s) and peak RSS (<500 MB) on ~2,000 records
    report1 = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    assert chaos_run["elapsed"] < 60.0
    assert report1["resources"]["peak_rss_mb"] < 500.0
