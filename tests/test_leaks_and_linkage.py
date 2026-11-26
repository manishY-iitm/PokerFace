"""Track 4 Canary Zero-Leak & Linkage Precision/Recall Verifier (Categories 2 & 4)."""
from __future__ import annotations

import base64
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path


_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def _decode_all_text(root: Path) -> str:
    parts = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts.append(p.name)
        txt = p.read_text(encoding="utf-8", errors="replace")
        parts.append(txt)
        for m in _B64_RE.finditer(txt):
            tok = m.group(0)
            try:
                parts.append(base64.b64decode(tok + "=" * (-len(tok) % 4)).decode("utf-8", errors="ignore"))
            except Exception:  # noqa: BLE001
                pass
    return "\n".join(parts)


def test_zero_canary_and_direct_pii_leaks(chaos_run):
    out = Path(chaos_run["output"])
    manifest = chaos_run["manifest"]

    # 1. Zero leaks in stdout / stderr
    combined_logs = (chaos_run["stdout"] + "\n" + chaos_run["stderr"]).lower()
    for c in manifest["canaries"]:
        assert c.lower() not in combined_logs, f"Canary leaked to logs: {c}"

    # 2. Scan EVERY byte of sanitized/, records.jsonl, and report.json (including decoded Base64!)
    full_out_text = _decode_all_text(out)
    for c in manifest["canaries"]:
        assert c not in full_out_text, f"Canary leaked in output: {c}"

    # 3. Direct PII scan across all 500 patients (phones, aadhaars, pans, full names)
    for pt in manifest["patients"]:
        assert pt["phone"] not in full_out_text, f"Phone leaked for {pt['pid']}"
        assert pt["aadhaar"] not in full_out_text, f"Aadhaar leaked for {pt['pid']}"
        assert pt["pan"] not in full_out_text, f"PAN leaked for {pt['pid']}"
        assert pt["mrn"] not in full_out_text, f"MRN leaked for {pt['pid']}"
        assert pt["full_en"] not in full_out_text, f"Full EN name leaked for {pt['pid']}"
        assert pt["full_hi"] not in full_out_text, f"Full HI name leaked for {pt['pid']}"


def test_linkage_precision_recall_and_quarantine(chaos_run):
    out = Path(chaos_run["output"])
    manifest = chaos_run["manifest"]

    recs_lines = [
        json.loads(ln)
        for ln in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    rec_map = {(r["file"].replace("sanitized/", "", 1), r["loc"]): r["patient"] for r in recs_lines}

    # 1. Verify every non-null PSN appears inside the sanitized file itself!
    file_text_cache: dict[str, str] = {}
    for r in recs_lines:
        if r["patient"] is not None:
            fp = str(out / r["file"])
            if fp not in file_text_cache:
                file_text_cache[fp] = Path(fp).read_text(encoding="utf-8")
            assert r["patient"] in file_text_cache[fp], f"Missing PSN inside {r['file']} ({r['loc']})"

    # 2. Anonymous notes must remain unlinked (patient: None)
    for u in manifest["units"]:
        if u["pid"] is None:
            assert rec_map.get((u["file"], u["loc"])) is None

    # 3. Pairwise Precision (0% false merge rate!) & Cross-Script Recall (>90%)
    by_true = defaultdict(list)
    by_psn = defaultdict(set)
    for u in manifest["units"]:
        if u["pid"] is None:
            continue
        psn = rec_map.get((u["file"], u["loc"]))
        by_true[u["pid"]].append(psn)
        if psn:
            by_psn[psn].add(u["pid"])

    # False merges: ANY pseudonym containing >1 distinct true pid -> MUST be 0!
    merged_clusters = [psn for psn, pids in by_psn.items() if len(pids) > 1]
    assert len(merged_clusters) == 0, f"False merges detected: {len(merged_clusters)}"

    # Explicit trap checks: spouses (PT-0001 vs PT-0002), twins (PT-0003 vs PT-0004), homonyms (PT-0005 vs PT-0006)
    psn_set = lambda pid: {p for p in by_true[pid] if p is not None}
    assert psn_set("PT-0001") and psn_set("PT-0002") and not (psn_set("PT-0001") & psn_set("PT-0002"))
    assert psn_set("PT-0003") and psn_set("PT-0004") and not (psn_set("PT-0003") & psn_set("PT-0004"))
    assert psn_set("PT-0005") and psn_set("PT-0006") and not (psn_set("PT-0005") & psn_set("PT-0006"))

    # Cross-script hero patient (PT-0000: EN CSV + unseen TSV + Initials JSON + FHIR + Hindi TXT) must all share ONE PSN!
    hero_psns = [p for p in by_true["PT-0000"] if p is not None]
    assert len(hero_psns) == len(by_true["PT-0000"])
    assert len(set(hero_psns)) == 1

    # Pairwise recall across all patients
    tp = total_true_pairs = 0
    for pid, psns in by_true.items():
        n = len(psns)
        total_true_pairs += n * (n - 1) // 2
        for a, b in itertools.combinations(psns, 2):
            if a is not None and a == b:
                tp += 1
    recall = tp / total_true_pairs
    assert recall > 0.90, f"Linkage recall too low: {recall:.3f}"

    # 4. Quarantine verification
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    q_entries = [f for f in report["files"] if f["status"] == "quarantined"]
    assert len(q_entries) == len(manifest["quarantine_files"])
    for q_rel in manifest["quarantine_files"]:
        assert not (out / "sanitized" / q_rel).exists(), f"Quarantined file leaked into sanitized/: {q_rel}"
    for qe in q_entries:
        assert qe["output"] is None and qe.get("reason")
