"""Track 4 Medical Utility & Contract Preservation Checker (Categories 3 & 5)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_medical_preservation_los_age_bands_and_formats(chaos_run):
    inp = Path(chaos_run["input"])
    out = Path(chaos_run["output"])
    manifest = chaos_run["manifest"]

    # 1. Verify --input directory was NEVER modified
    # Compare current input file count & ensure all non-quarantined files exist in report
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    inp_hashes = _tree_hashes(inp)
    report_shas = {f["input_sha256"] for f in report["files"]}
    assert set(inp_hashes.values()) == report_shas

    # 2. Verify 100% survival of doctors, eponyms, drugs, doses, vitals, H/o HTN, SNOMED/ICD-10
    must_preserve_phrases = [
        "Dr. A. K. Sharma",
        "H/o HTN",
        "Dolo 650 mg",
        "1-0-1",
        "Glycomet 500mg BD",
        "Pan-D",
        "Augmentin",
        "Parkinson's disease",
        "Ringer's lactate",
        "Foley catheter",
        "Widal test",
        "Crohn's",
        "BP 120/80",
        "SpO2 98%",
        "GCS 15/15",
        "373942005",
    ]
    all_san_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted((out / "sanitized").rglob("*"))
        if p.is_file()
    )
    for phrase in must_preserve_phrases:
        assert phrase in all_san_text, f"Medical value lost: {phrase}"

    # 3. Verify 100% exact Hospital Length of Stay (LOS) & truthful age bands in CSV
    san_csv_rows = list(csv.DictReader(io.StringIO((out / "sanitized" / "csv" / "reg_english.csv").read_text(encoding="utf-8"))))
    for idx, (row, pt) in enumerate(zip(san_csv_rows, manifest["patients"])):
        d_adm = dt.datetime.strptime(row["admission_date"], "%d/%m/%Y").date()
        d_dis = dt.datetime.strptime(row["discharge_date"], "%d/%m/%Y").date()
        assert (d_dis - d_adm).days == pt["los"], f"LOS mismatch at row {idx}"

        ab = row["age"]
        true_age = pt["age"]
        if ab in ("*", "SUPPRESSED"):
            continue
        elif ab.endswith("+"):
            assert true_age >= int(ab[:-1])
        else:
            lo, hi = (int(x) for x in ab.split("-"))
            assert lo <= true_age <= hi, f"Age band {ab} excludes {true_age}"

    # 4. Validate ALL exported CSV, TSV, JSON, and FHIR R4 Bundles
    bd_rx = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$")
    for p in sorted((out / "sanitized").rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".csv":
            rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8"))))
            assert len(rows) > 0
        elif p.suffix == ".tsv":
            rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8")), delimiter="\t"))
            assert len(rows) > 0
        elif p.suffix == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("resourceType") == "Bundle":
                urls = {e.get("fullUrl") for e in obj.get("entry", []) if e.get("fullUrl")}
                for e in obj.get("entry", []):
                    res = e["resource"]
                    if res["resourceType"] == "Patient" and "birthDate" in res:
                        assert bd_rx.match(str(res["birthDate"]))
                    if res["resourceType"] == "Encounter":
                        p_start = dt.datetime.fromisoformat(res["period"]["start"]).date()
                        p_end = dt.datetime.fromisoformat(res["period"]["end"]).date()
                        # Check exact LOS preserved in FHIR Encounter too
                        b_idx = int(p.stem.split("_")[1])
                        assert (p_end - p_start).days == manifest["patients"][b_idx]["los"]
                    # Check all urn:uuid references resolve
                    txt_res = json.dumps(res)
                    for ref in re.findall(r'"reference":\s*"(urn:uuid:[^"]+)"', txt_res):
                        assert ref in urls, f"Unresolved FHIR reference {ref} in {p.name}"
