"""4-Pass Project RAKSHA Pipeline Orchestrator."""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

from src.config import RuntimeConfig, load_runtime_config
from src.crypto import sha256_bytes
from src.ingest.csv_handler import parse_csv_file, sanitize_csv_file
from src.ingest.fhir_handler import parse_fhir_bundle, sanitize_fhir_bundle
from src.ingest.json_handler import parse_json_file, sanitize_json_file
from src.ingest.router import check_magic_bytes, decode_safe, detect_file_type, sanitize_rel_output_path
from src.ingest.text_handler import parse_text_file, sanitize_text_file
from src.linkage.cluster import link_and_assign_clusters
from src.models import FileEntry, PatientRecord, RunStats
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiNer
from src.privacy.audit import audit_secret_free, run_zero_leak_audit, sanitize_report_strings
from src.privacy.kanon import apply_kanonymity


def _peak_rss_mb() -> float:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / 1048576 if sys.platform == "darwin" else rss / 1024, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def run_pipeline(input_dir: Path, output_dir: Path, cfg: RuntimeConfig | None = None) -> int:
    t0 = time.monotonic()
    sys.setrecursionlimit(5000)

    cfg = cfg or load_runtime_config()
    guard = MedicalGuard(cfg)
    ner = PiiNer(cfg, guard)

    san_dir = output_dir / "sanitized"
    san_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ PASS 1: Ingest & Extract
    file_entries: list[FileEntry] = []
    all_records: list[PatientRecord] = []

    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(input_dir).as_posix()
        raw = path.read_bytes()
        sha = sha256_bytes(raw)
        try:
            check_magic_bytes(raw)
            text, encoding, has_bom = decode_safe(raw)
            if text is None:
                raise ValueError("undecodable_text")
            ftype = detect_file_type(rel, text)
            if ftype in ("csv", "tsv"):
                fe = parse_csv_file(rel, text, "\t" if ftype == "tsv" else ",", sha, encoding, has_bom)
            elif ftype == "fhir":
                fe = parse_fhir_bundle(rel, text, sha, encoding, has_bom)
            elif ftype == "json":
                fe = parse_json_file(rel, text, sha, encoding, has_bom)
            else:
                fe = parse_text_file(rel, text, sha, encoding, has_bom, guard, ner)

            file_entries.append(fe)
            all_records.extend(fe.records)
        except (ValueError, RecursionError, UnicodeError, json.JSONDecodeError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc).isidentifier() else type(exc).__name__
            file_entries.append(
                FileEntry(
                    rel_input_path=rel,
                    rel_output_path=None,
                    input_sha256=sha,
                    status="quarantined",
                    reason=reason,
                )
            )

    # ------------------------------------------------------------------ PASS 2: Link & Cluster
    link_and_assign_clusters(all_records, cfg)

    # Build in-memory entity lookup maps & per-cluster token sets for Pass 3
    mrn_to_psn: dict[str, str] = {}
    patient_names: set[str] = set()
    relative_names: set[str] = set()
    villages: set[str] = set()
    filename_name_tokens: set[str] = set()
    cluster_tokens_map: dict[str, set[str]] = {}

    for rec in all_records:
        ctoks: set[str] | None = None
        if rec.cluster_id:
            ctoks = cluster_tokens_map.setdefault(rec.cluster_id, set())
        if rec.pseudonym:
            for mrn in rec.all_mrns:
                if mrn:
                    mrn_to_psn[mrn.strip().upper()] = rec.pseudonym
                    if ctoks is not None:
                        ctoks.add(mrn.strip())
        for nm in rec.names_raw:
            patient_names.add(nm)
            for tok in normalize_text(nm).split():
                tok_clean = tok.strip(".,;:")
                if len(tok_clean) >= 4 and not guard.is_medical_or_geo_phrase(tok_clean):
                    filename_name_tokens.add(tok_clean.lower())
                if ctoks is not None and len(tok_clean) >= 3 and not guard.is_medical_or_geo_phrase(tok_clean):
                    ctoks.add(tok_clean)
        for gn in rec.guardian_names_raw:
            relative_names.add(gn)
            if ctoks is not None:
                for tok in normalize_text(gn).split():
                    tok_clean = tok.strip(".,;:")
                    if len(tok_clean) >= 3 and not guard.is_medical_or_geo_phrase(tok_clean):
                        ctoks.add(tok_clean)
        if rec.village:
            villages.add(rec.village)
            if ctoks is not None:
                for tok in normalize_text(rec.village).split():
                    tok_clean = tok.strip(".,;:")
                    if len(tok_clean) >= 3 and not guard.is_medical_or_geo_phrase(tok_clean):
                        ctoks.add(tok_clean)

    for rec in all_records:
        if rec.cluster_id and rec.cluster_id in cluster_tokens_map:
            rec.extra["cluster_tokens"] = cluster_tokens_map[rec.cluster_id]

    ner.register_known_entities(patient_names, relative_names, villages)

    # ------------------------------------------------------------------ PASS 3: k-Anonymity & Sanitize
    apply_kanonymity(all_records, cfg)

    exported_records: list[dict[str, object]] = []
    report_files: list[dict[str, object]] = []
    stats = RunStats(files_total=len(file_entries))

    for fe in file_entries:
        if fe.status != "sanitized":
            stats.files_quarantined += 1
            report_files.append(
                {
                    "input_sha256": fe.input_sha256,
                    "output": None,
                    "status": fe.status,
                    "reason": fe.reason or "unsupported_format",
                }
            )
            continue

        out_rel = sanitize_rel_output_path(fe.rel_input_path, cfg.secret, filename_name_tokens)
        fe.rel_output_path = out_rel

        try:
            if fe.file_type in ("csv", "tsv"):
                out_text = sanitize_csv_file(fe, cfg, guard, ner, mrn_to_psn)
            elif fe.file_type == "fhir":
                out_text = sanitize_fhir_bundle(fe, cfg, guard, ner)
            elif fe.file_type == "json":
                out_text = sanitize_json_file(fe, cfg, guard, ner, mrn_to_psn)
            else:
                out_text = sanitize_text_file(fe, cfg, guard, ner)

            dest = san_dir / out_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(out_text, encoding="utf-8", newline="")

            out_file_ref = f"sanitized/{out_rel}"
            report_files.append(
                {
                    "input_sha256": fe.input_sha256,
                    "output": out_file_ref,
                    "status": "sanitized",
                }
            )
            stats.files_sanitized += 1
            stats.identifiers_detected += fe.identifiers_detected

            for rec in fe.records:
                entry: dict[str, object] = {
                    "file": out_file_ref,
                    "loc": rec.loc,
                    "patient": rec.pseudonym,
                }
                if rec.was_generalized:
                    entry["status"] = "generalized"
                    stats.records_generalized += 1
                if rec.was_suppressed:
                    stats.records_suppressed += 1
                if rec.pseudonym is not None:
                    stats.records_linked += 1
                exported_records.append(entry)
        except Exception as exc:  # noqa: BLE001
            dest = san_dir / out_rel
            if dest.exists():
                dest.unlink()
            fe.status = "quarantined"
            fe.reason = type(exc).__name__
            stats.files_quarantined += 1
            report_files.append(
                {
                    "input_sha256": fe.input_sha256,
                    "output": None,
                    "status": "quarantined",
                    "reason": fe.reason,
                }
            )

    stats.records_processed = len(exported_records)
    exported_records.sort(key=lambda r: (str(r["file"]), str(r["loc"])))

    # ------------------------------------------------------------------ PASS 4: Export & Audit
    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for row in exported_records:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema": "raksha-report/1",
        "files": report_files,
        "counts": {
            "files_total": stats.files_total,
            "files_sanitized": stats.files_sanitized,
            "files_quarantined": stats.files_quarantined,
            "records_processed": stats.records_processed,
            "identifiers_detected": stats.identifiers_detected,
            "records_linked": stats.records_linked,
            "records_generalized": stats.records_generalized,
            "records_suppressed": stats.records_suppressed,
            "records_quarantined": stats.records_quarantined,
        },
        "timing": {"seconds": round(time.monotonic() - t0, 2)},
        "resources": {"peak_rss_mb": _peak_rss_mb()},
    }
    report = sanitize_report_strings(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit_res = run_zero_leak_audit(output_dir, cfg.secret, auto_scrub=True)
    if audit_res["secret_leaks"] > 0:
        raise RuntimeError("secret_audit_failed")

    print(
        f"files {stats.files_total} sanitized {stats.files_sanitized} "
        f"quarantined {stats.files_quarantined} records {stats.records_processed}"
    )
    return 0
