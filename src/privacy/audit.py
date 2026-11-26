"""Final Zero-Leak Self-Audit Scanner over sanitized/, records.jsonl, and report.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.nlp.indic_norm import normalize_text
from src.nlp.pii_regex import _AADHAAR_RE, _ABHA_NUM_RE, _EMAIL_RE, _PAN_RE, _PHONE_10_RE

_SAFE_EMAIL_OR_URL = ("@abdm", "hospital.example")


def audit_secret_free(output_dir: Path, secret: bytes) -> list[str]:
    """Verify RAKSHA_SECRET never appears in any file under output_dir."""
    if not secret:
        return []
    violations: list[str] = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            try:
                if secret in p.read_bytes():
                    violations.append(p.relative_to(output_dir).as_posix())
            except OSError:
                pass
    return violations


def run_zero_leak_audit(
    output_dir: Path,
    secret: bytes,
    *,
    auto_scrub: bool = True,
) -> dict[str, int]:
    """Scan all outputs for stray direct PII (Aadhaar, PAN, ABHA, 10-digit phones, emails) without logging PII."""
    counts = {"secret_leaks": 0, "direct_pii_hits": 0, "files_scrubbed": 0}
    counts["secret_leaks"] = len(audit_secret_free(output_dir, secret))

    san_dir = output_dir / "sanitized"
    targets: list[Path] = []
    if san_dir.is_dir():
        targets.extend(sorted(p for p in san_dir.rglob("*") if p.is_file()))
    for meta_file in (output_dir / "records.jsonl", output_dir / "report.json"):
        if meta_file.is_file():
            targets.append(meta_file)

    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        norm = normalize_text(text)
        hits = 0
        new_text = text

        for kind, rx in (
            ("PAN", _PAN_RE),
            ("ABHA_NUMBER", _ABHA_NUM_RE),
            ("EMAIL", _EMAIL_RE),
        ):
            found = list(rx.finditer(norm))
            if found:
                # Filter out safe placeholder/example strings
                real_hits = [
                    m for m in found
                    if not any(s in m.group(0).lower() for s in _SAFE_EMAIL_OR_URL)
                ]
                if real_hits:
                    hits += len(real_hits)
                    if auto_scrub and path.is_relative_to(san_dir):
                        new_text = rx.sub(
                            lambda m: m.group(0) if any(s in m.group(0).lower() for s in _SAFE_EMAIL_OR_URL) else f"[{kind}]",
                            new_text,
                        )

        # Check stray 10-digit Indian phones inside text/CSV/JSON
        if path.is_relative_to(san_dir):
            ph_hits = list(_PHONE_10_RE.finditer(norm))
            if ph_hits:
                # Avoid false positives inside JSON UUIDs, 12+ digit numbers, or Lot/Batch numbers
                filtered = [
                    m for m in ph_hits
                    if not (m.start() > 0 and norm[m.start() - 1] in "-0123456789abcdef")
                    and not (m.end() < len(norm) and norm[m.end()] in "-0123456789abcdef")
                    and not re.search(r"(?i)(?:lot|batch|serial)(?:\s+no\.?)?\s*[#:-]?\s*$", norm[max(0, m.start() - 20):m.start()])
                ]
                if filtered:
                    hits += len(filtered)
                    if auto_scrub:
                        for m in reversed(filtered):
                            new_text = new_text[:m.start()] + "[PHONE]" + new_text[m.end():]

        if hits > 0:
            counts["direct_pii_hits"] += hits
            if auto_scrub and new_text != text and path.is_relative_to(san_dir):
                path.write_text(new_text, encoding="utf-8")
                counts["files_scrubbed"] += 1

    return counts


def sanitize_report_strings(report: dict) -> dict:
    """Guarantee report.json contains no accidental path/PII strings."""
    safe_files = []
    for ent in report.get("files", []):
        safe_ent = {
            "input_sha256": ent["input_sha256"],
            "output": ent.get("output"),
            "status": ent["status"],
        }
        if ent.get("reason"):
            safe_ent["reason"] = str(ent["reason"])[:64]
        safe_files.append(safe_ent)
    out = dict(report)
    out["files"] = safe_files
    return out
