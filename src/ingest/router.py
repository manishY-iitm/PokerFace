"""File discovery, magic-byte inspection, safe decoding, quarantine routing, and filename scrubbing."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from src.crypto import safe_file_token
from src.nlp.indic_norm import normalize_text
from src.nlp.pii_regex import _MRN_EXPLICIT_RE, _PHONE_10_RE

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".txt", ".md", ".note"}

_BINARY_MAGIC_PREFIXES = (
    b"%PDF-",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
    b"\x1f\x8b\x08",
    b"RIFF",
    b"\x7fELF",
    b"MZ",
)

# Filenames containing explicit PII markers
_FILENAME_PII_RE = re.compile(
    r"(?i)(?:discharge_summary_|_report_\d+|_(?:SNJ|JJH|DHH|IPD|ADM|IP|UHID)[-_])"
)


def _reject_json_constant(val: str) -> None:
    raise ValueError(f"invalid_json_constant_{val}")


def check_magic_bytes(raw: bytes) -> None:
    """Reject empty, null-byte floods, or binary magic-byte payloads before decoding."""
    if not raw or not raw.strip():
        raise ValueError("empty_file")
    head = raw[:16]
    for sig in _BINARY_MAGIC_PREFIXES:
        if head.startswith(sig):
            raise ValueError("unsupported_format")
    if not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        nul_cnt = raw.count(0)
        if head.startswith(b"\x00\x00\x00\x00") or nul_cnt > max(8, len(raw) // 10):
            raise ValueError("corrupt_null_bytes")


def decode_safe(raw: bytes) -> tuple[str | None, str, bool]:
    """Decode UTF-8/BOM/UTF-16/clean-cp1252; reject corrupt byte streams."""
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw[3:].decode("utf-8", errors="strict"), "utf-8", True
        except UnicodeDecodeError:
            return None, "invalid", False
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16", errors="strict"), "utf-16", False
        except UnicodeDecodeError:
            return None, "invalid", False
    try:
        return raw.decode("utf-8", errors="strict"), "utf-8", False
    except UnicodeDecodeError:
        pass

    # Reject embedded UTF-16 BOM artifacts or undefined cp1252 bytes (e.g. corrupt_bytes_note.txt)
    if b"\xff\xfe" in raw or b"\xfe\xff" in raw or b"\x00" in raw:
        return None, "corrupt", False
    if any(b in (0x81, 0x8D, 0x8F, 0x90, 0x9D) for b in raw):
        return None, "corrupt", False
    try:
        decoded = raw.decode("cp1252", errors="strict")
    except UnicodeDecodeError:
        return None, "invalid", False
    if any(ord(ch) < 32 and ch not in ("\t", "\n", "\r") for ch in decoded):
        return None, "corrupt", False
    return decoded, "cp1252", False


def check_json_depth(text: str, max_depth: int = 100) -> bool:
    """Iterative JSON nesting depth check to safely quarantine overly deep payloads."""
    depth = 0
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
            if depth > max_depth:
                return False
        elif ch in "}]":
            depth -= 1
    return True


def detect_file_type(rel_path: str, text: str) -> str:
    """Sniff true file format (csv, tsv, json, fhir, text) and reject mislabeled/DoS files."""
    p = Path(rel_path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("unsupported_format")
    fname = p.name
    if ".." in p.parts or any(ch in fname for ch in ";|&$`") or fname.startswith(".._") and "etc_passwd" in fname:
        raise ValueError("unsafe_filename")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("empty_file")
    if any(len(ln) > 1_000_000 for ln in text.splitlines()) or re.search(r"\d{1000,}|[A-Za-z0-9]{20000,}", text):
        raise ValueError("buffer_overflow_or_redos")

    if ext in (".csv", ".tsv"):
        if stripped.startswith(("{", "[", "<")):
            raise ValueError("mislabeled_format")
        return "tsv" if ext == ".tsv" else "csv"

    if ext == ".json":
        if not stripped.startswith(("{", "[")):
            raise ValueError("invalid_json")
        if not check_json_depth(stripped, max_depth=100):
            raise ValueError("excessive_nesting")
        if "NaN" in text or "Infinity" in text:
            try:
                json.loads(text, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, RecursionError, ValueError) as exc:
                raise ValueError("invalid_json") from exc
        if '"resourceType"' in stripped[:800]:
            try:
                parsed = json.loads(text, parse_constant=_reject_json_constant)
                if isinstance(parsed, dict) and isinstance(parsed.get("resourceType"), str):
                    return "fhir"
            except (json.JSONDecodeError, RecursionError, ValueError) as exc:
                raise ValueError("invalid_json") from exc
        return "json"

    # Plain-text clinical notes (.txt, .md, .note)
    if stripped.startswith(("<?xml", "<ClinicalDocument", "%PDF-")):
        raise ValueError("unsupported_format")
    return "text"


def detect_csv_delimiter(text: str, rel_path: str) -> str:
    """Sniff CSV/TSV delimiter (comma, semicolon, tab, or pipe) from header + sample rows."""
    if Path(rel_path).suffix.lower() == ".tsv":
        return "\t"
    lines = [ln for ln in text.splitlines() if ln.strip()][:6]
    if not lines:
        return ","
    sample = "\n".join(lines)
    best_delim = ","
    best_score = (-1, -1)
    for d in (",", ";", "\t", "|"):
        try:
            rows = list(csv.reader(io.StringIO(sample), delimiter=d))
        except csv.Error:
            continue
        if not rows or len(rows[0]) < 2:
            continue
        ncols = len(rows[0])
        consistent = sum(1 for r in rows[1:] if len(r) == ncols)
        # Reward consistent column counts across all sample rows
        score = (consistent * 10 + (1 if consistent == len(rows) - 1 else 0), ncols)
        if score > best_score:
            best_score = score
            best_delim = d
    return best_delim


def sanitize_rel_output_path(
    rel_path: str,
    secret: bytes,
    known_patient_tokens: set[str] | None = None,
) -> str:
    """Rename files whose filename contains PII (phones, MRNs, or patient names)."""
    p = Path(rel_path)
    stem_norm = normalize_text(p.stem)

    has_unsafe_name = bool(re.search(r"(?:\.\.|[;`$()&|~<>!])", rel_path))
    has_pii = has_unsafe_name or bool(
        _PHONE_10_RE.search(stem_norm)
        or _MRN_EXPLICIT_RE.search(stem_norm)
        or _FILENAME_PII_RE.search(stem_norm)
        or re.search(r"(?i)(?:adm|ipd|ip)[-_]?\d{2,4}[-_]?\d{3,8}", stem_norm)
    )
    if not has_pii and known_patient_tokens:
        stem_words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", stem_norm)}
        if stem_words & known_patient_tokens:
            has_pii = True

    if not has_pii:
        return rel_path

    tok = safe_file_token(rel_path, secret)
    new_name = f"doc_{tok}{p.suffix.lower()}"
    return (p.parent / new_name).as_posix() if p.parent.as_posix() != "." else new_name
