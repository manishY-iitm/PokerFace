"""Clinical text note, discharge summary, prescription, and WhatsApp forward handler."""
from __future__ import annotations

import re

from src.config import RuntimeConfig
from src.ingest.csv_handler import _parse_gender
from src.models import FileEntry, PatientRecord
from src.nlp.date_shifter import parse_date_safe
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiNer
from src.nlp.pii_regex import detect_regex_pii
from src.nlp.redactor import redact_free_text

_AGE_EXTRACT_RE = re.compile(
    r"(?iu)(?:Age(?:\s*/\s*Sex)?\s*[:=-]\s*|आयु\s*[:=-]\s*|Report\s+of\s+[^\n]+\s+)(\d{1,3})\s*(?:Yrs?|Years?|Y|वर्ष)?\s*(?:/\s*([MF]|महिला|पुरुष))?"
)
_DOB_EXTRACT_RE = re.compile(
    r"(?iu)(?:DOB|Date\s+of\s+Birth|Janm\s+Tithi|जन्म\s*तिथि)\s*[:=-]\s*([^\n,;|]+)"
)
_DIST_EXTRACT_RE = re.compile(
    r"(?iu)(?:Distt\.?|District|जिला)\s*[:.]?\s*([A-Za-z\u0900-\u097f]+(?:\s+[A-Za-z\u0900-\u097f]+)?)"
)
_DIAG_LINE_RE = re.compile(r"(?iu)(?:Diagnosis|निदान)\s*[:=-]\s*([^\n]+)")


def _detect_text_date_hint(rel_path: str, norm: str) -> str:
    low = (rel_path + "\n" + norm[:300]).lower()
    if "arogya" in low or "nursing_adm" in low or "adm0" in low:
        return "mdy"
    return "dmy"


def parse_text_file(
    rel_path: str,
    text: str,
    sha256: str,
    encoding: str,
    has_bom: bool,
    guard: MedicalGuard,
    ner: PiiNer,
) -> FileEntry:
    norm = normalize_text(text)
    loc = "doc"
    is_clean = rel_path.startswith("clean/") or rel_path.endswith("README_export.txt")
    is_visitor = "visitor_log" in rel_path.lower() or norm.lstrip().upper().startswith("VISITOR LOG")

    date_hint = _detect_text_date_hint(rel_path, norm)
    rec = PatientRecord(
        record_uid=f"{rel_path}::{loc}",
        rel_input_path=rel_path,
        rel_output_path=rel_path,
        file_type="text",
        loc=loc,
        raw_payload=norm,
        date_hint=date_hint,
        has_patient_signal=(not is_clean and not is_visitor),
    )

    if rec.has_patient_signal:
        protected = guard.find_protected_spans(norm)
        regex_hits = detect_regex_pii(norm, protected)
        ner_hits = ner.detect(norm, protected, allow_unlabeled_fallback=False)

        for hit in regex_hits + ner_hits:
            if hit.kind == "PATIENT_NAME":
                # Strip honorific suffix when storing name for linkage
                cleaned = re.sub(
                    r"(?i)\s+(?:ji|sahab|saheb|didi|behenji|bahenji|bhai|bhaiya|babu|amma|chachu|mama|papa)$",
                    "",
                    hit.norm_value.strip(),
                ).strip()
                if cleaned:
                    rec.names_raw.append(cleaned)
            elif hit.kind == "RELATIVE_NAME":
                cleaned = re.sub(
                    r"(?i)\s+(?:bhaiya|chachu|mama|papa|ji|sahab)$",
                    "",
                    hit.norm_value.strip(),
                ).strip()
                if cleaned:
                    rec.guardian_names_raw.append(cleaned)
            elif hit.kind == "PHONE":
                digs = re.sub(r"\D", "", hit.norm_value)
                if len(digs) >= 8:
                    rec.phones.append(digs)
            elif hit.kind == "AADHAAR":
                digs = re.sub(r"\D", "", hit.norm_value)
                if len(digs) == 12:
                    rec.gov_ids["aadhaar"] = digs
            elif hit.kind == "ABHA_NUMBER":
                digs = re.sub(r"\D", "", hit.norm_value)
                if len(digs) == 14:
                    rec.gov_ids["abha"] = digs
            elif hit.kind == "PAN":
                rec.gov_ids["pan"] = hit.norm_value.upper()
            elif hit.kind == "EMAIL":
                rec.emails.append(hit.norm_value.lower())
            elif hit.kind == "MRN":
                rec.all_mrns.append(hit.norm_value.upper())
                rec.gov_ids.setdefault("mrn", hit.norm_value.upper())
            elif hit.kind == "VILLAGE":
                rec.village = hit.norm_value.strip()
            elif hit.kind == "PIN":
                rec.pin_code = hit.norm_value.strip()

        m_age = _AGE_EXTRACT_RE.search(norm)
        if m_age:
            av = int(m_age.group(1))
            if 0 <= av <= 120:
                rec.reported_age = av
            if m_age.group(2):
                rec.gender = _parse_gender(m_age.group(2))

        m_dob = _DOB_EXTRACT_RE.search(norm)
        if m_dob:
            d = parse_date_safe(m_dob.group(1), hint=date_hint)
            if d:
                rec.dob = d

        m_dist = _DIST_EXTRACT_RE.search(norm)
        if m_dist:
            rec.district = m_dist.group(1).strip()

        m_diag = _DIAG_LINE_RE.search(norm)
        if m_diag:
            rec.diagnoses.append(m_diag.group(1).strip())

        # Reference date for age-at-encounter comparisons
        for line in norm.splitlines()[:10]:
            if any(k in line.lower() for k in ("date", "तारीख", "visit:", "admission", "भर्ती")):
                d = parse_date_safe(line, hint=date_hint)
                if d:
                    rec.clinical_dates.append(d)
                    if rec.reference_date is None:
                        rec.reference_date = d

    return FileEntry(
        rel_input_path=rel_path,
        rel_output_path=rel_path,
        input_sha256=sha256,
        status="sanitized",
        file_type="text",
        encoding=encoding,
        has_bom=has_bom,
        records=[rec],
        raw_data=norm,
    )


def sanitize_text_file(
    fentry: FileEntry,
    cfg: RuntimeConfig,
    guard: MedicalGuard,
    ner: PiiNer,
) -> str:
    if fentry.rel_input_path.startswith("clean/") or fentry.rel_input_path.endswith("README_export.txt"):
        return fentry.raw_data

    rec = fentry.records[0]
    extra_terms: list[tuple[str, str]] = (
        [(nm, "PATIENT_NAME") for nm in rec.names_raw]
        + [(gn, "RELATIVE_NAME") for gn in rec.guardian_names_raw]
    )
    if rec.village:
        extra_terms.append((rec.village, "VILLAGE"))

    sanitized, n_det = redact_free_text(
        fentry.raw_data,
        guard,
        ner,
        allow_unlabeled_fallback=True,
        extra_terms=extra_terms,
        known_patient_tokens=rec.extra.get("cluster_tokens"),
        shift_days=rec.date_shift_days,
        date_hint=rec.date_hint,
        patient_pseudonym=rec.pseudonym,
        age_band_width=cfg.age_band_years,
        high_risk=rec.is_rare_risk,
        generalize_inline_age=True,
    )
    fentry.identifiers_detected += n_det

    # Ensure the assigned pseudonym explicitly appears inside the sanitized record
    if rec.pseudonym and rec.pseudonym not in sanitized:
        sanitized = f"Patient: {rec.pseudonym}\n" + sanitized

    return sanitized
