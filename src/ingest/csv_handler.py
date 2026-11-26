"""CSV and TSV ingestion and sanitization handler."""
from __future__ import annotations

import csv
import io
import re

from src.config import RuntimeConfig
from src.ingest.schema_infer import infer_csv_column_roles
from src.models import FileEntry, PatientRecord
from src.nlp.date_shifter import parse_date_safe, shift_date_str
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiNer
from src.nlp.pii_regex import _AADHAAR_RE, _ABHA_NUM_RE, _MRN_EXPLICIT_RE, _PHONE_10_RE
from src.nlp.redactor import generalize_age_value, generalize_pin_value, redact_free_text


def _parse_gender(val: str) -> str | None:
    v = normalize_text(val).strip().lower()
    if v in ("m", "male", "पुरुष"):
        return "M"
    if v in ("f", "female", "महिला"):
        return "F"
    if v in ("o", "other"):
        return "O"
    return None


def _parse_age_sex(val: str) -> tuple[int | None, str | None]:
    v = normalize_text(val).strip()
    m = re.search(r"(\d{1,3})\s*(?:yrs?|years?|y|वर्ष)?\s*(?:/\s*([mf]|male|female|महिला|पुरुष))?", v, re.I)
    if not m:
        return None, None
    age = int(m.group(1))
    g = _parse_gender(m.group(2)) if m.group(2) else None
    return (age if 0 <= age <= 120 else None), g


def _detect_csv_hints(rel_path: str, header: list[str]) -> tuple[str, str | None]:
    low = rel_path.lower()
    hdr_joined = " ".join(header).lower()
    if "mm/dd/yyyy" in hdr_joined or "arogya_legacy" in low:
        return "mdy", "arogya"
    if "sanjeevani" in low:
        return "dmy", "sanjeevani"
    if "jeevanjyoti" in low:
        return "dmy", "jeevanjyoti"
    if "dhh" in low:
        return "dmy", "dhh"
    return "dmy", None


def parse_csv_file(
    rel_path: str,
    text: str,
    delimiter: str,
    sha256: str,
    encoding: str,
    has_bom: bool,
) -> FileEntry:
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not rows:
        raise ValueError("empty_file")

    header = rows[0]
    body = rows[1:]
    col_roles = infer_csv_column_roles(header, body)
    date_hint, hosp_hint = _detect_csv_hints(rel_path, header)

    is_clean_ref = rel_path.startswith("clean/")
    has_patient_cols = any(
        r in ("mrn", "patient_name", "first_name", "last_name", "phone", "aadhaar", "abha", "pan", "insurance")
        for r in col_roles
    )

    records: list[PatientRecord] = []
    for i, row in enumerate(body):
        loc = f"row:{i}"
        uid = f"{rel_path}::{loc}"
        rec = PatientRecord(
            record_uid=uid,
            rel_input_path=rel_path,
            rel_output_path=rel_path,
            file_type="tsv" if delimiter == "\t" else "csv",
            loc=loc,
            raw_payload=row,
            date_hint=date_hint,
            hospital=hosp_hint,
            has_patient_signal=(not is_clean_ref and has_patient_cols),
        )
        first_part = ""
        last_part = ""
        for idx, cell in enumerate(row):
            role = col_roles[idx] if idx < len(col_roles) else "unknown"
            val = normalize_text(cell).strip()
            if not val:
                continue
            if role == "mrn":
                rec.all_mrns.append(val.upper())
                rec.gov_ids.setdefault("mrn", val.upper())
            elif role == "first_name":
                first_part = val
            elif role == "last_name":
                last_part = val
            elif role == "patient_name":
                rec.names_raw.append(val)
            elif role == "relative_name":
                clean_rel = re.sub(r"(?i)^(?:s/o|d/o|w/o|c/o)\s*", "", val).strip()
                if clean_rel and clean_rel.lower() not in ("na", "n/a"):
                    rec.guardian_names_raw.append(clean_rel)
            elif role == "phone":
                digs = re.sub(r"\D", "", val)
                if len(digs) >= 8:
                    rec.phones.append(digs)
            elif role == "email" and "@" in val:
                rec.emails.append(val.lower())
            elif role == "aadhaar":
                digs = re.sub(r"\D", "", val)
                if len(digs) == 12:
                    rec.gov_ids["aadhaar"] = digs
            elif role == "abha":
                digs = re.sub(r"\D", "", val)
                if len(digs) == 14:
                    rec.gov_ids["abha"] = digs
            elif role == "pan":
                rec.gov_ids["pan"] = val.upper()
            elif role == "insurance":
                rec.gov_ids["insurance"] = val.upper()
            elif role == "gender":
                rec.gender = _parse_gender(val)
            elif role == "age":
                age_v, _ = _parse_age_sex(val)
                if age_v is not None:
                    rec.reported_age = age_v
            elif role == "age_sex":
                age_v, g_v = _parse_age_sex(val)
                if age_v is not None:
                    rec.reported_age = age_v
                if g_v:
                    rec.gender = g_v
            elif role == "dob":
                d = parse_date_safe(val, hint=date_hint)
                if d:
                    rec.dob = d
            elif role == "village":
                rec.village = val
            elif role == "tehsil":
                rec.tehsil = val
            elif role == "district":
                rec.district = val
            elif role == "pin":
                digs = re.sub(r"\D", "", val)
                if len(digs) == 6:
                    rec.pin_code = digs
            elif role == "address":
                rec.address_raw = val
            elif role == "clinical_date":
                d = parse_date_safe(val, hint=date_hint)
                if d:
                    rec.clinical_dates.append(d)
                    if rec.reference_date is None:
                        rec.reference_date = d
            elif role == "medical_preserve":
                h = normalize_text(header[idx]).lower() if idx < len(header) else ""
                if any(w in h for w in ("procedure", "diagnosis")):
                    rec.diagnoses.append(val)

        if first_part or last_part:
            full = f"{first_part} {last_part}".strip()
            if full:
                rec.names_raw.insert(0, full)

        records.append(rec)

    return FileEntry(
        rel_input_path=rel_path,
        rel_output_path=rel_path,
        input_sha256=sha256,
        status="sanitized",
        file_type="tsv" if delimiter == "\t" else "csv",
        encoding=encoding,
        delimiter=delimiter,
        has_bom=has_bom,
        records=records,
        raw_data=rows,
        header=header,
        col_roles=col_roles,
        date_hint=date_hint,
        hospital_hint=hosp_hint,
    )


def sanitize_csv_file(
    fentry: FileEntry,
    cfg: RuntimeConfig,
    guard: MedicalGuard,
    ner: PiiNer,
    mrn_to_psn: dict[str, str],
) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=fentry.delimiter, lineterminator="\n")

    # Sanitize header if it contains MRNs or patient names (e.g., ward_vitals_sheet.csv)
    clean_header: list[str] = []
    for h in fentry.header:
        hn = normalize_text(h)
        m_mrn = _MRN_EXPLICIT_RE.search(hn)
        if m_mrn:
            mrn_key = m_mrn.group(0).upper()
            rep = mrn_to_psn.get(mrn_key, "[PATIENT_ID]")
            clean_header.append(rep)
            fentry.identifiers_detected += 1
        else:
            # Check if header itself contains a known patient name
            red_h, n_det = redact_free_text(hn, guard, ner, allow_unlabeled_fallback=False)
            clean_header.append(red_h)
            fentry.identifiers_detected += n_det
    is_clean_ref = fentry.rel_input_path.startswith("clean/")
    has_mrn_col = "mrn" in fentry.col_roles
    has_name_col = "patient_name" in fentry.col_roles or "first_name" in fentry.col_roles
    need_pid_col = (not is_clean_ref and not has_mrn_col and not has_name_col and any(r.pseudonym for r in fentry.records))
    if need_pid_col:
        clean_header.append("patient_id")
    w.writerow(clean_header)

    for rec in fentry.records:
        row: list[str] = rec.raw_payload
        if is_clean_ref:
            w.writerow([normalize_text(c) for c in row])
            continue

        extra_terms: list[tuple[str, str]] = []
        for nm in rec.names_raw:
            extra_terms.append((nm, "PATIENT_NAME"))
        for gnm in rec.guardian_names_raw:
            extra_terms.append((gnm, "RELATIVE_NAME"))
        if rec.village:
            extra_terms.append((rec.village, "VILLAGE"))

        out_row: list[str] = []
        psn_written = False
        for idx, cell in enumerate(row):
            role = fentry.col_roles[idx] if idx < len(fentry.col_roles) else "unknown"
            val = normalize_text(cell)
            if not val.strip():
                out_row.append(val)
                continue

            if role == "mrn":
                out_row.append(rec.pseudonym or "[MRN]")
                psn_written = True
                fentry.identifiers_detected += 1
            elif role in ("patient_name", "first_name", "last_name"):
                if not has_mrn_col and not psn_written and rec.pseudonym:
                    out_row.append(rec.pseudonym)
                    psn_written = True
                else:
                    out_row.append("[PATIENT_NAME]")
                fentry.identifiers_detected += 1
            elif role == "relative_name":
                out_row.append("[RELATIVE_NAME]")
                fentry.identifiers_detected += 1
            elif role == "phone":
                out_row.append("[PHONE]")
                fentry.identifiers_detected += 1
            elif role == "email":
                out_row.append("[EMAIL]")
                fentry.identifiers_detected += 1
            elif role == "aadhaar":
                out_row.append("[AADHAAR_MASKED]" if "x" in val.lower() else "[AADHAAR]")
                fentry.identifiers_detected += 1
            elif role == "abha":
                out_row.append("[ABHA_NUMBER]")
                fentry.identifiers_detected += 1
            elif role == "abha_address":
                out_row.append("[ABHA_ADDRESS]")
                fentry.identifiers_detected += 1
            elif role == "pan":
                out_row.append("[PAN]")
                fentry.identifiers_detected += 1
            elif role == "insurance":
                out_row.append("[INSURANCE_ID]")
                fentry.identifiers_detected += 1
            elif role == "dob":
                if rec.age_band:
                    out_row.append(rec.age_band)
                elif rec.dob:
                    ref_y = rec.reference_date.year if rec.reference_date else 2026
                    approx_age = max(0, ref_y - rec.dob.year)
                    out_row.append(generalize_age_value(approx_age, cfg.age_band_years, high_risk=rec.is_rare_risk))
                else:
                    out_row.append("[DOB]")
                fentry.identifiers_detected += 1
            elif role == "age":
                age_v, _ = _parse_age_sex(val)
                if age_v is not None:
                    band = rec.age_band or generalize_age_value(age_v, cfg.age_band_years, high_risk=rec.is_rare_risk)
                    out_row.append(f"{band} वर्ष" if "वर्ष" in val else band)
                else:
                    out_row.append(val)
            elif role == "age_sex":
                age_v, g_v = _parse_age_sex(val)
                if age_v is not None:
                    band = rec.age_band or generalize_age_value(age_v, cfg.age_band_years, high_risk=rec.is_rare_risk)
                    out_row.append(f"{band}/{g_v}" if g_v else band)
                else:
                    out_row.append(val)
            elif role == "pin":
                out_row.append(rec.extra.get("pin_generalized") or generalize_pin_value(val, high_risk=rec.is_rare_risk))
                fentry.identifiers_detected += 1
            elif role in ("village", "tehsil"):
                if rec.was_suppressed:
                    out_row.append("SUPPRESSED")
                elif rec.is_rare_risk and rec.geo_generalized:
                    out_row.append(rec.geo_generalized)
                else:
                    out_row.append(rec.district or rec.geo_generalized or "[VILLAGE]")
                fentry.identifiers_detected += 1
            elif role == "district":
                if rec.was_suppressed:
                    out_row.append("SUPPRESSED")
                elif rec.is_rare_risk and rec.geo_generalized:
                    out_row.append(rec.geo_generalized)
                else:
                    out_row.append(val)
            elif role == "address":
                red, n_det = redact_free_text(
                    val, guard, ner, is_address_field=True, allow_unlabeled_fallback=False,
                    extra_terms=extra_terms, high_risk=rec.is_rare_risk,
                )
                out_row.append(red)
                fentry.identifiers_detected += n_det
            elif role == "clinical_date":
                out_row.append(shift_date_str(val, rec.date_shift_days, hint=rec.date_hint))
            elif role == "doctor":
                out_row.append(val)
            elif role == "medical_preserve":
                # Still guard against sneaky phone/aadhaar inside medical cells
                if _PHONE_10_RE.search(val) or _AADHAAR_RE.search(val) or _ABHA_NUM_RE.search(val):
                    red, n_det = redact_free_text(val, guard, ner, allow_unlabeled_fallback=False)
                    out_row.append(red)
                    fentry.identifiers_detected += n_det
                else:
                    out_row.append(val)
            else:
                red, n_det = redact_free_text(
                    val, guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms,
                    known_patient_tokens=rec.extra.get("cluster_tokens"),
                    shift_days=rec.date_shift_days,
                    date_hint=rec.date_hint, patient_pseudonym=None,
                    age_band_width=cfg.age_band_years, high_risk=rec.is_rare_risk,
                )
                out_row.append(red)
                fentry.identifiers_detected += n_det

        if need_pid_col:
            out_row.append(rec.pseudonym or "")
        elif rec.pseudonym and not psn_written and not has_name_col and out_row:
            out_row[0] = f"{out_row[0]} ({rec.pseudonym})"
        w.writerow(out_row)

    return buf.getvalue()
