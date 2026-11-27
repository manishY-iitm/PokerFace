"""JSON ingestion and sanitization handler (OPD, registration JSON, labs, misc)."""
from __future__ import annotations

import base64
import binascii
import copy
import gzip
import json
import re
import urllib.parse
from typing import Any

from src.config import RuntimeConfig
from src.ingest.csv_handler import _SIMPLE_SCALAR_RE, _parse_age_sex, _parse_gender, _strip_honorific
from src.ingest.router import _reject_json_constant
from src.ingest.schema_infer import classify_field_name, infer_json_container_roles
from src.models import FileEntry, PatientRecord
from src.nlp.date_shifter import parse_date_safe, shift_date_str
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiNer
from src.nlp.pii_regex import _MRN_EXPLICIT_RE
from src.nlp.redactor import generalize_age_value, generalize_pin_value, redact_free_text


def _extract_from_obj(
    obj: Any,
    rec: PatientRecord,
    parent_key: str = "",
    key_roles: dict[str, str] | None = None,
) -> None:
    if isinstance(obj, dict):
        first_p = ""
        last_p = ""
        for k, v in obj.items():
            role = classify_field_name(str(k), parent_key)
            if role == "unknown" and key_roles:
                role = key_roles.get(str(k), "unknown")
            if isinstance(v, (dict, list)):
                _extract_from_obj(v, rec, parent_key=str(k).lower(), key_roles=key_roles)
                continue
            if v is None:
                continue
            s = normalize_text(str(v)).strip()
            if not s:
                continue
            if role == "mrn":
                rec.all_mrns.append(s.upper())
                rec.gov_ids.setdefault("mrn", s.upper())
            elif role == "first_name":
                first_p = s
            elif role == "last_name":
                last_p = s
            elif role == "patient_name":
                cleaned_nm = _strip_honorific(s)
                if cleaned_nm:
                    rec.names_raw.append(cleaned_nm)
            elif role == "relative_name":
                clean_rel = re.sub(r"(?i)^(?:s/o|d/o|w/o|c/o)\s*", "", s).strip()
                if clean_rel:
                    rec.guardian_names_raw.append(clean_rel)
            elif role == "phone":
                digs = re.sub(r"\D", "", s)
                if len(digs) >= 8:
                    rec.phones.append(digs)
            elif role == "email" and "@" in s:
                rec.emails.append(s.lower())
            elif role == "aadhaar":
                digs = re.sub(r"\D", "", s)
                if len(digs) == 12:
                    rec.gov_ids["aadhaar"] = digs
            elif role == "abha":
                digs = re.sub(r"\D", "", s)
                if len(digs) == 14:
                    rec.gov_ids["abha"] = digs
            elif role == "abha_address" and "@" in s:
                rec.gov_ids["abha_address"] = s.lower()
            elif role == "pan":
                rec.gov_ids["pan"] = s.upper()
            elif role == "insurance":
                if s.lower() not in ("true", "false", "yes", "no", "0", "1", "na", "n/a"):
                    rec.gov_ids["insurance"] = s.upper()
            elif role == "gender":
                rec.gender = _parse_gender(s)
            elif role == "age":
                av, _ = _parse_age_sex(s)
                if av is not None:
                    rec.reported_age = av
            elif role == "age_sex":
                av, gv = _parse_age_sex(s)
                if av is not None:
                    rec.reported_age = av
                if gv:
                    rec.gender = gv
            elif role == "dob":
                d = parse_date_safe(s, hint=rec.date_hint)
                if d:
                    rec.dob = d
            elif role == "village":
                rec.village = s
            elif role == "tehsil":
                rec.tehsil = s
            elif role == "district":
                rec.district = s
            elif role == "state":
                rec.state = s
            elif role == "pin":
                digs = re.sub(r"\D", "", s)
                if len(digs) == 6:
                    rec.pin_code = digs
            elif role == "address":
                rec.address_raw = s
            elif role == "clinical_date":
                d = parse_date_safe(s, hint=rec.date_hint)
                if d:
                    rec.clinical_dates.append(d)
                    if rec.reference_date is None:
                        rec.reference_date = d
            elif role == "medical_preserve" and str(k).lower() in ("display", "procedure", "diagnosis", "nidan", "surgery"):
                rec.diagnoses.append(s)
        if first_p or last_p:
            full = f"{first_p} {last_p}".strip()
            if full:
                if parent_key in ("kin", "relative", "guardian"):
                    rec.guardian_names_raw.insert(0, full)
                else:
                    rec.names_raw.insert(0, full)
    elif isinstance(obj, list):
        for item in obj:
            _extract_from_obj(item, rec, parent_key=parent_key, key_roles=key_roles)


def _find_record_container(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    for k, v in data.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            # Non-patient sub-arrays (panels in diagnostic reports, wards in census, tickets)
            if k.lower() in ("wards", "panels", "rx", "diagnosis", "tickets"):
                return None
            return k, v
    return None


def parse_json_file(
    rel_path: str,
    text: str,
    sha256: str,
    encoding: str,
    has_bom: bool,
) -> FileEntry:
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid_json") from exc

    is_clean = rel_path.startswith("clean/")
    is_meta_index = rel_path.endswith("patient_index.json") or rel_path.endswith("helpdesk_tickets.json") or rel_path.endswith("scanned_consent_form.json")

    records: list[PatientRecord] = []
    if isinstance(data, list):
        key_roles = infer_json_container_roles(data)
        for i, item in enumerate(data):
            loc = f"/{i}"
            rec = PatientRecord(
                record_uid=f"{rel_path}::{loc}",
                rel_input_path=rel_path,
                rel_output_path=rel_path,
                file_type="json",
                loc=loc,
                raw_payload=item,
                has_patient_signal=(not is_clean and isinstance(item, dict)),
            )
            if isinstance(item, dict) and rec.has_patient_signal:
                _extract_from_obj(item, rec, key_roles=key_roles)
            records.append(rec)
        return FileEntry(
            rel_input_path=rel_path,
            rel_output_path=rel_path,
            input_sha256=sha256,
            status="sanitized",
            file_type="json",
            encoding=encoding,
            has_bom=has_bom,
            records=records,
            raw_data=data,
        )

    container = _find_record_container(data) if isinstance(data, dict) and not rel_path.endswith("helpdesk_tickets.json") else None

    if container is None:
        loc = "doc"
        rec = PatientRecord(
            record_uid=f"{rel_path}::{loc}",
            rel_input_path=rel_path,
            rel_output_path=rel_path,
            file_type="json",
            loc=loc,
            raw_payload=data,
            has_patient_signal=(not is_clean and not is_meta_index),
        )
        if isinstance(data, dict) and rec.has_patient_signal:
            _extract_from_obj(data, rec)
        records.append(rec)
    else:
        key_name, items = container
        key_roles = infer_json_container_roles(items)
        pointer_key = key_name.replace("~", "~0").replace("/", "~1")
        for i, item in enumerate(items):
            loc = f"/{pointer_key}/{i}"
            rec = PatientRecord(
                record_uid=f"{rel_path}::{loc}",
                rel_input_path=rel_path,
                rel_output_path=rel_path,
                file_type="json",
                loc=loc,
                raw_payload=item,
                has_patient_signal=not is_clean,
            )
            if not is_clean:
                _extract_from_obj(item, rec, key_roles=key_roles)
            records.append(rec)

    return FileEntry(
        rel_input_path=rel_path,
        rel_output_path=rel_path,
        input_sha256=sha256,
        status="sanitized",
        file_type="json",
        encoding=encoding,
        has_bom=has_bom,
        records=records,
        raw_data=data,
    )


_SAFE_SCHEMA_KEYS = frozenset({
    "system", "hospital", "exported_by", "generated_at", "source_path",
    "_export_meta", "export_meta", "export_info", "record_count", "tool", "rows",
    "visits", "patients", "results", "facility", "schema", "beneficiaries", "members",
    "exported_at", "by_uhid", "by_name", "generated", "wards", "tickets", "map",
    "form", "attachment", "contenttype", "data", "__proto__", "constructor",
    "prototype", "doc_type", "gzip_base64", "double_base64", "hex_payload",
    "url_encoded", "windows_path", "bp", "diagnosis", "village", "pin",
    "district", "location", "contact", "identifiers", "encounter", "patient",
    "extensions", "hinglish_note", "medications", "vitals", "hba1c", "doctor",
    "admit_date", "discharge_date", "visit_id", "mrn", "uhid", "abha", "name",
    "guardian", "age", "gender", "phone", "email", "verified", "insurance",
    "notes", "clinical_notes", "kin", "pt", "beneficiary", "abode", "residence",
    "state_code", "slip", "bed", "diet", "t", "v", "u", "d", "n", "id",
})


def _sanitize_key(
    key: str,
    guard: MedicalGuard,
    ner: PiiNer,
    mrn_to_psn: dict[str, str],
    fentry: FileEntry,
    known_tokens: set[str] | None = None,
) -> str:
    kn = normalize_text(key)
    m = _MRN_EXPLICIT_RE.search(kn)
    if m:
        fentry.identifiers_detected += 1
        psn_rep = mrn_to_psn.get(m.group(0).upper(), "[MRN]")
        # If the key is purely an MRN (or MRN + extra PII like "KR-6361141 Anirban Murmu"),
        # return psn_rep or redact remainder so no patient name leaks in the key.
        rest = (kn[:m.start()] + " " + kn[m.end():]).strip()
        if not rest:
            return psn_rep
        red_rest, n_det = redact_free_text(
            rest, guard, ner, allow_unlabeled_fallback=True, known_patient_tokens=known_tokens,
        )
        fentry.identifiers_detected += n_det
        return f"{psn_rep} {red_rest}".strip() if n_det > 0 else psn_rep
    kn_low = kn.lower()
    if kn_low in _SAFE_SCHEMA_KEYS:
        return key
    # Check underscore-separated unknown keys like patient_Mohammad_Arif_meta
    if not kn_low.startswith("patient_") and classify_field_name(kn) != "unknown":
        return key
    red, n_det = redact_free_text(
        kn, guard, ner, allow_unlabeled_fallback=True, known_patient_tokens=known_tokens,
    )
    if n_det > 0:
        fentry.identifiers_detected += n_det
        return red
    return key


def _scrub_encoded_payload(
    s: str,
    guard: MedicalGuard,
    ner: PiiNer,
    fentry: FileEntry,
    extra_terms: list[tuple[str, str]],
    shift_days: int,
    psn: str | None,
    *,
    depth: int = 0,
) -> str | None:
    """Recursively decode and scrub Base64, Gzip+Base64, Double-Base64, Hex, or URL-encoded PII."""
    if depth > 3 or len(s) < 16:
        return None
    stripped = s.strip()

    # 1. URL-encoded string
    if "%" in stripped and re.search(r"%[0-9a-fA-F]{2}", stripped):
        dec = urllib.parse.unquote(stripped)
        if dec != stripped:
            red, n_det = redact_free_text(
                dec, guard, ner, allow_unlabeled_fallback=True,
                extra_terms=extra_terms, shift_days=shift_days, patient_pseudonym=psn,
            )
            red = re.sub(r"_[0-9]{5,8}\b", "_[ID]", red)
            fentry.identifiers_detected += max(n_det, 1)
            return urllib.parse.quote(red, safe="=&:/? []_-")

    # 2. Hex-encoded UTF-8 string
    if len(stripped) >= 24 and len(stripped) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", stripped):
        try:
            dec = bytes.fromhex(stripped).decode("utf-8", errors="strict")
            if sum(c.isprintable() for c in dec) / len(dec) > 0.9 and any(c.isalpha() for c in dec):
                red, n_det = redact_free_text(
                    dec, guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms, shift_days=shift_days, patient_pseudonym=psn,
                )
                fentry.identifiers_detected += n_det
                return red.encode("utf-8").hex()
        except (ValueError, UnicodeDecodeError):
            pass

    # 3. Base64 / Gzip+Base64 / Double-Base64
    if len(stripped) >= 24 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", stripped):
        try:
            raw_b = base64.b64decode(stripped + "=" * (-len(stripped) % 4), validate=True)
            if raw_b.startswith(b"\x1f\x8b"):
                dec = gzip.decompress(raw_b).decode("utf-8", errors="strict")
                red, n_det = redact_free_text(
                    dec, guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms, shift_days=shift_days, patient_pseudonym=psn,
                )
                fentry.identifiers_detected += n_det
                return base64.b64encode(gzip.compress(red.encode("utf-8"))).decode("ascii")
            dec = raw_b.decode("utf-8", errors="strict")
            if sum(c.isprintable() or c in "\r\n\t" for c in dec) / max(1, len(dec)) > 0.85:
                inner = _scrub_encoded_payload(dec, guard, ner, fentry, extra_terms, shift_days, psn, depth=depth + 1)
                if inner is not None:
                    return base64.b64encode(inner.encode("utf-8")).decode("ascii")
                red, n_det = redact_free_text(
                    dec, guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms, shift_days=shift_days, patient_pseudonym=psn,
                )
                fentry.identifiers_detected += n_det
                return base64.b64encode(red.encode("utf-8")).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    return None


def _sanitize_json_node(
    node: Any,
    rec: PatientRecord | None,
    parent_key: str,
    cfg: RuntimeConfig,
    guard: MedicalGuard,
    ner: PiiNer,
    mrn_to_psn: dict[str, str],
    fentry: FileEntry,
    extra_terms: list[tuple[str, str]],
    key_roles: dict[str, str] | None = None,
    unk_cache: dict[tuple[str, int, bool, str | None], tuple[str, int]] | None = None,
) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        psn = rec.pseudonym if rec else None
        ktoks = rec.extra.get("cluster_tokens") if rec else None
        for k, v in node.items():
            sk = _sanitize_key(str(k), guard, ner, mrn_to_psn, fentry, known_tokens=ktoks)
            role = classify_field_name(str(k), parent_key)
            if role == "unknown" and key_roles:
                role = key_roles.get(str(k), "unknown")

            if isinstance(v, (dict, list)):
                out[sk] = _sanitize_json_node(
                    v, rec, str(k).lower(), cfg, guard, ner, mrn_to_psn, fentry, extra_terms, key_roles=key_roles, unk_cache=unk_cache
                )
                continue

            if v is None or isinstance(v, bool):
                out[sk] = v
                continue

            # Numeric handling (e.g. Aadhaar stored as int under aadhar_no, or age int)
            if isinstance(v, (int, float)):
                if role == "aadhaar":
                    fentry.identifiers_detected += 1
                    out[sk] = "[AADHAAR]"
                elif role == "abha":
                    fentry.identifiers_detected += 1
                    out[sk] = "[ABHA_NUMBER]"
                elif role == "phone":
                    fentry.identifiers_detected += 1
                    out[sk] = "[PHONE]"
                elif role == "age" and isinstance(v, int):
                    out[sk] = (rec.age_band if rec and rec.age_band else generalize_age_value(v, cfg.age_band_years, high_risk=bool(rec and rec.is_rare_risk)))
                elif role == "pin":
                    fentry.identifiers_detected += 1
                    out[sk] = generalize_pin_value(str(int(v)), high_risk=bool(rec and rec.is_rare_risk))
                else:
                    out[sk] = v
                continue

            # String handling
            s = normalize_text(str(v))
            if not s.strip():
                out[sk] = s
                continue

            if role == "metadata_pii" or "path" in str(k).lower() or "url_encoded" in str(k).lower():
                enc_scrubbed = _scrub_encoded_payload(s, guard, ner, fentry, extra_terms, rec.date_shift_days if rec else 0, psn)
                if enc_scrubbed is not None:
                    out[sk] = enc_scrubbed
                    continue
                red, n_det = redact_free_text(s, guard, ner, allow_unlabeled_fallback=True, extra_terms=extra_terms)
                red = re.sub(r"_[0-9]{5,8}\b", "_[ID]", red)
                if "source_path" in str(k).lower() or "\\" in s or "/" in s:
                    red = re.sub(r"(?i)(opd[/\\])([^/\\]+)([/\\])", r"\1[PATIENT_DIR]\3", red)
                    n_det += 1
                fentry.identifiers_detected += n_det
                out[sk] = red
            elif role == "base64_data" or (
                role == "unknown" and (
                    (len(s.strip()) >= 24 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", s.strip()))
                    or (len(s.strip()) >= 24 and len(s.strip()) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", s.strip()))
                )
            ):
                enc_scrubbed = _scrub_encoded_payload(
                    s, guard, ner, fentry, extra_terms,
                    rec.date_shift_days if rec else 0, psn,
                )
                out[sk] = enc_scrubbed if enc_scrubbed is not None else s
            elif role == "mrn":
                fentry.identifiers_detected += 1
                out[sk] = psn or mrn_to_psn.get(s.strip().upper(), "[MRN]")
            elif role in ("patient_name", "first_name", "last_name"):
                fentry.identifiers_detected += 1
                out[sk] = "[PATIENT_NAME]" if role != "patient_name" or (rec and rec.all_mrns) else (psn or "[PATIENT_NAME]")
            elif role == "relative_name":
                fentry.identifiers_detected += 1
                out[sk] = "[RELATIVE_NAME]"
            elif role == "phone":
                fentry.identifiers_detected += 1
                out[sk] = "[PHONE]"
            elif role == "email":
                fentry.identifiers_detected += 1
                out[sk] = "[EMAIL]"
            elif role == "aadhaar":
                fentry.identifiers_detected += 1
                out[sk] = "[AADHAAR]"
            elif role == "abha":
                fentry.identifiers_detected += 1
                out[sk] = "[ABHA_NUMBER]"
            elif role == "abha_address":
                fentry.identifiers_detected += 1
                out[sk] = "[ABHA_ADDRESS]"
            elif role == "pan":
                fentry.identifiers_detected += 1
                out[sk] = "[PAN]"
            elif role == "dob":
                fentry.identifiers_detected += 1
                if rec and rec.age_band:
                    out[sk] = rec.age_band
                elif rec and rec.dob:
                    ref_y = rec.reference_date.year if rec.reference_date else 2026
                    out[sk] = generalize_age_value(max(0, ref_y - rec.dob.year), cfg.age_band_years, high_risk=rec.is_rare_risk)
                else:
                    out[sk] = "[DOB]"
            elif role == "age":
                av, _ = _parse_age_sex(s)
                if av is not None:
                    out[sk] = rec.age_band if rec and rec.age_band else generalize_age_value(av, cfg.age_band_years, high_risk=bool(rec and rec.is_rare_risk))
                else:
                    out[sk] = s
            elif role == "age_sex":
                av, gv = _parse_age_sex(s)
                if av is not None:
                    band = rec.age_band if rec and rec.age_band else generalize_age_value(av, cfg.age_band_years, high_risk=bool(rec and rec.is_rare_risk))
                    out[sk] = f"{band}/{gv}" if gv else band
                else:
                    out[sk] = s
            elif role == "pin":
                fentry.identifiers_detected += 1
                out[sk] = (rec.extra.get("pin_generalized") if rec else None) or generalize_pin_value(s, high_risk=bool(rec and rec.is_rare_risk))
            elif role in ("village", "tehsil"):
                fentry.identifiers_detected += 1
                if rec and rec.was_suppressed:
                    out[sk] = "SUPPRESSED"
                elif rec and rec.is_rare_risk and rec.geo_generalized:
                    out[sk] = rec.geo_generalized
                else:
                    out[sk] = (rec.district if rec and rec.district else (rec.geo_generalized if rec and rec.geo_generalized else "[VILLAGE]"))
            elif role == "district":
                if rec and rec.was_suppressed:
                    out[sk] = "SUPPRESSED"
                else:
                    out[sk] = (rec.geo_generalized if rec and rec.is_rare_risk and rec.geo_generalized else s)
            elif role == "address":
                red, n_det = redact_free_text(
                    s, guard, ner, is_address_field=True, allow_unlabeled_fallback=False,
                    extra_terms=extra_terms, high_risk=bool(rec and rec.is_rare_risk),
                    geo_generalized=rec.geo_generalized if rec else None,
                )
                fentry.identifiers_detected += n_det
                out[sk] = red
            elif role == "clinical_date":
                shift = rec.date_shift_days if rec else 0
                out[sk] = shift_date_str(s, shift, hint=rec.date_hint if rec else "dmy")
            elif role == "doctor":
                out[sk] = s
            elif role == "medical_preserve":
                out[sk] = s
            else:
                # Could be value inside by_name (e.g., "Zainab Siddiqui": "JJH-0077112")
                if _MRN_EXPLICIT_RE.fullmatch(s.strip()):
                    fentry.identifiers_detected += 1
                    out[sk] = mrn_to_psn.get(s.strip().upper(), "[MRN]")
                elif _SIMPLE_SCALAR_RE.match(s.strip()):
                    out[sk] = s
                elif unk_cache is not None and not extra_terms and not (rec and rec.extra.get("cluster_tokens")):
                    ck = (s, rec.date_shift_days if rec else 0, bool(rec and rec.is_rare_risk), rec.geo_generalized if rec else None)
                    cached = unk_cache.get(ck)
                    if cached is None:
                        cached = redact_free_text(
                            s, guard, ner, allow_unlabeled_fallback=True,
                            extra_terms=extra_terms,
                            known_patient_tokens=None,
                            shift_days=rec.date_shift_days if rec else 0,
                            date_hint=rec.date_hint if rec else "dmy",
                            patient_pseudonym=None,
                            age_band_width=cfg.age_band_years,
                            high_risk=bool(rec and rec.is_rare_risk),
                            geo_generalized=rec.geo_generalized if rec else None,
                        )
                        if len(unk_cache) < 50_000:
                            unk_cache[ck] = cached
                    red, n_det = cached
                    fentry.identifiers_detected += n_det
                    out[sk] = red
                else:
                    red, n_det = redact_free_text(
                        s, guard, ner, allow_unlabeled_fallback=True,
                        extra_terms=extra_terms,
                        known_patient_tokens=rec.extra.get("cluster_tokens") if rec else None,
                        shift_days=rec.date_shift_days if rec else 0,
                        date_hint=rec.date_hint if rec else "dmy",
                        patient_pseudonym=None,
                        age_band_width=cfg.age_band_years,
                        high_risk=bool(rec and rec.is_rare_risk),
                        geo_generalized=rec.geo_generalized if rec else None,
                    )
                    fentry.identifiers_detected += n_det
                    out[sk] = red
        return out

    if isinstance(node, list):
        return [
            _sanitize_json_node(x, rec, parent_key, cfg, guard, ner, mrn_to_psn, fentry, extra_terms, key_roles=key_roles, unk_cache=unk_cache)
            for x in node
        ]
    return node


def sanitize_json_file(
    fentry: FileEntry,
    cfg: RuntimeConfig,
    guard: MedicalGuard,
    ner: PiiNer,
    mrn_to_psn: dict[str, str],
) -> str:
    if fentry.rel_input_path.startswith("clean/"):
        return json.dumps(fentry.raw_data, ensure_ascii=False, indent=2) + "\n"

    data = copy.deepcopy(fentry.raw_data)
    if isinstance(data, list):
        key_roles = infer_json_container_roles(data)
        san_root = []
        for idx, item in enumerate(data):
            rec = fentry.records[idx] if idx < len(fentry.records) else None
            extra = []
            if rec:
                extra = [(nm, "PATIENT_NAME") for nm in rec.names_raw] + [(gn, "RELATIVE_NAME") for gn in rec.guardian_names_raw]
                if rec.village:
                    extra.append((rec.village, "VILLAGE"))
            san_root.append(_sanitize_json_node(item, rec, "", cfg, guard, ner, mrn_to_psn, fentry, extra, key_roles=key_roles))
        return json.dumps(san_root, ensure_ascii=False, indent=2) + "\n"

    if len(fentry.records) == 1 and fentry.records[0].loc == "doc":
        rec = fentry.records[0]
        extra = [(nm, "PATIENT_NAME") for nm in rec.names_raw] + [(gn, "RELATIVE_NAME") for gn in rec.guardian_names_raw]
        if rec.village:
            extra.append((rec.village, "VILLAGE"))
        sanitized = _sanitize_json_node(data, rec, "", cfg, guard, ner, mrn_to_psn, fentry, extra)
        return json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n"

    # Multi-record container JSON (e.g. visits, patients, results, rows, members)
    container = _find_record_container(data)
    if container is None:
        sanitized = _sanitize_json_node(data, None, "", cfg, guard, ner, mrn_to_psn, fentry, [])
        return json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n"

    container_key, items = container
    key_roles = infer_json_container_roles(items)
    unk_cache: dict[tuple[str, int, bool, str | None], tuple[str, int]] = {}
    sanitized_top: dict[str, Any] = {}
    for k, v in data.items():
        if k == container_key and isinstance(v, list):
            san_list = []
            for idx, item in enumerate(v):
                rec = fentry.records[idx] if idx < len(fentry.records) else None
                extra = []
                if rec:
                    extra = [(nm, "PATIENT_NAME") for nm in rec.names_raw] + [(gn, "RELATIVE_NAME") for gn in rec.guardian_names_raw]
                    if rec.village:
                        extra.append((rec.village, "VILLAGE"))
                san_list.append(
                    _sanitize_json_node(item, rec, k.lower(), cfg, guard, ner, mrn_to_psn, fentry, extra, key_roles=key_roles, unk_cache=unk_cache)
                )
            sanitized_top[k] = san_list
        else:
            sanitized_top[k] = _sanitize_json_node(v, None, k.lower(), cfg, guard, ner, mrn_to_psn, fentry, [])

    return json.dumps(sanitized_top, ensure_ascii=False, indent=2) + "\n"
