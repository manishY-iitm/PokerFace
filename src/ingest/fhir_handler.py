"""FHIR R4 Bundle & single-resource ingestion and structural-preserving sanitization."""
from __future__ import annotations

import base64
import binascii
import copy
import datetime as dt
import html
import json
import re
from typing import Any

from src.config import RuntimeConfig
from src.crypto import safe_fhir_uuid
from src.ingest.csv_handler import _parse_gender
from src.models import FileEntry, PatientRecord
from src.nlp.date_shifter import parse_date_safe, shift_date_str
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiNer
from src.nlp.redactor import generalize_pin_value, redact_free_text

_UUID_REF_RE = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_from_fhir_resource(res: dict[str, Any], rec: PatientRecord) -> None:
    rtype = res.get("resourceType", "")
    if rtype == "Patient":
        for ident in res.get("identifier") or []:
            val = normalize_text(str(ident.get("value", ""))).strip()
            sys_url = str(ident.get("system", "")).lower()
            if not val:
                continue
            if "abdm" in sys_url or "healthid" in sys_url:
                digs = re.sub(r"\D", "", val)
                if len(digs) == 14:
                    rec.gov_ids["abha"] = digs
            elif "aadhaar" in sys_url or "uidai" in sys_url:
                digs = re.sub(r"\D", "", val)
                if len(digs) == 12:
                    rec.gov_ids["aadhaar"] = digs
            else:
                rec.all_mrns.append(val.upper())
                rec.gov_ids.setdefault("mrn", val.upper())
        for nm in res.get("name") or []:
            t = normalize_text(str(nm.get("text", ""))).strip()
            if t:
                rec.names_raw.append(t)
            else:
                full = " ".join((nm.get("given") or []) + ([nm["family"]] if nm.get("family") else []))
                if full.strip():
                    rec.names_raw.append(normalize_text(full))
        for tel in res.get("telecom") or []:
            val = normalize_text(str(tel.get("value", ""))).strip()
            if tel.get("system") == "phone" or (val and any(c.isdigit() for c in val)):
                digs = re.sub(r"\D", "", val)
                if len(digs) >= 8:
                    rec.phones.append(digs)
            elif "@" in val:
                rec.emails.append(val.lower())
        if res.get("gender"):
            rec.gender = _parse_gender(str(res["gender"]))
        if res.get("birthDate"):
            d = parse_date_safe(str(res["birthDate"]), hint="ymd")
            if d:
                rec.dob = d
        for addr in res.get("address") or []:
            if addr.get("city"):
                rec.village = normalize_text(str(addr["city"])).strip()
            if addr.get("district"):
                rec.district = normalize_text(str(addr["district"])).strip()
            if addr.get("postalCode"):
                digs = re.sub(r"\D", "", str(addr["postalCode"]))
                if len(digs) == 6:
                    rec.pin_code = digs
        for ct in res.get("contact") or []:
            cn = ((ct.get("name") or {}).get("text")) or ""
            if cn:
                rec.guardian_names_raw.append(normalize_text(str(cn)).strip())

    elif rtype == "Encounter":
        for ident in res.get("identifier") or []:
            val = normalize_text(str(ident.get("value", ""))).strip()
            if val:
                rec.all_mrns.append(val.upper())
        period = res.get("period") or {}
        for k in ("start", "end"):
            if period.get(k):
                d = parse_date_safe(str(period[k]), hint="ymd")
                if d:
                    rec.clinical_dates.append(d)
                    if rec.reference_date is None:
                        rec.reference_date = d

    elif rtype in ("Condition", "Procedure"):
        disp = ((res.get("code") or {}).get("text")) or ""
        if not disp:
            for c in ((res.get("code") or {}).get("coding") or []):
                if c.get("display"):
                    disp = c["display"]
                    break
        if disp:
            rec.diagnoses.append(str(disp))

    for cont in res.get("contained") or []:
        if isinstance(cont, dict):
            _extract_from_fhir_resource(cont, rec)


def parse_fhir_bundle(
    rel_path: str,
    text: str,
    sha256: str,
    encoding: str,
    has_bom: bool,
) -> FileEntry:
    data = json.loads(text)
    loc = "doc"
    rec = PatientRecord(
        record_uid=f"{rel_path}::{loc}",
        rel_input_path=rel_path,
        rel_output_path=rel_path,
        file_type="fhir",
        loc=loc,
        raw_payload=data,
        has_patient_signal=True,
    )

    meta_src = ((data.get("meta") or {}).get("source")) or ""
    if isinstance(meta_src, str) and meta_src.startswith("#") and len(meta_src) > 2:
        rec.all_mrns.append(meta_src[1:].upper())

    if data.get("resourceType") == "Bundle":
        for entry in data.get("entry") or []:
            res = entry.get("resource") or {}
            if isinstance(res, dict):
                _extract_from_fhir_resource(res, rec)
    else:
        _extract_from_fhir_resource(data, rec)

    return FileEntry(
        rel_input_path=rel_path,
        rel_output_path=rel_path,
        input_sha256=sha256,
        status="sanitized",
        file_type="fhir",
        encoding=encoding,
        has_bom=has_bom,
        records=[rec],
        raw_data=data,
    )


def _scrub_deep_fields(
    obj: Any,
    rec: PatientRecord,
    guard: MedicalGuard,
    ner: PiiNer,
    extra_terms: list[tuple[str, str]],
    fentry: FileEntry,
    psn: str,
    ref_map: dict[str, str],
    parent_key: str = "",
) -> None:
    """Recursively scrub extension[], note[].text, and Reference.display / Reference.reference."""
    if isinstance(obj, dict):
        # Remap non-UUID Reference.reference if it pointed to old Patient/MRN
        if "reference" in obj and isinstance(obj["reference"], str):
            old_ref = obj["reference"]
            if old_ref in ref_map:
                obj["reference"] = ref_map[old_ref]
            elif old_ref.startswith("Patient/"):
                obj["reference"] = f"Patient/{psn}"

        # Scrub Reference.display on non-practitioner/non-organization references
        if "reference" in obj and isinstance(obj.get("display"), str) and parent_key not in ("author", "serviceProvider", "performer", "requester"):
            red, n_det = redact_free_text(
                obj["display"], guard, ner, allow_unlabeled_fallback=True,
                extra_terms=extra_terms, shift_days=rec.date_shift_days,
                patient_pseudonym=psn, high_risk=rec.is_rare_risk,
            )
            if n_det > 0 or parent_key in ("subject", "patient"):
                obj["display"] = psn if parent_key in ("subject", "patient") else red
                fentry.identifiers_detected += max(n_det, 1 if parent_key in ("subject", "patient") else 0)

        for k, v in list(obj.items()):
            if k == "extension" and isinstance(v, list):
                for ext in v:
                    if isinstance(ext, dict) and isinstance(ext.get("valueString"), str):
                        red, n_det = redact_free_text(
                            ext["valueString"], guard, ner, allow_unlabeled_fallback=True,
                            extra_terms=extra_terms, shift_days=rec.date_shift_days,
                            patient_pseudonym=psn,
                        )
                        ext["valueString"] = red
                        fentry.identifiers_detected += n_det
            elif k == "note" and isinstance(v, list):
                for nt in v:
                    if isinstance(nt, dict) and isinstance(nt.get("text"), str):
                        red, n_det = redact_free_text(
                            nt["text"], guard, ner, allow_unlabeled_fallback=True,
                            extra_terms=extra_terms, shift_days=rec.date_shift_days,
                            patient_pseudonym=psn,
                        )
                        nt["text"] = red
                        fentry.identifiers_detected += n_det
            elif isinstance(v, (dict, list)):
                _scrub_deep_fields(v, rec, guard, ner, extra_terms, fentry, psn, ref_map, parent_key=str(k))
    elif isinstance(obj, list):
        for item in obj:
            _scrub_deep_fields(item, rec, guard, ner, extra_terms, fentry, psn, ref_map, parent_key=parent_key)


def sanitize_fhir_bundle(
    fentry: FileEntry,
    cfg: RuntimeConfig,
    guard: MedicalGuard,
    ner: PiiNer,
) -> str:
    bundle = copy.deepcopy(fentry.raw_data)
    rec = fentry.records[0]
    psn = rec.pseudonym or "PSN-ANON"
    shift = rec.date_shift_days

    extra_terms: list[tuple[str, str]] = (
        [(nm, "PATIENT_NAME") for nm in rec.names_raw]
        + [(gn, "RELATIVE_NAME") for gn in rec.guardian_names_raw]
    )
    if rec.village:
        extra_terms.append((rec.village, "VILLAGE"))

    # Build reference remap for any non-UUID fullUrl (prevents PII in fullUrl)
    ref_map: dict[str, str] = {}
    if bundle.get("resourceType") == "Bundle":
        for entry in bundle.get("entry") or []:
            furl = entry.get("fullUrl")
            if isinstance(furl, str) and not _UUID_REF_RE.match(furl):
                new_furl = f"urn:uuid:{safe_fhir_uuid(furl, cfg.secret)}"
                ref_map[furl] = new_furl
                entry["fullUrl"] = new_furl

    meta = bundle.get("meta")
    if isinstance(meta, dict):
        if isinstance(meta.get("source"), str) and meta["source"].startswith("#"):
            meta["source"] = f"#{psn}"
            fentry.identifiers_detected += 1
        if isinstance(meta.get("lastUpdated"), str):
            meta["lastUpdated"] = shift_date_str(meta["lastUpdated"], shift, hint="ymd")
    if isinstance(bundle.get("timestamp"), str):
        bundle["timestamp"] = shift_date_str(bundle["timestamp"], shift, hint="ymd")

    def _sanitize_resource(res: dict[str, Any]) -> None:
        rtype = res.get("resourceType", "")

        # Scrub text.div XHTML narrative on ALL resources
        if isinstance(res.get("text"), dict) and isinstance(res["text"].get("div"), str):
            if rtype == "Patient":
                res["text"]["div"] = f'<div xmlns="http://www.w3.org/1999/xhtml">{html.escape(psn)}</div>'
                fentry.identifiers_detected += 1
            elif rtype not in ("Practitioner", "Organization"):
                plain = _HTML_TAG_RE.sub(" ", res["text"]["div"])
                red, n_det = redact_free_text(
                    plain, guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms, shift_days=shift, patient_pseudonym=psn,
                    high_risk=rec.is_rare_risk,
                )
                fentry.identifiers_detected += n_det
                res["text"]["div"] = f'<div xmlns="http://www.w3.org/1999/xhtml">{html.escape(red)}</div>'

        if rtype == "Patient":
            if res.get("id") and not _UUID_REF_RE.match(f"urn:uuid:{res['id']}"):
                old_pid = str(res["id"])
                ref_map[f"Patient/{old_pid}"] = f"Patient/{psn}"
                res["id"] = psn
            for ident in res.get("identifier") or []:
                if isinstance(ident, dict) and "value" in ident:
                    fentry.identifiers_detected += 1
                    sys_url = str(ident.get("system", "")).lower()
                    if "abdm" in sys_url or "healthid" in sys_url or "aadhaar" in sys_url:
                        ident["value"] = "[REDACTED]"
                    else:
                        ident["value"] = psn
            res["name"] = [{"text": psn}]
            fentry.identifiers_detected += 1
            for tel in res.get("telecom") or []:
                if isinstance(tel, dict) and "value" in tel:
                    tel["value"] = "[EMAIL]" if tel.get("system") == "email" else "[PHONE]"
                    fentry.identifiers_detected += 1
            if "photo" in res:
                res["photo"] = [{"contentType": "image/png", "title": "[REDACTED]"}]
                fentry.identifiers_detected += 1
            if "birthDate" in res:
                fentry.identifiers_detected += 1
                if rec.is_rare_risk:
                    del res["birthDate"]
                elif rec.dob is not None:
                    shifted_dob = rec.dob + dt.timedelta(days=shift)
                    res["birthDate"] = shifted_dob.isoformat()
                else:
                    del res["birthDate"]
            for addr in res.get("address") or []:
                if isinstance(addr, dict):
                    if "line" in addr:
                        addr["line"] = ["[ADDRESS]"]
                        fentry.identifiers_detected += 1
                    if "city" in addr:
                        addr["city"] = (
                            "SUPPRESSED" if rec.was_suppressed
                            else (rec.geo_generalized if rec.is_rare_risk and rec.geo_generalized else (rec.district or rec.geo_generalized or "[CITY]"))
                        )
                        fentry.identifiers_detected += 1
                    if "district" in addr and (rec.is_rare_risk or rec.was_suppressed):
                        addr["district"] = "SUPPRESSED" if rec.was_suppressed else (rec.geo_generalized or "Uttar Pradesh")
                    if "postalCode" in addr:
                        addr["postalCode"] = rec.extra.get("pin_generalized") or generalize_pin_value(str(addr["postalCode"]), high_risk=rec.is_rare_risk)
                        fentry.identifiers_detected += 1
            for ct in res.get("contact") or []:
                if isinstance(ct, dict):
                    if isinstance(ct.get("name"), dict):
                        ct["name"] = {"text": "[RELATIVE_NAME]"}
                        fentry.identifiers_detected += 1
                    for tel in ct.get("telecom") or []:
                        if isinstance(tel, dict) and "value" in tel:
                            tel["value"] = "[PHONE]"
                            fentry.identifiers_detected += 1

        elif rtype == "Composition":
            if isinstance(res.get("subject"), dict) and "display" in res["subject"]:
                res["subject"]["display"] = psn
                fentry.identifiers_detected += 1
            if isinstance(res.get("date"), str):
                res["date"] = shift_date_str(res["date"], shift, hint="ymd")

        elif rtype == "Encounter":
            for ident in res.get("identifier") or []:
                if isinstance(ident, dict) and "value" in ident:
                    ident["value"] = psn
                    fentry.identifiers_detected += 1
            if isinstance(res.get("period"), dict):
                for k in ("start", "end"):
                    if isinstance(res["period"].get(k), str):
                        res["period"][k] = shift_date_str(res["period"][k], shift, hint="ymd")

        elif rtype in ("Condition", "Observation", "MedicationRequest", "Procedure", "DiagnosticReport"):
            for dk in ("recordedDate", "effectiveDateTime", "authoredOn", "performedDateTime", "issued"):
                if isinstance(res.get(dk), str):
                    res[dk] = shift_date_str(res[dk], shift, hint="ymd")
            if isinstance(res.get("conclusion"), str):
                red, n_det = redact_free_text(
                    res["conclusion"], guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms, shift_days=shift, patient_pseudonym=psn,
                )
                res["conclusion"] = red
                fentry.identifiers_detected += n_det

        elif rtype == "DocumentReference":
            if isinstance(res.get("date"), str):
                res["date"] = shift_date_str(res["date"], shift, hint="ymd")
            for content in res.get("content") or []:
                att = (content or {}).get("attachment")
                if isinstance(att, dict) and isinstance(att.get("data"), str):
                    try:
                        decoded = base64.b64decode(att["data"] + "=" * (-len(att["data"]) % 4)).decode("utf-8")
                        red_dec, n_det = redact_free_text(
                            decoded, guard, ner, allow_unlabeled_fallback=True,
                            extra_terms=extra_terms,
                            known_patient_tokens=rec.extra.get("cluster_tokens"),
                            shift_days=shift, patient_pseudonym=psn,
                            high_risk=rec.is_rare_risk,
                        )
                        fentry.identifiers_detected += n_det
                        att["data"] = base64.b64encode(red_dec.encode("utf-8")).decode("ascii")
                    except (binascii.Error, UnicodeDecodeError):
                        att["data"] = base64.b64encode(b"[REDACTED_ATTACHMENT]").decode("ascii")

        elif rtype == "Binary" and isinstance(res.get("data"), str):
            try:
                decoded = base64.b64decode(res["data"] + "=" * (-len(res["data"]) % 4)).decode("utf-8")
                red_dec, n_det = redact_free_text(
                    decoded, guard, ner, allow_unlabeled_fallback=True,
                    extra_terms=extra_terms, shift_days=shift, patient_pseudonym=psn,
                )
                fentry.identifiers_detected += n_det
                res["data"] = base64.b64encode(red_dec.encode("utf-8")).decode("ascii")
            except (binascii.Error, UnicodeDecodeError):
                res["data"] = base64.b64encode(b"[REDACTED_ATTACHMENT]").decode("ascii")

        for cont in res.get("contained") or []:
            if isinstance(cont, dict):
                _sanitize_resource(cont)

        _scrub_deep_fields(res, rec, guard, ner, extra_terms, fentry, psn, ref_map)

    if bundle.get("resourceType") == "Bundle":
        for entry in bundle.get("entry") or []:
            res = entry.get("resource")
            if isinstance(res, dict):
                _sanitize_resource(res)
    else:
        _sanitize_resource(bundle)

    return json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
