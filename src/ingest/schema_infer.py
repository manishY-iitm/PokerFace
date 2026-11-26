"""Schema-agnostic semantic column & JSON key classifier (fuzzy headers + value sniffing)."""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

from src.nlp.indic_norm import normalize_text

# Regexes for value-distribution content sniffing on unseen/opaque column names (e.g., col_7)
_VAL_PHONE_RE = re.compile(r"^(?:\+?91[\s.-]?|0)?[6-9]\d{4}[\s.-]?\d{5}$|^(?:\+?91[\s.-]?|0)?[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}$")
_VAL_AADHAAR_RE = re.compile(r"^(?:\d{4}[\s-]?\d{4}[\s-]?\d{4}|[Xx*]{4}[\s-]?[Xx*]{4}[\s-]?\d{4})$")
_VAL_ABHA_RE = re.compile(r"^(?:91[\s-]?)?\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$")
_VAL_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_VAL_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w.-]+)+$")
_VAL_PIN_RE = re.compile(r"^[1-8]\d{5}$")
_VAL_DATE_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}(?:T.*)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$")
_VAL_GENDER_RE = re.compile(r"^(?:m|f|o|male|female|other|महिला|पुरुष)$", re.I)
_VAL_AGE_SEX_RE = re.compile(r"^\d{1,3}\s*/\s*(?:m|f|male|female)$", re.I)
_VAL_REL_RE = re.compile(r"^(?:s/o|d/o|w/o|c/o)\s+\S+", re.I)
_VAL_MRN_RE = re.compile(r"^(?:SNJ/\d{2}/\d+|JJH-\d+|DHH\d{4}-\d+|UHID-\d+|IPD-\d{4}-\d+|ADM\d+)$", re.I)

_FUZZY_KEY_TARGETS = {
    "patient_name": ("patient name", "rogi naam", "rogi ka naam", "mareez naam", "naam", "full name"),
    "relative_name": ("relative name", "guardian name", "pita ka naam", "pati ka naam", "s o", "w o"),
    "phone": ("mobile number", "phone number", "sampark", "sampark sutra", "contact number", "doorbhash"),
    "dob": ("date of birth", "janm tithi", "janam tithi", "birth date"),
    "age": ("umar", "aayu", "patient age"),
    "gender": ("ling", "gender", "sex"),
    "address": ("pata", "full address", "residential address", "sthan"),
    "village": ("gaon", "gram", "village mohalla"),
    "district": ("jila", "district", "distt"),
    "pin": ("pin code", "postal code", "dak code", "zip code"),
}


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", normalize_text(name).lower()).strip()


def classify_field_name(name: str, parent_key: str = "") -> str:
    """Classify a CSV header or JSON key into a semantic role without hardcoding schemas."""
    k = _norm_key(name)
    pk = _norm_key(parent_key)

    if not k:
        return "unknown"

    # Explicit metadata path / clerk
    if any(w in k for w in ("source path", "file path", "filepath", "exported by")):
        return "metadata_pii"

    # Identifiers (check before generic "hospital" / "name" checks!)
    if any(w in k for w in ("aadhaar", "aadhar", "uidai")):
        return "aadhaar"
    if "abha" in k and "address" in k:
        return "abha_address"
    if "abha" in k or "healthid" in k:
        return "abha"
    if k in ("pan", "payer pan") or "pan " in k or k.endswith(" pan"):
        return "pan"
    if any(w in k for w in ("pmjay", "policy", "insurance")):
        return "insurance"
    if any(w in k for w in ("uhid", "mrn", "ipd", "reg no", "registration no", "patient id", "patientid", "patient ref", "hospital uhid")):
        return "mrn"

    # Preserved clinical/reference fields (check before generic name checks!)
    if any(w in k for w in ("doctor", "consultant", "referred by", "ref doctor", "practitioner")):
        return "doctor"
    if any(w in k for w in (
        "diagnosis", "procedure", "package", "loinc", "vitals", "bp", "pulse", "temp",
        "spo2", "weight", "drug", "strength", "frequency", "duration", "complaints",
        "history", "department", "amount", "ref range", "reference range", "ward",
        "beds", "occupied", "dengue", "facility", "hospital", "lab", "panel", "test",
        "unit", "result", "status", "bill no", "visit id", "sample id", "trip",
    )):
        return "medical_preserve"

    # Contact
    if any(w in k for w in ("phone", "mobile", "mob", "tel", "callback", "contact no", "sampark", "doorbhash")):
        return "phone"
    if "email" in k or "e mail" in k:
        return "email"

    # Relative / Guardian (word boundary for short Hindi words like 'pati'/'pita')
    if re.search(r"\b(?:relative|guardian|father|husband|spouse|pita|pati|next of kin)\b", k):
        return "relative_name"
    if k == "relation":
        return "relation"

    # Names
    if k in ("first", "firstname", "first name", "given"):
        return "first_name"
    if k in ("last", "lastname", "last name", "family", "surname"):
        return "last_name"
    if k == "local" and pk == "name":
        return "patient_name"
    if k == "name" and pk in ("panels", "panel", "facility", "hospital", "organization", "ward", "wards", "department", "drug", "tests"):
        return "medical_preserve"
    if any(w in k for w in ("patient name", "patients name", "rogi naam", "rogi ka naam", "naam", "raised by")) or k in ("name", "patient", "pt name"):
        return "patient_name"

    # Demographics
    if any(w in k for w in ("dob", "birth", "birthdate", "janm tithi", "janam tithi", "date of birth")):
        return "dob"
    if any(w in k for w in ("age sex", "age gender")):
        return "age_sex"
    if k in ("age", "umar", "aayu", "patient age") or k.startswith("age "):
        return "age"
    if k in ("gender", "sex", "ling"):
        return "gender"

    # Geography
    if any(w in k for w in ("pin", "pincode", "pin code", "postal", "zip")):
        return "pin"
    if any(w in k for w in ("district", "distt", "jila")):
        return "district"
    if any(w in k for w in ("tehsil", "taluka", "block")):
        return "tehsil"
    if any(w in k for w in ("village", "vill", "gaon", "gram")) or (k == "city" and pk != "facility"):
        return "village"
    if any(w in k for w in ("address", "pata", "street", "mohalla", "line")):
        return "address"

    # Dates
    if any(w in k for w in (
        "date", "registered", "collected", "reported", "admission", "discharge",
        "authoredon", "effective", "recorded", "performed",
    )):
        return "clinical_date"

    # Free text & attachments
    if k in ("notes", "note", "remarks", "remark", "reviewed by", "text"):
        return "free_text"
    if k == "data" and pk in ("attachment", ""):
        return "base64_data"

    # Fuzzy header fallback for unseen transliterated/variant headers
    best_role = "unknown"
    best_score = 84
    for role, prototypes in _FUZZY_KEY_TARGETS.items():
        for proto in prototypes:
            sc = fuzz.ratio(k, proto)
            if sc > best_score:
                best_score = sc
                best_role = role
    return best_role


def infer_role_from_values(values: list[str]) -> str:
    """Sniff semantic field role from non-empty sample values when header/key is opaque."""
    non_empty = [normalize_text(v).strip() for v in values if v and normalize_text(v).strip()]
    if not non_empty:
        return "unknown"
    sample = non_empty[:50]
    n = len(sample)
    counts: Counter[str] = Counter()
    for s in sample:
        if _VAL_EMAIL_RE.match(s):
            counts["email"] += 1
        elif _VAL_PAN_RE.match(s):
            counts["pan"] += 1
        elif _VAL_ABHA_RE.match(s):
            counts["abha"] += 1
        elif _VAL_AADHAAR_RE.match(s):
            counts["aadhaar"] += 1
        elif _VAL_PHONE_RE.match(s):
            counts["phone"] += 1
        elif _VAL_MRN_RE.match(s):
            counts["mrn"] += 1
        elif _VAL_AGE_SEX_RE.match(s):
            counts["age_sex"] += 1
        elif _VAL_GENDER_RE.match(s):
            counts["gender"] += 1
        elif _VAL_REL_RE.match(s):
            counts["relative_name"] += 1
        elif _VAL_DATE_RE.match(s):
            counts["clinical_date"] += 1
        elif _VAL_PIN_RE.match(s):
            counts["pin"] += 1

    if not counts:
        return "unknown"
    top_role, top_cnt = counts.most_common(1)[0]
    if top_cnt / n >= 0.55:
        return top_role
    return "unknown"


def infer_csv_column_roles(header: list[str], rows: list[list[str]]) -> list[str]:
    """Combine header classification with column value-distribution sniffing."""
    roles = [classify_field_name(h) for h in header]
    for idx, role in enumerate(roles):
        if role == "unknown":
            col_vals = [r[idx] for r in rows if idx < len(r)]
            sniffed = infer_role_from_values(col_vals)
            if sniffed != "unknown":
                roles[idx] = sniffed
    return roles
