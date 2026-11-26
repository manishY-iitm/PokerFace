"""Schema-agnostic semantic column & JSON key classifier (fuzzy headers + value sniffing)."""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

from src.nlp.indic_norm import devanagari_to_latin, has_devanagari, normalize_text

# Regexes for value-distribution content sniffing on unseen/opaque column names (e.g., col_7, n, id)
_VAL_PHONE_RE = re.compile(
    r"^(?:(?:\+?91[\s.-]?|0091[\s.-]?|0)?(?:[6-9]\d{4}[\s.-]?\d{5}|[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}|[6-9]\d{3}[\s.-]?\d{6})|\(\d{3}\)\s*\d{3}[-\s]\d{4}|0\d{2,5}[-\s]\d{5,8}|[6-9]\.\d{3,10}[eE]\+?0?9)$"
)
_VAL_AADHAAR_RE = re.compile(r"^(?:\d{4}[\s.-]?\d{4}[\s.-]?\d{4}|[Xx*]{4}[\s.-]?[Xx*]{4}[\s.-]?\d{4})$")
_VAL_ABHA_RE = re.compile(r"^(?:91[\s-]?)?\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$")
_VAL_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_VAL_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w.-]+)+$")
_VAL_PIN_RE = re.compile(r"^[1-8]\d{5}$")
_VAL_DATE_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}(?:T.*)?|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})$")
_VAL_GENDER_RE = re.compile(r"^(?:m|f|o|male|female|other|महिला|पुरुष)$", re.I)
_VAL_AGE_SEX_RE = re.compile(r"^\d{1,3}\s*/\s*(?:m|f|male|female)$", re.I)
_VAL_REL_RE = re.compile(r"^(?:s/o|d/o|w/o|c/o|पुत्र|पुत्री|पत्नी)\s+\S+", re.I)
_CLINICAL_CO_HINT_RE = re.compile(r"(?i)(?:fever|cough|pain|weakness|bp\s+\d|mg\b|days\b|syndrome|palsy|apgar|ringer|lactate|;)")
_VAL_MRN_RE = re.compile(
    r"^(?:SNJ/\d{2}/\d+|JJH-\d+|DHH\d{4}-\d+|UHID-\d+|IPD[-/]\d{2,4}[-/]\d+|ADM-?\d+|"
    r"[A-Z]{2,6}(?:[-/.][A-Z0-9]{1,8}){1,3}[-/.]\d{4,8}|[A-Z]{2,4}-\d{5,9}|[A-Z]{2,4}\d{7,10})$",
    re.I,
)
_VAL_PERSON_NAME_RE = re.compile(
    r"^(?:[A-Z][a-z.'-]{1,22}(?:\s+[A-Z][a-z.'-]{1,22}){1,2}|[\u0900-\u097f]{2,20}(?:\s+[\u0900-\u097f]{2,20}){1,2})$"
)
_MED_OR_PLACE_HINT = frozenset({
    "haemoglobin", "hemoglobin", "glucose", "sugar", "blood", "platelet", "count",
    "normal", "abnormal", "inconclusive", "improved", "recovered", "referred",
    "dengue", "malaria", "typhoid", "fever", "hypertension", "diabetes", "cancer",
    "routine", "profile", "panel", "medical", "management", "emergency", "elective",
})

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
    n = normalize_text(name)
    if has_devanagari(n):
        n = devanagari_to_latin(n)
    return re.sub(r"[^a-z0-9]+", " ", n.lower()).strip()


def classify_field_name(name: str, parent_key: str = "") -> str:
    """Classify a CSV header or JSON key into a semantic role without hardcoding schemas."""
    k = _norm_key(name)
    pk = _norm_key(parent_key)

    if not k:
        return "unknown"

    # Explicit metadata path / clerk / export info
    if any(w in k for w in ("source path", "file path", "filepath", "source file", "exported by", "export info", "export meta")):
        return "metadata_pii"

    # Identifiers (check before generic "hospital" / "name" checks!)
    if any(w in k for w in ("aadhaar", "aadhar", "aadhr", "uidai", "uid number", "natl id")) or k == "uid":
        return "aadhaar"
    if "abha" in k and "address" in k:
        return "abha_address"
    if "abha" in k or "healthid" in k or "health id" in k:
        return "abha"
    if k in ("pan", "payer pan", "pan no") or "pan " in k or k.endswith(" pan"):
        return "pan"
    if any(w in k for w in ("pmjay", "abpmjay", "policy", "scheme id")) or k in ("card", "card no") or (
        "insurance" in k and not any(w in k for w in ("covered", "coverage", "status", "flag", "provider", "company", "plan", "type", "name"))
    ):
        return "insurance"
    if any(w in k for w in (
        "uhid", "mrn", "ipd", "reg no", "reg code", "registration no", "patient id", "patientid",
        "patient ref", "hospital uhid", "hosp id", "hosp no", "ssh no", "vzc id", "mrno", "mr no",
        "kramank", "kramank", "panjikaran", "kramank", "kramaank", "kramanka",
    )) or k in ("ref", "id", "kramank", "kramaank"):
        return "mrn"

    # Names / Patient labels (explicit before substring 'lab' match)
    if any(w in k for w in ("patient name", "patients name", "patient label", "pt label", "rogi naam", "rogi ka naam", "rogi", "mareez", "mariz")) or k in ("beneficiary", "insured", "insured person"):
        return "patient_name"

    # Preserved clinical/reference fields (check before generic name checks!)
    if any(w in k for w in ("doctor", "consultant", "referred by", "ref doctor", "practitioner", "ref by")):
        return "doctor"
    if any(w in k for w in (
        "diagnosis", "procedure", "package", "loinc", "vitals", "bp", "pulse", "temp",
        "spo2", "weight", "drug", "strength", "frequency", "duration", "complaints", "complaint",
        "symptoms", "symptom", "treatment", "outcome", "imaging", "findings", "glucose",
        "hba1c", "cholesterol", "bmi", "insurance covered", "insurance provider", "socioeconomic",
        "blood type", "blood group", "medical condition", "admission type", "medication", "medications", "room number",
        "surgery", "pkg code", "nidan", "claim",
        "history", "department", "amount", "ref range", "reference range", "ward",
        "beds", "occupied", "dengue", "facility", "hospital", "panel", "test",
        "unit", "result", "status", "bill no", "visit id", "sample id", "trip", "prescription",
    )) or re.search(r"\b(?:lab|labs|lab name)\b", k):
        return "medical_preserve"

    # Contact
    if any(w in k for w in ("phone", "mobile", "mob", "tel", "callback", "contact no", "sampark", "dooraabhash", "doorbhash")) or k in ("cell", "ph", "contact"):
        return "phone"
    if "email" in k or "e mail" in k:
        return "email"

    # Relative / Guardian (word boundary for short Hindi words like 'pati'/'pita')
    if re.search(r"\b(?:relative|guardian|father|husband|spouse|pita|pati|next of kin|kin|sambandhi|sanbandhi|relnm|rel nm)\b", k) or (k == "name" and pk in ("kin", "relative", "guardian")):
        return "relative_name"
    if k == "relation":
        return "relation"

    # Names
    if k in ("first", "firstname", "first name", "given", "ptfirst", "givenname", "given name"):
        return "first_name"
    if k in ("last", "lastname", "last name", "family", "surname", "ptlast", "familyname", "family name"):
        return "last_name"
    if k == "local" and pk == "name":
        return "patient_name"
    if k == "name" and pk in ("panels", "panel", "facility", "hospital", "organization", "ward", "wards", "department", "drug", "tests"):
        return "medical_preserve"
    if any(w in k for w in ("patient name", "patients name", "patient label", "pt label", "rogi naam", "rogi ka naam", "naam", "raised by")) or k in ("name", "patient", "pt name", "pt", "nm", "member", "members", "person"):
        return "patient_name"

    # Demographics
    if any(w in k for w in ("dob", "birth", "birthdate", "birthdt", "janm tithi", "janam tithi", "janm din", "janm", "janma", "date of birth", "dt of birth", "d o b")) or k == "born":
        return "dob"
    if any(w in k for w in ("age sex", "age gender")):
        return "age_sex"
    if k in ("age", "umar", "aayu", "aayuu", "patient age", "yrs", "age yr", "age y") or k.startswith("age "):
        return "age"
    if k in ("gender", "sex", "ling", "linga", "sx", "g"):
        return "gender"

    # Geography
    if any(w in k for w in ("pin", "pincode", "pin code", "pinkod", "postal", "zip", "postcode")) or k == "pc":
        return "pin"
    if any(w in k for w in ("district", "distt", "jila", "zilla", "zila")) or k == "dist":
        return "district"
    if k in ("state", "region"):
        return "state"
    if any(w in k for w in ("tehsil", "tahasil", "taluka", "block", "subdiv")) or k == "tal":
        return "tehsil"
    if any(w in k for w in ("village", "vill", "gaon", "ganv", "gram", "wadi", "hamlet", "locality")) or (k == "city" and pk != "facility"):
        return "village"
    if any(w in k for w in ("address", "pata", "street", "mohalla", "line", "abode", "residence", "nivas", "addr1", "addr")):
        return "address"

    # Dates
    if any(w in k for w in (
        "date", "registered", "regd on", "collected", "reported", "admission", "discharge",
        "admitted", "discharged", "tithi", "dinank", "tarikh", "dtreg", "doa", "dod",
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


def infer_role_from_values(values: list[str], key_hint: str = "") -> str:
    """Sniff semantic field role from non-empty sample values when header/key is opaque."""
    non_empty = [normalize_text(v).strip() for v in values if v and normalize_text(v).strip()]
    if not non_empty:
        return "unknown"
    sample = non_empty[:50]
    n = len(sample)
    counts: Counter[str] = Counter()
    kh = _norm_key(key_hint)
    opaque_key = (len(kh) <= 4 or kh.startswith("col ") or kh.startswith("field"))
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
        elif _VAL_REL_RE.match(s) and not (s.lower().startswith("c/o") and _CLINICAL_CO_HINT_RE.search(s)):
            counts["relative_name"] += 1
        elif _VAL_DATE_RE.match(s):
            counts["clinical_date"] += 1
        elif _VAL_PIN_RE.match(s):
            counts["pin"] += 1
        elif opaque_key and _VAL_PERSON_NAME_RE.match(s):
            toks = {w.lower() for w in s.split()}
            if not (toks & _MED_OR_PLACE_HINT):
                counts["patient_name"] += 1

    if not counts:
        return "unknown"
    top_role, top_cnt = counts.most_common(1)[0]
    thresh = 0.75 if top_role == "patient_name" else 0.55
    if top_cnt / n >= thresh:
        return top_role
    return "unknown"


def infer_csv_column_roles(header: list[str], rows: list[list[str]]) -> list[str]:
    """Combine header classification with column value-distribution sniffing."""
    roles = [classify_field_name(h) for h in header]
    for idx, role in enumerate(roles):
        if role == "unknown":
            col_vals = [r[idx] for r in rows[:80] if idx < len(r)]
            sniffed = infer_role_from_values(col_vals, key_hint=header[idx] if idx < len(header) else "")
            if sniffed != "unknown":
                roles[idx] = sniffed
    return roles


def infer_json_container_roles(items: list[dict]) -> dict[str, str]:
    """Sniff semantic roles for opaque/short keys across a list of JSON records."""
    by_key: dict[str, list[str]] = {}
    for item in items[:60]:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            if v is not None and not isinstance(v, (dict, list)):
                by_key.setdefault(str(k), []).append(str(v))
    inferred: dict[str, str] = {}
    for k, vals in by_key.items():
        if classify_field_name(k) == "unknown":
            role = infer_role_from_values(vals, key_hint=k)
            if role != "unknown":
                inferred[k] = role
    return inferred
