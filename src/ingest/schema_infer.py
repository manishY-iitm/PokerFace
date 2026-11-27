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
_VAL_AGE_RE = re.compile(r"^\d{1,3}\s*(?:yrs?|years?|y|months?|mons?|days?|d|वर्ष|साल)$", re.I)
_VAL_AGE_SEX_RE = re.compile(r"^\d{1,3}\s*/\s*(?:m|f|male|female)$", re.I)
_VAL_REL_RE = re.compile(r"^(?:s/o|d/o|w/o|c/o|पुत्र|पुत्री|पत्नी)\s+\S+", re.I)
_CLINICAL_CO_HINT_RE = re.compile(r"(?i)(?:fever|cough|pain|weakness|bp\s+\d|mg\b|days\b|syndrome|palsy|apgar|ringer|lactate|;)")
_VAL_MRN_RE = re.compile(
    r"^(?:SNJ/\d{2}/\d+|JJH-\d+|DHH\d{4}-\d+|UHID-\d+|IPD[-/]\d{2,4}[-/]\d+|ADM-?\d+|"
    r"[A-Z]{2,6}\d{2,4}[-/][A-Z]{1,6}[-/]\d{4,8}|"
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
    if any(w in k for w in ("aadhaar", "aadhar", "aadhr", "uidai", "uid number", "natl id")) or re.search(r"\buid\b", k):
        return "aadhaar"
    if re.search(r"\babha\b", k) and "address" in k:
        return "abha_address"
    if re.search(r"\b(?:abha|healthid|health id)\b", k):
        return "abha"
    if re.search(r"\bpan\b", k):
        return "pan"
    if any(w in k for w in ("pmjay", "abpmjay", "policy", "scheme id")) or k in ("card", "card no") or (
        "insurance" in k and not any(w in k for w in ("covered", "coverage", "status", "flag", "provider", "company", "plan", "type", "name"))
    ):
        return "insurance"
    is_date_word = any(w in k for w in ("tithi", "dinank", "tarikh", "date", "birth", "janm"))
    if (not is_date_word) and (
        any(w in k for w in (
            "uhid", "mrn", "ipd", "reg no", "reg code", "reg id", "regid", "registration no",
            "patient id", "patientid", "patient ref", "hospital uhid", "hosp id", "hosp no",
            "ssh no", "vzc id", "mrno", "mr no", "laabharthi code",
            "kramank", "kramaank", "kramanka", "kram san", "panjikaran",
        )) or k in ("ref", "id")
    ):
        return "mrn"

    # Names / Patient labels (explicit before substring 'lab' match)
    if any(w in k for w in ("patient name", "patients name", "patient label", "pt label", "rogi naam", "rogi ka naam", "rogi", "mareez", "mariz")) or k in ("beneficiary", "insured", "insured person"):
        return "patient_name"

    # Preserved clinical/reference fields (check before generic name/contact checks!)
    if any(w in k for w in ("doctor", "consultant", "referred by", "ref doctor", "practitioner", "ref by")):
        return "doctor"
    if any(w in k for w in (
        "diagnosis", "procedure", "package", "loinc", "vitals", "bp", "pulse", "temp",
        "spo2", "weight", "drug", "strength", "frequency", "duration", "complaints", "complaint",
        "symptoms", "symptom", "treatment", "outcome", "imaging", "findings", "glucose",
        "hba1c", "cholesterol", "bmi", "platelet", "platelets", "hemoglobin", "haemoglobin",
        "bilirubin", "creatinine", "insurance covered", "insurance provider", "socioeconomic",
        "blood type", "blood group", "medical condition", "admission type", "medication", "medications", "room number",
        "surgery", "pkg code", "nidan", "claim",
        "history", "department", "amount", "ref range", "reference range", "ward",
        "beds", "occupied", "dengue", "facility", "hospital", "panel", "test",
        "unit", "result", "status", "bill no", "visit id", "sample id", "trip", "prescription",
    )) or re.search(r"\b(?:lab|labs|lab name|wbc|rbc)\b", k):
        return "medical_preserve"

    # Contact
    if re.search(r"\b(?:phone|mobile|mob|tel|callback|contact|sampark|sanpark|dooraabhash|doorbhash|durabhash|cell|ph)\b", k):
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
    if k in ("age", "umar", "aayu", "aayuu", "patient age", "yrs", "age yr", "age y") or k.startswith("age ") or re.search(r"\b(?:umar|aayu|aayuu)\b", k):
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
    date_years: list[int] = []
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
        elif _VAL_AGE_RE.match(s):
            counts["age"] += 1
        elif _VAL_GENDER_RE.match(s):
            counts["gender"] += 1
        elif _VAL_REL_RE.match(s) and not (s.lower().startswith("c/o") and _CLINICAL_CO_HINT_RE.search(s)):
            counts["relative_name"] += 1
        elif _VAL_DATE_RE.match(s):
            counts["clinical_date"] += 1
            ym = re.search(r"\b((?:19|20)\d{2})\b", s)
            if ym:
                date_years.append(int(ym.group(1)))
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
        if top_role == "clinical_date" and date_years:
            birth_era = sum(1 for y in date_years if y <= 2012)
            if birth_era / len(date_years) >= 0.60:
                return "dob"
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
        elif role == "patient_name":
            col_vals = [normalize_text(r[idx]).strip() for r in rows[:60] if idx < len(r) and normalize_text(r[idx]).strip()]
            if col_vals:
                rel_hits = sum(
                    1 for s in col_vals
                    if _VAL_REL_RE.match(s) and not (s.lower().startswith("c/o") and _CLINICAL_CO_HINT_RE.search(s))
                )
                if rel_hits / len(col_vals) >= 0.50:
                    roles[idx] = "relative_name"
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
