"""Unicode regex PII detection with Verhoeff Aadhaar validation & offline phonenumbers check."""
from __future__ import annotations

from dataclasses import dataclass

import phonenumbers
import regex as re

from src.nlp.indic_norm import map_span_to_original, normalize_with_map
from src.nlp.medical_guard import is_in_spans


@dataclass(frozen=True)
class PiiMatch:
    start: int
    end: int
    kind: str
    text: str
    norm_value: str = ""
    priority: int = 50


# Verhoeff tables for 12-digit Aadhaar checksum
_V_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_V_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    if len(digits) != 12 or not digits.isdigit() or digits[0] in ("0", "1"):
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _V_D[c][_V_P[i % 8][int(ch)]]
    return c == 0


# Core identifier patterns
_ABHA_NUM_RE = re.compile(r"(?<!\d)(?:91[\s-]?)?\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_ABHA_ADDR_RE = re.compile(r"\b[\w.+-]+@(?:abdm|ndhm|sbx)\b", re.I)
_UPI_RE = re.compile(r"\b[\w.+-]+@(?:upi|okicici|okaxis|oksbi|okhdfcbank|ybl|paytm|ibl|axl|sbi|icici|hdfcbank)\b", re.I)
_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_AADHAAR_LABELED_RE = re.compile(r"(?iu)(?:aadhaar|aadhar|uid|आधार)\s*(?:no\.?|number|सं\.?)?\s*[:=#-]?\s*(\d{4}[\s-]?\d{4}[\s-]?\d{4})")
_AADHAAR_MASKED_RE = re.compile(r"\b[X*x]{4}[\s-]?[X*x]{4}[\s-]?\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_EPIC_RE = re.compile(r"\b[A-Z]{3}\d{7}\b")
_DL_RE = re.compile(r"\b[A-Z]{2}\d{2}[-\s]?(?:19|20)\d{2}\d{7}\b")
_PASSPORT_LABELED_RE = re.compile(r"(?iu)(?:passport|पासपोर्ट)\s*(?:no\.?)?\s*[:=#-]?\s*([A-PR-WY][1-9]\d{6})\b")
_IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_BANK_AC_RE = re.compile(r"(?iu)(?:a/c|acct|account|खाता)\s*(?:no\.?)?\s*[:=#-]?\s*(\d{9,18})\b")
_VEHICLE_RE = re.compile(r"\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{4}\b")
_EMAIL_RE = re.compile(r"(?:mailto:)?([\w.+-]+@[\w-]+(?:\.[\w.-]+)+)", re.I)
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"'()]+", re.I)
_IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
_INSURANCE_RE = re.compile(r"\b(?:PMJAY-[A-Z]{2}-\d{6,14}|POL/\d{2,4}/\d{5,12})\b")

# Phones: +91/0 mobile, STD landline, tel: URI, anchored phone
_TEL_URI_RE = re.compile(r"(?i)\btel:([+\d][\d\s.-]{6,15}\d)")
_PHONE_10_RE = re.compile(
    r"(?<![\d/.-])(?:\+?91[\s.-]?|0)?([6-9]\d{4}[\s.-]?\d{5}|[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4})(?![\d/.-])"
)
_STD_LANDLINE_RE = re.compile(r"(?<![\d/.-])(0\d{2,4}[-\s]\d{6,8})(?![\d/.-])")
_PHONE_ANCHORED_RE = re.compile(
    r"(?iu)(?:mob(?:ile)?|phone|ph|tel|call(?:\s+karo)?|callback|contact|sampark|मोबाइल|फ़ोन|फोन|संपर्क)\s*[:.#-]?\s*"
    r"((?:\+?91[\s.-]?|0)?\d[\d\s.-]{6,13}\d)"
)
_PHONE_MASKED_RE = re.compile(r"\b\d{2}[X*x]{4,6}\d{2,4}\b")

# MRN / UHID / IPD / ADM IDs
_MRN_EXPLICIT_RE = re.compile(
    r"\b(?:SNJ/\d{2}/\d{4,8}|JJH-\d{5,9}|DHH\d{4}-\d{4,8}|DHH-IPD-\d{4,8}|"
    r"UHID-\d{4,8}|IPD-\d{4}-\d{4,8}|IP/\d{2}/\d{4,8}|ADM\d{5,9}|"
    r"MRN[-/]\d{4,10}|PID[-/]\d{4,10}|[A-Z]{2,6}\d{2,4}[-/]\d{4,8}|[A-Z]{3,6}-[A-Z0-9]{2,10}-\d{4,8})\b",
    re.I,
)
_MRN_LABELED_RE = re.compile(
    r"(?iu)(?:uhid|mrn|ipd(?:\s*no\.?)?|opd(?:\s*no\.?)?|reg(?:\.|istration)?\s*no\.?|policy\s*no\.?|patient[_\s]*(?:id|ref)|"
    r"hospital[_\s]*uhid|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:=#-]?\s*([A-Z0-9][A-Z0-9/_-]{4,20})"
)

# Address components (EN + HI + Hinglish)
_HOUSE_RE = re.compile(
    r"(?iu)(?:"
    r"(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Flat(?:\s+No\.?)?|Plot(?:\s+No\.?)?|Makan\s+No\.?|Gali\s+No\.?|Sector|"
    r"मकान\s+नं\.?|मकान\s+सं\.?|गली\s+नं\.?)\s*[:.]?\s*[\w/-]+"
    r")"
)
_LOCALITY_RE = re.compile(
    r"(?iu)(?:"
    r"(?:Mohalla|मोहल्ला|Colony|कॉलोनी|Nagar|Sector|Basti|Gali|Street|Road|Niwasi|निवासी)\s+[^,\n;|]{2,40}"
    r"|\bWard\s+\d+(?:\s*\([^)]+\))?"
    r")"
)
_LANDMARK_RE = re.compile(
    r"(?iu)(?:"
    r"(?:near|opp\.?|opposite|behind|beside)\s+[^,\n;|]{2,45}"
    r"|(?:[^\s,]+(?:\s+[^\s,]+){0,3}\s+के\s+(?:पीछे|पास|सामने))"
    r")"
)
_VILLAGE_RE = re.compile(r"(?iu)(?:Vill\.?|Village|Gram|Gaon|गाँव|ग्राम)\s*[:.]?\s*([^,\n;|]{2,40})")
_TEHSIL_RE = re.compile(r"(?iu)(?:Tehsil|Taluka|Block|Thana|तहसील|थाना)\s*[:.]?\s*([^,\n;|]{2,35})")
_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")

_DATE_LIKE_RE = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b")
_SAFE_URL_HOSTS = ("snomed.info", "loinc.org", "hl7.org", "w3.org", "nrces.in", "abdm.gov.in", "hospital.example")


def _digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _is_valid_indian_phone(raw_num: str) -> bool:
    digs = _digits_only(raw_num)
    if len(digs) == 10 and digs[0] in "6789":
        return True
    try:
        parsed = phonenumbers.parse(raw_num, "IN")
        return phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return len(digs) >= 8


def detect_regex_pii(
    text: str,
    protected_spans: list[tuple[int, int]],
    *,
    is_address_field: bool = False,
) -> list[PiiMatch]:
    """Detect structured PII spans in normalized text, respecting protected clinical spans."""
    norm, _imap = normalize_with_map(text)
    matches: list[PiiMatch] = []
    date_spans = [m.span() for m in _DATE_LIKE_RE.finditer(norm)]

    def _add(start: int, end: int, kind: str, norm_val: str = "", prio: int = 50, ignore_protected: bool = False) -> None:
        if start >= end:
            return
        if not ignore_protected and is_in_spans(start, end, protected_spans):
            return
        matches.append(PiiMatch(start, end, kind, norm[start:end], norm_val or norm[start:end], prio))

    # 1. Emails, UPI, ABHA addresses, URLs, IPs
    for m in _ABHA_ADDR_RE.finditer(norm):
        _add(m.start(), m.end(), "ABHA_ADDRESS", m.group(0).lower(), 96, ignore_protected=True)
    for m in _UPI_RE.finditer(norm):
        _add(m.start(), m.end(), "UPI_ID", m.group(0).lower(), 95, ignore_protected=True)
    for m in _EMAIL_RE.finditer(norm):
        _add(m.start(), m.end(), "EMAIL", m.group(1).lower(), 92, ignore_protected=True)
    for m in _URL_RE.finditer(norm):
        url_low = m.group(0).lower()
        if not any(h in url_low for h in _SAFE_URL_HOSTS):
            _add(m.start(), m.end(), "URL", m.group(0), 85, ignore_protected=False)
    for m in _IP_RE.finditer(norm):
        _add(m.start(), m.end(), "IP_ADDRESS", m.group(0), 85, ignore_protected=False)

    # 2. Gov / Financial IDs
    for m in _PAN_RE.finditer(norm):
        _add(m.start(), m.end(), "PAN", m.group(0).upper(), 92, ignore_protected=True)
    for m in _INSURANCE_RE.finditer(norm):
        _add(m.start(), m.end(), "INSURANCE_ID", m.group(0).upper(), 90, ignore_protected=True)
    for m in _DL_RE.finditer(norm):
        _add(m.start(), m.end(), "GOV_ID", m.group(0).upper(), 90, ignore_protected=True)
    for m in _PASSPORT_LABELED_RE.finditer(norm):
        _add(m.start(1), m.end(1), "GOV_ID", m.group(1).upper(), 90, ignore_protected=True)
    for m in _IFSC_RE.finditer(norm):
        _add(m.start(), m.end(), "IFSC", m.group(0).upper(), 88, ignore_protected=False)
    for m in _BANK_AC_RE.finditer(norm):
        _add(m.start(1), m.end(1), "BANK_ACCOUNT", m.group(1), 88, ignore_protected=True)
    for m in _VEHICLE_RE.finditer(norm):
        if not is_in_spans(m.start(), m.end(), date_spans):
            _add(m.start(), m.end(), "VEHICLE_REG", m.group(0).upper(), 80, ignore_protected=False)
    for m in _EPIC_RE.finditer(norm):
        if not is_in_spans(m.start(), m.end(), date_spans):
            _add(m.start(), m.end(), "GOV_ID", m.group(0).upper(), 84, ignore_protected=False)

    # 3. ABHA (14 digits) before Aadhaar (12 digits)
    abha_spans: list[tuple[int, int]] = []
    for m in _ABHA_NUM_RE.finditer(norm):
        digs = _digits_only(m.group(0))
        if len(digs) == 14:
            abha_spans.append(m.span())
            _add(m.start(), m.end(), "ABHA_NUMBER", digs, 93, ignore_protected=True)

    # 4. Aadhaar (labeled OR Verhoeff valid OR 12-digit unseparated outside protected spans)
    for m in _AADHAAR_LABELED_RE.finditer(norm):
        digs = _digits_only(m.group(1))
        if len(digs) == 12:
            _add(m.start(1), m.end(1), "AADHAAR", digs, 92, ignore_protected=True)
    for m in _AADHAAR_RE.finditer(norm):
        if is_in_spans(m.start(), m.end(), abha_spans):
            continue
        digs = _digits_only(m.group(0))
        if len(digs) == 12:
            prio = 91 if verhoeff_valid(digs) else 88
            _add(m.start(), m.end(), "AADHAAR", digs, prio)
    for m in _AADHAAR_MASKED_RE.finditer(norm):
        _add(m.start(), m.end(), "AADHAAR_MASKED", m.group(0).upper(), 86, ignore_protected=True)

    # 5. Phones (tel: URI, anchored, 10-digit Indian mobile, STD landline)
    for m in _TEL_URI_RE.finditer(norm):
        digs = _digits_only(m.group(1))
        if len(digs) >= 7:
            _add(m.start(), m.end(), "PHONE", digs[-10:] if len(digs) >= 10 else digs, 90, ignore_protected=True)

    for m in _PHONE_ANCHORED_RE.finditer(norm):
        raw_num = m.group(1).strip()
        digs = _digits_only(raw_num)
        if 7 <= len(digs) <= 13:
            last10 = digs[-10:] if len(digs) >= 10 else digs
            _add(m.start(1), m.end(1), "PHONE", last10, 87, ignore_protected=True)

    for m in _PHONE_10_RE.finditer(norm):
        if is_in_spans(m.start(), m.end(), abha_spans):
            continue
        digs = _digits_only(m.group(1))
        if len(digs) == 10 and digs[0] in "6789":
            left = norm[m.start() - 1] if m.start() > 0 else " "
            right = norm[m.end()] if m.end() < len(norm) else " "
            if left.isdigit() or right.isdigit():
                continue
            _add(m.start(), m.end(), "PHONE", digs, 85, ignore_protected=True)

    for m in _STD_LANDLINE_RE.finditer(norm):
        if _is_valid_indian_phone(m.group(1)):
            _add(m.start(1), m.end(1), "PHONE", _digits_only(m.group(1)), 82, ignore_protected=False)

    for m in _PHONE_MASKED_RE.finditer(norm):
        _add(m.start(), m.end(), "PHONE", m.group(0), 80)

    # 6. MRN / UHID / IPD / ADM
    for m in _MRN_EXPLICIT_RE.finditer(norm):
        if not is_in_spans(m.start(), m.end(), date_spans):
            _add(m.start(), m.end(), "MRN", m.group(0).upper(), 83, ignore_protected=True)

    for m in _MRN_LABELED_RE.finditer(norm):
        val = m.group(1)
        if any(ch.isdigit() for ch in val) and not is_in_spans(m.start(1), m.end(1), date_spans):
            _add(m.start(1), m.end(1), "MRN", val.upper(), 84, ignore_protected=True)

    # 7. Address parts
    for m in _HOUSE_RE.finditer(norm):
        _add(m.start(), m.end(), "HOUSE", m.group(0), 75, ignore_protected=is_address_field)

    for m in _LOCALITY_RE.finditer(norm):
        seg = m.group(0)
        if seg.lower().startswith("ward"):
            ctx = norm[max(0, m.start() - 40):min(len(norm), m.end() + 60)].lower()
            if not is_address_field and not re.search(r"\b(?:h\.?\s*no|makan|mohalla|near|opp|tehsil|distt|vill|gram|address|c/o)\b", ctx):
                continue
        _add(m.start(), m.end(), "LOCALITY", seg.strip(), 70, ignore_protected=is_address_field)

    for m in _LANDMARK_RE.finditer(norm):
        seg_low = m.group(0).lower()
        if any(w in seg_low for w in ("normal", "discharge", "admission", "future", "term")):
            continue
        _add(m.start(), m.end(), "LANDMARK", m.group(0).strip(), 68, ignore_protected=is_address_field)

    for m in _VILLAGE_RE.finditer(norm):
        val = m.group(1).strip()
        val = re.split(r"(?i)\s+(?:Tehsil|Distt|District|तहसील|जिला)\b", val)[0].strip(" ,.")
        if val:
            _add(m.start(1), m.start(1) + len(val), "VILLAGE", val, 72, ignore_protected=is_address_field)

    for m in _TEHSIL_RE.finditer(norm):
        val = m.group(1).strip(" ,.")
        val = re.split(r"(?i)\s+(?:Distt|District|जिला)\b", val)[0].strip(" ,.")
        if val:
            _add(m.start(1), m.start(1) + len(val), "TEHSIL", val, 65, ignore_protected=is_address_field)

    for m in _PIN_RE.finditer(norm):
        ctx = norm[max(0, m.start() - 60):min(len(norm), m.end() + 30)].lower()
        if is_address_field or re.search(r"(?:pin|zip|distt|district|tehsil|vill|gram|address|पता|जिला|-\s*$)", ctx) or re.search(r"-\s*" + m.group(1), norm[max(0, m.start() - 5):m.end()]):
            if not re.search(r"\b(?:platelet|wbc|/ul|range|amount|inr|rs)\b", ctx):
                _add(m.start(1), m.end(1), "PIN", m.group(1), 70, ignore_protected=is_address_field)

    return matches


def map_matches_to_original(
    matches: list[PiiMatch],
    index_map: list[int],
    orig_len: int,
) -> list[PiiMatch]:
    """Map normalized match spans back to exact original-string character offsets."""
    mapped: list[PiiMatch] = []
    for m in matches:
        os_, oe_ = map_span_to_original(m.start, m.end, index_map, orig_len)
        mapped.append(PiiMatch(os_, oe_, m.kind, m.text, m.norm_value, m.priority))
    return mapped
