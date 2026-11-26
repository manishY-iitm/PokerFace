"""Fast compiled deterministic date parser & exact-offset clinical date shifter."""
from __future__ import annotations

import datetime as dt
import re

from src.nlp.indic_norm import normalize_digits

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_NAMES_FULL = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_NAMES_SHORT = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_HINDI_MONTH_MAP = {
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4,
    "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
    "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11,
    "दिसंबर": 12, "दिसम्बर": 12,
}
_HINDI_MONTH_NAMES = [
    "", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]

_ISO_DT_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?)?\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DASH_DATE_RE = re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b")
_WA_DATE_RE = re.compile(r"\[(\d{1,2})/(\d{1,2})/(\d{2}),(\s*\d{1,2}:\d{2}\s*(?:am|pm))\]", re.I)
_TEXT_DATE_FULL_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b",
    re.I,
)
_MON_DD_YYYY_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),\s*(\d{4})\b",
    re.I,
)
_TEXT_DATE_NOYR_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b(?!\s+\d{4})",
    re.I,
)
_HINDI_DATE_RE = re.compile(
    r"(?u)\b(\d{1,2})\s+(" + "|".join(_HINDI_MONTH_MAP.keys()) + r")\s+(\d{4})\b"
)


def parse_date_safe(text: str, hint: str = "dmy") -> dt.date | None:
    s = normalize_digits(text).strip()
    m = _ISO_DT_RE.search(s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    for rx in (_SLASH_DATE_RE, _DASH_DATE_RE):
        m = rx.search(s)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a > 12:
                d, mo = a, b
            elif b > 12:
                mo, d = a, b
            elif hint == "mdy":
                mo, d = a, b
            else:
                d, mo = a, b
            try:
                return dt.date(y, mo, d)
            except ValueError:
                pass
    m = _WA_DATE_RE.search(s)
    if m:
        a, b, yy = int(m.group(1)), int(m.group(2)), 2000 + int(m.group(3))
        d, mo = (b, a) if (b > 12 or (hint == "mdy" and a <= 12)) else (a, b)
        try:
            return dt.date(yy, mo, d)
        except ValueError:
            pass
    m = _TEXT_DATE_FULL_RE.search(s)
    if m:
        d, mo, y = int(m.group(1)), _MONTH_MAP[m.group(2).lower()], int(m.group(3))
        try:
            return dt.date(y, mo, d)
        except ValueError:
            pass
    m = _MON_DD_YYYY_RE.search(s)
    if m:
        mo, d, y = _MONTH_MAP[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        try:
            return dt.date(y, mo, d)
        except ValueError:
            pass
    m = _HINDI_DATE_RE.search(s)
    if m:
        d, mo, y = int(m.group(1)), _HINDI_MONTH_MAP[m.group(2)], int(m.group(3))
        try:
            return dt.date(y, mo, d)
        except ValueError:
            pass
    return None


def compute_truthful_age_band(
    dob: dt.date,
    ref_date: dt.date | None = None,
    band_width: int = 5,
    *,
    high_risk: bool = False,
) -> str:
    """Compute exact age at ref_date and return a truthful enclosing age band (never shifting DOB)."""
    ref = ref_date or dt.date(2026, 1, 1)
    age = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    age = max(0, age)
    if high_risk or age >= 85:
        if age >= 80:
            return "80+"
        w = max(10, band_width * 2)
        lo = (age // w) * w
        return f"{lo}-{lo + w - 1}"
    lo = (age // band_width) * band_width
    return f"{lo}-{lo + band_width - 1}"


def shift_date_str(text: str, offset_days: int, hint: str = "dmy") -> str:
    """Shift all clinical dates inside a string by exact offset_days, preserving format."""
    if offset_days == 0:
        return text
    s = normalize_digits(text)
    delta = dt.timedelta(days=offset_days)

    def _sub_iso(m: re.Match[str]) -> str:
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) + delta
            suffix = m.group(4) or ""
            return f"{d.isoformat()}{suffix}"
        except ValueError:
            return m.group(0)

    def _sub_sep(sep: str, m: re.Match[str]) -> str:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12:
            d, mo, order = a, b, "dmy"
        elif b > 12:
            mo, d, order = a, b, "mdy"
        elif hint == "mdy":
            mo, d, order = a, b, "mdy"
        else:
            d, mo, order = a, b, "dmy"
        try:
            nd = dt.date(y, mo, d) + delta
            if order == "mdy":
                return f"{nd.month:02d}{sep}{nd.day:02d}{sep}{nd.year:04d}"
            return f"{nd.day:02d}{sep}{nd.month:02d}{sep}{nd.year:04d}"
        except ValueError:
            return m.group(0)

    def _sub_wa(m: re.Match[str]) -> str:
        a, b, yy = int(m.group(1)), int(m.group(2)), 2000 + int(m.group(3))
        d, mo = (b, a) if (b > 12 or (hint == "mdy" and a <= 12)) else (a, b)
        try:
            nd = dt.date(yy, mo, d) + delta
            return f"[{nd.day:02d}/{nd.month:02d}/{nd.year % 100:02d},{m.group(4)}]"
        except ValueError:
            return m.group(0)

    def _sub_text_full(m: re.Match[str]) -> str:
        d, mo_str, y = int(m.group(1)), m.group(2), int(m.group(3))
        mo = _MONTH_MAP[mo_str.lower()]
        try:
            nd = dt.date(y, mo, d) + delta
            mname = _MONTH_NAMES_SHORT[nd.month] if len(mo_str) <= 3 else _MONTH_NAMES_FULL[nd.month]
            return f"{nd.day} {mname} {nd.year}"
        except ValueError:
            return m.group(0)

    def _sub_mon_dd_yyyy(m: re.Match[str]) -> str:
        mo_str, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        mo = _MONTH_MAP[mo_str.lower()]
        try:
            nd = dt.date(y, mo, d) + delta
            mname = _MONTH_NAMES_SHORT[nd.month] if len(mo_str) <= 3 else _MONTH_NAMES_FULL[nd.month]
            return f"{mname} {nd.day}, {nd.year}"
        except ValueError:
            return m.group(0)

    def _sub_text_noyr(m: re.Match[str]) -> str:
        d, mo = int(m.group(1)), _MONTH_MAP[m.group(2).lower()]
        try:
            nd = dt.date(2026, mo, d) + delta
            return f"{nd.day} {_MONTH_NAMES_FULL[nd.month]}"
        except ValueError:
            return m.group(0)

    def _sub_hindi(m: re.Match[str]) -> str:
        d, mo, y = int(m.group(1)), _HINDI_MONTH_MAP[m.group(2)], int(m.group(3))
        try:
            nd = dt.date(y, mo, d) + delta
            return f"{nd.day} {_HINDI_MONTH_NAMES[nd.month]} {nd.year}"
        except ValueError:
            return m.group(0)

    s = _ISO_DT_RE.sub(_sub_iso, s)
    s = _WA_DATE_RE.sub(_sub_wa, s)
    s = _SLASH_DATE_RE.sub(lambda m: _sub_sep("/", m), s)
    s = _DASH_DATE_RE.sub(lambda m: _sub_sep("-", m), s)
    s = _TEXT_DATE_FULL_RE.sub(_sub_text_full, s)
    s = _MON_DD_YYYY_RE.sub(_sub_mon_dd_yyyy, s)
    s = _TEXT_DATE_NOYR_RE.sub(_sub_text_noyr, s)
    s = _HINDI_DATE_RE.sub(_sub_hindi, s)
    return s
