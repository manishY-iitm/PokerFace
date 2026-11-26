"""Unified span-merging redactor and inline quasi-identifier anonymizer."""
from __future__ import annotations

import regex as re

from src.nlp.date_shifter import shift_date_str
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard, is_in_spans
from src.nlp.pii_ner import PiiNer
from src.nlp.pii_regex import PiiMatch, detect_regex_pii

_DOB_LINE_RE = re.compile(
    r"(?iu)(?:DOB|Date\s+of\s+Birth|Janm\s+Tithi|जन्म\s*तिथि)\s*[:=-]\s*([^\n,;|]+)"
)
_AGE_INLINE_RE = re.compile(
    r"(?iu)\b(\d{1,3})\s*(?:Yrs?|Years?|Y)\s*(?:/\s*([MF]|Male|Female|Other))?\b"
)
_AGE_HINDI_RE = re.compile(r"(?u)(?:आयु|उम्र)\s*[:=-]\s*(\d{1,3})\s*(वर्ष|साल)?")
_AGE_LABEL_RE = re.compile(r"(?iu)(?:Age(?:\s*/\s*Sex)?)\s*[:=-]\s*(\d{1,3})(?:\s*(?:/\s*([MF]))|\s*(Yrs?|Years?))?")


def generalize_age_value(age: int, band_width: int = 5, *, high_risk: bool = False) -> str:
    if high_risk or age >= 85:
        if age >= 80:
            return "80+"
        w = max(10, band_width * 2)
        lo = (age // w) * w
        return f"{lo}-{lo + w - 1}"
    lo = (age // band_width) * band_width
    return f"{lo}-{lo + band_width - 1}"


def generalize_pin_value(pin: str, *, high_risk: bool = False) -> str:
    digs = "".join(c for c in pin if c.isdigit())
    if len(digs) != 6:
        return "[PIN]"
    if high_risk:
        return "[PIN]"
    return f"{digs[:3]}***"


def _merge_matches(matches: list[PiiMatch]) -> list[PiiMatch]:
    ordered = sorted(matches, key=lambda m: (m.start, -m.priority, -(m.end - m.start)))
    chosen: list[PiiMatch] = []
    for m in ordered:
        overlap = False
        for c in chosen:
            if m.start < c.end and m.end > c.start:
                overlap = True
                break
        if not overlap:
            chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


def _replacement_for(
    kind: str,
    text: str,
    *,
    patient_pseudonym: str | None = None,
    high_risk: bool = False,
) -> str:
    if kind in ("MRN", "PATIENT_NAME") and patient_pseudonym:
        return patient_pseudonym
    if kind == "PIN":
        return generalize_pin_value(text, high_risk=high_risk)
    return f"[{kind}]"


def redact_free_text(
    text: str,
    guard: MedicalGuard,
    ner: PiiNer,
    *,
    is_address_field: bool = False,
    allow_unlabeled_fallback: bool = True,
    extra_terms: list[tuple[str, str]] | None = None,
    known_patient_tokens: set[str] | None = None,
    shift_days: int = 0,
    date_hint: str = "dmy",
    patient_pseudonym: str | None = None,
    age_band_width: int = 5,
    high_risk: bool = False,
    generalize_inline_age: bool = True,
) -> tuple[str, int]:
    """Normalize, detect PII outside medical guard spans, replace, generalize, and shift dates."""
    norm = normalize_text(text)
    if not norm:
        return norm, 0

    protected = guard.find_protected_spans(norm, is_address_field=is_address_field)
    matches: list[PiiMatch] = []

    # 1. DOB anchors first (so DOB date is replaced with age band or [DOB], never shifted as clinical date!)
    for m in _DOB_LINE_RE.finditer(norm):
        matches.append(PiiMatch(m.start(1), m.end(1), "DOB", m.group(1), m.group(1), 96))

    # 2. Structured regex & contextual + cluster-boosted NER
    matches.extend(detect_regex_pii(norm, protected, is_address_field=is_address_field))
    matches.extend(
        ner.detect(
            norm,
            protected,
            is_address_field=is_address_field,
            allow_unlabeled_fallback=allow_unlabeled_fallback,
            known_patient_tokens=known_patient_tokens,
        )
    )

    # 3. Record-specific extra terms
    if extra_terms:
        for term, kind in extra_terms:
            t_norm = normalize_text(term).strip()
            if len(t_norm) < 3 or guard.is_medical_or_geo_phrase(t_norm):
                continue
            rx = re.compile(rf"(?iu)(?<![\w\u0900-\u097f]){re.escape(t_norm)}(?![\w\u0900-\u097f])")
            for m in rx.finditer(norm):
                if not is_in_spans(m.start(), m.end(), protected):
                    matches.append(PiiMatch(m.start(), m.end(), kind, m.group(0), m.group(0), 89))

    chosen = _merge_matches(matches)
    if chosen:
        parts: list[str] = []
        cursor = 0
        for m in chosen:
            parts.append(norm[cursor:m.start])
            rep = _replacement_for(
                m.kind,
                m.text,
                patient_pseudonym=patient_pseudonym,
                high_risk=high_risk,
            )
            parts.append(rep)
            cursor = m.end
        parts.append(norm[cursor:])
        out = "".join(parts)
    else:
        out = norm

    # 4. Generalize inline ages cleanly
    if generalize_inline_age:
        def _sub_age_lbl(m: re.Match[str]) -> str:
            age_v = int(m.group(1))
            if age_v > 120:
                return m.group(0)
            band = generalize_age_value(age_v, age_band_width, high_risk=high_risk)
            prefix = m.group(0)[: m.start(1) - m.start()]
            suffix = m.group(0)[m.end(1) - m.start():]
            return f"{prefix}{band}{suffix}"

        def _sub_age_inline(m: re.Match[str]) -> str:
            age_v = int(m.group(1))
            if age_v > 120:
                return m.group(0)
            ctx_before = out[max(0, m.start() - 6):m.start()].lower()
            ctx_after = out[m.end():min(len(out), m.end() + 8)].lower()
            if "x " in ctx_before or "back" in ctx_after or "for " in ctx_before:
                return m.group(0)
            band = generalize_age_value(age_v, age_band_width, high_risk=high_risk)
            sex = f" / {m.group(2)}" if m.group(2) else ""
            unit = "Yrs" if "yrs" in m.group(0).lower() else ("Years" if "year" in m.group(0).lower() else "Y")
            return f"{band} {unit}{sex}"

        def _sub_age_hindi(m: re.Match[str]) -> str:
            age_v = int(m.group(1))
            if age_v > 120:
                return m.group(0)
            band = generalize_age_value(age_v, age_band_width, high_risk=high_risk)
            unit = f" {m.group(2)}" if m.group(2) else ""
            return f"आयु: {band}{unit}"

        out = _AGE_LABEL_RE.sub(_sub_age_lbl, out)
        out = _AGE_HINDI_RE.sub(_sub_age_hindi, out)
        out = _AGE_INLINE_RE.sub(_sub_age_inline, out)

    # 5. Shift clinical dates
    if shift_days != 0:
        out = shift_date_str(out, shift_days, hint=date_hint)

    return out, len(chosen)


# Track 2 alias
redact_text = redact_free_text
