"""Medical context protection so doctors, eponyms, drugs, doses, vitals, and history survive."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from src.config import DATA_DIR, RuntimeConfig
from src.nlp.indic_norm import normalize_text

# Clinicians / Doctors (unless preceded by Patient Name:)
_DOCTOR_RE = re.compile(
    r"(?iu)(?:^|(?<=[\s,;(]))"
    r"(?:dr\.?|doctor|डॉ\.?|डॉक्टर|consultant|attending|surgeon|referred\s+by\s+dr\.?|signed\s+by)"
    r"\s*[:.]?\s*(?:[A-Z\u0900-\u097f][\w.\u0900-\u097f'-]*"
    r"(?:\s+[A-Z\u0900-\u097f][\w.\u0900-\u097f'-]*){0,3})"
)

_DOCTOR_STOP_WORDS = frozenset({
    "reviewed", "advised", "prescribed", "ordered", "examined", "saw", "noted",
    "took", "said", "told", "bola", "ne", "ko", "for", "and", "the", "patient",
    "with", "in", "on", "at", "after", "before", "from", "to",
})

# Clinical Ward / Bed: "Ward 8 (Medicine)", "Ward 2 (Surgery) / Bed 29", "Bed 12"
_CLINICAL_WARD_RE = re.compile(
    r"(?i)\bWard\s+\d+\s*\([A-Za-z &/-]+\)(?:\s*/\s*Bed\s+\d+)?|\bBed\s+\d+\b"
)

# Diagnosis / निदान lines
_DIAG_LINE_RE = re.compile(r"(?iu)(?:Diagnosis|निदान)\s*[:=-]\s*[^\n]+")

# K/C/O or H/O clinical history (CRITICAL: H/o DM, H/o HTN = History of, NOT Husband of!)
_HISTORY_RE = re.compile(
    r"(?i)(?:\b(?:no\s+)?k\s*/\s*c\s*/\s*o\b|\bh\s*/\s*o\b)\s+[^.\n;]+"
)

# C/O complaint vs C/O relative (never match inside K/C/O)
_CO_RE = re.compile(r"(?i)(?<![A-Za-z/])c\s*/\s*o\s+([^.,\n;]+)")

# Clinical values: doses, frequencies (1-0-1), BP (120/80), vision (20/20), pain (7/10), GCS (15/15),
# SNOMED/ICD-10/LOINC codes, numbers + clinical units
_CLINICAL_VALUE_RE = re.compile(
    r"(?i)(?:"
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|gm|ml|l|iu|units?|puffs?|drops?|%|mmhg|bpm|g/dl|mg/dl|u/l|iu/l|mmol/l|meq/l|miu/l|/ul|kg|cm|mm)\b"
    r"|\b[012]-[012]-[012](?:-[012])?\b"
    r"|\b(?:OD|BD|TDS|QID|HS|SOS|PRN)\b(?:\s*\([^)]+\))?(?:\s*x\s*\d+\s*(?:days?|wks?|weeks?|months?|yrs?))?"
    r"|\bx\s*\d+\s*(?:days?|d|wks?|weeks?|months?|yrs?|years?)\b"
    r"|\b\d+\s*(?:yrs?|years?|days?|wks?|weeks?|months?)\s+back\b"
    r"|\b(?:0\.\d+|[1-9]\d{0,2}(?:\.\d+)?)\s*-\s*(?:[1-9]\d{0,2}(?:\.\d+)?)\b"
    r"|\b(?:4000-11000|150000-410000)\b"
    r"|\bBP\s*[:=]?\s*\d{2,3}/\d{2,3}\b"
    r"|\b(?:vision|va|pain|gcs)\s*[:=]?\s*\d{1,2}/\d{1,2}\b"
    r"|\b(?:20/20|15/15|120/80)\b"
    r"|\b(?:SNOMED|ICD-?10|LOINC)\s*[:#-]?\s*[A-Z0-9.-]+\b"
    r"|\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b"  # ICD-10 like E27.1, N39.0, J18.9, B54
    r")"
)

_ASHA_RE = re.compile(r"(?i)\bASHA\s+(?:worker|didi|behen|supervisor)\b|\breferred\s+by\s+ASHA\b")

_EPONYM_SUFFIX_RE = re.compile(
    r"(?i)\b([A-Za-z][A-Za-z'-]+(?:'s|s')?)\s+"
    r"(?:disease|syndrome|procedure|lymphoma|sign|triad|phenomenon|test|score|scale|palsy|reflex|nodes?|tumor|tumour|lactate|catheter|tube|tendon|fracture|ulcer|angina|repair|triad|cirrhosis|ataxia)\b"
)

_DEPARTMENTS = frozenset({
    "cardiology", "opd", "ipd", "icu", "nicu", "picu", "ccu", "ent", "obg",
    "obgyn", "pediatrics", "paediatrics", "medicine", "surgery", "pulmonology",
    "neurology", "nephrology", "endocrinology", "dermatology", "psychiatry",
    "orthopaedics", "orthopedics", "ophthalmology", "gynaecology", "obstetrics",
    "oncology", "urology", "gastroenterology", "radiology", "pathology",
})

_SYMPTOM_HINTS = frozenset({
    "fever", "cough", "pain", "headache", "giddiness", "dizziness", "vomiting",
    "nausea", "loose", "stools", "diarrhoea", "diarrhea", "weakness", "swelling",
    "breathlessness", "dyspnoea", "dyspnea", "chest", "abdomen", "abdominal",
    "burning", "micturition", "itching", "rash", "bleeding", "chills", "rigors",
    "joint", "backache", "bodyache", "cold", "sore", "throat", "malaise",
    "fatigue", "anorexia", "loss", "weight", "yellow", "jaundice", "edema",
    "htn", "dm", "dm2", "tb", "cad", "ckd", "copd", "uti", "asthma",
    "बुखार", "खांसी", "दर्द", "चक्कर", "उल्टी", "दस्त", "कमजोरी", "सूजन", "सांस",
})

_ADDRESS_AFTER_CO = re.compile(
    r"(?i)(?:h\.?\s*no|house|makan|ward|mohalla|vill|village|gram|tehsil|distt|district|\b\d{6}\b)"
)


def _load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        ln.strip().lower()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans)
    if not ordered:
        return []
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def is_in_spans(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and end > s:
            return True
        if s >= end:
            break
    return False


class MedicalGuard:
    """Identifies protected clinical spans that must never be redacted as PII."""

    def __init__(self, cfg: RuntimeConfig | None = None) -> None:
        med = cfg.medical_terms if cfg else {}
        self.eponym_phrases: set[str] = _load_lines(DATA_DIR / "medical_eponyms.txt")
        self.eponyms: set[str] = {w.lower() for w in med.get("eponyms", [])}
        for ep in self.eponym_phrases:
            for tok in re.findall(r"[a-z]{3,}", ep):
                if tok not in ("disease", "syndrome", "sign", "test", "score", "tube", "node", "cells", "cell", "tumor"):
                    self.eponyms.add(tok)

        self.drug_phrases: set[str] = _load_lines(DATA_DIR / "indian_drugs.txt")
        self.drugs: set[str] = {w.lower() for w in med.get("drugs", [])} | {
            tok for dp in self.drug_phrases for tok in dp.split() if len(tok) >= 3
        }
        self.abbreviations: set[str] = {w.lower() for w in med.get("abbreviations", [])}
        self.clinical_words: set[str] = {w.lower() for w in med.get("clinical_words", [])} | _DEPARTMENTS
        self.all_medical: set[str] = (
            self.eponyms | self.drugs | self.abbreviations | self.clinical_words
        )
        geo = cfg.indian_geo if cfg else {}
        self.geo_words: set[str] = (
            {w.lower() for w in geo.get("states", [])}
            | {w.lower() for w in geo.get("districts", [])}
        )
        # Precompile multi-word eponym + drug regex for fast span protection
        multi_eponyms = sorted({p for p in self.eponym_phrases if " " in p or "'" in p}, key=lambda x: -len(x))
        if multi_eponyms:
            self._multi_ep_rx: re.Pattern[str] | None = re.compile(
                r"(?i)(?<![A-Za-z])(?:" + "|".join(re.escape(p) for p in multi_eponyms) + r")(?![A-Za-z])"
            )
        else:
            self._multi_ep_rx = None

    def find_protected_spans(self, text: str, *, is_address_field: bool = False) -> list[tuple[int, int]]:
        norm = normalize_text(text)
        spans: list[tuple[int, int]] = []

        # 1. Doctor / clinician names (unless line starts with Patient Name:)
        for m in _DOCTOR_RE.finditer(norm):
            s, e = m.span()
            before = norm[max(0, s - 16):s].lower()
            if "patient" in before:
                continue
            seg = norm[s:e]
            toks = seg.split()
            while len(toks) > 2 and toks[-1].rstrip(".,:;").lower() in _DOCTOR_STOP_WORDS:
                toks.pop()
            trimmed = " ".join(toks)
            spans.append((s, s + len(trimmed)))

        # 2. Clinical Ward/Bed & Diagnosis lines
        if not is_address_field:
            for m in _CLINICAL_WARD_RE.finditer(norm):
                ctx = norm[max(0, m.start() - 40):min(len(norm), m.end() + 60)].lower()
                if re.search(r"\b(?:h\.?\s*no|makan|mohalla|near|opp|tehsil|distt|vill|gram)\b", ctx):
                    continue
                spans.append(m.span())
            for m in _DIAG_LINE_RE.finditer(norm):
                spans.append(m.span())

        # 3. K/C/O and H/O clinical history ("H/o HTN", "H/o DM" -> History of, NOT Husband of!)
        for m in _HISTORY_RE.finditer(norm):
            spans.append(m.span())

        # 4. C/O complaint (protect only when NOT care-of person/address)
        if not is_address_field:
            for m in _CO_RE.finditer(norm):
                rhs = m.group(1).strip()
                rhs_low = rhs.lower()
                if _ADDRESS_AFTER_CO.search(rhs):
                    continue
                words = [w.strip(".,:;()") for w in rhs_low.split()]
                if any(w in _SYMPTOM_HINTS or w in self.clinical_words for w in words):
                    spans.append(m.span())
                elif re.search(r"\bx\s*\d+\s*(?:days?|wks?|weeks?|months?|yrs?)", rhs_low):
                    spans.append(m.span())
                elif rhs and rhs[0].islower():
                    spans.append(m.span())

        # 5. Clinical values (doses, vitals, labs, ICD-10, SNOMED, LOINC)
        for m in _CLINICAL_VALUE_RE.finditer(norm):
            spans.append(m.span())

        # 6. ASHA worker
        for m in _ASHA_RE.finditer(norm):
            spans.append(m.span())

        # 7. Eponyms (gazetteer multi-word + suffix regex)
        if self._multi_ep_rx is not None:
            for m in self._multi_ep_rx.finditer(norm):
                spans.append(m.span())
        for m in _EPONYM_SUFFIX_RE.finditer(norm):
            stem = m.group(1).lower().rstrip("'s").rstrip("'")
            if stem in self.eponyms or m.group(1).lower() in self.eponyms:
                spans.append(m.span())

        return merge_spans(spans)

    def is_medical_or_geo_phrase(self, phrase: str) -> bool:
        norm_p = normalize_text(phrase).strip().lower()
        if norm_p in self.eponym_phrases or norm_p in self.drug_phrases:
            return True
        toks = [t.strip(".,:;()[]{}").lower() for t in norm_p.split() if t.strip()]
        if not toks:
            return True
        if all(t in self.all_medical or t in self.geo_words for t in toks):
            return True
        if len(toks) == 1 and toks[0].rstrip("'s") in self.eponyms:
            return True
        return False
