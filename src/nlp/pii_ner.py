"""Context-aware multilingual NER with cluster-boosted redaction & medical guard."""
from __future__ import annotations

import regex as re
from rapidfuzz import fuzz

from src.config import RuntimeConfig
from src.nlp.indic_norm import normalize_text
from src.nlp.medical_guard import MedicalGuard, is_in_spans
from src.nlp.pii_regex import PiiMatch

# Explicit labeled name anchors (EN / HI / Hinglish)
_PATIENT_LABEL_RE = re.compile(
    r"(?iu)(?:(?:Patient\s*Name|Patient|Pt\.?|Name|Naam|रोगी\s*का\s*नाम|मरीज़\s*का\s*नाम|नाम)\s*[:=-]|Mera\s+naam\s*[:=-]?)"
    r"\s*([^\n,;|()]+?)(?=\s*(?:Age|Sex|Gender|UHID|IPD|Reg|DOB|Mobile|Mob|आयु|लिंग|hai|हूँ|\n|,|;|\||$))"
)

# Honorific title prefixes ("Mr. Ramesh Kumar", "Smt. Sunita Devi", "Baby of Meera", "Master Aryan")
_TITLE_PREFIX_NAME_RE = re.compile(
    r"(?u)\b(?:Mr\.?|Mrs\.?|Ms\.?|Shri|Sh\.?|Smt\.?|Km\.?|Kumari|Master|Baby(?:\s+of)?|Late)"
    r"\s+([A-Z\u0900-\u097f][^\n,;.()]{1,35}?)(?=\s*(?:Age|Sex|W/o|S/o|D/o|,|\n|;|\(|$))"
)

# Relation labels (EN / HI / Hinglish)
_RELATIVE_LABEL_RE = re.compile(
    r"(?iu)(?:S\s*/\s*o|D\s*/\s*o|W\s*/\s*o|Father(?:'s)?\s*Name|Husband(?:'s)?\s*Name|Guardian|Attendant|Next\s+of\s+Kin|"
    r"pita\s+ka\s+naam|pati\s+ka\s+naam|"
    r"पिता/पति\s*का\s*नाम|पति\s*का\s*नाम|पिता\s*का\s*नाम|माता\s*का\s*नाम|पुत्र|पुत्री|पत्नी)"
    r"\s*[:.]?\s*([^\n,;|()]+?)(?=\s*(?:Address|Age|Sex|Mob|Mobile|Phone|आयु|पता|\n|,|;|\(|\)|$))"
)

# Care-of person (C/O complaints are already protected by medical_guard!)
_CARE_OF_RE = re.compile(
    r"(?iu)\bC\s*/\s*O\s+([A-Z\u0900-\u097f][\w\u0900-\u097f.-]*(?:\s+[A-Z\u0900-\u097f][\w\u0900-\u097f.-]*){0,2})"
)

# Fragmented "<Name>, W/o <RelName>"
_FRAG_WO_RE = re.compile(
    r"(?u)(?:^|(?<=[\n.]))[ \t]*([A-Z\u0900-\u097f][^\n,;]{1,35}?)"
    r"[ \t]*,[ \t]*(?i:S\s*/\s*o|D\s*/\s*o|W\s*/\s*o)[ \t]+([A-Z\u0900-\u097f][^\n,;.()]{1,35})"
)

# Honorific / Hinglish / WhatsApp patterns (case-sensitive leading capital for Latin!)
_HONORIFIC_PATIENT_RE = re.compile(
    r"(?u)\b([A-Z][a-z]{1,24}(?:\s+[A-Z][a-z]{1,24})?)\s+"
    r"(?i:ji|sahab|saheb|didi|behenji|bahenji|bhai|bhaiya|babu|amma|chachu|mama|papa)\b"
)

_REPORT_OF_RE = re.compile(
    r"(?iu)(?:Report\s+of|OPD\s+note\s+for|Rx\s+of|witnessed\s+by|witness|called\s+by|raised\s+by|Referral\s*:)\s+"
    r"([A-Z\u0900-\u097f][\w\u0900-\u097f.-]{1,25}(?:\s+[A-Z\u0900-\u097f][\w\u0900-\u097f.-]{1,25}){0,2})"
)

_POSSESSIVE_ATTENDANT_RE = re.compile(
    r"(?iu)\b([A-Z][a-z]{2,20})(?:'s|’s)\s+(?:attendant|wife|husband|son|daughter|father|mother)\b"
)

_HINDI_KO_RE = re.compile(
    r"(?u)(?:^|(?<=[\n।.]))\s*([\u0900-\u097f]{2,20}(?:\s+[\u0900-\u097f]{2,20}){0,2})\s+को\s+\d+"
)

_HINDI_GRAM_RE = re.compile(r"(?u)(?:ग्राम|गाँव)\s*[:.]?\s*([^\n,|;]{2,40})")

_TITLE_NAME_RE = re.compile(r"\b([A-Z][a-z]{1,20}(?:[-'][A-Z][a-z]{1,20})?(?:\s+[A-Z][a-z]{1,20}(?:[-'][A-Z][a-z]{1,20})?){1,2})\b")
_DEVA_NAME_RE = re.compile(r"([\u0900-\u097f]{2,20}(?:\s+[\u0900-\u097f]{2,20}){1,2})")
_WHATSAPP_SENDER_RE = re.compile(r"\]\s*([^:\n]{2,40}):")
_WORD_TOK_RE = re.compile(r"[A-Za-z\u0900-\u097f]{3,}")


class PiiNer:
    """Context-aware multilingual NER with medical guard and cluster-boosted matching."""

    def __init__(self, cfg: RuntimeConfig, guard: MedicalGuard) -> None:
        self.guard = guard
        hon = cfg.honorifics or {}
        self.honorifics = {w.lower() for w in hon.get("honorifics", [])}
        self.non_name_words = {w.lower() for w in hon.get("non_name_words", [])}
        self._known_name_rx: re.Pattern[str] | None = None

    def register_known_entities(
        self,
        patient_names: set[str],
        relative_names: set[str],
        villages: set[str],
    ) -> None:
        """Build an in-memory compiled pattern for dataset-level names discovered in Pass 1."""
        terms: set[str] = set()
        for src in (patient_names, relative_names, villages):
            for item in src:
                s = normalize_text(item).strip()
                if len(s) >= 3 and not self.guard.is_medical_or_geo_phrase(s):
                    terms.add(s)
                    for tok in re.findall(r"[A-Za-z\u0900-\u097f]{4,}", s):
                        tl = tok.lower()
                        if (
                            tl not in self.guard.all_medical
                            and tl not in self.guard.geo_words
                            and tl not in self.non_name_words
                            and tl not in self.honorifics
                        ):
                            terms.add(tok)
        if not terms:
            self._known_name_rx = None
            return
        ordered = sorted(terms, key=lambda x: (-len(x), x))
        escaped = [re.escape(t) for t in ordered]
        self._known_name_rx = re.compile(
            r"(?iu)(?<![\w\u0900-\u097f])(?:" + "|".join(escaped) + r")(?![\w\u0900-\u097f])"
        )

    def _looks_like_non_name(self, candidate: str) -> bool:
        toks = [t.strip(".,:;()[]{}").lower() for t in candidate.split() if t.strip()]
        if not toks:
            return True
        if self.guard.is_medical_or_geo_phrase(candidate):
            return True
        if all(
            t in self.non_name_words
            or t in self.honorifics
            or t in self.guard.all_medical
            or t in self.guard.geo_words
            for t in toks
        ):
            return True
        if any(any(c.isdigit() for c in t) for t in toks):
            return True
        return False

    def detect(
        self,
        text: str,
        protected_spans: list[tuple[int, int]],
        *,
        is_address_field: bool = False,
        allow_unlabeled_fallback: bool = True,
        known_patient_tokens: set[str] | None = None,
    ) -> list[PiiMatch]:
        norm = normalize_text(text)
        matches: list[PiiMatch] = []

        def _add(start: int, end: int, kind: str, prio: int = 75) -> None:
            if start >= end:
                return
            if is_in_spans(start, end, protected_spans):
                return
            seg = norm[start:end].strip(" ,.:;()[]")
            if len(seg) < 2 or self._looks_like_non_name(seg):
                return
            matches.append(PiiMatch(start, end, kind, norm[start:end], seg, prio))

        # 1. Explicit patient labels
        for m in _PATIENT_LABEL_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 85)

        # 2. Honorific title prefixes ("Mr. Ramesh Kumar", "Baby of Meera")
        for m in _TITLE_PREFIX_NAME_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 84)

        # 3. Explicit relative labels
        for m in _RELATIVE_LABEL_RE.finditer(norm):
            val = m.group(1).strip()
            if val.lower() not in ("na", "n/a", "none", "-"):
                _add(m.start(1), m.end(1), "RELATIVE_NAME", 84)

        # 4. C/O <Name>
        for m in _CARE_OF_RE.finditer(norm):
            _add(m.start(1), m.end(1), "RELATIVE_NAME", 80)

        # 5. Fragmented "<Name>, W/o <RelName>"
        for m in _FRAG_WO_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 82)
            _add(m.start(2), m.end(2), "RELATIVE_NAME", 82)

        # 6. WhatsApp sender "[date] Kamal Bhaiya:"
        wa_sender_spans: list[tuple[int, int]] = []
        for m in _WHATSAPP_SENDER_RE.finditer(norm):
            wa_sender_spans.append(m.span(1))
            _add(m.start(1), m.end(1), "RELATIVE_NAME", 80)

        # 7. Honorific patient names ("Shahid sahab", "Asha ji", "Sunita didi")
        for m in _HONORIFIC_PATIENT_RE.finditer(norm):
            if is_in_spans(m.start(), m.end(), wa_sender_spans):
                continue
            _add(m.start(), m.end(), "PATIENT_NAME", 78)

        # 8. "Report of X", "called by X", "witnessed by X", "raised by X"
        for m in _REPORT_OF_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 81)

        # 9. "Asha's attendant"
        for m in _POSSESSIVE_ATTENDANT_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 77)

        # 10. Hindi "<Name> को 9 दिन से..."
        for m in _HINDI_KO_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 80)

        # 11. Hindi village "ग्राम: ..."
        for m in _HINDI_GRAM_RE.finditer(norm):
            _add(m.start(1), m.end(1), "VILLAGE", 78)

        # 12. Cluster-Boosted Redaction (exact + fuzzy match against linked patient tokens, guarded by medical_guard!)
        if known_patient_tokens:
            clean_cluster_toks = {
                normalize_text(t).strip().lower()
                for t in known_patient_tokens
                if len(normalize_text(t).strip()) >= 3
                and not self.guard.is_medical_or_geo_phrase(t)
                and normalize_text(t).strip().lower() not in self.non_name_words
                and normalize_text(t).strip().lower() not in self.honorifics
            }
            if clean_cluster_toks:
                for m in _WORD_TOK_RE.finditer(norm):
                    if is_in_spans(m.start(), m.end(), protected_spans):
                        continue
                    w = m.group(0)
                    wl = w.lower()
                    if wl in self.guard.all_medical or wl in self.guard.geo_words or wl in self.non_name_words:
                        continue
                    if wl in clean_cluster_toks:
                        _add(m.start(), m.end(), "PATIENT_NAME", 88)
                    elif len(wl) >= 5:
                        for ctok in clean_cluster_toks:
                            if len(ctok) >= 5 and abs(len(ctok) - len(wl)) <= 2 and fuzz.ratio(wl, ctok) >= 88:
                                _add(m.start(), m.end(), "PATIENT_NAME", 86)
                                break

        # 13. Known dataset-level names/villages discovered in Pass 1
        if self._known_name_rx is not None:
            for m in self._known_name_rx.finditer(norm):
                _add(m.start(), m.end(), "PATIENT_NAME", 74)

        # 14. Unlabeled Title-Case / Devanagari multi-word name fallback (catches unseen canaries)
        if allow_unlabeled_fallback and not is_address_field:
            for m in _TITLE_NAME_RE.finditer(norm):
                cand = m.group(1)
                toks = cand.split()
                if any(
                    t.lower() in self.guard.all_medical
                    or t.lower() in self.guard.geo_words
                    or t.lower() in self.non_name_words
                    for t in toks
                ):
                    continue
                _add(m.start(1), m.end(1), "PATIENT_NAME", 60)

            for m in _DEVA_NAME_RE.finditer(norm):
                cand = m.group(1)
                toks = cand.split()
                if any(t in self.guard.all_medical or t in self.guard.geo_words or t in self.honorifics for t in toks):
                    continue
                if len(toks) >= 2:
                    _add(m.start(1), m.end(1), "PATIENT_NAME", 58)

        return matches
