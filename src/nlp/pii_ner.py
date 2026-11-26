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
    r"\s*([^\n,;|()]+?)(?=\s*(?:Age|Sex|Gender|UHID|IPD|Reg|DOB|Mobile|Mob|आयु|लिंग|hai|हूँ|\n|,|;|\||\(|\)|$))"
)
_PT_INLINE_RE = re.compile(
    r"(?iu)(?:^|(?<=[\s(\[|]))Pt\.?\s+([A-Z\u0900-\u097f][^\n,;|()]{1,40}?)"
    r"(?=\s*\(\s*\d{1,3}\s*(?:Yrs?|Years?|Y)?\s*(?:/\s*)?[MF]\s*\))"
)

# Honorific title prefixes ("Mr. Ramesh Kumar", "Smt. Sunita Devi", "Baby of Meera", "Master Aryan")
_TITLE_PREFIX_NAME_RE = re.compile(
    r"(?u)\b(?:Mr\.?|Mrs\.?|Ms\.?|Shri|Sh\.?|Smt\.?|Km\.?|Kumari|Master|Baby(?:\s+of)?|Late)"
    r"\s+([A-Z\u0900-\u097f][^\n,;.()]{1,35}?)(?=\s*(?:Age|Sex|W/o|S/o|D/o|,|\n|;|\(|$))"
)

# Relation labels (EN / HI / Hinglish)
_RELATIVE_LABEL_RE = re.compile(
    r"(?iu)(?:(?:S\s*/\s*o|D\s*/\s*o|W\s*/\s*o|Father(?:'s)?\s*Name|Husband(?:'s)?\s*Name|Guardian|Attendant|Next\s+of\s+Kin|"
    r"pita\s+ka\s+naam|pati\s+ka\s+naam|"
    r"(?:पिता/पति|पति|पिता|माता)\s*का\s*नाम|पुत्र|पुत्री|पत्नी)\s*[:.]?|"
    r"(?:पिता/पति|पति|पिता|माता)\s*[:=-])"
    r"\s*([^\n,;|()]+?)(?=\s*(?:\s{2,}|Address|Age|Sex|Mob|Mobile|Phone|Ph\b|Tel\b|counselled|advised|noted|bataya|आयु|पता|\n|,|;|\||\(|\)|$))"
)

_RESIDENT_OF_RE = re.compile(
    r"(?iu)\bresident\s+of\s+([A-Z\u0900-\u097f][\w\u0900-\u097f'-]{2,30}(?:[ \t]+[A-Z\u0900-\u097f][\w\u0900-\u097f'-]{2,20})?)(?=\s*(?:[.,;\n|]|$))"
)

_URL_QUERY_NAME_RE = re.compile(
    r"(?iu)[?&](patient|patient_name|pt|name|fname|lname|user|father|husband|guardian|relative)=([A-Za-z0-9%._+-]{2,80}?)(?=[&\s]|$)"
)
_PERCENT_NAME_RE = re.compile(
    r"(?u)(?<![A-Za-z])([A-Z][a-z]{1,20}(?:%20[A-Z][a-z]{1,20}){1,2})(?![A-Za-z])"
)
_MIXED_SCRIPT_NAME_RE = re.compile(
    r"(?u)(?:(?<![\w\u0900-\u097f])[\u0900-\u097f]{2,20}[ \t]+[A-Z][a-z]{1,20}(?:[-'][A-Z][a-z]{1,20})?(?![\w\u0900-\u097f])|"
    r"(?<![\w\u0900-\u097f])[A-Z][a-z]{1,20}(?:[-'][A-Z][a-z]{1,20})?[ \t]+[\u0900-\u097f]{2,20}(?![\w\u0900-\u097f]))"
)

# Care-of person (C/O complaints are already protected by medical_guard!)
_CARE_OF_RE = re.compile(
    r"(?iu)\bC\s*/\s*O\s+([A-Z\u0900-\u097f][\w\u0900-\u097f.-]*(?:\s+[A-Z\u0900-\u097f][\w\u0900-\u097f.-]*){0,2})"
)

# Fragmented "<Name>, W/o <RelName>"
_FRAG_WO_RE = re.compile(
    r"(?u)(?:^|(?<=[\n.]))[ \t]*(?:(?i:Patient(?:\s*Name)?|Pt\.?|Name|Naam)\s*[:=-][ \t]*)?([A-Z\u0900-\u097f][^\n,;]{1,35}?)"
    r"[ \t]*,[ \t]*(?i:S\s*/\s*o|D\s*/\s*o|W\s*/\s*o)[ \t]+([A-Z\u0900-\u097f][^\n,;.()]{1,35})"
)

# Honorific / Hinglish / WhatsApp patterns (case-sensitive leading capital for Latin!)
_HONORIFIC_PATIENT_RE = re.compile(
    r"(?u)\b([A-Z][a-z]{1,24}(?:\s+[A-Z][a-z]{1,24})?)\s+"
    r"(?i:ji|sahab|saheb|didi|behenji|bahenji|bhai|bhaiya|babu|amma|chachu|mama|papa|garu|anna|akka|ben)\b"
)

_HINGLISH_KIN_RE = re.compile(
    r"(?iu)\b(?:unke|unki|mere|meri|apne)\s+(?:pati|pita|patni|mata|beta|beti|bhai|behen|mama|chacha)\s+"
    r"([A-Z\u0900-\u097f][a-z\u0900-\u097f]{1,24}(?:\s+[A-Z\u0900-\u097f][a-z\u0900-\u097f]{1,24})?)"
)

_BROUGHT_BY_RE = re.compile(
    r"(?iu)\b(?:brought\s+by|accompanied\s+by)\s+(?:his|her|the)?\s*"
    r"(?:son|daughter|wife|husband|father|mother|brother|sister|relative|attendant)?\s*"
    r"([A-Z\u0900-\u097f][a-z\u0900-\u097f]{1,24}(?:\s+[A-Z\u0900-\u097f][a-z\u0900-\u097f]{1,24})?)"
)

_REPORT_OF_RE = re.compile(
    r"(?iu)(?:Report\s+of|OPD\s+note\s+for|Rx\s+of|witnessed\s+by|witness|called\s+by|raised\s+by|signed\s+by(?!\s+dr\.?)|Referral\s*:)\s+"
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
_UNDERSCORE_NAME_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][a-z]{1,20}(?:_[A-Z][a-z]{1,20}){1,2})(?![A-Za-z0-9])")
_DEVA_NAME_RE = re.compile(r"([\u0900-\u097f]{2,20}(?:\s+[\u0900-\u097f]{2,20}){1,2})")
_WHATSAPP_SENDER_RE = re.compile(r"\]\s*([^:\n]{2,40}):")
_WORD_TOK_RE = re.compile(r"[A-Za-z\u0900-\u097f]{3,}")


_EXTRA_NON_NAME_WORDS = frozenset({
    "bataya", "gaya", "agar", "koi", "subah", "raat", "dawai", "bukhar", "khana", "lena",
    "theek", "hai", "bed", "pe", "se", "ka", "ki", "ke", "ne", "ko", "baat", "hui",
    "diya", "liya", "li", "kiya", "nahi", "resident", "seen", "today", "follow", "followup",
    "patient", "advice", "advised", "referred", "report", "discharge", "summary",
    "admission", "notes", "note", "nursing", "progress", "ward", "ref", "self", "none",
    "male", "female", "year", "years", "yrs", "months", "days", "day",
    "hand", "hygiene", "isolate", "escalate", "infection", "control", "sop", "workers",
    "hospital", "trust", "care", "mission", "diagnostics", "rural", "chikitsalaya",
    "unit", "medicine", "cases", "proven", "sepsis", "masks", "visits", "vaccine", "batch", "serial",
    "ज़िला", "जिला", "तहसील", "थाना", "ग्राम", "गाँव", "मकान", "गली", "वार्ड",
    "मोहल्ला", "कॉलोनी", "चौक", "लेन", "पेठ", "टोला", "पता", "रोगी", "मरीज़",
    "पिता", "पति", "माता", "पुत्र", "पुत्री", "पत्नी", "भर्ती", "छुट्टी", "निदान",
    "दवा", "सारांश", "आयु", "वर्ष", "महिला", "पुरुष", "साल", "डॉ", "अस्पताल",
    "चिकित्सालय", "दिनांक", "तारीख", "सं", "नं",
})


def _compile_or_index_terms(
    raw_items: set[str],
    guard: MedicalGuard,
    non_name_words: set[str],
    honorifics: set[str],
) -> tuple[re.Pattern[str] | None, set[str], dict[str, list[str]]]:
    terms: set[str] = set()
    for item in raw_items:
        s = normalize_text(item).strip()
        if len(s) >= 3 and not guard.is_medical_or_geo_phrase(s):
            terms.add(s)
            for tok in re.findall(r"[A-Za-z\u0900-\u097f]{4,}", s):
                tl = tok.lower()
                if (
                    tl not in guard.all_medical
                    and tl not in guard.geo_words
                    and tl not in non_name_words
                    and tl not in honorifics
                ):
                    terms.add(tok)
    if not terms:
        return None, set(), {}
    if len(terms) <= 800:
        ordered = sorted(terms, key=lambda x: (-len(x), x))
        escaped = [re.escape(t) for t in ordered]
        rx = re.compile(
            r"(?iu)(?<![\w\u0900-\u097f])(?:" + "|".join(escaped) + r")(?![\w\u0900-\u097f])"
        )
        return rx, set(), {}
    single: set[str] = set()
    multi: dict[str, list[str]] = {}
    for t in terms:
        tl = t.lower()
        parts = _WORD_TOK_RE.findall(tl)
        if len(parts) <= 1:
            if len(tl) >= 4:
                single.add(tl)
        else:
            multi.setdefault(parts[0], []).append(tl)
    return None, single, multi


class PiiNer:
    """Context-aware multilingual NER with medical guard and cluster-boosted matching."""

    def __init__(self, cfg: RuntimeConfig, guard: MedicalGuard) -> None:
        self.guard = guard
        hon = cfg.honorifics or {}
        self.honorifics = {w.lower() for w in hon.get("honorifics", [])} | {"garu", "anna", "akka", "ben"}
        self.non_name_words = {w.lower() for w in hon.get("non_name_words", [])} | _EXTRA_NON_NAME_WORDS
        self._known_patient_rx: re.Pattern[str] | None = None
        self._known_patient_single: set[str] = set()
        self._known_patient_multi: dict[str, list[str]] = {}
        self._known_relative_rx: re.Pattern[str] | None = None
        self._known_relative_single: set[str] = set()
        self._known_relative_multi: dict[str, list[str]] = {}
        self._known_village_rx: re.Pattern[str] | None = None
        self._known_village_single: set[str] = set()
        self._known_village_multi: dict[str, list[str]] = {}

    def register_known_entities(
        self,
        patient_names: set[str],
        relative_names: set[str],
        villages: set[str],
    ) -> None:
        """Build per-role in-memory index/pattern for dataset-level entities discovered in Pass 1."""
        self._known_patient_rx, self._known_patient_single, self._known_patient_multi = _compile_or_index_terms(
            patient_names, self.guard, self.non_name_words, self.honorifics
        )
        self._known_relative_rx, self._known_relative_single, self._known_relative_multi = _compile_or_index_terms(
            relative_names, self.guard, self.non_name_words, self.honorifics
        )
        self._known_village_rx, self._known_village_single, self._known_village_multi = _compile_or_index_terms(
            villages, self.guard, self.non_name_words, self.honorifics
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
        for m in _PT_INLINE_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 85)

        # 1b. URL query parameters carrying names (?name=..., &father=...)
        for m in _URL_QUERY_NAME_RE.finditer(norm):
            raw_v = m.group(2)
            decoded_v = raw_v.replace("%20", " ").replace("+", " ").strip()
            if not any(ch.isdigit() for ch in decoded_v) and not self._looks_like_non_name(decoded_v):
                rel_keys = ("father", "husband", "guardian", "relative")
                kind = "RELATIVE_NAME" if m.group(1).lower() in rel_keys else "PATIENT_NAME"
                prio = 85 if kind == "RELATIVE_NAME" else 86
                if not is_in_spans(m.start(2), m.end(2), protected_spans):
                    matches.append(PiiMatch(m.start(2), m.end(2), kind, raw_v, decoded_v, prio))

        # 2. Honorific title prefixes ("Mr. Ramesh Kumar", "Baby of Meera")
        for m in _TITLE_PREFIX_NAME_RE.finditer(norm):
            _add(m.start(1), m.end(1), "PATIENT_NAME", 84)

        # 3. Explicit relative labels
        for m in _RELATIVE_LABEL_RE.finditer(norm):
            val = m.group(1).strip()
            if val.lower() not in ("na", "n/a", "none", "-"):
                _add(m.start(1), m.end(1), "RELATIVE_NAME", 84)

        # 3b. Hinglish/English kin narratives ("unke pati Selvaraj", "brought by son Ramesh")
        for m in _HINGLISH_KIN_RE.finditer(norm):
            _add(m.start(1), m.end(1), "RELATIVE_NAME", 82)
        for m in _BROUGHT_BY_RE.finditer(norm):
            _add(m.start(1), m.end(1), "RELATIVE_NAME", 82)

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

        # 11. Hindi village "ग्राम: ..." & English "resident of <Village>"
        for m in _HINDI_GRAM_RE.finditer(norm):
            _add(m.start(1), m.end(1), "VILLAGE", 78)
        for m in _RESIDENT_OF_RE.finditer(norm):
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
                        _add(m.start(), m.end(), "PATIENT_NAME", 76)
                    elif len(wl) >= 5:
                        for ctok in clean_cluster_toks:
                            if len(ctok) >= 5 and abs(len(ctok) - len(wl)) <= 2 and fuzz.ratio(wl, ctok) >= 88:
                                _add(m.start(), m.end(), "PATIENT_NAME", 75)
                                break

        # 13. Known dataset-level names/villages discovered in Pass 1 (role-preserving)
        norm_low = norm.lower()
        for rx, single_set, multi_map, ent_kind, ent_prio in (
            (self._known_patient_rx, self._known_patient_single, self._known_patient_multi, "PATIENT_NAME", 74),
            (self._known_village_rx, self._known_village_single, self._known_village_multi, "VILLAGE", 73),
            (self._known_relative_rx, self._known_relative_single, self._known_relative_multi, "RELATIVE_NAME", 72),
        ):
            if rx is not None:
                for m in rx.finditer(norm):
                    _add(m.start(), m.end(), ent_kind, ent_prio)
            elif single_set or multi_map:
                for m in _WORD_TOK_RE.finditer(norm):
                    if is_in_spans(m.start(), m.end(), protected_spans):
                        continue
                    wl = m.group(0).lower()
                    if wl in multi_map:
                        for phrase in multi_map[wl]:
                            end_pos = m.start() + len(phrase)
                            if end_pos <= len(norm_low) and norm_low[m.start():end_pos] == phrase:
                                _add(m.start(), end_pos, ent_kind, ent_prio)
                                break
                    if wl in single_set:
                        _add(m.start(), m.end(), ent_kind, ent_prio)

        # 14. Unlabeled Title-Case / Mixed-Script / Devanagari / Percent / Underscore name fallback
        if allow_unlabeled_fallback and not is_address_field:
            for m in _PERCENT_NAME_RE.finditer(norm):
                raw_v = m.group(1)
                toks = raw_v.split("%20")
                if any(
                    t.lower() in self.guard.all_medical
                    or t.lower() in self.guard.geo_words
                    or t.lower() in self.non_name_words
                    or t.lower() in self.honorifics
                    for t in toks
                ):
                    continue
                if not is_in_spans(m.start(1), m.end(1), protected_spans):
                    matches.append(PiiMatch(m.start(1), m.end(1), "PATIENT_NAME", raw_v, raw_v.replace("%20", " "), 68))
            for m in _UNDERSCORE_NAME_RE.finditer(norm):
                _add(m.start(1), m.end(1), "PATIENT_NAME", 62)
            for m in _MIXED_SCRIPT_NAME_RE.finditer(norm):
                cand = m.group(0)
                toks = cand.split()
                if any(
                    t.lower() in self.guard.all_medical
                    or t.lower() in self.guard.geo_words
                    or t.lower() in self.non_name_words
                    or t.lower() in self.honorifics
                    for t in toks
                ):
                    continue
                _add(m.start(), m.end(), "PATIENT_NAME", 63)
            for m in _TITLE_NAME_RE.finditer(norm):
                cand = m.group(1)
                toks = cand.split()
                if any(
                    t.lower() in self.guard.all_medical
                    or t.lower() in self.guard.geo_words
                    or t.lower() in self.non_name_words
                    or t.lower() in self.honorifics
                    for t in toks
                ):
                    continue
                _add(m.start(1), m.end(1), "PATIENT_NAME", 60)

            for m in _DEVA_NAME_RE.finditer(norm):
                cand = m.group(1)
                toks = cand.split()
                if any(
                    t in self.guard.all_medical
                    or t in self.guard.geo_words
                    or t in self.non_name_words
                    or t in self.honorifics
                    for t in toks
                ):
                    continue
                if len(toks) >= 2:
                    _add(m.start(1), m.end(1), "PATIENT_NAME", 58)

        return matches
