"""Unicode normalization, index mapping, homoglyph folding, and Devanagari-Latin phonetic transliteration."""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH_CHARS = frozenset({
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff", "\u00ad", "\u2060",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
})
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff\u00ad]")
_BIDI_RLO_RE = re.compile(r"\u202e([^\u202c\r\n]{1,64})\u202c?")

_HOMOGLYPHS = {
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X", "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0406": "I",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X", "\u03b1": "a",
    "\u03bf": "o", "\u03c1": "p",
}
_HOMOGLYPH_TRANS = str.maketrans(_HOMOGLYPHS)
_INDIC_DIGITS = str.maketrans("०१२३४५६७८९٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f",
    "य़": "y",
}

_DEVA_INDEP_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऑ": "o", "ॲ": "e", "ऍ": "e",
}

_DEVA_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}

_DEVA_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}

_HONORIFIC_TOKENS = frozenset({
    "ji", "sahab", "saheb", "didi", "behenji", "bahenji", "bahanji",
    "bhai", "bhaiya", "babu", "amma", "chacha", "chachu", "mama",
    "papa", "tau", "kaka", "bua", "mausi", "nana", "nani", "dada", "dadi",
    "garu", "anna", "akka", "ben",
    "mr", "mrs", "ms", "miss", "shri", "sri", "smt", "shrimati", "kumari",
    "master", "baby", "late", "sh", "km", "dr", "sriman",
    "so", "do", "wo", "co", "s/o", "d/o", "w/o", "c/o",
})

_WEAK_TOKENS = frozenset({"devi", "kumar", "kumari", "ji"})


def strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text)


def fold_homoglyphs(text: str) -> str:
    return text.translate(_HOMOGLYPH_TRANS)


def normalize_digits(text: str) -> str:
    return text.translate(_INDIC_DIGITS)


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize digits/homoglyphs/zero-width chars while preserving an exact index map back to `text`."""
    norm_chars: list[str] = []
    index_map: list[int] = []
    for orig_idx, ch in enumerate(text):
        if ch in _ZERO_WIDTH_CHARS:
            continue
        if ch == "\x00":
            norm_chars.append(" ")
            index_map.append(orig_idx)
            continue
        mapped = _HOMOGLYPHS.get(ch, ch).translate(_INDIC_DIGITS)
        nfkc = unicodedata.normalize("NFKC", mapped)
        for out_ch in nfkc:
            norm_chars.append(out_ch)
            index_map.append(orig_idx)
    return "".join(norm_chars), index_map


def map_span_to_original(start: int, end: int, index_map: list[int], orig_len: int) -> tuple[int, int]:
    """Map a [start, end) span in normalized text back to [orig_start, orig_end) in original text."""
    if not index_map or start >= end or start >= len(index_map):
        return start, end
    orig_start = index_map[start]
    last_idx = min(end - 1, len(index_map) - 1)
    orig_end = min(orig_len, index_map[last_idx] + 1)
    return orig_start, orig_end


def normalize_text(text: str) -> str:
    """Resolve Bidi RLO spans, NFKC normalize, strip zero-width/Bidi chars, fold homoglyphs, map Indic digits."""
    if "\x00" in text:
        text = text.replace("\x00", " ")
    if "\u202e" in text:
        text = _BIDI_RLO_RE.sub(lambda m: m.group(1)[::-1], text)
    s = unicodedata.normalize("NFKC", text)
    s = strip_zero_width(s)
    s = fold_homoglyphs(s)
    s = normalize_digits(s)
    return s


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def devanagari_to_latin(text: str) -> str:
    """Schwa-aware Devanagari to Latin phonetic transliterator (e.g., राजेश -> rajesh, शर्मा -> sharma)."""
    s = normalize_text(text)
    out: list[str] = []
    n = len(s)
    i = 0
    word_chars = 0
    while i < n:
        ch = s[i]
        if i + 1 < n and s[i + 1] == "\u093c" and (ch + "\u093c") in _DEVA_CONSONANTS:
            cons = _DEVA_CONSONANTS[ch + "\u093c"]
            i += 2
            step_consumed = True
        elif ch in _DEVA_CONSONANTS:
            cons = _DEVA_CONSONANTS[ch]
            i += 1
            step_consumed = True
        else:
            step_consumed = False

        if step_consumed:
            word_chars += 1
            if i < n and s[i] == "\u094d":  # virama (halant)
                out.append(cons)
                i += 1
            elif i < n and s[i] in _DEVA_MATRAS:
                out.append(cons + _DEVA_MATRAS[s[i]])
                i += 1
            else:
                # Word-final schwa deletion when word has >1 akshara
                is_word_end = i >= n or not ("\u0900" <= s[i] <= "\u097f")
                # Also drop medial schwa before a consonant+matra when 3+ consonants (e.g. पार्वती, सुखविंदर)
                if is_word_end and word_chars > 1:
                    out.append(cons)
                else:
                    out.append(cons + "a")
            continue

        if ch in _DEVA_INDEP_VOWELS:
            word_chars += 1
            out.append(_DEVA_INDEP_VOWELS[ch])
        elif ch in _DEVA_SIGNS:
            out.append(_DEVA_SIGNS[ch])
        elif ch == "\u093c":
            pass
        else:
            word_chars = 0
            out.append(ch)
        i += 1
    return "".join(out)


def phonetic_token(tok: str) -> str:
    """Fold Indian/English transliteration variants into a single canonical phonetic key."""
    t = devanagari_to_latin(tok).lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9]", "", t)
    if not t:
        return ""
    if t in ("md", "mohd", "mhmd", "mohammad", "mohammed", "muhammad", "muhammed"):
        return "mohamad"
    if t in ("kum", "km"):
        return "kumar"
    if t in ("nair", "nayar"):
        return "nayar"
    if t in ("iyer", "aiyar", "ayar", "ayyar"):
        return "ayar"
    # Phonetic equivalences: v<->w, ee<->i, oo<->u, ph<->f, sh<->s, z<->j, k<->q, doubled letters
    t = t.replace("ph", "f").replace("gh", "g").replace("kh", "k")
    t = t.replace("bh", "b").replace("dh", "d").replace("th", "t")
    t = t.replace("jh", "j").replace("chh", "ch").replace("sh", "s")
    t = t.replace("w", "v").replace("q", "k").replace("z", "j")
    t = t.replace("ee", "i").replace("oo", "u").replace("aa", "a")
    t = t.replace("au", "o").replace("ai", "e").replace("ou", "o")
    t = t.replace("ao", "o").replace("uo", "u")
    t = re.sub(r"(.)\1+", r"\1", t)
    if len(t) > 4 and t.endswith("ey"):
        t = t[:-1]
    elif len(t) > 5 and t.endswith("eya"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("iya"):
        t = t[:-1]
    if len(t) > 3:
        t = t[0] + re.sub(r"h", "", t[1:])
    return t


def canonical_name_tokens(name: str, drop_honorifics: bool = True) -> list[str]:
    """Return phonetic tokens for a Latin or Devanagari name."""
    lat = devanagari_to_latin(name)
    raw_tokens = re.findall(r"[A-Za-z0-9]+", lat.lower())
    out: list[str] = []
    for tok in raw_tokens:
        if drop_honorifics and tok in _HONORIFIC_TOKENS:
            continue
        pt = phonetic_token(tok)
        if pt and len(pt) >= 2:
            out.append(pt)
    return out


def parse_indian_name(name: str) -> dict[str, object]:
    """Parse North/South Indian name tokens into (first, middle, last, initials)."""
    lat = devanagari_to_latin(name)
    raw = re.findall(r"[A-Za-z]+", lat)
    cleaned = [t for t in raw if t.lower() not in _HONORIFIC_TOKENS]
    if not cleaned:
        return {"first": "", "middle": "", "last": "", "initials": []}

    initials = [t.upper() for t in cleaned if len(t) == 1]
    words = [t for t in cleaned if len(t) > 1]

    if not words:
        return {"first": "", "middle": "", "last": "", "initials": initials}
    if len(words) == 1:
        # Handle "K. Venkatraman" or "Venkatraman K."
        return {"first": words[0].lower(), "middle": "", "last": "", "initials": initials}
    if len(words) == 2:
        return {"first": words[0].lower(), "middle": "", "last": words[1].lower(), "initials": initials}
    return {
        "first": words[0].lower(),
        "middle": " ".join(w.lower() for w in words[1:-1]),
        "last": words[-1].lower(),
        "initials": initials,
    }


def canonical_name_key(name: str) -> str:
    toks = canonical_name_tokens(name)
    if not toks:
        return ""
    return " ".join(sorted(toks))


def strong_name_tokens(name: str) -> list[str]:
    lat = devanagari_to_latin(name)
    raw_tokens = re.findall(r"[A-Za-z0-9]+", lat.lower())
    out: list[str] = []
    for tok in raw_tokens:
        if tok in _HONORIFIC_TOKENS or tok in _WEAK_TOKENS:
            continue
        pt = phonetic_token(tok)
        if pt and len(pt) >= 2:
            out.append(pt)
    return out


def is_honorific_only(name: str) -> bool:
    return len(strong_name_tokens(name)) == 0


def canonical_place(place: str) -> str:
    return " ".join(canonical_name_tokens(place, drop_honorifics=False))
