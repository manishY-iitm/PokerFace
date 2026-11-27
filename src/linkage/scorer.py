"""Fellegi-Sunter probabilistic pairwise scorer + Hard Must-Not-Link (MNL) Conflict Detector."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.models import PatientRecord
from src.nlp.indic_norm import (
    canonical_name_key,
    canonical_place,
    parse_indian_name,
    phonetic_token,
    strong_name_tokens,
)

DEFAULT_LINK_THRESHOLD = 5.0


@dataclass(frozen=True)
class ScoreResult:
    score: float
    conflict: bool
    reason: str = ""


def _mrn_prefix(mrn: str) -> str:
    m = mrn.strip().upper()
    if m.startswith(("IPD-", "IP/", "ADM", "DHH-IPD-")):
        return ""
    if m.startswith("SNJ/"):
        return "SNJ"
    if m.startswith("JJH-"):
        return "JJH"
    if m.startswith("DHH"):
        return "DHH"
    if m.startswith("UHID-"):
        return "UHID"
    return re.split(r"[-/]", m)[0]


def _is_swapped_dob(d1: dt.date, d2: dt.date) -> bool:
    return d1.year == d2.year and d1.day <= 12 and d2.day <= 12 and d1.day == d2.month and d1.month == d2.day


def _dobs_conflict(d1: dt.date, d2: dt.date) -> tuple[bool, bool]:
    """Return (is_conflict, is_swapped_dd_mm)."""
    if d1 == d2:
        return False, False
    if _is_swapped_dob(d1, d2):
        return False, True
    if abs(d1.year - d2.year) > 1:
        return True, False
    # Same/adjacent year: both month & day differ and not swapped -> conflict
    return True, False


def _consonant_skel(s: str) -> str:
    return re.sub(r"[aeiou]", "", s)


def _name_pair_analysis(na: str, nb: str) -> tuple[str, float]:
    """Compare two Indian names across scripts/initials.

    Returns (status, bonus):
      - status: 'exact', 'initial_surname', 'first_match', 'fuzzy', or 'conflict'
    """
    cka = canonical_name_key(na)
    ckb = canonical_name_key(nb)
    ta = strong_name_tokens(na)
    tb = strong_name_tokens(nb)
    pa = parse_indian_name(na)
    pb = parse_indian_name(nb)

    if not ta or not tb:
        return "neutral", 0.0

    # Exact phonetic match (handles First Last vs Last First & Devanagari vs Latin)
    if cka == ckb and len(ta) >= 2 and len(tb) >= 2:
        return "exact", 4.5

    # Token set overlap (e.g. "Asha" vs "Asha Rajput", or "Shahid" vs "Mohd Shahid")
    sa, sb = set(ta), set(tb)
    if len(ta) == 1 or len(tb) == 1:
        single = ta[0] if len(ta) == 1 else tb[0]
        other = tb if len(ta) == 1 else ta
        # Check exact token match or initial compatibility
        if single in other or any(fuzz.ratio(single, o) >= 88 for o in other):
            return "first_match", 2.5
        # Check initial match against other's first/last
        init_a: list[str] = list(pa.get("initials") or [])  # type: ignore[arg-type]
        init_b: list[str] = list(pb.get("initials") or [])  # type: ignore[arg-type]
        if (init_a and any(o.startswith(i.lower()) for i in init_a for o in tb)) or (
            init_b and any(o.startswith(i.lower()) for i in init_b for o in ta)
        ):
            return "initial_surname", 2.0
        return "conflict", 0.0

    # Both have >= 2 tokens: check initials + surname (e.g. "R. K. Sharma" vs "Rajesh Kumar Sharma" / "राजेश शर्मा")
    init_a = list(pa.get("initials") or [])  # type: ignore[arg-type]
    init_b = list(pb.get("initials") or [])  # type: ignore[arg-type]
    last_a = phonetic_token(str(pa.get("last") or ""))
    last_b = phonetic_token(str(pb.get("last") or ""))
    first_a = phonetic_token(str(pa.get("first") or ""))
    first_b = phonetic_token(str(pb.get("first") or ""))

    # Handle "R. K. Sharma" -> parse_indian_name puts "sharma" in first if only 1 full word!
    if (init_a or init_b) and (sa & sb):
        # Surname matches AND leading initial matches a given token (middle initials optional if omitted)
        inits = init_a if init_a else init_b
        full_toks = tb if init_a else ta
        non_surname = [ft for ft in full_toks if ft not in (sa & sb)] or full_toks
        if any(ft.startswith(inits[0].lower()) for ft in non_surname):
            return "initial_surname", 3.0

    # Full token_sort fuzzy ratio
    ratio = fuzz.token_sort_ratio(cka, ckb)
    if ratio >= 88:
        return "fuzzy", 3.5

    # Check if given names or surnames conflict (stops spouses/twins & same-first-name collisions!)
    if sa <= sb or sb <= sa:
        return "fuzzy", 2.5
    if first_a and first_b and last_a and last_b:
        first_ok = first_a == first_b or fuzz.ratio(first_a, first_b) >= 82
        last_ok = (
            last_a == last_b
            or fuzz.ratio(last_a, last_b) >= 78
            or fuzz.ratio(_consonant_skel(last_a), _consonant_skel(last_b)) >= 78
        )
        if first_ok and last_ok:
            return "fuzzy", 2.5
        return "conflict", 0.0
    if first_a and first_b:
        if first_a == first_b or fuzz.ratio(first_a, first_b) >= 82 or (sa == sb):
            return "fuzzy", 2.0
    if last_a and last_b and last_a == last_b:
        # Same surname but different first names (e.g., Rajesh Sharma vs Sunita Sharma, Aarav vs Arjun)
        return "conflict", 0.0
    if not (sa & sb) and ratio < 75:
        return "conflict", 0.0

    return "fuzzy", 1.5


def _names_compare(a: PatientRecord, b: PatientRecord) -> tuple[bool, float]:
    """Return (has_conflict, best_name_score)."""
    if not a.names_raw or not b.names_raw:
        return False, 0.0
    best = -1.0
    any_compat = False
    for na in a.names_raw:
        for nb in b.names_raw:
            status, sc = _name_pair_analysis(na, nb)
            if status != "conflict":
                any_compat = True
                if sc > best:
                    best = sc
    if not any_compat:
        return True, 0.0
    return False, max(0.0, best)


def _guardians_match(a: PatientRecord, b: PatientRecord) -> tuple[bool, bool]:
    """Return (match, definite_conflict)."""
    if not a.guardian_names_raw or not b.guardian_names_raw:
        return False, False
    for ga in a.guardian_names_raw:
        cka = canonical_name_key(ga)
        ta = strong_name_tokens(ga)
        for gb in b.guardian_names_raw:
            ckb = canonical_name_key(gb)
            tb = strong_name_tokens(gb)
            if not ta or not tb:
                continue
            if cka == ckb or fuzz.token_sort_ratio(cka, ckb) >= 85:
                return True, False
            if ta[0] == tb[0] and (len(ta) == 1 or len(tb) == 1 or ta[-1] == tb[-1]):
                return True, False
    return False, True


def score_pair_detailed(
    a: PatientRecord,
    b: PatientRecord,
    *,
    weights: dict[str, float] | None = None,
) -> ScoreResult:
    """Fellegi-Sunter probabilistic score with strict Hard Must-Not-Link Conflict Guards."""
    if not a.has_patient_signal or not b.has_patient_signal:
        return ScoreResult(-1000.0, True, "no_patient_signal")

    w = weights or {}

    # 1. Hard MNL: Gender conflict
    if a.gender and b.gender and a.gender != b.gender:
        return ScoreResult(-1000.0, True, "gender_conflict")

    # Check exact Gov ID / MRN early so boundary age glitches (e.g. age=0) don't veto exact ID+name pairs
    exact_gov_agree = any(
        a.gov_ids.get(k) and b.gov_ids.get(k) and a.gov_ids.get(k) == b.gov_ids.get(k)
        for k in ("aadhaar", "abha", "pan", "passport", "voter")
    )
    mrns_a_early = {m.strip().upper() for m in a.all_mrns if m}
    mrns_b_early = {m.strip().upper() for m in b.all_mrns if m}
    exact_mrn_agree = bool(mrns_a_early & mrns_b_early)

    # 2. Hard MNL: Age conflict (adjusted for encounter year difference, max 3 years)
    if a.reported_age is not None and b.reported_age is not None:
        year_gap = 0
        if a.reference_date and b.reference_date:
            year_gap = abs(a.reference_date.year - b.reference_date.year)
        if abs(a.reported_age - b.reported_age) > 3 + year_gap:
            if not (exact_gov_agree or exact_mrn_agree):
                return ScoreResult(-1000.0, True, "age_conflict")

    # 3. Hard MNL: Exact DOB conflict (unless DD/MM <-> MM/DD swap)
    dob_swapped = False
    if a.dob and b.dob:
        dob_conf, dob_swapped = _dobs_conflict(a.dob, b.dob)
        if dob_conf:
            return ScoreResult(-1000.0, True, "dob_conflict")

    # 4. Compare names early (needed both for MNL first-name conflict and typo exception)
    name_conflict, name_bonus = _names_compare(a, b)
    if name_conflict:
        return ScoreResult(-1000.0, True, "first_name_conflict")

    # Phone sets
    ph_a = {"".join(c for c in p if c.isdigit())[-10:] for p in a.phones if len("".join(c for c in p if c.isdigit())) >= 10}
    ph_b = {"".join(c for c in p if c.isdigit())[-10:] for p in b.phones if len("".join(c for c in p if c.isdigit())) >= 10}
    shared_phone = bool(ph_a & ph_b)

    ph8_a = {"".join(c for c in p if c.isdigit())[:8] for p in a.phones if len("".join(c for c in p if c.isdigit())) >= 8}
    ph8_b = {"".join(c for c in p if c.isdigit())[:8] for p in b.phones if len("".join(c for c in p if c.isdigit())) >= 8}
    shared_phone8 = bool(ph8_a & ph8_b)

    # 5. Hard MNL: Verified Gov IDs conflict (unless 1-char typo AND name + DOB + phone all strongly match)
    shared_gov_id = False
    for k in ("aadhaar", "abha", "pan", "passport", "voter"):
        va, vb = a.gov_ids.get(k), b.gov_ids.get(k)
        if va and vb:
            if va == vb:
                shared_gov_id = True
            else:
                is_typo = (
                    Levenshtein.distance(va, vb) == 1
                    and name_bonus >= 3.0
                    and (a.dob and b.dob and a.dob == b.dob)
                    and shared_phone
                )
                if not is_typo:
                    return ScoreResult(-1000.0, True, f"gov_id_conflict:{k}")

    # Shared MRN / Insurance / Email
    mrns_a = {m.strip().upper() for m in a.all_mrns if m}
    mrns_b = {m.strip().upper() for m in b.all_mrns if m}
    shared_mrn = bool(mrns_a & mrns_b)
    shared_ins = bool(a.gov_ids.get("insurance") and a.gov_ids.get("insurance") == b.gov_ids.get("insurance"))
    shared_email = bool(set(a.emails) & set(b.emails))

    strong_match = shared_gov_id or shared_mrn or shared_ins or shared_email

    # 6. Guardian / Geo / Same-hospital primary UHID conflict guards
    rel_match, rel_conflict = _guardians_match(a, b)
    if rel_conflict and not (shared_gov_id or shared_phone):
        return ScoreResult(-1000.0, True, "guardian_conflict")

    geo_mismatch = False
    if a.district and b.district:
        da, db = canonical_place(a.district), canonical_place(b.district)
        if da and db and da != db and fuzz.token_sort_ratio(da, db) < 75:
            if not (shared_gov_id or shared_phone or shared_mrn):
                return ScoreResult(-1000.0, True, "district_conflict")
            geo_mismatch = True

    if a.village and b.village:
        va_p, vb_p = canonical_place(a.village), canonical_place(b.village)
        if va_p and vb_p and fuzz.token_sort_ratio(va_p, vb_p) < 75:
            if not (shared_gov_id or shared_phone or shared_mrn):
                return ScoreResult(-1000.0, True, "village_conflict")
            geo_mismatch = True

    if not strong_match and not shared_phone:
        pref_a = {_mrn_prefix(m): m for m in mrns_a if _mrn_prefix(m)}
        pref_b = {_mrn_prefix(m): m for m in mrns_b if _mrn_prefix(m)}
        for p in set(pref_a) & set(pref_b):
            if pref_a[p] != pref_b[p]:
                return ScoreResult(-1000.0, True, "same_hospital_uhid_conflict")

    # 7. Weighted Probabilistic Score
    score = 0.0
    if shared_gov_id:
        score += float(w.get("exact_gov_id", 12.0))
    if shared_mrn or shared_ins or shared_email:
        score += float(w.get("exact_mrn", 10.0))
    if shared_phone:
        score += float(w.get("exact_phone", 4.0))
    elif shared_phone8:
        score += 2.5
    elif ph_a and ph_b:
        score += float(w.get("phone_mismatch", -1.0))

    if a.dob and b.dob:
        if a.dob == b.dob:
            score += float(w.get("exact_dob", 5.0))
        elif dob_swapped:
            score += float(w.get("swapped_dob", 3.0))

    score += name_bonus

    if rel_match:
        score += float(w.get("guardian_match", 3.0))

    same_pin = bool(a.pin_code and b.pin_code and a.pin_code == b.pin_code)
    same_vil = bool(
        a.village and b.village
        and fuzz.token_sort_ratio(canonical_place(a.village), canonical_place(b.village)) >= 80
    )
    if same_pin or same_vil:
        score += float(w.get("same_pin_city", 1.5))
    elif a.pin_code and b.pin_code and a.pin_code != b.pin_code:
        score += float(w.get("geo_mismatch", -0.5))
    elif geo_mismatch:
        score += float(w.get("geo_mismatch", -0.5))

    if a.gender and b.gender and a.gender == b.gender:
        score += float(w.get("same_gender", 1.0))
    if a.reported_age is not None and b.reported_age is not None:
        age_diff = abs(a.reported_age - b.reported_age)
        if age_diff == 0:
            score += float(w.get("exact_age", 1.5))
        elif age_diff <= 2:
            score += float(w.get("compatible_age", 0.5))

    # Rule: Shared phone or shared address ALONE without compatible name/age/gender cannot exceed link threshold
    if not strong_match and name_bonus == 0.0:
        if not (a.dob and b.dob and a.dob == b.dob):
            score = min(score, 4.0)

    return ScoreResult(round(score, 3), False, "")


def score_pair(a: PatientRecord, b: PatientRecord, *, weights: dict[str, float] | None = None) -> float:
    res = score_pair_detailed(a, b, weights=weights)
    return -1000.0 if res.conflict else res.score
