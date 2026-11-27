"""Deterministic 5-pass multi-key blocking with giant/placeholder block protection."""
from __future__ import annotations

from collections import defaultdict

from src.models import PatientRecord
from src.nlp.indic_norm import (
    canonical_name_key,
    canonical_place,
    parse_indian_name,
    phonetic_token,
    strong_name_tokens,
)

_PLACEHOLDER_DIGITS = frozenset({
    "0000000000", "9999999999", "1111111111", "1234567890",
    "000000000000", "999999999999", "111111111111",
    "unknown", "na", "none", "null", "xxxx",
})


def _is_placeholder(val: str) -> bool:
    v = val.strip().lower()
    if not v or v in _PLACEHOLDER_DIGITS:
        return True
    if len(set(v)) == 1 and len(v) >= 6:
        return True
    return False


def generate_blocks(records: list[PatientRecord], max_block_size: int = 200) -> list[tuple[int, int]]:
    """Return sorted deterministic candidate pairs (i, j) with i < j in O(N*B) time."""
    buckets: dict[str, list[int]] = defaultdict(list)

    for idx, rec in enumerate(records):
        if not rec.has_patient_signal:
            continue

        # Block 1: Exact normalized government IDs & facility-scoped/global MRNs
        for id_type, id_val in rec.gov_ids.items():
            v = id_val.strip().upper()
            if v and not _is_placeholder(v):
                buckets[f"b1:{id_type}:{v}"].append(idx)
        for mrn in rec.all_mrns:
            mv = mrn.strip().upper()
            if mv and not _is_placeholder(mv):
                buckets[f"b1:mrn:{mv}"].append(idx)
        for em in rec.emails:
            ev = em.strip().lower()
            if ev and not _is_placeholder(ev):
                buckets[f"b1:email:{ev}"].append(idx)

        # Block 2: Normalized 10-digit Indian phone number (or 8+ digit prefix)
        for ph in rec.phones:
            digs = "".join(c for c in ph if c.isdigit())
            if len(digs) >= 10:
                last10 = digs[-10:]
                if not _is_placeholder(last10):
                    buckets[f"b2:ph:{last10}"].append(idx)
            elif len(digs) >= 8 and not _is_placeholder(digs):
                buckets[f"b2:ph8:{digs[:8]}"].append(idx)

        # Block 3: (normalized_dob, gender) and (dd_mm_swapped_dob, gender)
        g_tag = rec.gender or "*"
        if rec.dob:
            buckets[f"b3:dob:{rec.dob.isoformat()}|{g_tag}"].append(idx)
            if rec.dob.day <= 12 and rec.dob.month <= 12 and rec.dob.day != rec.dob.month:
                swapped = f"{rec.dob.year:04d}-{rec.dob.day:02d}-{rec.dob.month:02d}"
                buckets[f"b3:dob:{swapped}|{g_tag}"].append(idx)
            if g_tag != "*":
                buckets[f"b3:dob:{rec.dob.isoformat()}|*"].append(idx)

        # Block 4 & Block 5: Phonetic Indian name blocks
        age_5yr = str(rec.reported_age // 5) if rec.reported_age is not None else (
            str(max(0, (rec.reference_date.year if rec.reference_date else 2026) - rec.dob.year) // 5)
            if rec.dob else "*"
        )
        for nm in rec.names_raw:
            ck = canonical_name_key(nm)
            if ck and not _is_placeholder(ck):
                buckets[f"b4:full:{ck}"].append(idx)

            parsed = parse_indian_name(nm)
            first_raw = str(parsed.get("first") or "")
            last_raw = str(parsed.get("last") or "")
            initials: list[str] = list(parsed.get("initials") or [])  # type: ignore[arg-type]

            first_ph = phonetic_token(first_raw) if first_raw else ""
            last_ph = phonetic_token(last_raw) if last_raw else ""

            # Block 4: (phonetic_last_name, phonetic_first_initial, gender)
            # and (phonetic_first_name, phonetic_last_initial, gender)
            if last_ph and (first_ph or initials):
                init_ch = first_ph[0] if first_ph else initials[0].lower()
                buckets[f"b4:li:{last_ph}|{init_ch}|{g_tag}"].append(idx)
                if g_tag != "*":
                    buckets[f"b4:li:{last_ph}|{init_ch}|*"].append(idx)
            if first_ph and last_ph:
                buckets[f"b4:fi:{first_ph}|{last_ph[0]}|{g_tag}"].append(idx)
                if g_tag != "*":
                    buckets[f"b4:fi:{first_ph}|{last_ph[0]}|*"].append(idx)
            elif first_ph and not last_ph and initials:
                # Single full word + initial ("F. Sharma" or "Vikram S."): index under both li and fi
                init_ch = initials[0].lower()
                buckets[f"b4:li:{first_ph}|{init_ch}|{g_tag}"].append(idx)
                buckets[f"b4:fi:{first_ph}|{init_ch}|{g_tag}"].append(idx)
                if g_tag != "*":
                    buckets[f"b4:li:{first_ph}|{init_ch}|*"].append(idx)
                    buckets[f"b4:fi:{first_ph}|{init_ch}|*"].append(idx)

            # Block 5: (phonetic_first_name, age_5yr_bucket, gender) + guardian corroboration
            stoks = strong_name_tokens(nm)
            for fn in stoks[:1]:
                if age_5yr != "*":
                    buckets[f"b5:fa:{fn}|{age_5yr}|{g_tag}"].append(idx)
                    if g_tag != "*":
                        buckets[f"b5:fa:{fn}|{age_5yr}|*"].append(idx)
                for rnm in rec.guardian_names_raw:
                    rt = strong_name_tokens(rnm)
                    if rt:
                        buckets[f"b5:fg:{fn}|{rt[0]}"].append(idx)
                if rec.village:
                    vc = canonical_place(rec.village)
                    if vc:
                        buckets[f"b5:fv:{fn}|{vc}"].append(idx)

    pairs: set[tuple[int, int]] = set()
    for bkey in sorted(buckets):
        members = buckets[bkey]
        if len(members) < 2 or len(members) > max_block_size:
            continue
        for p in range(len(members)):
            i = members[p]
            for q in range(p + 1, len(members)):
                j = members[q]
                pairs.add((i, j) if i < j else (j, i))
    return sorted(pairs)
