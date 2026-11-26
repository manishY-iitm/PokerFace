"""Global Privacy Engine: iterative bottom-up k-anonymity & l-diversity generalization."""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

from src.config import DATA_DIR, RuntimeConfig, load_json_data
from src.models import PatientRecord
from src.nlp.date_shifter import compute_truthful_age_band
from src.nlp.indic_norm import canonical_place

_PIN_HIERARCHY: dict | None = None


def _get_pin_hierarchy() -> dict:
    global _PIN_HIERARCHY
    if _PIN_HIERARCHY is None:
        if (DATA_DIR / "indian_pin_hierarchy.json").exists():
            _PIN_HIERARCHY = load_json_data("indian_pin_hierarchy.json")
        else:
            _PIN_HIERARCHY = {}
    return _PIN_HIERARCHY


def _age_band_at_level(age: int | None, level: int, ladder: list[int]) -> str:
    if age is None:
        return "*"
    if age >= 85:
        return "85+" if level == 0 else ("80+" if level == 1 else "*")
    if level >= len(ladder):
        return "*"
    w = ladder[level]
    lo = (age // w) * w
    return f"{lo}-{lo + w - 1}"


def _geo_at_level(pin: str | None, state: str | None, district: str | None, level: int) -> tuple[str, str]:
    """Return (geo_label, pin_generalized) for geo ladder level 0..4.

    Ladder:
      0: PIN-3 (560***)
      1: PIN-2 (56****)
      2: State (e.g. Uttar Pradesh)
      3: Region (e.g. North India)
      4: *
    """
    hier = _get_pin_hierarchy()
    pin2_map = hier.get("pin2_to_state_region") or {}
    st_reg_map = hier.get("state_to_region") or {}

    digs = "".join(c for c in (pin or "") if c.isdigit())
    p2 = digs[:2] if len(digs) >= 2 else ""
    inferred = pin2_map.get(p2, {})
    eff_state = state or inferred.get("state") or ("Uttar Pradesh" if district else "India")
    eff_region = inferred.get("region") or st_reg_map.get(eff_state.lower(), "India")

    if level == 0:
        if len(digs) == 6:
            return f"{digs[:3]}***", f"{digs[:3]}***"
        return (district or eff_state), "[PIN]"
    if level == 1:
        if len(digs) >= 2:
            return f"{digs[:2]}****", f"{digs[:2]}****"
        return eff_state, "[PIN]"
    if level == 2:
        return eff_state, "[PIN]"
    if level == 3:
        return eff_region, "[PIN]"
    return "*", "SUPPRESSED"


def apply_kanonymity(records: list[PatientRecord], cfg: RuntimeConfig) -> None:
    """Enforce k-anonymity & l-diversity across linked clusters via bottom-up generalization."""
    priv = cfg.privacy_cfg
    k_target = int(priv.get("k", cfg.min_group_size))
    l_target = int(priv.get("l", cfg.l_diversity))
    age_ladder = [int(x) for x in priv.get("age_band_ladder", [5, 10, 20])]
    if not age_ladder:
        age_ladder = [5, 10, 20]

    clusters: dict[str, list[PatientRecord]] = defaultdict(list)
    for r in records:
        if r.has_patient_signal:
            key = r.cluster_id or r.record_uid
            clusters[key].append(r)

    if not clusters:
        return

    # 1. Extract baseline per-cluster QIs (computed from ORIGINAL dates/ages BEFORE shifting!)
    cluster_meta: dict[str, dict] = {}
    diag_global_freq: Counter[str] = Counter()

    for ckey, grp in clusters.items():
        rep_age: int | None = None
        for r in grp:
            if r.dob is not None:
                ref = r.reference_date or dt.date(2026, 1, 1)
                a = ref.year - r.dob.year - ((ref.month, ref.day) < (r.dob.month, r.dob.day))
                rep_age = max(0, a)
                break
        if rep_age is None:
            ages = [r.reported_age for r in grp if r.reported_age is not None]
            if ages:
                rep_age = max(ages)

        genders = [r.gender for r in grp if r.gender]
        pins = [r.pin_code for r in grp if r.pin_code]
        states = [r.state for r in grp if r.state]
        districts = [r.district for r in grp if r.district]
        diags = {d.strip().lower() for r in grp for d in r.diagnoses if d.strip()}
        for d in diags:
            diag_global_freq[d] += 1

        cluster_meta[ckey] = {
            "age": rep_age,
            "gender": genders[0] if genders else "*",
            "pin": pins[0] if pins else None,
            "state": states[0] if states else None,
            "district": districts[0] if districts else None,
            "diags": diags,
            "age_lvl": 1 if (rep_age is not None and rep_age >= 85) else 0,
            "geo_lvl": 1 if (rep_age is not None and rep_age >= 85) else 0,
            "suppressed": False,
        }

    # 2. Iterative bottom-up generalization (alternate Geo ladder -> Age ladder until k & l satisfied)
    max_age_lvl = len(age_ladder)  # == '*'
    max_geo_lvl = 4                # == '*'

    for _ in range(max_age_lvl + max_geo_lvl + 2):
        eq_classes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for ckey, m in cluster_meta.items():
            if m["suppressed"]:
                eq_key = ("*", "*", "*")
            else:
                ab = _age_band_at_level(m["age"], m["age_lvl"], age_ladder)
                gl, _ = _geo_at_level(m["pin"], m["state"], m["district"], m["geo_lvl"])
                eq_key = (ab, m["gender"], gl)
            eq_classes[eq_key].append(ckey)

        changed = False
        for eq_key, members in eq_classes.items():
            if eq_key == ("*", "*", "*"):
                continue
            k_size = len(members)
            distinct_diags = set()
            has_rare = False
            for ckey in members:
                ds = cluster_meta[ckey]["diags"]
                distinct_diags.update(ds)
                if any(diag_global_freq[d] <= 2 for d in ds):
                    has_rare = True

            # Only require l-diversity when diagnosis data is present in the equivalence class
            l_violated = bool(distinct_diags) and (len(distinct_diags) < min(l_target, k_size) or (has_rare and k_size < k_target))
            k_violated = k_size < k_target

            if k_violated or l_violated:
                for ckey in members:
                    m = cluster_meta[ckey]
                    # Generalize geo first, then age, then suppress extreme outlier
                    if m["geo_lvl"] < max_geo_lvl and (m["geo_lvl"] <= m["age_lvl"] + 1 or m["age_lvl"] >= max_age_lvl):
                        m["geo_lvl"] += 1
                        changed = True
                    elif m["age_lvl"] < max_age_lvl:
                        m["age_lvl"] += 1
                        changed = True
                    elif m["gender"] != "*":
                        m["gender"] = "*"
                        changed = True
                    else:
                        m["suppressed"] = True
                        changed = True
        if not changed:
            break

    # 3. Assign final consistent generalized QIs back onto every record in each cluster
    for ckey, grp in clusters.items():
        m = cluster_meta[ckey]
        has_rare_diag = any(diag_global_freq[d] <= 2 for d in m["diags"])
        high_risk = m["suppressed"] or m["geo_lvl"] >= 2 or m["age_lvl"] >= 1 or has_rare_diag

        for r in grp:
            # Ensure per-record age band ALWAYS contains THAT record's exact true age!
            rec_age = r.reported_age
            if rec_age is None and r.dob is not None:
                ref = r.reference_date or dt.date(2026, 1, 1)
                rec_age = max(0, ref.year - r.dob.year - ((ref.month, ref.day) < (r.dob.month, r.dob.day)))

            if m["suppressed"]:
                r.was_suppressed = True
                r.was_generalized = True
                r.is_rare_risk = True
                r.age_band = "*" if rec_age is None else compute_truthful_age_band(
                    r.dob, r.reference_date, band_width=age_ladder[-1], high_risk=True
                ) if r.dob else _age_band_at_level(rec_age, max_age_lvl - 1, age_ladder)
                r.geo_generalized = "*"
                r.extra["pin_generalized"] = "SUPPRESSED"
                continue

            if rec_age is not None:
                r.age_band = _age_band_at_level(rec_age, min(m["age_lvl"], max_age_lvl - 1), age_ladder)
                r.was_generalized = True

            geo_lbl, pin_gen = _geo_at_level(
                r.pin_code or m["pin"], r.state or m["state"], r.district or m["district"], m["geo_lvl"]
            )
            r.geo_generalized = geo_lbl
            r.extra["pin_generalized"] = pin_gen
            if r.village or r.district or r.pin_code or r.dob:
                r.was_generalized = True
            r.is_rare_risk = high_risk
