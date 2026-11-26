"""Track 4 Red-Team Automated Re-Identification Attacker (Category 1)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _age_matches(age_band: str, true_age: int) -> bool:
    ab = age_band.strip()
    if ab == "*" or ab == "SUPPRESSED":
        return True
    m = re.match(r"^(\d+)-(\d+)", ab)
    if m:
        return int(m.group(1)) <= true_age <= int(m.group(2))
    m_plus = re.match(r"^(\d+)\+", ab)
    if m_plus:
        return true_age >= int(m_plus.group(1))
    if ab.isdigit():
        return int(ab) == true_age
    return True


def _geo_matches(san_pin: str, san_city: str, voter_pin: str, voter_city: str) -> bool:
    sp = san_pin.strip()
    sc = san_city.strip()
    if sp in ("*", "SUPPRESSED", "[PIN]", ""):
        pin_ok = True
    elif sp.endswith("****") and sp[:2].isdigit():
        pin_ok = voter_pin.startswith(sp[:2])
    elif sp.endswith("***") and sp[:3].isdigit():
        pin_ok = voter_pin.startswith(sp[:3])
    else:
        pin_ok = (sp == voter_pin)

    if sc in ("*", "SUPPRESSED", "[VILLAGE]", "[CITY]", "India", "North India", "South India", "West India", "East India", "Central India", "Northeast India", ""):
        city_ok = True
    elif sc.lower() in ("uttar pradesh", "karnataka", "maharashtra", "rajasthan", "bihar", "madhya pradesh"):
        city_ok = True
    else:
        city_ok = (sc.lower() == voter_city.lower())

    return pin_ok and city_ok


def test_attacker_reidentification_resistance(chaos_run):
    out = Path(chaos_run["output"])
    manifest = chaos_run["manifest"]
    voter_rows = list(csv.DictReader(Path(manifest["voter_registry"]).open(encoding="utf-8")))
    facts = [json.loads(ln) for ln in Path(manifest["known_facts"]).read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Before sanitization: unique (gender, dob, pin) in voter registry -> ~100% re-identification
    raw_buckets = Counter((r["gender"], r["dob"], r["pin"]) for r in voter_rows)
    re_id_rate_before = sum(1 for r in voter_rows if raw_buckets[(r["gender"], r["dob"], r["pin"])] == 1) / len(voter_rows)

    # Read sanitized reg_english.csv
    san_csv = list(csv.DictReader(io.StringIO((out / "sanitized" / "csv" / "reg_english.csv").read_text(encoding="utf-8"))))
    assert len(san_csv) == len(voter_rows)

    # Attack A: Quasi-identifier linkage (Gender + AgeBand + Geo/PIN) against Voter Register
    attack_a_hits = 0
    eq_classes: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for row, true_v in zip(san_csv, voter_rows):
        ab = row["age"]
        g = row["gender"]
        pin_s = row["pin"]
        city_s = row["city"]
        eq_classes[(ab, g, f"{pin_s}|{city_s}")].append(row["diagnosis"])

        cands = [
            v for v in voter_rows
            if (g in ("*", "SUPPRESSED") or v["gender"] == g)
            and _age_matches(ab, int(v["age"]))
            and _geo_matches(pin_s, city_s, v["pin"], v["city"])
        ]
        # Re-identified if true person is uniquely singled out or top candidate probability >= 0.5
        if len(cands) <= 2 and any(v["pid"] == true_v["pid"] for v in cands):
            attack_a_hits += 1

    re_id_rate_after = attack_a_hits / len(voter_rows)

    # Attack B: Known Visit Date + Gender/Age (global shift recovery test)
    # Count how many patients share the same date offset (global shift fails if max bucket is small)
    inp_csv = list(csv.DictReader(io.StringIO((Path(chaos_run["input"]) / "csv" / "reg_english.csv").read_text(encoding="utf-8"))))
    offsets = []
    for ir, sr in zip(inp_csv, san_csv):
        d_in = dt.datetime.strptime(ir["admission_date"], "%d/%m/%Y").date()
        d_out = dt.datetime.strptime(sr["admission_date"], "%d/%m/%Y").date()
        offsets.append((d_out - d_in).days)
    # Ensure offsets are spread across many distinct values (no global shift!)
    assert len(set(offsets)) > 100

    attack_b_hits = 0
    for f_row in facts:
        true_v = next(v for v in voter_rows if v["pid"] == f_row["pid"])
        true_adm = dt.date.fromisoformat(f_row["visit_date"])
        # Without knowing the per-patient shift, matching exact visit date fails
        matched = [
            sr for sr in san_csv
            if dt.datetime.strptime(sr["admission_date"], "%d/%m/%Y").date() == true_adm
            and _age_matches(sr["age"], int(true_v["age"]))
            and sr["gender"] == true_v["gender"]
        ]
        if len(matched) == 1:
            attack_b_hits += 1

    # Attack C: Rare Disease + LOS + QI bucket singling-out
    singling_out_count = 0
    for f_row in facts:
        if not f_row["is_rare"]:
            continue
        true_v = next(v for v in voter_rows if v["pid"] == f_row["pid"])
        cands = []
        for sr in san_csv:
            d1 = dt.datetime.strptime(sr["admission_date"], "%d/%m/%Y").date()
            d2 = dt.datetime.strptime(sr["discharge_date"], "%d/%m/%Y").date()
            if sr["diagnosis"] == f_row["diag"] and (d2 - d1).days == f_row["los"]:
                if _age_matches(sr["age"], int(true_v["age"])) and _geo_matches(sr["pin"], sr["city"], true_v["pin"], true_v["city"]):
                    cands.append(sr)
        # Verify rare-disease patients were NOT left in a unique (1-person) QI bucket on reg_english.csv
        if len(cands) == 1:
            # Check if their QI bucket itself was broadened (city/pin generalized)
            sr0 = cands[0]
            if sr0["pin"] == true_v["pin"] or sr0["city"].lower() == true_v["city"].lower():
                singling_out_count += 1

    min_k_achieved = min(len(v) for v in eq_classes.values()) if eq_classes else 0
    min_l_achieved = min(len(set(v)) for v in eq_classes.values()) if eq_classes else 0

    print(
        f"\n[ATTACKER METRICS] re_id_rate_before={re_id_rate_before:.3f} "
        f"re_id_rate_after={re_id_rate_after:.3f} "
        f"attack_b_hits={attack_b_hits}/{len(facts)} "
        f"min_k_achieved={min_k_achieved} min_l_achieved={min_l_achieved} "
        f"singling_out_count={singling_out_count}"
    )

    assert re_id_rate_before > 0.90
    assert re_id_rate_after <= 0.15
    assert attack_b_hits <= 2
    assert singling_out_count == 0
