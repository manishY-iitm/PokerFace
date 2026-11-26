"""Unit tests for Track 3: crypto determinism, linkage conflict guards, and k-anonymity."""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_runtime_config
from src.crypto import make_date_shift, make_pseudonym
from src.linkage.cluster import link_and_assign_clusters
from src.linkage.scorer import score_pair_detailed
from src.models import PatientRecord
from src.privacy.kanon import apply_kanonymity


def _rec(uid: str, **kwargs) -> PatientRecord:
    return PatientRecord(
        record_uid=uid,
        rel_input_path=uid.split("::")[0],
        rel_output_path=uid.split("::")[0],
        file_type="csv",
        loc=uid.split("::")[1],
        **kwargs,
    )


def test_crypto_determinism_and_secret_sensitivity():
    p1 = make_pseudonym("secret-A", "cluster-1")
    p2 = make_pseudonym("secret-A", "cluster-1")
    p3 = make_pseudonym("secret-B", "cluster-1")
    assert p1 == p2 and p1.startswith("PSN-")
    assert p1 != p3

    s1 = make_date_shift("secret-A", "cluster-1", min_days=-365, max_days=-31)
    s2 = make_date_shift("secret-A", "cluster-1", min_days=-365, max_days=-31)
    s3 = make_date_shift("secret-B", "cluster-1", min_days=-365, max_days=-31)
    assert s1 == s2
    assert -365 <= s1 <= -31
    assert s1 != s3


def test_twins_and_spouses_sharing_phone_and_address_never_merge():
    cfg = load_runtime_config()
    # Spouses sharing same mobile + PIN + address
    husband = _rec(
        "reg.csv::row:0",
        names_raw=["Rajesh Sharma"],
        phones=["9876543210"],
        pin_code="226010",
        gender="M",
        reported_age=42,
    )
    wife = _rec(
        "reg.csv::row:1",
        names_raw=["Sunita Sharma"],
        phones=["9876543210"],
        pin_code="226010",
        gender="F",
        reported_age=39,
    )
    # Same-sex twins sharing same DOB + phone + PIN + surname
    twin_a = _rec(
        "reg.csv::row:2",
        names_raw=["Aarav Verma"],
        phones=["9123456789"],
        dob=dt.date(2005, 6, 15),
        gender="M",
        pin_code="226010",
    )
    twin_b = _rec(
        "reg.csv::row:3",
        names_raw=["Arjun Verma"],
        phones=["9123456789"],
        dob=dt.date(2005, 6, 15),
        gender="M",
        pin_code="226010",
    )
    assert score_pair_detailed(husband, wife).conflict is True
    assert score_pair_detailed(twin_a, twin_b).conflict is True

    link_and_assign_clusters([husband, wife, twin_a, twin_b], cfg)
    assert husband.pseudonym != wife.pseudonym
    assert twin_a.pseudonym != twin_b.pseudonym


def test_cross_script_hindi_initials_and_fhir_merge():
    cfg = load_runtime_config()
    r_hi = _rec(
        "rx.txt::doc",
        names_raw=["राजेश शर्मा"],
        phones=["9811122233"],
        gender="M",
        reported_age=45,
        pin_code="560001",
    )
    r_init = _rec(
        "opd.json::/visits/0",
        names_raw=["R. K. Sharma"],
        phones=["9811122233"],
        gender="M",
        reported_age=45,
    )
    r_fhir = _rec(
        "fhir.json::doc",
        names_raw=["Rajesh Kumar Sharma"],
        dob=dt.date(1981, 3, 10),
        gender="M",
        phones=["9811122233"],
        pin_code="560001",
    )
    link_and_assign_clusters([r_hi, r_init, r_fhir], cfg)
    assert r_hi.pseudonym is not None
    assert r_hi.pseudonym == r_init.pseudonym == r_fhir.pseudonym
    assert r_hi.date_shift_days == r_init.date_shift_days == r_fhir.date_shift_days


def test_transitive_abc_conflict_blocked():
    cfg = load_runtime_config()
    # A shares phone with B (no gender on B), B shares phone with C, but A (M) and C (F) conflict!
    a = _rec("f1.csv::row:0", names_raw=["Ramesh Gupta"], phones=["9000011111"], gender="M", reported_age=50)
    b = _rec("f2.csv::row:0", names_raw=["R. Gupta"], phones=["9000011111"])
    c = _rec("f3.csv::row:0", names_raw=["Ramesh Gupta"], phones=["9000011111"], gender="F", reported_age=50)

    link_and_assign_clusters([a, b, c], cfg)
    # A and C MUST NOT end up in the same cluster!
    assert a.pseudonym != c.pseudonym


def test_kanonymity_and_truthful_age_bands():
    cfg = load_runtime_config()
    # Create 6 patients in Lucknow / UP (226010) with ages 41..44 and distinct diagnoses
    recs = []
    for idx in range(6):
        recs.append(
            _rec(
                f"reg.csv::row:{idx}",
                names_raw=[f"Patient{idx} Kumar"],
                all_mrns=[f"MRN-{idx:04d}"],
                gender="M",
                reported_age=40 + (idx % 5),
                pin_code="226010",
                district="Lucknow",
                diagnoses=["Hypertension" if idx % 2 == 0 else "Diabetes"],
            )
        )
    # Add an extreme 92-year-old outlier with a rare disease
    outlier = _rec(
        "reg.csv::row:99",
        names_raw=["Outlier Singh"],
        all_mrns=["MRN-0099"],
        gender="F",
        reported_age=92,
        pin_code="560001",
        district="Bengaluru",
        diagnoses=["Whipple procedure"],
    )
    recs.append(outlier)

    link_and_assign_clusters(recs, cfg)
    apply_kanonymity(recs, cfg)

    # Check normal 6-patient equivalence class satisfies k>=5 and truthful age band
    for idx in range(6):
        ab = recs[idx].age_band
        assert ab is not None and "-" in ab
        lo, hi = (int(x) for x in ab.split("-"))
        assert lo <= recs[idx].reported_age <= hi

    # Check outlier is generalized/suppressed and truthful ("85+" or "80+" or "*")
    assert outlier.is_rare_risk is True
    assert outlier.age_band in ("85+", "80+", "*")
    assert outlier.geo_generalized in ("Karnataka", "South India", "*")
