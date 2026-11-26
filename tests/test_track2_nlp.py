"""Unit tests for Track 2: Indic normalization, medical guard, multilingual PII, cluster redaction, and date shifting."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_runtime_config
from src.nlp.date_shifter import compute_truthful_age_band, parse_date_safe, shift_date_str
from src.nlp.indic_norm import (
    canonical_name_key,
    devanagari_to_latin,
    map_span_to_original,
    normalize_with_map,
    parse_indian_name,
    phonetic_token,
)
from src.nlp.medical_guard import MedicalGuard
from src.nlp.pii_ner import PiiNer
from src.nlp.pii_regex import verhoeff_valid
from src.nlp.redactor import redact_text


def _setup():
    cfg = load_runtime_config()
    guard = MedicalGuard(cfg)
    ner = PiiNer(cfg, guard)
    return guard, ner


def test_indic_norm_index_map_and_phonetics():
    raw = "Mob: ७८\u200b८८०२\ufeff३९२३"
    norm, imap = normalize_with_map(raw)
    assert norm == "Mob: 7888023923"
    s = norm.index("7888023923")
    orig_s, orig_e = map_span_to_original(s, s + 10, imap, len(raw))
    assert raw[orig_s:orig_e] == "७८\u200b८८०२\ufeff३९२३"

    assert devanagari_to_latin("राजेश").lower() == "rajesh"
    assert devanagari_to_latin("शर्मा").lower() == "sharma"
    assert canonical_name_key("Md. Shahid") == canonical_name_key("मोहम्मद शाहिद")
    assert phonetic_token("Aggarwal") == phonetic_token("Agarwal")
    assert phonetic_token("Sukhwinder") == phonetic_token("Sukhvinder")

    p1 = parse_indian_name("R. K. Sharma")
    assert p1["first"] == "sharma" and p1["initials"] == ["R", "K"]
    p2 = parse_indian_name("K. Venkatraman")
    assert p2["first"] == "venkatraman" and p2["initials"] == ["K"]


def test_medical_guard_preserves_doctors_eponyms_drugs_vitals_history():
    guard, ner = _setup()
    clinical = (
        "Consultant: Dr. Ramesh Srivastava reviewed the patient in Cardiology OPD.\n"
        "H/o HTN and H/o DM2 x 8 yrs. K/C/O Parkinson's disease and Addison's disease.\n"
        "Also rule out Hodgkin lymphoma, Bell's palsy, Murphy's sign, McBurney's point.\n"
        "Vitals: BP 120/80 mmHg, vision 20/20, pain 7/10, GCS 15/15.\n"
        "Rx: Tab Dolo 650 mg 1-0-1, Glycomet 500 mg BD, Pan-D, Lantus 10 IU, Ringer's lactate, Foley catheter.\n"
        "Codes: SNOMED: 373942005, ICD-10: E27.1, LOINC: 718-7, Hb 13.5 g/dL, RBS 145 mg/dL."
    )
    out, count = redact_text(clinical, guard, ner)
    assert count == 0
    for must_keep in (
        "Dr. Ramesh Srivastava", "Cardiology", "H/o HTN", "H/o DM2",
        "Parkinson's disease", "Addison's disease", "Hodgkin lymphoma",
        "Bell's palsy", "Murphy's sign", "McBurney's point",
        "BP 120/80", "20/20", "7/10", "15/15", "Dolo 650 mg", "1-0-1",
        "Glycomet 500 mg BD", "Pan-D", "Lantus 10 IU", "Ringer's lactate",
        "Foley catheter", "373942005", "E27.1", "718-7", "13.5 g/dL", "145 mg/dL",
    ):
        assert must_keep in out, f"Lost clinical token: {must_keep}"


def test_multilingual_pii_zero_width_canaries_and_cluster_boost():
    guard, ner = _setup()
    # Verhoeff check (999999990019 is valid UIDAI test number; 999999990018 is invalid)
    assert verhoeff_valid("999999990019") is True
    assert verhoeff_valid("999999990018") is False
    # Zero-width obfuscated canary + Devanagari digits + Hinglish + Dr. Sharma survival
    note = (
        "रोगी का नाम: राजेश शर्मा, पिता: सुरेश शर्मा\n"
        "Mera naam Anita Sharma hai, S/o Mohan Lal.\n"
        "Aadhaar: 3972 0583 9628, ABHA: 91-7938-3729-8403, PAN: ABCDE1234F, UPI: rajesh@okaxis\n"
        "Mobile: +91 98765 43210, Alt: ७८८८०२३९२३, Address: H.No. 45, Sector 12, PIN 226010\n"
        "Visitor: Z\u200borawar Q\u200badri (attendant). Seen by Dr. Sharma for H/o HTN.\n"
        "Note mentions Sharma returned for follow-up."
    )
    out, count = redact_text(
        note,
        guard,
        ner,
        known_patient_tokens={"anita", "sharma"},
        patient_pseudonym="PSN-TEST01",
    )
    assert count > 0
    for forbidden in (
        "राजेश", "शर्मा", "सुरेश", "Anita", "Mohan Lal",
        "3972 0583 9628", "91-7938-3729-8403", "ABCDE1234F", "rajesh@okaxis",
        "98765 43210", "7888023923", "H.No. 45", "226010", "Zorawar", "Qadri",
    ):
        assert forbidden not in out, f"Leaked PII: {forbidden}"
    assert "Dr. Sharma" in out
    assert "H/o HTN" in out


def test_date_shifter_hindi_formats_los_and_dob():
    s = "Admission: 14 मार्च 2024, Visit: Mar 20, 2024, Follow-up: 2024-03-25"
    shifted = shift_date_str(s, -31)
    assert "12 फरवरी 2024" in shifted or "12 फ़रवरी 2024" in shifted
    assert "Feb 18, 2024" in shifted
    assert "2024-02-23" in shifted

    # Verify exact LOS preserved
    adm = parse_date_safe("१४ मार्च २०२४")
    dis = parse_date_safe("2024-03-25")
    assert adm == dt.date(2024, 3, 14)
    assert (dis - adm).days == 11

    shifted_adm = parse_date_safe(shift_date_str("14/03/2024", -90))
    shifted_dis = parse_date_safe(shift_date_str("25/03/2024", -90))
    assert (shifted_dis - shifted_adm).days == 11

    # Truthful DOB -> age band (never shifted DOB)
    assert compute_truthful_age_band(dt.date(1977, 4, 12), dt.date(2026, 1, 1)) == "45-49"
    assert compute_truthful_age_band(dt.date(1930, 8, 23), dt.date(2026, 1, 1)) == "80+"
