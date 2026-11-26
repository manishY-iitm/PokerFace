"""Synthetic Chaos & Hidden-Dataset-3 Simulator (~500 patients, ~2,000 records, 100% synthetic)."""
from __future__ import annotations

import base64
import csv
import datetime as dt
import json
import random
import uuid
from pathlib import Path

_FIRST_M = [
    ("Rajesh", "राजेश"), ("Amit", "अमित"), ("Vikram", "विक्रम"), ("Suresh", "सुरेश"),
    ("Anil", "अनिल"), ("Deepak", "दीपक"), ("Manoj", "मनोज"), ("Rohan", "रोहन"),
    ("Karan", "करण"), ("Nitin", "नितिन"), ("Sanjay", "संजय"), ("Arjun", "अर्जुन"),
    ("Aarav", "आरव"), ("Prakash", "प्रकाश"), ("Vivek", "विवेक"), ("Harish", "हरीश"),
]
_FIRST_F = [
    ("Priya", "प्रिया"), ("Sunita", "सुनीता"), ("Anita", "अनीता"), ("Kavita", "कविता"),
    ("Meera", "मीरा"), ("Pooja", "पूजा"), ("Neha", "नेहा"), ("Rekha", "रेखा"),
    ("Geeta", "गीता"), ("Asha", "आशा"), ("Shanti", "शांति"), ("Suman", "सुमन"),
    ("Divya", "दिव्या"), ("Kiran", "किरण"), ("Radha", "राधा"), ("Seema", "सीमा"),
]
_LAST = [
    ("Sharma", "शर्मा"), ("Verma", "वर्मा"), ("Gupta", "गुप्ता"), ("Patel", "पटेल"),
    ("Singh", "सिंह"), ("Yadav", "यादव"), ("Mishra", "मिश्रा"), ("Tiwari", "तिवारी"),
    ("Pandey", "पांडे"), ("Chauhan", "चौहान"), ("Agarwal", "अग्रवाल"), ("Joshi", "जोशी"),
]
_CITIES = [
    ("Lucknow", "Uttar Pradesh", "226010"),
    ("Kanpur", "Uttar Pradesh", "208001"),
    ("Varanasi", "Uttar Pradesh", "221001"),
    ("Bengaluru", "Karnataka", "560001"),
    ("Pune", "Maharashtra", "411001"),
    ("Jaipur", "Rajasthan", "302001"),
    ("Patna", "Bihar", "800001"),
    ("Bhopal", "Madhya Pradesh", "462001"),
]
_COMMON_DIAGS = [
    ("I10", "Essential hypertension"),
    ("E11.9", "Type 2 diabetes mellitus"),
    ("J06.9", "Acute upper respiratory infection"),
    ("A09", "Infectious gastroenteritis"),
    ("K29.7", "Gastritis, unspecified"),
]
_RARE_DIAGS = [
    ("A82.9", "Rabies, unspecified"),
    ("B55.0", "Visceral leishmaniasis"),
    ("E27.1", "Addison's disease"),
]
_HINDI_MONTHS = [
    "", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]
_DEVA_TRANS = str.maketrans("0123456789", "०१२३४५६७८९")


def _try_faker():
    try:
        from faker import Faker
        return Faker("en_IN"), Faker("hi_IN")
    except Exception:  # noqa: BLE001
        return None, None


def generate_chaos_dataset(base_dir: Path, num_patients: int = 500, seed: int = 42) -> tuple[Path, Path]:
    """Create input/ dataset + external truth_manifest.json, voter_registry.csv, known_facts.jsonl."""
    rng = random.Random(seed)
    fake_en, fake_hi = _try_faker()
    if fake_en:
        fake_en.seed_instance(seed)
    if fake_hi:
        fake_hi.seed_instance(seed)

    inp = base_dir / "input"
    for sub in ("csv", "tsv", "json", "fhir", "notes", "misc"):
        (inp / sub).mkdir(parents=True, exist_ok=True)

    patients = []
    used_phones: set[str] = set()
    used_aadhaar: set[str] = set()

    def _gen_phone() -> str:
        while True:
            p = f"{rng.choice('6789')}{rng.randint(100000000, 999999999)}"
            if p not in used_phones:
                used_phones.add(p)
                return p

    def _gen_aadhaar() -> str:
        while True:
            a = f"{rng.randint(2, 9)}{rng.randint(10000000000, 99999999999)}"
            if a not in used_aadhaar:
                used_aadhaar.add(a)
                return a

    for i in range(num_patients):
        gender = "M" if i % 2 == 0 else "F"
        fn_en, fn_hi = rng.choice(_FIRST_M if gender == "M" else _FIRST_F)
        ln_en, ln_hi = rng.choice(_LAST)
        mid_en = rng.choice(["Kumar", "Prasad", "Devi", "Nath"])
        city, state, pin_base = _CITIES[i % len(_CITIES)]
        pin = f"{pin_base[:4]}{rng.randint(10, 99)}"
        dob = dt.date(rng.randint(1945, 2005), rng.randint(1, 12), rng.randint(1, 28))
        # Make 2 rare-disease outlier patients
        is_rare = i in (10, 20)
        if is_rare:
            dob = dt.date(1930 + (i // 10), 4, 15)
        diag_code, diag_name = rng.choice(_RARE_DIAGS if is_rare else _COMMON_DIAGS)
        adm = dt.date(2025, rng.randint(1, 10), rng.randint(1, 20))
        los = rng.randint(2, 12)
        dis = adm + dt.timedelta(days=los)

        patients.append({
            "pid": f"PT-{i:04d}",
            "first": fn_en, "middle": mid_en, "last": ln_en,
            "full_en": f"{fn_en} {mid_en} {ln_en}",
            "full_hi": f"{fn_hi} {ln_hi}",
            "initials_en": f"{fn_en[0]}. {mid_en[0]}. {ln_en}",
            "gender": gender,
            "dob": dob,
            "age_2025": 2025 - dob.year - ((11, 1) < (dob.month, dob.day)),
            "phone": _gen_phone(),
            "aadhaar": _gen_aadhaar(),
            "pan": f"{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{rng.randint(1000,9999)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
            "mrn": f"RK{2025}-{i+10000:05d}",
            "city": city, "state": state, "pin": pin,
            "diag_code": diag_code, "diag_name": diag_name,
            "is_rare": is_rare,
            "adm": adm, "dis": dis, "los": los,
            "trap_group": None,
        })

    # Inject explicit Hard Linkage Traps on indices 0..7
    # 0: Cross-script hero patient (Rajesh Kumar Sharma / R. K. Sharma / राजेश शर्मा)
    patients[0].update({
        "first": "Rajesh", "middle": "Kumar", "last": "Sharma",
        "full_en": "Rajesh Kumar Sharma", "full_hi": "राजेश शर्मा",
        "initials_en": "R. K. Sharma", "gender": "M",
        "phone": "9876543210", "dob": dt.date(1980, 3, 14),
        "age_2025": 45,
    })
    # 1 & 2: Husband + Wife + Child sharing EXACT same phone & PIN/city!
    shared_fam_phone = "9811223344"
    patients[1].update({
        "first": "Suresh", "last": "Verma", "full_en": "Suresh Kumar Verma", "full_hi": "सुरेश वर्मा",
        "initials_en": "S. K. Verma", "gender": "M", "phone": shared_fam_phone,
        "pin": "226010", "city": "Lucknow", "trap_group": "family_spouse_1",
    })
    patients[2].update({
        "first": "Sunita", "last": "Verma", "full_en": "Sunita Devi Verma", "full_hi": "सुनीता वर्मा",
        "initials_en": "S. D. Verma", "gender": "F", "phone": shared_fam_phone,
        "pin": "226010", "city": "Lucknow", "trap_group": "family_spouse_2",
    })
    # 3 & 4: Identical twins sharing DOB, surname, phone, address (different first name / gender)
    shared_twin_phone = "9799887766"
    twin_dob = dt.date(2004, 7, 19)
    patients[3].update({
        "first": "Aarav", "last": "Gupta", "full_en": "Aarav Prasad Gupta", "full_hi": "आरव गुप्ता",
        "initials_en": "A. P. Gupta", "gender": "M", "dob": twin_dob, "age_2025": 21,
        "phone": shared_twin_phone, "pin": "560001", "city": "Bengaluru", "trap_group": "twin_a",
    })
    patients[4].update({
        "first": "Arjun", "last": "Gupta", "full_en": "Arjun Prasad Gupta", "full_hi": "अर्जुन गुप्ता",
        "initials_en": "A. P. Gupta", "gender": "M", "dob": twin_dob, "age_2025": 21,
        "phone": shared_twin_phone, "pin": "560001", "city": "Bengaluru", "trap_group": "twin_b",
    })
    # 5 & 6: Common Indian name homonyms in different cities & Aadhaars
    patients[5].update({
        "first": "Amit", "last": "Kumar", "full_en": "Amit Kumar", "full_hi": "अमित कुमार",
        "initials_en": "A. Kumar", "gender": "M", "city": "Lucknow", "pin": "226010", "trap_group": "homonym_a",
    })
    patients[6].update({
        "first": "Amit", "last": "Kumar", "full_en": "Amit Kumar", "full_hi": "अमित कुमार",
        "initials_en": "A. Kumar", "gender": "M", "city": "Pune", "pin": "411001", "trap_group": "homonym_b",
    })

    truth_units = []
    canaries = []

    # 1. reg_english.csv (500 rows)
    with (inp / "csv" / "reg_english.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "uhid", "patient_name", "gender", "dob", "age", "phone", "aadhaar",
            "pan", "city", "state", "pin", "admission_date", "discharge_date", "diagnosis", "remarks",
        ])
        for idx, pt in enumerate(patients):
            rem = "H/o HTN; Dr. A. K. Sharma advised Tab Dolo 650 mg 1-0-1"
            if idx == 0:
                canary_tok = "CNRY-AADHAAR-9981"
                canaries.append(canary_tok)
                rem = f"H/o HTN; ref {canary_tok}; Dr. A. K. Sharma"
            w.writerow([
                pt["mrn"], pt["full_en"], pt["gender"], pt["dob"].strftime("%d/%m/%Y"),
                pt["age_2025"], pt["phone"], pt["aadhaar"], pt["pan"],
                pt["city"], pt["state"], pt["pin"],
                pt["adm"].strftime("%d/%m/%Y"), pt["dis"].strftime("%d/%m/%Y"),
                pt["diag_name"], rem,
            ])
            truth_units.append({
                "file": "csv/reg_english.csv", "loc": f"row:{idx}",
                "pid": pt["pid"], "age": pt["age_2025"], "los": pt["los"],
                "adm": pt["adm"].isoformat(), "dis": pt["dis"].isoformat(),
                "script": "en",
            })

    # 2. reg_unseen.tsv (Unseen-Schema Chaos for all 500 patients)
    with (inp / "tsv" / "reg_unseen.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uhid_ref", "rogi_ka_naam", "ling", "janam_tithi", "sampark_sankhya", "col_x", "meta_blob"])
        for idx, pt in enumerate(patients):
            c_name = ""
            meta_blob = f"BP 120/80 SpO2 98% GCS 15/15"
            if idx == 1:
                c_name = "Z\u200borawar Q\u200badri"
                canaries.append("Zorawar Qadri")
                meta_blob = f"verified by {c_name}; BP 120/80"
            w.writerow([
                pt["mrn"], pt["full_en"], pt["gender"], pt["dob"].isoformat(),
                pt["phone"], pt["pan"], meta_blob,
            ])
            truth_units.append({
                "file": "tsv/reg_unseen.tsv", "loc": f"row:{idx}",
                "pid": pt["pid"], "age": pt["age_2025"], "los": None,
                "script": "unseen",
            })

    # 3. opd_visits.json (500 records using initials + unseen keys)
    opd_visits = []
    for idx, pt in enumerate(patients):
        opd_visits.append({
            "visit_id": f"OPD-{idx:05d}",
            "rogi_ka_naam": pt["initials_en"],
            "sampark_sankhya": pt["phone"],
            "gender": pt["gender"],
            "age": pt["age_2025"],
            "visit_date": pt["adm"].isoformat(),
            "doctor": "Dr. A. K. Sharma",
            "diagnosis": pt["diag_name"],
            "notes": "Tab Glycomet 500mg BD, Pan-D, Augmentin. H/o HTN. Parkinson's disease ruled out.",
        })
        truth_units.append({
            "file": "json/opd_visits.json", "loc": f"/visits/{idx}",
            "pid": pt["pid"], "age": pt["age_2025"], "los": None,
            "script": "initials",
        })
    (inp / "json" / "opd_visits.json").write_text(
        json.dumps({"_meta": {"exported_by": "clerk_1", "source_path": "/srv/exports/Amit_Sharma_001/"}, "visits": opd_visits}, indent=2),
        encoding="utf-8",
    )

    # 4. FHIR R4 Bundles (100 bundles covering patients 0..99, with ALL hidden FHIR traps!)
    for idx in range(100):
        pt = patients[idx]
        pat_uuid = str(uuid.UUID(int=idx + 1000))
        enc_uuid = str(uuid.UUID(int=idx + 2000))
        doc_uuid = str(uuid.UUID(int=idx + 3000))
        prac_uuid = str(uuid.UUID(int=idx + 4000))
        canary_phone_fhir = f"990011{idx:04d}"
        if idx < 3:
            canaries.append(f"CanaryFHIR{idx} Khan")
        b64_note = base64.b64encode(
            f"Referral for {pt['full_en']} ({pt['mrn']}), phone {pt['phone']}. Ringer's lactate, Foley catheter, Widal test negative.".encode("utf-8")
        ).decode("ascii")
        bundle = {
            "resourceType": "Bundle",
            "id": f"bundle-{idx:04d}",
            "type": "document",
            "timestamp": f"{pt['dis'].isoformat()}T10:00:00+05:30",
            "entry": [
                {
                    "fullUrl": f"urn:uuid:{pat_uuid}",
                    "resource": {
                        "resourceType": "Patient",
                        "id": pat_uuid,
                        "text": {
                            "status": "generated",
                            "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">{pt['full_en']} {pt['phone']}</div>",
                        },
                        "identifier": [{"system": "https://hospital.example/mrn", "value": pt["mrn"]}],
                        "name": [{"text": pt["full_en"], "given": [pt["first"]], "family": pt["last"]}],
                        "telecom": [{"system": "phone", "value": pt["phone"]}],
                        "gender": "male" if pt["gender"] == "M" else "female",
                        "birthDate": pt["dob"].isoformat(),
                        "address": [{"line": ["H.No 12"], "city": pt["city"], "state": pt["state"], "postalCode": pt["pin"]}],
                        "contact": [{"name": {"text": f"CanaryFHIR{idx} Khan" if idx < 3 else "Relative Person"}, "telecom": [{"system": "phone", "value": canary_phone_fhir}]}],
                        "photo": [{"contentType": "image/jpeg", "title": pt["full_en"]}],
                        "extension": [{"url": "https://example.org/ext", "valueString": f"Note for {pt['full_en']}"}],
                    },
                },
                {
                    "fullUrl": f"urn:uuid:{prac_uuid}",
                    "resource": {
                        "resourceType": "Practitioner",
                        "id": prac_uuid,
                        "name": [{"text": "Dr. A. K. Sharma"}],
                    },
                },
                {
                    "fullUrl": f"urn:uuid:{enc_uuid}",
                    "resource": {
                        "resourceType": "Encounter",
                        "id": enc_uuid,
                        "status": "finished",
                        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP"},
                        "subject": {"reference": f"urn:uuid:{pat_uuid}", "display": pt["full_en"]},
                        "period": {"start": f"{pt['adm'].isoformat()}T09:00:00+05:30", "end": f"{pt['dis'].isoformat()}T14:00:00+05:30"},
                        "contained": [
                            {"resourceType": "Patient", "id": "cont-1", "name": [{"text": pt["full_en"]}]}
                        ],
                    },
                },
                {
                    "fullUrl": f"urn:uuid:{doc_uuid}",
                    "resource": {
                        "resourceType": "DocumentReference",
                        "id": doc_uuid,
                        "status": "current",
                        "subject": {"reference": f"urn:uuid:{pat_uuid}", "display": pt["full_en"]},
                        "content": [{"attachment": {"contentType": "text/plain", "data": b64_note}}],
                    },
                },
            ],
        }
        rel = f"fhir/bundle_{idx:04d}.json"
        (inp / rel).write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        truth_units.append({
            "file": rel, "loc": "doc", "pid": pt["pid"], "age": pt["age_2025"],
            "los": pt["los"], "adm": pt["adm"].isoformat(), "dis": pt["dis"].isoformat(),
            "script": "fhir",
        })

    # 5. Hindi / Hinglish clinical .txt notes (400 notes covering patients 0..399, + 20 anonymous notes)
    for idx in range(400):
        pt = patients[idx]
        adm_hi = f"{str(pt['adm'].day).translate(_DEVA_TRANS)} {_HINDI_MONTHS[pt['adm'].month]} {str(pt['adm'].year).translate(_DEVA_TRANS)}"
        dis_hi = f"{str(pt['dis'].day).translate(_DEVA_TRANS)} {_HINDI_MONTHS[pt['dis'].month]} {str(pt['dis'].year).translate(_DEVA_TRANS)}"
        ph_hi = pt["phone"].translate(_DEVA_TRANS)
        note_txt = (
            f"रोगी का नाम: {pt['full_hi']}\n"
            f"आयु: {pt['age_2025']} वर्ष / {pt['gender']}\n"
            f"मोबाइल: {ph_hi}\n"
            f"भर्ती की तारीख: {adm_hi}\n"
            f"छुट्टी की तारीख: {dis_hi}\n"
            f"Consultant: Dr. A. K. Sharma\n"
            f"H/o HTN. Diagnosed with {pt['diag_name']} ({pt['diag_code']}). SNOMED: 373942005.\n"
            f"Treatment: Tab Dolo 650 mg 1-0-1, Glycomet 500mg BD, Pan-D, Augmentin, Ringer's lactate, Foley catheter, Widal test, Crohn's check.\n"
            f"Vitals: BP 120/80, SpO2 98%, GCS 15/15.\n"
        )
        rel = f"notes/note_{idx:04d}.txt"
        (inp / rel).write_text(note_txt, encoding="utf-8")
        truth_units.append({
            "file": rel, "loc": "doc", "pid": pt["pid"], "age": pt["age_2025"],
            "los": pt["los"], "adm": pt["adm"].isoformat(), "dis": pt["dis"].isoformat(),
            "script": "hi",
        })

    # Anonymous clinical snippets (expected patient = None)
    for a_idx in range(10):
        rel = f"notes/anon_{a_idx:02d}.txt"
        (inp / rel).write_text(
            "PROTOCOL NOTE\nH/o HTN. Monitor BP 120/80, SpO2 98%, GCS 15/15.\nRx: Dolo 650 mg 1-0-1.\n",
            encoding="utf-8",
        )
        truth_units.append({"file": rel, "loc": "doc", "pid": None, "age": None, "los": None, "script": "anon"})

    # 6. Quarantine Traps (5 broken/unsupported files)
    quarantine_files = [
        "misc/corrupt.json",
        "misc/raw_blob.bin",
        "misc/fake.pdf",
        "misc/bad_utf8.txt",
        "misc/empty.csv",
    ]
    (inp / "misc" / "corrupt.json").write_text('{"patient": "Leak Name", "broken": [1, 2', encoding="utf-8")
    (inp / "misc" / "raw_blob.bin").write_bytes(b"\x89PNG\r\n\x1a\nLeakSecretName9876543210")
    (inp / "misc" / "fake.pdf").write_bytes(b"%PDF-1.4\n(Patient: SecretPdfName Mob 9876501234)\n%%EOF")
    (inp / "misc" / "bad_utf8.txt").write_bytes(b"Patient LeakUtf8 \xff\xfe\xc3( Mob 9988776655\n")
    (inp / "misc" / "empty.csv").write_bytes(b"")
    canaries.extend(["SecretPdfName", "LeakUtf8"])

    # External Voter Registry (with extra decoys) & Known Facts (OUTSIDE input/)
    voter_path = base_dir / "voter_registry.csv"
    with voter_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "name", "gender", "dob", "age", "pin", "city"])
        for pt in patients:
            w.writerow([pt["pid"], pt["full_en"], pt["gender"], pt["dob"].isoformat(), pt["age_2025"], pt["pin"], pt["city"]])

    facts_path = base_dir / "known_facts.jsonl"
    with facts_path.open("w", encoding="utf-8") as f:
        for pt in patients[:150]:
            f.write(json.dumps({
                "pid": pt["pid"],
                "visit_date": pt["adm"].isoformat(),
                "los": pt["los"],
                "diag": pt["diag_name"],
                "is_rare": pt["is_rare"],
            }) + "\n")

    manifest = {
        "patients": [
            {
                "pid": pt["pid"],
                "full_en": pt["full_en"],
                "full_hi": pt["full_hi"],
                "phone": pt["phone"],
                "aadhaar": pt["aadhaar"],
                "pan": pt["pan"],
                "mrn": pt["mrn"],
                "age": pt["age_2025"],
                "gender": pt["gender"],
                "pin": pt["pin"],
                "city": pt["city"],
                "diag_name": pt["diag_name"],
                "los": pt["los"],
                "is_rare": pt["is_rare"],
                "trap_group": pt["trap_group"],
            }
            for pt in patients
        ],
        "units": truth_units,
        "canaries": canaries,
        "quarantine_files": quarantine_files,
        "voter_registry": str(voter_path),
        "known_facts": str(facts_path),
    }
    manifest_path = base_dir / "truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return inp, manifest_path
