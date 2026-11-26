# Project RAKSHA — Team PokerFace

**Yugaantar DevForge · Track 3 ("Secret Wards" Privacy Engineering)**
Repo: `https://github.com/manishY-iitm/PokerFace` · Brief: [project_brief.html](project_brief.html) / [docs/brief.pdf](docs/brief.pdf)

An **offline, CPU-only, deterministic privacy firewall** for heterogeneous Indian healthcare records (CSV, TSV, nested JSON, FHIR R4 Bundles, and English/Hindi/Hinglish clinical notes). It removes direct and hidden-channel PII, resists voter-list re-identification attacks, links patients accurately across scripts without merging twins or spouses, and preserves clinical utility (exact lengths of stay, truthful age bands, diagnoses, medications, and vitals).

---

## 1. 4-Pass System Architecture

```text
+-----------------------------------------------------------------------------+
|  READ-ONLY INPUT (--input)                                                  |
|  CSV / TSV (UTF-8, BOM, UTF-16, cp1252) | JSON | FHIR R4 | EN/HI/Hinglish   |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|  PASS 1: Ingestion, Magic-Byte Quarantine & Schema-Agnostic Extraction      |
|  - router.py: SHA-256, magic bytes, depth guard (>100), encoding validation |
|  - schema_infer.py: fuzzy EN/HI/Hinglish headers + value-sniffing (col_x)   |
|  - csv / json / fhir / text handlers -> in-memory PatientRecord + FileEntry |
+--------------------------------------+--------------------------------------+
                                       |  (Corrupt/unsupported -> quarantine)
                                       v
+-----------------------------------------------------------------------------+
|  PASS 2: Multilingual NLP & Constrained Cross-Script Linkage                |
|  - indic_norm.py: NFKC, zero-width map, homoglyphs, schwa-aware Deva->Lat   |
|  - medical_guard.py: protects Dr., 200+ eponyms, drugs, doses, H/o, SNOMED  |
|  - blocking.py + scorer.py: 5-pass blocking + Fellegi-Sunter + Hard MNL     |
|  - cluster.py: conflict-free agglomerative clustering + ambiguity holdout   |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|  PASS 3: Secret-Keyed Crypto, Adaptive k-Anon / l-Div & Sanitization        |
|  - crypto.py: HMAC-SHA256(RAKSHA_SECRET) PSN-XXXXXX + shift in [-365, -31]  |
|  - kanon.py: bottom-up k=5, l=2 ladder (Age 5->10->20, PIN-3->PIN-2->State) |
|  - Cluster-boosted redaction + FHIR/Base64/header/key/filename scrubbing    |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|  PASS 4: Zero-Leak Self-Audit & Deterministic Export (--output)             |
|  - audit.py: scans outputs for RAKSHA_SECRET & stray direct PII             |
|  - Outputs: sanitized/ (UTF-8), records.jsonl (sorted), report.json         |
+-----------------------------------------------------------------------------+
```

---

## 2. Offline Setup & Execution

```bash
# 1. Install offline-safe dependencies (zero LLMs or pretrained models)
pip install -r requirements.txt

# 2. Run the pipeline (no internet required during execution)
RAKSHA_SECRET=your-release-secret python main.py \
  --input /path/to/input \
  --output /path/to/output \
  --config config/policy.yaml

# 3. Validate output contract & self-check (using Secret-Wards-ToolKit)
python3 ../Secret-Wards-ToolKit/validate_output.py --input /path/to/input --output /path/to/output
# -> CONTRACT OK
```

### Offline Container Run (2 CPUs, 4 GB RAM, No Network)
```bash
docker build -t raksha-pokerface .
docker run --rm --network none --cpus=2 --memory=4g \
  -e RAKSHA_SECRET=release-secret \
  -v /abs/path/to/input:/data/input:ro \
  -v /abs/path/to/output:/data/output \
  raksha-pokerface --input /data/input --output /data/output
```

---

## 3. Explicit Model & Library Declaration

Per competition ground rules, we declare **every** library, gazetteer, and algorithm used:

| Component | Package / File | Purpose |
|---|---|---|
| Fuzzy & phonetic string matching | `rapidfuzz>=3.0.0` | Cross-script name similarity & Levenshtein-1 typo checks |
| Unicode regex engine | `regex>=2023.0.0` | Devanagari/Latin contextual PII & medical-guard patterns |
| Offline phone validation | `phonenumbers>=8.13.0` | Indian mobile & STD landline validation (no network calls) |
| Policy config parsing | `PyYAML` (optional, stdlib fallback) | Loads `config/policy.yaml` thresholds |
| Core runtime & crypto | Python 3.10+ stdlib (`hmac`, `hashlib`, `csv`, `json`, `unicodedata`) | Deterministic HMAC-SHA256 pseudonyms, date shifts, I/O |
| Bundled offline gazetteers | `data/medical_eponyms.txt` (270+ eponyms), `data/indian_drugs.txt` (200+ drugs), `data/medical_terms.json`, `data/indian_geo.json`, `data/indian_pin_hierarchy.json`, `data/honorifics_and_stopwords.json` | Clinical context protection & PIN-to-State/Region generalization |
| Synthetic test suite | `pytest` (`tests/`) | Internal red-team attacker, chaos generator, medical & determinism tests |

- **NO LLMs & NO Generative Models:** Zero GPT, Claude, Gemini, Llama, Mistral, or API-hosted models.
- **NO Pretrained Transformers >50M Parameters:** Zero HuggingFace/neural transformer weights (`0` model parameters).
- **100% Offline & CPU-Only:** Zero runtime network requests; deterministic rule, gazetteer, transliteration, and Fellegi–Sunter linkage logic only.

---

## 4. Measured Benchmarks

| Benchmark | Practice Dataset (`dev/input`, 200 files / 816 records) | Unseen Hard Gauntlet (`sets/H7` Pool A / `sets/H11` Held-Out Pool B) | Stress & Red-Team Suites (`pytest` / `dataset-2-test`) |
|---|---|---|---|
| Overall Score | `CONTRACT OK` · `READY TO SUBMIT` | **`98.95 / 100`** (`H7`) · **`98.81 / 100`** (`H11`, `-0.0` penalties) | **`99.51 / 100.00`** (`dataset-2-test`, `36/36` matrix `OK`) |
| Personal Data Leaked | **`0 / 3,230` (`0.0%`)** (Latin `0.0%`, Deva `0.0%`, Hinglish `0.0%`, all 6 channels) | **`0 / 8,569`** (`H7`) · **`0 / 8,760`** (`H11`); **`0.0`** strict token leak rate | **`0 / 3,063`** (`dataset-2-test`) · **`0`** leaks (`tests/`) |
| Canary Leaks | **`0 / 4`** | **`0 / 8`** (`0` across outputs, stdout/stderr, TMP/HOME) | **`0`** (`dataset-2-test` & `tests/`) |
| Medical Values Preserved | **`4,772 / 4,855` (`98.3%`)** | **`4,899 / 5,006`** (`H7`) · **`4,950 / 5,070`** (`H11`) (`ctx-precision 1.0`) | **`2,370 / 2,370`** (`dataset-2-test`) · **`100%`** (`tests/`) |
| Exact LOS & Truthful Age Bands | **`100%` exact LOS** | `LOS 64/64` (`H7`), `76/76` (`H11`); `0 WRONG` age bands | **`100%`** exact LOS & truthful age bands |
| Linkage Precision / Recall / False Merges | **`1.00` / `0.95` / `0` false merges** (`F0.5 = 0.99`) | **`1.00` / `0.97` / `0` false merges** (`F0.5 = 0.99`, twins & father-son separated) | **`F1 = 0.9671`**, **`0`** homonym merges (`dataset-2-test`) |
| Adversarial Re-ID Risk | Rare/elderly generalized (`560` generalized records) | `A3 0.0317–0.0402`, **`A4 0.0172`**, `A5 0.0251–0.0352`, `A7 0.0` | **`0 / 67`** Re-ID (`dataset-2-test`) · `99.6% → 0.4%` (`tests/`) |
| Runtime & Peak RSS | **`3.4s`**, **`42.7 MB` peak RSS** | `27.12s` (`H7`), `42.77s` (`H11`) (3 runs) | **`2.84s` / `42.0 MB`** (`dataset-2-test`) · `14/14` pytest in **`24.79s`** |

---

## 5. Security, Determinism & Known Limitations

- **Keyed Determinism:** `make_pseudonym` and `make_date_shift` use `HMAC-SHA256` keyed by `RAKSHA_SECRET`. Identical input + secret yields byte-identical `sanitized/` and `records.jsonl`; a different secret rotates 100% of `PSN-XXXXXX` IDs and negative date shifts (`[-365, -31]`).
- **Fail-Safe & Zero-Leak Hygiene:** Input is treated read-only. Corrupt, binary, mislabeled, undecodable, or overly nested (`>100`) files are quarantined (`"output": null`) without copying raw bytes. No PII or `RAKSHA_SECRET` is ever written to stdout, stderr, temp files, or `report.json`.
- **Known Limitations:** Binary attachments outside JSON/FHIR text (PDFs, JPEGs) and HL7 XML documents are safely quarantined rather than OCR'd. Sparse standalone notes whose initials and shared family phone tie evenly across twins remain unlinked (`"patient": null`) to preserve `0%` false merges.
