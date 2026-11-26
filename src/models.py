"""Shared data contracts for Project RAKSHA (Secret Wards)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class PatientRecord:
    """In-memory representation of a single patient-bearing record across all 4 passes."""

    record_uid: str
    rel_input_path: str
    rel_output_path: str
    file_type: str  # csv | tsv | json | fhir | text
    loc: str
    raw_payload: Any = None
    names_raw: list[str] = field(default_factory=list)
    guardian_names_raw: list[str] = field(default_factory=list)
    gov_ids: dict[str, str] = field(default_factory=dict)
    phones: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    gender: str | None = None
    dob: date | None = None
    reported_age: int | None = None
    reference_date: date | None = None
    pin_code: str | None = None
    city: str | None = None
    state: str | None = None
    address_raw: str | None = None
    clinical_dates: list[date] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    cluster_id: str | None = None
    pseudonym: str | None = None
    date_shift_days: int = 0
    age_band: str | None = None
    geo_generalized: str | None = None
    was_generalized: bool = False
    was_suppressed: bool = False

    # Internal structured metadata (never serialized directly to outputs)
    all_mrns: list[str] = field(default_factory=list)
    district: str | None = None
    tehsil: str | None = None
    village: str | None = None
    hospital: str | None = None
    date_hint: str = "dmy"
    has_patient_signal: bool = True
    is_rare_risk: bool = False
    out_loc: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileEntry:
    """Tracks input-to-output state for each file in the release."""

    rel_input_path: str
    rel_output_path: str | None
    input_sha256: str
    status: str  # sanitized | quarantined | suppressed
    reason: str | None = None
    file_type: str = "unknown"
    encoding: str = "utf-8"
    delimiter: str = ","
    has_bom: bool = False
    records: list[PatientRecord] = field(default_factory=list)
    raw_data: Any = None
    header: list[str] = field(default_factory=list)
    col_roles: list[str] = field(default_factory=list)
    date_hint: str = "dmy"
    hospital_hint: str | None = None
    identifiers_detected: int = 0


@dataclass
class RunStats:
    """Cross-checked run counters for report.json."""

    files_total: int = 0
    files_sanitized: int = 0
    files_quarantined: int = 0
    records_processed: int = 0
    identifiers_detected: int = 0
    records_linked: int = 0
    records_generalized: int = 0
    records_suppressed: int = 0
    records_quarantined: int = 0
    entity_counts: dict[str, int] = field(default_factory=dict)
