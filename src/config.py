"""Configuration and offline gazetteer loader."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_POLICY_PATH = ROOT_DIR / "config" / "policy.yaml"


def _parse_scalar(val: str) -> Any:
    v = val.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lstrip("-").isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def load_policy(path: Path) -> dict[str, Any]:
    """Load policy.yaml using PyYAML if available, else deterministic stdlib fallback."""
    defaults: dict[str, Any] = {
        "k": 5,
        "l": 2,
        "min_group_size": 5,
        "age_band_years": 5,
        "date_shift_min": -365,
        "date_shift_max": -31,
        "date_shift_max_days": 180,
        "privacy": {
            "k": 5,
            "l": 2,
            "age_band_years": 5,
            "age_band_ladder": [5, 10, 20],
            "pin_ladder": [3, 2, 0],
            "date_shift_min": -365,
            "date_shift_max": -31,
            "pseudonym_prefix": "PSN-",
            "pseudonym_length": 6,
        },
        "linkage": {
            "merge_threshold": 70.0,
            "review_threshold": 55.0,
            "max_block_size": 200,
            "max_age_diff_years": 3,
        },
    }
    if not path.exists():
        return defaults

    raw_text = path.read_text(encoding="utf-8")
    loaded: dict[str, Any] = {}
    try:
        import yaml  # type: ignore[import-untyped]
        parsed = yaml.safe_load(raw_text)
        if isinstance(parsed, dict):
            loaded = parsed
    except Exception:  # noqa: BLE001
        cur_section: str | None = None
        for raw_line in raw_text.splitlines():
            clean = raw_line.split("#", 1)[0].rstrip()
            if not clean.strip():
                continue
            indent = len(clean) - len(clean.lstrip(" "))
            if ":" not in clean:
                continue
            k, v = (part.strip() for part in clean.strip().split(":", 1))
            if indent == 0 and v == "":
                cur_section = k
                loaded.setdefault(cur_section, {})
            elif indent > 0 and cur_section and isinstance(loaded.get(cur_section), dict):
                loaded[cur_section][k] = _parse_scalar(v)
            else:
                cur_section = None
                loaded[k] = _parse_scalar(v)

    merged = dict(defaults)
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def load_json_data(name: str) -> dict[str, Any]:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable run configuration; secret is kept strictly in memory."""

    policy: dict[str, Any] = field(default_factory=dict)
    secret: bytes = b""
    medical_terms: dict[str, Any] = field(default_factory=dict)
    indian_geo: dict[str, Any] = field(default_factory=dict)
    honorifics: dict[str, Any] = field(default_factory=dict)

    @property
    def privacy_cfg(self) -> dict[str, Any]:
        return self.policy.get("privacy") or {}

    @property
    def age_band_years(self) -> int:
        return int(self.privacy_cfg.get("age_band_years", self.policy.get("age_band_years", 5)))

    @property
    def date_shift_min(self) -> int:
        return int(self.privacy_cfg.get("date_shift_min", self.policy.get("date_shift_min", -365)))

    @property
    def date_shift_max(self) -> int:
        return int(self.privacy_cfg.get("date_shift_max", self.policy.get("date_shift_max", -31)))

    @property
    def date_shift_max_days(self) -> int:
        return int(self.policy.get("date_shift_max_days", 180))

    @property
    def min_group_size(self) -> int:
        return int(self.privacy_cfg.get("k", self.policy.get("min_group_size", 5)))

    @property
    def l_diversity(self) -> int:
        return int(self.privacy_cfg.get("l", self.policy.get("l", 2)))


def load_runtime_config(policy_path: Path | None = None) -> RuntimeConfig:
    p_path = policy_path or Path(os.environ.get("RAKSHAK_POLICY", str(DEFAULT_POLICY_PATH)))
    secret_str = os.environ.get("RAKSHA_SECRET", "")
    return RuntimeConfig(
        policy=load_policy(p_path),
        secret=secret_str.encode("utf-8"),
        medical_terms=load_json_data("medical_terms.json"),
        indian_geo=load_json_data("indian_geo.json"),
        honorifics=load_json_data("honorifics_and_stopwords.json"),
    )
