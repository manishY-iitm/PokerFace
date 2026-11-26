"""Keyed HMAC-SHA256 pseudonyms, deterministic negative date shifts, and collision checks."""
from __future__ import annotations

import hashlib
import hmac
import uuid

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 (no I, L, O, U)
_UUID_NS = uuid.UUID("7f3d5c1a-9b42-4e8a-a1d0-8c3b2e9f0123")


def _to_secret_bytes(secret: str | bytes) -> bytes:
    return secret if isinstance(secret, bytes) else secret.encode("utf-8")


def make_pseudonym(
    secret: str | bytes,
    canonical_cluster_key: str,
    *,
    used_pseudonyms: dict[str, str] | None = None,
    prefix: str = "PSN-",
    length: int = 6,
) -> str:
    """Return deterministic non-reversible PSN-XXXXXX keyed by RAKSHA_SECRET, collision-checked."""
    sec_b = _to_secret_bytes(secret)
    attempt = 0
    while True:
        payload = f"psn|{canonical_cluster_key}" if attempt == 0 else f"psn|{attempt}|{canonical_cluster_key}"
        digest = hmac.new(sec_b, payload.encode("utf-8"), hashlib.sha256).digest()
        num = int.from_bytes(digest[:8], "big")
        code = "".join(_ALPHABET[(num >> (5 * i)) & 31] for i in range(length))
        psn = f"{prefix}{code}"
        if used_pseudonyms is None:
            return psn
        existing = used_pseudonyms.get(psn)
        if existing is None or existing == canonical_cluster_key:
            used_pseudonyms[psn] = canonical_cluster_key
            return psn
        attempt += 1


def make_date_shift(
    secret: str | bytes,
    canonical_cluster_key: str,
    min_days: int = -365,
    max_days: int = -31,
) -> int:
    """Return a deterministic non-zero integer day offset in [min_days, max_days]."""
    sec_b = _to_secret_bytes(secret)
    if min_days > max_days:
        min_days, max_days = max_days, min_days
    digest = hmac.new(sec_b, f"shift|{canonical_cluster_key}".encode("utf-8"), hashlib.sha256).digest()
    span = max_days - min_days + 1
    offset = min_days + (int.from_bytes(digest[:4], "big") % span)
    if offset == 0:
        offset = min_days if min_days != 0 else -1
    return offset


# Backwards-compatible aliases used by existing modules
def pseudonym(cluster_key: str, secret: bytes, prefix: str = "PSN-", length: int = 6) -> str:
    return make_pseudonym(secret, cluster_key, prefix=prefix, length=length)


def date_shift_days(
    shift_key: str,
    secret: bytes,
    max_days: int = 180,
    *,
    min_shift: int | None = None,
    max_shift: int | None = None,
) -> int:
    lo = min_shift if min_shift is not None else -abs(max_days)
    hi = max_shift if max_shift is not None else -31
    if lo > hi:
        lo, hi = -365, -31
    return make_date_shift(secret, shift_key, min_days=lo, max_days=hi)


def safe_file_token(rel_path: str, secret: str | bytes) -> str:
    """Deterministic short token for renaming PII-bearing filenames."""
    sec_b = _to_secret_bytes(secret)
    digest = hmac.new(sec_b, b"file|" + rel_path.encode("utf-8"), hashlib.sha256).digest()
    num = int.from_bytes(digest[:5], "big")
    return "".join(_ALPHABET[(num >> (5 * i)) & 31] for i in range(8)).lower()


def safe_fhir_uuid(key: str, secret: str | bytes) -> str:
    """Deterministic UUIDv5 derived from keyed HMAC."""
    sec_b = _to_secret_bytes(secret)
    digest = hmac.new(sec_b, b"fhir|" + key.encode("utf-8"), hashlib.sha256).hexdigest()
    return str(uuid.uuid5(_UUID_NS, digest))


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
