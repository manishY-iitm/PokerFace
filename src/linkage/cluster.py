"""Constrained agglomerative clustering without transitive over-merging."""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from src.config import RuntimeConfig
from src.crypto import make_date_shift, make_pseudonym
from src.linkage.blocking import generate_blocks
from src.linkage.scorer import DEFAULT_LINK_THRESHOLD, score_pair_detailed
from src.models import PatientRecord
from src.nlp.indic_norm import normalize_text, strong_name_tokens


def _is_weak_identity(rec: PatientRecord) -> bool:
    """True for sparse notes with only a single given-name token and zero corroborating IDs/demographics."""
    if (
        rec.all_mrns
        or rec.gov_ids
        or rec.phones
        or rec.emails
        or rec.dob is not None
        or rec.reported_age is not None
        or rec.gender
        or rec.pin_code
        or rec.village
        or rec.district
    ):
        return False
    return all(len(strong_name_tokens(nm)) < 2 for nm in rec.names_raw)


def link_and_assign_clusters(records: list[PatientRecord], cfg: RuntimeConfig) -> None:
    """Cluster records safely without transitive false merges and assign PSN + negative date shift."""
    n = len(records)
    if n == 0:
        return

    link_cfg = cfg.policy.get("linkage") or {}
    link_threshold = float(link_cfg.get("link_threshold", link_cfg.get("merge_threshold", DEFAULT_LINK_THRESHOLD)))
    ambiguity_margin = float(link_cfg.get("ambiguity_margin", 1.5))
    max_block_size = int(link_cfg.get("max_block_size", 200))
    weights = link_cfg.get("weights") or {}

    pairs = generate_blocks(records, max_block_size=max_block_size)

    # Precompute non-conflicting edge scores
    edge_scores: dict[tuple[int, int], float] = {}
    edges: list[tuple[float, int, int]] = []
    for i, j in pairs:
        res = score_pair_detailed(records[i], records[j], weights=weights)
        if not res.conflict and res.score > 0:
            edge_scores[(i, j)] = res.score
            if res.score >= link_threshold:
                edges.append((res.score, i, j))

    # Deterministic order: highest score first, then lexicographical record_uid tiebreak
    edges.sort(key=lambda e: (-e[0], records[e[1]].record_uid, records[e[2]].record_uid))

    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Ambiguity protection: detect weak singletons (no strong gov_id/mrn/phone) that match
    # two incompatible high-confidence targets with score gap < ambiguity_margin
    weak_ambiguous: set[int] = set()
    adj_above: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for sc, i, j in edges:
        adj_above[i].append((sc, j))
        adj_above[j].append((sc, i))
    for idx, rec in enumerate(records):
        if rec.all_mrns or any(k in rec.gov_ids for k in ("aadhaar", "abha", "pan", "passport", "voter")):
            continue
        cands = sorted(adj_above.get(idx, []), key=lambda x: -x[0])
        if len(cands) >= 2:
            top_sc, top_nb = cands[0]
            for alt_sc, alt_nb in cands[1:]:
                if top_sc - alt_sc < ambiguity_margin:
                    res_alt = score_pair_detailed(records[top_nb], records[alt_nb], weights=weights)
                    if res_alt.conflict:
                        weak_ambiguous.add(idx)
                        break
    weak_ambig_uids = {records[i].record_uid for i in weak_ambiguous}

    for _score, i, j in edges:
        if (
            i in weak_ambiguous
            or j in weak_ambiguous
            or _is_weak_identity(records[i])
            or _is_weak_identity(records[j])
        ):
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if len(members[ri]) < len(members[rj]):
            ri, rj = rj, ri

        # 1. Zero Hard MNL conflicts across ANY pair in C1 x C2
        conflict = False
        comparable_scores: list[float] = []
        max_cross = 0.0
        for mi in members[ri]:
            for mj in members[rj]:
                pair_key = (mi, mj) if mi < mj else (mj, mi)
                if pair_key in edge_scores:
                    sc = edge_scores[pair_key]
                    comparable_scores.append(sc)
                    if sc > max_cross:
                        max_cross = sc
                else:
                    res = score_pair_detailed(records[mi], records[mj], weights=weights)
                    if res.conflict:
                        conflict = True
                        break
                    if res.score > 0:
                        comparable_scores.append(res.score)
                        if res.score > max_cross:
                            max_cross = res.score
            if conflict:
                break
        if conflict:
            continue

        # 2. Cross-cluster compatibility check
        avg_comp = sum(comparable_scores) / len(comparable_scores) if comparable_scores else 0.0
        if max_cross < link_threshold or avg_comp < max(2.0, link_threshold * 0.5):
            continue

        parent[rj] = ri
        members[ri].extend(members[rj])
        del members[rj]

    # Group clusters deterministically
    clusters: dict[int, list[PatientRecord]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.has_patient_signal:
            clusters[find(idx)].append(rec)

    used_pseudonyms: dict[str, str] = {}
    psn_prefix = str((cfg.privacy_cfg or {}).get("pseudonym_prefix", "PSN-"))
    psn_len = int((cfg.privacy_cfg or {}).get("pseudonym_length", 6))

    for _root, grp in sorted(clusters.items(), key=lambda kv: min(r.record_uid for r in kv[1])):
        grp.sort(key=lambda r: r.record_uid)

        # Anonymous, weak-singleton, or ambiguous records get pseudonym = None and per-record date shift
        has_id = any(r.names_raw or r.all_mrns or r.phones or r.gov_ids for r in grp)
        if has_id and len(grp) == 1 and _is_weak_identity(grp[0]):
            has_id = False
        is_ambig = len(grp) == 1 and grp[0].record_uid in weak_ambig_uids if len(grp) == 1 else False
        if not has_id or is_ambig:
            for r in grp:
                r.cluster_id = None
                r.pseudonym = None
                r.date_shift_days = make_date_shift(
                    cfg.secret, r.record_uid, cfg.date_shift_min, cfg.date_shift_max
                )
            continue

        # Canonical cluster key: sorted record_uids (stable & deterministic under same input + secret)
        ckey = "|".join(r.record_uid for r in grp)
        psn = make_pseudonym(
            cfg.secret,
            ckey,
            used_pseudonyms=used_pseudonyms,
            prefix=psn_prefix,
            length=psn_len,
        )
        shift = make_date_shift(cfg.secret, ckey, cfg.date_shift_min, cfg.date_shift_max)

        # Avoid accidental shift landing onto the patient's own DOB
        dobs = {r.dob for r in grp if r.dob}
        cl_dates = {d for r in grp for d in r.clinical_dates}
        if dobs and cl_dates:
            for _ in range(30):
                if any((d + dt.timedelta(days=shift)) in dobs for d in cl_dates):
                    shift = shift + 1 if shift < cfg.date_shift_max else cfg.date_shift_min
                else:
                    break

        # Build per-cluster known_patient_tokens for Track 2 Cluster-Boosted Redaction
        cluster_tokens: set[str] = set()
        for r in grp:
            for nm in r.names_raw:
                for tok in normalize_text(nm).split():
                    if len(tok) >= 3:
                        cluster_tokens.add(tok)
            for gn in r.guardian_names_raw:
                for tok in normalize_text(gn).split():
                    if len(tok) >= 3:
                        cluster_tokens.add(tok)
            for mrn in r.all_mrns:
                if mrn:
                    cluster_tokens.add(mrn.strip())
            for ph in r.phones:
                if len(ph) >= 8:
                    cluster_tokens.add(ph[-10:])
            for gv in r.gov_ids.values():
                if gv:
                    cluster_tokens.add(gv)
            if r.village:
                for tok in normalize_text(r.village).split():
                    if len(tok) >= 3:
                        cluster_tokens.add(tok)

        for r in grp:
            r.cluster_id = ckey
            r.pseudonym = psn
            r.date_shift_days = shift
            r.extra["cluster_tokens"] = cluster_tokens

    # Ensure unlinked/non-patient records still get a deterministic per-record shift
    for rec in records:
        if rec.date_shift_days == 0:
            rec.date_shift_days = make_date_shift(
                cfg.secret, rec.record_uid, cfg.date_shift_min, cfg.date_shift_max
            )
