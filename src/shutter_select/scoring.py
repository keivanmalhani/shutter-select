"""Run-relative scoring, class weights, hard fails, and the decide step.

Every feature is percentile-normalized within the run and within class
(speech vs b-roll), the shutter-cull composite pattern: no absolute
thresholds on scene-dependent features. Composites therefore rank segments
against this shoot, not against an absolute notion of good.

Hard fails are the exception, absolute by design: clipped audio on a
speech take and very quiet speech are wrong regardless of what the rest of
the shoot looks like. Each hard fail carries a human-readable reason.

Decide posture, same as shutter-cull: mark the clear picks and the clear
rejects, leave the middle unmarked for human judgment.

Small-group rule (not in the spec, logged in _hq/mistakes.md 2026-08-05):
percentile thresholds are meaningless for tiny groups. A run with a single
clean interview take must not bury it below the 60th-percentile bar the
way shutter-cull's singleton bug buried no-burst shoots, so when a class
has 3 or fewer segments every non-hard-fail segment is a Select and only
hard fails Reject.
"""

from __future__ import annotations

import numpy as np

SELECT_PERCENTILE = 0.60
REJECT_PERCENTILE = 0.10
SMALL_GROUP_MAX = 3
MIN_SPEECH_RMS_DB = -35.0
SHARPNESS_FAIL_PERCENTILE = 0.02
SHARPNESS_FAIL_MIN_GROUP = 50
BROLL_DURATION_SWEET_LO = 4.0
BROLL_DURATION_SWEET_HI = 20.0

SPEECH_WEIGHTS = {
    "audio_quality": 0.40,
    "speech_density": 0.20,
    "sharpness": 0.20,
    "exposure": 0.10,
    "faces": 0.10,
}
BROLL_WEIGHTS = {
    "sharpness": 0.35,
    "motion": 0.30,
    "exposure": 0.20,
    "duration_fit": 0.15,
}
INTERVIEW_SPEECH_WEIGHTS = {
    "audio_quality": 0.50,
    "speech_density": 0.20,
    "sharpness": 0.15,
    "exposure": 0.05,
    "faces": 0.10,
}

PROFILES = {
    "auto": {"speech": SPEECH_WEIGHTS, "broll": BROLL_WEIGHTS},
    "event": {"speech": SPEECH_WEIGHTS, "broll": BROLL_WEIGHTS},
    "interview": {"speech": INTERVIEW_SPEECH_WEIGHTS, "broll": BROLL_WEIGHTS},
    "broll": {"speech": BROLL_WEIGHTS, "broll": BROLL_WEIGHTS},
}


def percentile_ranks(values: list[float]) -> list[float]:
    """Rank positions scaled to 0..1. A single value ranks 0.5."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return list(ranks / (n - 1))


def _exposure_quality(row: dict) -> float:
    return 1.0 - min(1.0, row.get("crushed_frac", 0.0) * 3.0 + row.get("blown_frac", 0.0) * 4.0)


def _duration_fit(duration: float) -> float:
    if duration <= 0:
        return 0.0
    if duration < BROLL_DURATION_SWEET_LO:
        return duration / BROLL_DURATION_SWEET_LO
    if duration <= BROLL_DURATION_SWEET_HI:
        return 1.0
    return max(0.2, 1.0 - (duration - BROLL_DURATION_SWEET_HI) / 40.0)


def _audio_quality_raw(row: dict) -> float:
    return row.get("noise_margin_db", 0.0) - 20.0 * row.get("silence_ratio", 0.0)


_RAW_EXTRACTORS = {
    "audio_quality": _audio_quality_raw,
    "speech_density": lambda row: row.get("words_per_second", 0.0),
    "sharpness": lambda row: row.get("sharpness", 0.0),
    "exposure": _exposure_quality,
    "motion": lambda row: row.get("motion", 0.0),
    "duration_fit": lambda row: _duration_fit(row.get("duration", 0.0)),
    "faces": lambda row: row.get("face_ratio"),
}


def _hard_fails(row: dict, min_speech_rms_db: float) -> list[str]:
    reasons = []
    if row["klass"] == "speech":
        if row.get("clipped"):
            reasons.append("clipped audio, distorted at full scale")
        if row.get("rms_db", 0.0) < min_speech_rms_db:
            reasons.append("speech far too quiet to use")
    if row.get("frames_sampled", 0) == 0:
        reasons.append("no frame could be decoded")
    return reasons


def score_run(
    rows: list[dict],
    profile: str = "auto",
    select_percentile: float = SELECT_PERCENTILE,
    reject_percentile: float = REJECT_PERCENTILE,
    min_speech_rms_db: float = MIN_SPEECH_RMS_DB,
) -> list[dict]:
    """Score and decide every segment row of a run. Returns new dicts.

    Rows must carry the feature keys produced by engine.analyze_file. The
    same list shape is accepted regardless of how many source files the
    rows span; percentiles are computed across the whole run per class.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choices: {sorted(PROFILES)}")
    weight_tables = PROFILES[profile]

    scored = [dict(row) for row in rows]
    by_class: dict[str, list[int]] = {"speech": [], "broll": []}
    for i, row in enumerate(scored):
        by_class.setdefault(row["klass"], []).append(i)

    for klass, indices in by_class.items():
        if not indices:
            continue
        weights = weight_tables.get(klass, BROLL_WEIGHTS)

        # Percentile-normalize each weighted feature within run and class.
        feature_percentiles: dict[str, dict[int, float | None]] = {}
        for feature in weights:
            raws = [(_RAW_EXTRACTORS[feature](scored[i]), i) for i in indices]
            present = [(value, i) for value, i in raws if value is not None]
            ranks = percentile_ranks([value for value, _ in present])
            per_index: dict[int, float | None] = {i: None for i in indices}
            for (_, i), rank in zip(present, ranks):
                per_index[i] = rank
            feature_percentiles[feature] = per_index

        composites: list[float] = []
        for i in indices:
            available = {
                feature: weight
                for feature, weight in weights.items()
                if feature_percentiles[feature][i] is not None
            }
            total = sum(available.values())
            if total <= 0:
                composite = 0.5
            else:
                composite = sum(
                    feature_percentiles[feature][i] * (weight / total)
                    for feature, weight in available.items()
                )
            scored[i]["composite"] = round(float(composite), 4)
            composites.append(float(composite))

        composite_ranks = percentile_ranks(composites)
        group_size = len(indices)
        for i, comp_p in zip(indices, composite_ranks):
            row = scored[i]
            row["composite_percentile"] = round(float(comp_p), 4)
            reasons = _hard_fails(row, min_speech_rms_db)
            if (
                group_size >= SHARPNESS_FAIL_MIN_GROUP
                and feature_percentiles.get("sharpness", {}).get(i) is not None
                and feature_percentiles["sharpness"][i] <= SHARPNESS_FAIL_PERCENTILE
            ):
                reasons.append("among the softest frames of the whole run")
            row["hard_fail"] = bool(reasons)

            if reasons:
                row["decision"] = "reject"
            elif group_size <= SMALL_GROUP_MAX:
                row["decision"] = "select"
                reasons = ["small group, kept every clean segment"]
            elif comp_p >= select_percentile:
                row["decision"] = "select"
                top_pct = max(1, round((1 - comp_p) * 100))
                reasons = [f"top {top_pct} percent of {klass} segments"]
            elif comp_p <= reject_percentile:
                row["decision"] = "reject"
                reasons = [f"bottom decile of {klass} segments"]
            else:
                row["decision"] = "none"
            row["reasons"] = reasons

    return scored
