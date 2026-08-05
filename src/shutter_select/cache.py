"""Per-source-file analysis cache: the idea 5 reuse contract.

One JSON per source file under <out>/cache/, keyed by a hash of the
resolved path and invalidated by source mtime plus size (the shutter-clip
incremental pattern). Raw features are stored, never only composites, so
downstream consumers (shutter-clip's social ranking) can re-weight without
re-running analysis.

Contract rules, from the spec:
- schema_version starts at 1 and gates every read.
- Additive evolution only. A removal or semantic change bumps the version
  and gets a mistakes.md ledger entry.

Note, logged in _hq/mistakes.md (2026-08-05): decisions are computed at
emit time, not stored here. Composite percentiles are run-relative, so a
cached decision would go stale the moment more footage joins the shoot.
The cache holds facts about one file; opinions about the run stay in the
emit stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from shutter_select import __version__
from shutter_select.probe import ProbeInfo

SCHEMA_VERSION = 1


def cache_key(source: Path) -> str:
    return hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: Path, source: Path) -> Path:
    return cache_dir / f"{cache_key(source)}.json"


def build_payload(
    source: Path,
    info: ProbeInfo,
    file_audio: dict,
    transcript: dict,
    segment_rows: list[dict],
) -> dict:
    stat = source.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": {"name": "shutter-select", "version": __version__},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(source.resolve()),
            "name": source.name,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "duration": info.duration,
            "fps": [info.fps.numerator, info.fps.denominator],
            "width": info.width,
            "height": info.height,
            "vcodec": info.vcodec,
            "acodec": info.acodec,
            "has_audio": info.has_audio,
            "rotation": info.rotation,
        },
        "audio": file_audio,
        "transcript": transcript,
        "segments": segment_rows,
    }


def write_cache(cache_dir: Path, source: Path, payload: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_path(cache_dir, source)
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(target)
    return target


def load_cache(cache_dir: Path, source: Path) -> dict | None:
    """Return the cached payload when present and still valid, else None."""
    target = cache_path(cache_dir, source)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    recorded = payload.get("source", {})
    try:
        stat = source.stat()
    except OSError:
        return None
    if abs(recorded.get("mtime", -1) - stat.st_mtime) > 1e-6:
        return None
    if recorded.get("size") != stat.st_size:
        return None
    return payload


def probe_from_payload(payload: dict) -> ProbeInfo:
    """Rebuild a ProbeInfo from a cached payload for emit-time timeline math."""
    from fractions import Fraction

    src = payload["source"]
    num, den = src["fps"]
    return ProbeInfo(
        path=src["path"],
        duration=src["duration"],
        fps=Fraction(num, den),
        width=src["width"],
        height=src["height"],
        has_audio=src["has_audio"],
        vcodec=src["vcodec"],
        acodec=src.get("acodec"),
        rotation=src.get("rotation", 0),
    )


__all__ = [
    "SCHEMA_VERSION",
    "asdict",
    "build_payload",
    "cache_key",
    "cache_path",
    "load_cache",
    "probe_from_payload",
    "write_cache",
]
