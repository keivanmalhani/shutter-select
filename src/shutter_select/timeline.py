"""Timeline, subtitle, and transcript emit.

Two OTIO timelines per run:
- selects.otio: only Select segments, chronological, one clip each.
- stringout.otio: every segment of every file in order, so each call the
  engine made is visible in context. Nothing is hidden.

Markers carry the verdict: green Select, red Reject, yellow unmarked, with
score, reasons, and a transcript snippet in the marker name and metadata.

Formats: .otio opens natively in DaVinci Resolve. selects.fcp7.xml (FCP7
xmeml via the fcp_xml adapter) targets Premiere import. selects.edl
(cmx_3600) is the cuts-only fallback; EDL barely supports markers, which
is documented rather than worked around.

Frame math uses each source's exact rational rate from ffprobe; seconds
are converted to integer frame counts at that rate, never through a
rounded fps.
"""

from __future__ import annotations

import json
from pathlib import Path

import opentimelineio as otio

MARKER_COLORS = {
    "select": otio.schema.MarkerColor.GREEN,
    "reject": otio.schema.MarkerColor.RED,
    "none": otio.schema.MarkerColor.YELLOW,
}
SNIPPET_CHARS = 90


def _rate_of(payload: dict) -> float:
    num, den = payload["source"]["fps"]
    return num / den


def _clip_for(payload: dict, row: dict) -> otio.schema.Clip:
    rate = _rate_of(payload)
    src = payload["source"]
    start_frame = round(row["t_in"] * rate)
    end_frame = max(start_frame + 1, round(row["t_out"] * rate))

    media = otio.schema.ExternalReference(
        target_url=Path(src["path"]).as_uri(),
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, rate),
            duration=otio.opentime.RationalTime(round(src["duration"] * rate), rate),
        ),
    )
    clip = otio.schema.Clip(
        name=f"{src['name']} {row['klass']} {_fmt_tc(row['t_in'])}",
        media_reference=media,
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start_frame, rate),
            duration=otio.opentime.RationalTime(end_frame - start_frame, rate),
        ),
    )

    snippet = (row.get("transcript") or "").strip()
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[: SNIPPET_CHARS - 3] + "..."
    label = {
        "select": "SELECT",
        "reject": "REJECT",
        "none": "?",
    }[row["decision"]]
    marker = otio.schema.Marker(
        name=f"{label} {row.get('composite', 0):.2f} {row['klass']}"
        + (f" | {snippet}" if snippet else ""),
        color=MARKER_COLORS[row["decision"]],
        marked_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start_frame, rate),
            duration=otio.opentime.RationalTime(1, rate),
        ),
        metadata={
            "shutter_select": {
                "decision": row["decision"],
                "composite": row.get("composite"),
                "composite_percentile": row.get("composite_percentile"),
                "reasons": row.get("reasons", []),
                "klass": row["klass"],
            }
        },
    )
    clip.markers.append(marker)
    return clip


def _timeline_from_rows(
    name: str, rows: list[tuple[dict, dict]]
) -> otio.schema.Timeline:
    timeline = otio.schema.Timeline(name=name)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)
    for payload, row in rows:
        track.append(_clip_for(payload, row))
    return timeline


def _ordered(rows: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    return sorted(rows, key=lambda pair: (pair[0]["source"]["path"], pair[1]["t_in"]))


def write_timelines(
    out_dir: Path, analyzed: list[tuple[dict, list[dict]]]
) -> tuple[list[Path], list[str]]:
    """Write selects and stringout timelines in every format. Returns paths plus warnings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    warnings: list[str] = []

    all_rows = _ordered(
        [(payload, row) for payload, rows in analyzed for row in rows]
    )
    select_rows = [(p, r) for p, r in all_rows if r["decision"] == "select"]

    selects = _timeline_from_rows("shutter-select selects", select_rows)
    stringout = _timeline_from_rows("shutter-select stringout", all_rows)

    for timeline, filename in ((selects, "selects.otio"), (stringout, "stringout.otio")):
        target = out_dir / filename
        otio.adapters.write_to_file(timeline, str(target))
        written.append(target)

    try:
        target = out_dir / "selects.fcp7.xml"
        otio.adapters.write_to_file(selects, str(target), adapter_name="fcp_xml")
        written.append(target)
    except Exception as exc:  # noqa: BLE001 - adapter failures degrade, not crash
        warnings.append(f"FCP7 XML export failed: {exc}")

    try:
        target = out_dir / "selects.edl"
        otio.adapters.write_to_file(selects, str(target), adapter_name="cmx_3600")
        written.append(target)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"EDL export failed (cuts-only format, markers unsupported): {exc}")

    return written, warnings


def _fmt_tc(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{secs:04.1f}"


def _fmt_srt_time(seconds: float) -> str:
    total_ms = round(max(0.0, seconds) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(out_dir: Path, payload: dict) -> Path | None:
    """One .srt per source file, from the whisper speech spans."""
    spans = payload.get("transcript", {}).get("spans", [])
    if not spans:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(payload["source"]["name"]).stem
    target = out_dir / f"{stem}.srt"
    lines = []
    for i, span in enumerate(spans, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(span['start'])} --> {_fmt_srt_time(span['end'])}")
        lines.append(span["text"].strip())
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def write_transcript_json(out_dir: Path, payloads: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "transcript.json"
    body = {
        "files": [
            {
                "path": payload["source"]["path"],
                "language": payload.get("transcript", {}).get("language"),
                "spans": payload.get("transcript", {}).get("spans", []),
                "words": payload.get("transcript", {}).get("words"),
            }
            for payload in payloads
        ]
    }
    target.write_text(json.dumps(body, indent=2, ensure_ascii=True), encoding="utf-8")
    return target
