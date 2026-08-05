from __future__ import annotations

import json

import opentimelineio as otio
import pytest

from shutter_select import timeline
from tests.conftest import make_row


def _payload(tmp_path, name="a.mp4", fps=(24, 1), duration=60.0, spans=None):
    return {
        "schema_version": 1,
        "source": {
            "path": str(tmp_path / name),
            "name": name,
            "duration": duration,
            "fps": list(fps),
            "width": 1920,
            "height": 1080,
            "vcodec": "h264",
            "acodec": "aac",
            "has_audio": True,
            "mtime": 0.0,
            "size": 1,
            "rotation": 0,
        },
        "audio": {"has_audio": True},
        "transcript": {"language": "en", "spans": spans or [], "words": None},
        "segments": [],
    }


def _scored(**overrides):
    row = make_row(**overrides)
    row.setdefault("composite", 0.8)
    row.setdefault("composite_percentile", 0.9)
    row.setdefault("decision", "select")
    row.setdefault("reasons", ["top 10 percent of speech segments"])
    return row


def test_selects_timeline_structure_and_frame_math(tmp_path):
    payload = _payload(tmp_path, fps=(30000, 1001))
    rows = [
        _scored(t_in=10.0, t_out=20.0),
        _scored(t_in=30.0, t_out=35.0, decision="none", reasons=[]),
        _scored(t_in=40.0, t_out=45.0, decision="reject", reasons=["bottom decile"]),
    ]
    written, warnings = timeline.write_timelines(tmp_path / "out", [(payload, rows)])
    names = {p.name for p in written}
    assert {"selects.otio", "stringout.otio"} <= names

    selects = otio.adapters.read_from_file(str(tmp_path / "out" / "selects.otio"))
    clips = list(selects.find_clips())
    assert len(clips) == 1  # only the select survives
    clip = clips[0]
    rate = 30000 / 1001
    assert clip.source_range.start_time.value == round(10.0 * rate)
    assert clip.source_range.start_time.rate == pytest.approx(rate)
    marker = clip.markers[0]
    assert marker.color == otio.schema.MarkerColor.GREEN
    assert marker.name.startswith("SELECT")
    assert marker.metadata["shutter_select"]["decision"] == "select"

    stringout = otio.adapters.read_from_file(str(tmp_path / "out" / "stringout.otio"))
    stringout_clips = list(stringout.find_clips())
    assert len(stringout_clips) == 3
    colors = [c.markers[0].color for c in stringout_clips]
    assert colors == [
        otio.schema.MarkerColor.GREEN,
        otio.schema.MarkerColor.YELLOW,
        otio.schema.MarkerColor.RED,
    ]


def test_premiere_and_edl_exports_write_or_warn(tmp_path):
    payload = _payload(tmp_path)
    rows = [_scored(t_in=1.0, t_out=5.0)]
    written, warnings = timeline.write_timelines(tmp_path / "out", [(payload, rows)])
    names = {p.name for p in written}
    if "selects.fcp7.xml" not in names:
        assert any("FCP7" in w for w in warnings)
    if "selects.edl" not in names:
        assert any("EDL" in w for w in warnings)


def test_chronological_order_across_files(tmp_path):
    a = _payload(tmp_path, name="a.mp4")
    b = _payload(tmp_path, name="b.mp4")
    rows_a = [_scored(t_in=50.0, t_out=55.0), _scored(t_in=5.0, t_out=9.0)]
    rows_b = [_scored(t_in=1.0, t_out=4.0)]
    timeline.write_timelines(tmp_path / "out", [(b, rows_b), (a, rows_a)])
    selects = otio.adapters.read_from_file(str(tmp_path / "out" / "selects.otio"))
    starts = [
        (c.media_reference.target_url, c.source_range.start_time.value)
        for c in selects.find_clips()
    ]
    assert starts == sorted(starts)


def test_srt_and_transcript_json(tmp_path):
    spans = [
        {"start": 0.0, "end": 2.5, "text": "First line spoken."},
        {"start": 3.0, "end": 5.75, "text": "Second line spoken."},
    ]
    payload = _payload(tmp_path, spans=spans)
    srt = timeline.write_srt(tmp_path / "out", payload)
    body = srt.read_text()
    assert "1\n00:00:00,000 --> 00:00:02,500\nFirst line spoken." in body
    assert "2\n00:00:03,000 --> 00:00:05,750\nSecond line spoken." in body

    combined = timeline.write_transcript_json(tmp_path / "out", [payload])
    data = json.loads(combined.read_text())
    assert data["files"][0]["language"] == "en"
    assert len(data["files"][0]["spans"]) == 2


def test_srt_skipped_when_no_speech(tmp_path):
    payload = _payload(tmp_path, spans=[])
    assert timeline.write_srt(tmp_path / "out", payload) is None
