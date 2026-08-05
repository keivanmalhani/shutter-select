"""CLI-level tests. The full pipeline runs with a fake transcriber so no
model download or real inference happens outside the integration marker."""

from __future__ import annotations

import json

import pytest

from shutter_select import cli, engine
from shutter_select.transcribe import SpeechSpan, TranscriptResult
from tests.conftest import make_video, needs_ffmpeg


class FakeTranscriber:
    """Pretends the sine tone is a two-span answer about the canyon."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, wav_path, words=False):
        return TranscriptResult(
            spans=[
                SpeechSpan(0.3, 1.2, "The canyon light was unbelievable."),
                SpeechSpan(1.5, 2.4, "We are coming back next season."),
            ],
            language="en",
            language_probability=0.99,
        )


@pytest.fixture()
def shoot(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "Transcriber", FakeTranscriber)
    root = tmp_path / "shoot"
    root.mkdir()
    make_video(root / "a_interview.mp4", duration=3.0)
    make_video(root / "b_broll.mp4", duration=3.0, with_audio=False)
    return root


@needs_ffmpeg
def test_scan_lists_footage(shoot, capsys):
    assert cli.main(["scan", str(shoot)]) == 0
    out = capsys.readouterr().out
    assert "a_interview.mp4" in out and "b_broll.mp4" in out
    assert "Total footage" in out


@needs_ffmpeg
def test_run_produces_all_outputs_and_cache_hits(shoot, capsys):
    assert cli.main(["run", str(shoot)]) == 0
    out_dir = shoot / "_selects"

    assert (out_dir / "selects.otio").exists()
    assert (out_dir / "stringout.otio").exists()
    assert (out_dir / "report.txt").exists()
    assert (out_dir / "transcripts" / "a_interview.srt").exists()
    assert (out_dir / "transcripts" / "transcript.json").exists()

    caches = list((out_dir / "cache").glob("*.json"))
    assert len(caches) == 2
    payload = json.loads(caches[0].read_text())
    assert payload["schema_version"] == 1

    report = (out_dir / "report.txt").read_text()
    assert "canyon" in report.lower()

    # Second run must hit the cache for both files.
    capsys.readouterr()
    assert cli.main(["-v", "run", str(shoot)]) == 0
    out = capsys.readouterr().out
    assert out.count("cached") == 2


@needs_ffmpeg
def test_emit_without_cache_explains_itself(shoot, capsys):
    assert cli.main(["emit", str(shoot)]) == 1
    out = capsys.readouterr().out
    assert "analyze" in out


@needs_ffmpeg
def test_outputs_land_outside_root_with_out_flag(shoot, tmp_path, capsys):
    elsewhere = tmp_path / "elsewhere"
    assert cli.main(["run", str(shoot), "--out", str(elsewhere)]) == 0
    assert (elsewhere / "selects.otio").exists()
    assert not (shoot / "_selects").exists()


def test_missing_root_is_a_clean_error(capsys):
    assert cli.main(["scan", "/definitely/not/a/real/path"]) == 1
    assert "error:" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
