from __future__ import annotations

import subprocess
from fractions import Fraction

import pytest

from shutter_select.probe import ProbeError, probe
from tests.conftest import make_video, needs_ffmpeg

pytestmark = needs_ffmpeg


def test_probe_reads_synthetic_clip(fixture_clip):
    info = probe(fixture_clip)
    assert info.width == 320 and info.height == 240
    assert info.fps == Fraction(24)
    assert 2.5 < info.duration < 3.5
    assert info.has_audio
    assert info.vcodec == "h264"
    assert info.rotation == 0


def test_probe_clip_without_audio(tmp_path):
    clip = make_video(tmp_path / "mute.mp4", duration=1.0, with_audio=False)
    info = probe(clip)
    assert not info.has_audio
    assert info.acodec is None


def test_probe_rejects_audio_only_file(tmp_path):
    audio_only = tmp_path / "audio.m4a"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:a", "aac", str(audio_only),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ProbeError):
        probe(audio_only)


def test_probe_rejects_non_media(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(ProbeError):
        probe(junk)
