"""End-to-end integration: real whisper tiny on espeak-generated speech.

Excluded from the default run and CI by the integration marker (pyproject
addopts). Run locally with: pytest -m integration
Requires: ffmpeg, espeak-ng, network on first run for the tiny model.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from shutter_select import engine
from shutter_select.transcribe import Transcriber
from tests.conftest import HAVE_FFMPEG

pytestmark = pytest.mark.integration

HAVE_ESPEAK = shutil.which("espeak-ng") is not None


def _speech_clip(path, text, duration=6.0):
    wav = path.with_suffix(".wav")
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "130", "-w", str(wav), text],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=duration={duration}:size=320x240:rate=24",
            "-i", str(wav),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.mark.skipif(not (HAVE_FFMPEG and HAVE_ESPEAK), reason="needs ffmpeg and espeak-ng")
def test_real_transcription_end_to_end(tmp_path):
    root = tmp_path / "shoot"
    root.mkdir()
    _speech_clip(
        root / "take.mov",
        "The light through the canyon at sunrise was unbelievable.",
    )

    transcriber = Transcriber(model="tiny", allow_download=True)
    payloads = engine.analyze_root(
        root, root / "_selects", transcriber=transcriber
    )
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["transcript"]["language"] == "en"
    joined = " ".join(s["text"] for s in payload["transcript"]["spans"]).lower()
    # Word-level accuracy is a model concern, not an engine concern: the
    # tiny model garbles synthetic speech sometimes. Two keyword hits prove
    # the audio really went through transcription.
    hits = sum(1 for word in ("light", "canyon", "sunrise", "unbelievable") if word in joined)
    assert hits >= 2, f"transcript too far off: {joined!r}"

    speech_rows = [r for r in payload["segments"] if r["klass"] == "speech"]
    assert speech_rows, "expected at least one speech take"

    analyzed = engine.decide(payloads)
    decisions = [row["decision"] for _, rows in analyzed for row in rows]
    assert "select" in decisions
