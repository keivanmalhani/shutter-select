"""Shared fixtures: synthetic footage, synthetic audio, feature-row factory.

Fixture encodes always use libx264 ultrafast (standing sandbox rule from
_hq/mistakes.md: software encoders only, keep test loops fast).
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def make_video(
    path: Path,
    duration: float = 3.0,
    fps: int = 24,
    size: str = "320x240",
    with_audio: bool = True,
    blur: bool = False,
    pattern: str = "testsrc2",
) -> Path:
    """Encode a small synthetic clip."""
    vsrc = f"{pattern}=duration={duration}:size={size}:rate={fps}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", vsrc]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    if blur:
        cmd += ["-vf", "gblur=sigma=8"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def make_two_scene_video(path: Path, half: float = 2.0, fps: int = 24) -> Path:
    """Two visually distinct halves joined by one hard cut."""
    filter_graph = (
        f"testsrc2=duration={half}:size=320x240:rate={fps}[a];"
        f"smptebars=duration={half}:size=320x240:rate={fps}[b];"
        "[a][b]concat=n=2:v=1:a=0"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        filter_graph,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> Path:
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(ints.tobytes())
    return path


def sine(duration: float, rate: int = 16000, freq: float = 440.0, amp: float = 0.5):
    t = np.arange(int(duration * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_row(**overrides) -> dict:
    """A complete segment feature row with sane speech-take defaults."""
    row = {
        "index": 0,
        "t_in": 0.0,
        "t_out": 10.0,
        "duration": 10.0,
        "klass": "speech",
        "speech_ratio": 0.9,
        "transcript": "we should film the canyon at sunrise",
        "words_per_second": 2.5,
        "rms_db": -18.0,
        "peak_db": -6.0,
        "clipped": False,
        "silence_ratio": 0.05,
        "noise_margin_db": 30.0,
        "sharpness": 250.0,
        "mean_luma": 118.0,
        "crushed_frac": 0.01,
        "blown_frac": 0.005,
        "motion": 0.04,
        "frames_sampled": 8,
        "face_ratio": None,
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="session")
def fixture_clip(tmp_path_factory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    root = tmp_path_factory.mktemp("clips")
    return make_video(root / "clip.mp4")


@pytest.fixture(scope="session")
def fixture_two_scene(tmp_path_factory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    root = tmp_path_factory.mktemp("scenes")
    return make_two_scene_video(root / "twoscene.mp4")
