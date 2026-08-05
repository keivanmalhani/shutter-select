"""ffprobe wrapper. Frame rates are kept as exact rationals.

29.97 and 23.976 material must never be rounded: timeline math in
timeline.py converts seconds to frame counts against the exact rational
rate, and a rounded rate drifts by a frame every few seconds on long
sources.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class ProbeError(Exception):
    """ffprobe missing, failed, or returned no usable video stream."""


@dataclass(frozen=True)
class ProbeInfo:
    path: str
    duration: float
    fps: Fraction
    width: int
    height: int
    has_audio: bool
    vcodec: str
    acodec: str | None
    rotation: int

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def require_binaries() -> None:
    for binary in ("ffprobe", "ffmpeg"):
        if shutil.which(binary) is None:
            raise ProbeError(
                f"{binary} was not found on PATH. Install ffmpeg first: "
                "https://ffmpeg.org/download.html (macOS: brew install ffmpeg)"
            )


def _parse_rate(raw: str | None) -> Fraction | None:
    if not raw or raw in ("0/0", "N/A"):
        return None
    try:
        rate = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _rotation_of(stream: dict) -> int:
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                return int(side["rotation"]) % 360
            except (TypeError, ValueError):
                continue
    tags = stream.get("tags") or {}
    try:
        return int(tags.get("rotate", 0)) % 360
    except (TypeError, ValueError):
        return 0


def probe(path: Path) -> ProbeInfo:
    """Probe one media file. Raises ProbeError when it cannot be analyzed."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out on {path.name}") from exc
    if out.returncode != 0:
        raise ProbeError(f"ffprobe failed on {path.name}: {out.stderr.strip()[:200]}")

    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    video = None
    audio = None
    for stream in streams:
        kind = stream.get("codec_type")
        disposition = stream.get("disposition") or {}
        if kind == "video" and video is None:
            # Ignore cover-art style streams the same way shutter-clip does.
            if disposition.get("attached_pic"):
                continue
            if stream.get("codec_name") in ("mjpeg", "png") and len(streams) > 1:
                continue
            video = stream
        elif kind == "audio" and audio is None:
            audio = stream
    if video is None:
        raise ProbeError(f"No usable video stream in {path.name}")

    fps = (
        _parse_rate(video.get("avg_frame_rate"))
        or _parse_rate(video.get("r_frame_rate"))
        or Fraction(30)
    )

    duration = 0.0
    for source in (video.get("duration"), (data.get("format") or {}).get("duration")):
        try:
            duration = float(source)
            break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        raise ProbeError(f"Could not determine duration of {path.name}")

    return ProbeInfo(
        path=str(path),
        duration=duration,
        fps=fps,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        has_audio=audio is not None,
        vcodec=str(video.get("codec_name") or "unknown"),
        acodec=str(audio.get("codec_name")) if audio else None,
        rotation=_rotation_of(video),
    )
