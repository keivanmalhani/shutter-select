"""Visual quality features from sampled frames.

Per segment, up to MAX_FRAMES frames are pulled at roughly 1 fps via fast
ffmpeg seeks, downscaled to 640 px wide, and scored for sharpness
(Laplacian variance, the same measure proven in shutter-mcp and
shutter-cull), exposure (crushed and blown histogram fractions), and motion
(mean absolute difference between consecutive samples).

One ffmpeg process per sampled frame is deliberate MVP simplicity: it is
seek-accurate and codec-agnostic. A single-pass sampler is a known phase 2
performance item once real 4K shoots make process spawn cost visible.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_FRAMES = 12
MIN_FRAMES = 3
SAMPLE_WIDTH = 640
EDGE_MARGIN = 0.1
CRUSHED_LEVEL = 8
BLOWN_LEVEL = 247


@dataclass(frozen=True)
class SegmentVisualFeatures:
    sharpness: float
    mean_luma: float
    crushed_frac: float
    blown_frac: float
    motion: float
    frames_sampled: int


def sample_times(t_in: float, t_out: float) -> list[float]:
    duration = max(0.0, t_out - t_in)
    count = int(np.clip(int(duration), MIN_FRAMES, MAX_FRAMES))
    margin = min(EDGE_MARGIN, duration / 4.0)
    return list(np.linspace(t_in + margin, t_out - margin, count))


def grab_frame(source: Path, at: float) -> np.ndarray | None:
    """Decode one frame as BGR, downscaled. None when decode fails."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, at):.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale={SAMPLE_WIDTH}:-2",
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def sample_segment(source: Path, t_in: float, t_out: float) -> list[np.ndarray]:
    frames = []
    for at in sample_times(t_in, t_out):
        frame = grab_frame(source, at)
        if frame is not None:
            frames.append(frame)
    return frames


def features_from_frames(frames: list[np.ndarray]) -> SegmentVisualFeatures:
    if not frames:
        return SegmentVisualFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    sharpness = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]))
    stacked = np.concatenate([g.ravel() for g in grays])
    mean_luma = float(np.mean(stacked))
    crushed = float(np.mean(stacked <= CRUSHED_LEVEL))
    blown = float(np.mean(stacked >= BLOWN_LEVEL))

    if len(grays) >= 2:
        diffs = [
            float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) / 255.0
            for a, b in zip(grays, grays[1:])
        ]
        motion = float(np.mean(diffs))
    else:
        motion = 0.0

    return SegmentVisualFeatures(
        sharpness=sharpness,
        mean_luma=mean_luma,
        crushed_frac=crushed,
        blown_frac=blown,
        motion=motion,
        frames_sampled=len(frames),
    )
