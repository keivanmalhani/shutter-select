"""Audio quality features from the 16 kHz mono wav extracted per source.

Everything here is computed from the waveform in numpy rather than by
parsing ffmpeg filter logs: silence spans, per-segment RMS level, peak,
clipping ratio, and a speech-to-noise margin estimated against the file's
quiet floor. Segment loudness is therefore RMS dBFS, deliberately named
rms_db and not LUFS. A file-level integrated LUFS from ffmpeg's ebur128
filter is stored per file for the report. This refinement of the spec's
audio-pass wording is logged in _hq/mistakes.md (2026-08-05).

The speech-to-noise margin is a heuristic: loud-part level minus quiet
floor. It tracks how far voices sit above room tone and hum, it is not a
calibrated SNR measurement.
"""

from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME_SECONDS = 0.05
CLIP_SAMPLE_THRESHOLD = 0.999
CLIP_RATIO_THRESHOLD = 1e-4
FLOOR_DB = -90.0


@dataclass(frozen=True)
class AudioFileFeatures:
    duration: float
    peak_db: float
    clipped_ratio: float
    noise_floor_db: float
    integrated_lufs: float | None
    silences: list[tuple[float, float]]


@dataclass(frozen=True)
class SegmentAudioFeatures:
    rms_db: float
    peak_db: float
    clipped: bool
    silence_ratio: float
    noise_margin_db: float


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("Expected 16-bit mono wav from extract_audio()")
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def _to_db(values: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(values, 10 ** (FLOOR_DB / 20.0)))


def frame_rms_db(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling frame RMS in dBFS plus the center time of each frame."""
    hop = max(1, int(rate * FRAME_SECONDS))
    n_frames = max(1, len(samples) // hop)
    trimmed = samples[: n_frames * hop]
    frames = trimmed.reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    times = (np.arange(n_frames) + 0.5) * FRAME_SECONDS
    return np.asarray(_to_db(rms)), times


def silence_spans(
    rms_db: np.ndarray,
    times: np.ndarray,
    threshold_db: float = -45.0,
    min_duration: float = 0.4,
) -> list[tuple[float, float]]:
    """Contiguous stretches below threshold_db lasting at least min_duration."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    half = FRAME_SECONDS / 2.0
    for level, center in zip(rms_db, times):
        if level < threshold_db:
            if start is None:
                start = center - half
        elif start is not None:
            end = center - half
            if end - start >= min_duration:
                spans.append((round(start, 3), round(end, 3)))
            start = None
    if start is not None:
        end = float(times[-1]) + half
        if end - start >= min_duration:
            spans.append((round(start, 3), round(end, 3)))
    return spans


def integrated_lufs(source: Path) -> float | None:
    """File-level integrated loudness via ffmpeg ebur128. None on any failure."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-map",
        "a:0",
        "-af",
        "ebur128",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = None
    for match in re.finditer(r"I:\s*(-?[\d.]+)\s*LUFS", result.stderr):
        pass
    return float(match.group(1)) if match else None


def file_features(
    samples: np.ndarray,
    rate: int,
    silence_threshold_db: float = -45.0,
    lufs: float | None = None,
) -> AudioFileFeatures:
    rms_db, times = frame_rms_db(samples, rate)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    clipped_ratio = (
        float(np.mean(np.abs(samples) >= CLIP_SAMPLE_THRESHOLD)) if len(samples) else 0.0
    )
    return AudioFileFeatures(
        duration=len(samples) / rate if rate else 0.0,
        peak_db=float(_to_db(peak)),
        clipped_ratio=clipped_ratio,
        noise_floor_db=float(np.percentile(rms_db, 10)) if len(rms_db) else FLOOR_DB,
        integrated_lufs=lufs,
        silences=silence_spans(rms_db, times, threshold_db=silence_threshold_db),
    )


def _overlap(spans: list[tuple[float, float]], t_in: float, t_out: float) -> float:
    total = 0.0
    for start, end in spans:
        total += max(0.0, min(end, t_out) - max(start, t_in))
    return total


def segment_features(
    samples: np.ndarray,
    rate: int,
    t_in: float,
    t_out: float,
    file_feats: AudioFileFeatures,
) -> SegmentAudioFeatures:
    lo = max(0, int(t_in * rate))
    hi = min(len(samples), int(t_out * rate))
    chunk = samples[lo:hi]
    duration = max(1e-6, t_out - t_in)
    if len(chunk) == 0:
        return SegmentAudioFeatures(FLOOR_DB, FLOOR_DB, False, 1.0, 0.0)

    rms = float(np.sqrt(np.mean(chunk**2)))
    peak = float(np.max(np.abs(chunk)))
    clipped = float(np.mean(np.abs(chunk) >= CLIP_SAMPLE_THRESHOLD)) > CLIP_RATIO_THRESHOLD
    frame_db, _ = frame_rms_db(chunk, rate)
    loud_db = float(np.percentile(frame_db, 90)) if len(frame_db) else FLOOR_DB
    return SegmentAudioFeatures(
        rms_db=float(_to_db(rms)),
        peak_db=float(_to_db(peak)),
        clipped=clipped,
        silence_ratio=min(1.0, _overlap(file_feats.silences, t_in, t_out) / duration),
        noise_margin_db=max(0.0, loud_db - file_feats.noise_floor_db),
    )
