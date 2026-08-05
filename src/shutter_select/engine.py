"""Pipeline orchestration: the importable API surface.

analyze_file() runs the full per-file pipeline and returns the cacheable
payload of raw features. analyze_root() adds discovery, caching, and
incremental skips. decide() applies run-relative scoring across every
file's segments at emit time.

This module is a small structural addition on top of the spec's file list
(logged in _hq/mistakes.md 2026-08-05): the spec promises a stable Python
API for idea 5 alongside the CLI, and that API needs a home that is not
cli.py.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from shutter_select import audio as audio_mod
from shutter_select import cache as cache_mod
from shutter_select import segments as segments_mod
from shutter_select import visual as visual_mod
from shutter_select.faces import FaceCounter, ensure_model
from shutter_select.probe import ProbeInfo, probe
from shutter_select.transcribe import (
    AudioExtractError,
    Transcriber,
    TranscriptResult,
    extract_audio,
)
from shutter_select.walk import discover, resolve_root

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:  # pragma: no cover - default sink
    return None


def analyze_file(
    source: Path,
    transcriber: Transcriber | None,
    *,
    scene_threshold: float = segments_mod.DEFAULT_SCENE_THRESHOLD,
    take_gap: float = segments_mod.GAP_SECONDS,
    words: bool = False,
    face_counter: FaceCounter | None = None,
    progress: Progress = _noop,
) -> dict:
    """Run the full per-file pipeline. Returns the cache payload dict."""
    info: ProbeInfo = probe(source)

    transcript = TranscriptResult()
    file_audio_dict: dict = {"has_audio": info.has_audio}
    samples = None
    rate = 0
    file_audio = None

    if info.has_audio:
        with tempfile.TemporaryDirectory(prefix="shutter-select-") as tmp:
            wav = Path(tmp) / "audio.wav"
            try:
                extract_audio(source, wav)
            except AudioExtractError:
                file_audio_dict["has_audio"] = False
            else:
                samples, rate = audio_mod.read_wav(wav)
                if transcriber is not None:
                    progress(f"  transcribing {source.name}")
                    transcript = transcriber.transcribe(wav, words=words)

    if samples is not None:
        lufs = audio_mod.integrated_lufs(source)
        file_audio = audio_mod.file_features(samples, rate, lufs=lufs)
        file_audio_dict.update(
            {
                "peak_db": round(file_audio.peak_db, 2),
                "clipped_ratio": file_audio.clipped_ratio,
                "noise_floor_db": round(file_audio.noise_floor_db, 2),
                "integrated_lufs": file_audio.integrated_lufs,
                "silences": file_audio.silences,
            }
        )

    progress(f"  detecting scenes in {source.name}")
    cuts = segments_mod.detect_scene_cuts(source, threshold=scene_threshold)
    segs = segments_mod.build_segments(
        transcript.spans, cuts, info.duration, gap=take_gap
    )

    rows: list[dict] = []
    for seg in segs:
        row = {
            "index": seg.index,
            "t_in": seg.t_in,
            "t_out": seg.t_out,
            "duration": round(seg.duration, 3),
            "klass": seg.klass,
            "speech_ratio": seg.speech_ratio,
            "transcript": seg.transcript,
            "words_per_second": round(seg.words_per_second, 3),
        }
        if samples is not None and file_audio is not None:
            row.update(asdict(audio_mod.segment_features(samples, rate, seg.t_in, seg.t_out, file_audio)))
            row["rms_db"] = round(row["rms_db"], 2)
            row["peak_db"] = round(row["peak_db"], 2)
            row["noise_margin_db"] = round(row["noise_margin_db"], 2)
            row["silence_ratio"] = round(row["silence_ratio"], 3)
        else:
            row.update(
                {
                    "rms_db": audio_mod.FLOOR_DB,
                    "peak_db": audio_mod.FLOOR_DB,
                    "clipped": False,
                    "silence_ratio": 1.0,
                    "noise_margin_db": 0.0,
                }
            )

        progress(f"  sampling frames {source.name} [{seg.t_in:.1f}-{seg.t_out:.1f}]")
        frames = visual_mod.sample_segment(source, seg.t_in, seg.t_out)
        vis = visual_mod.features_from_frames(frames)
        row.update(
            {
                "sharpness": round(vis.sharpness, 2),
                "mean_luma": round(vis.mean_luma, 2),
                "crushed_frac": round(vis.crushed_frac, 4),
                "blown_frac": round(vis.blown_frac, 4),
                "motion": round(vis.motion, 4),
                "frames_sampled": vis.frames_sampled,
            }
        )
        row["face_ratio"] = (
            face_counter.face_ratio(frames) if face_counter is not None else None
        )
        rows.append(row)

    transcript_dict = {
        "language": transcript.language,
        "language_probability": round(transcript.language_probability, 4),
        "spans": [
            {"start": s.start, "end": s.end, "text": s.text} for s in transcript.spans
        ],
        "words": transcript.words,
    }
    return cache_mod.build_payload(source, info, file_audio_dict, transcript_dict, rows)


def analyze_root(
    root: Path,
    out_dir: Path,
    *,
    model: str = "small",
    allow_download: bool = False,
    words: bool = False,
    faces: bool = False,
    scene_threshold: float = segments_mod.DEFAULT_SCENE_THRESHOLD,
    take_gap: float = segments_mod.GAP_SECONDS,
    transcriber: Transcriber | None = None,
    progress: Progress = _noop,
) -> list[dict]:
    """Analyze every discovered file under root, using the cache to skip."""
    sources = discover(root)
    if not sources:
        return []

    cache_dir = out_dir / "cache"
    if transcriber is None:
        transcriber = Transcriber(model=model, allow_download=allow_download)
    face_counter = None
    if faces:
        model_path = ensure_model("face_detector", allow_download=allow_download)
        face_counter = FaceCounter(model_path)

    payloads: list[dict] = []
    for source in sources:
        cached = cache_mod.load_cache(cache_dir, source)
        if cached is not None:
            progress(f"cached    {source.name}")
            payloads.append(cached)
            continue
        progress(f"analyzing {source.name}")
        payload = analyze_file(
            source,
            transcriber,
            scene_threshold=scene_threshold,
            take_gap=take_gap,
            words=words,
            face_counter=face_counter,
            progress=progress,
        )
        cache_mod.write_cache(cache_dir, source, payload)
        payloads.append(payload)
    return payloads


def load_cached_payloads(root: Path, out_dir: Path) -> list[dict]:
    """Emit-time loader: valid cache payloads for every discovered file."""
    payloads = []
    for source in discover(root):
        cached = cache_mod.load_cache(out_dir / "cache", source)
        if cached is not None:
            payloads.append(cached)
    return payloads


def decide(
    payloads: list[dict],
    *,
    profile: str = "auto",
    select_percentile: float | None = None,
    reject_percentile: float | None = None,
) -> list[tuple[dict, list[dict]]]:
    """Score all payloads as one run. Returns (payload, scored rows) pairs."""
    from shutter_select import scoring

    kwargs = {}
    if select_percentile is not None:
        kwargs["select_percentile"] = select_percentile
    if reject_percentile is not None:
        kwargs["reject_percentile"] = reject_percentile

    flat: list[dict] = []
    owners: list[int] = []
    for i, payload in enumerate(payloads):
        for row in payload["segments"]:
            flat.append(row)
            owners.append(i)

    scored = scoring.score_run(flat, profile=profile, **kwargs)
    grouped: list[list[dict]] = [[] for _ in payloads]
    for owner, row in zip(owners, scored):
        grouped[owner].append(row)
    return list(zip(payloads, grouped))


def resolve_out_dir(root: Path, out: str | None) -> Path:
    return Path(out).expanduser().resolve() if out else root / "_selects"


__all__ = [
    "analyze_file",
    "analyze_root",
    "decide",
    "load_cached_payloads",
    "resolve_out_dir",
    "resolve_root",
]
