"""The segment model: speech is the primary structure, scenes fill the rest.

Scene detection alone fails the primary use case (tripod interview footage
has zero visual cuts for whole cards), so:

1. Whisper speech spans are grouped into takes: gaps under GAP_SECONDS
   join, longer gaps split. Take boundaries pad outward by PAD_SECONDS,
   clamped to file bounds and to the midpoint between adjacent takes.
2. Regions with no speech are segmented by PySceneDetect cuts.
3. A scene cut never splits a speech take (known MVP simplification,
   logged in the spec).
4. Class: "speech" when speech covers at least half the segment, else
   "broll". Segments keep natural length, no maximum cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shutter_select.transcribe import SpeechSpan

GAP_SECONDS = 1.5
PAD_SECONDS = 0.3
MIN_BROLL_SECONDS = 1.0
MIN_REGION_SECONDS = 0.5
DEFAULT_SCENE_THRESHOLD = 27.0


@dataclass
class Segment:
    index: int
    t_in: float
    t_out: float
    klass: str  # "speech" | "broll"
    speech_ratio: float = 0.0
    transcript: str = ""
    spans: list[SpeechSpan] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.t_out - self.t_in)

    @property
    def words_per_second(self) -> float:
        words = len(self.transcript.split())
        return words / self.duration if self.duration > 0 else 0.0


def detect_scene_cuts(
    source: Path, threshold: float = DEFAULT_SCENE_THRESHOLD
) -> list[float]:
    """Interior scene-cut times in seconds via PySceneDetect ContentDetector."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(source), ContentDetector(threshold=threshold))
    return [start.seconds for start, _end in scenes[1:]]


def group_takes(
    spans: list[SpeechSpan],
    duration: float,
    gap: float = GAP_SECONDS,
    pad: float = PAD_SECONDS,
) -> list[tuple[float, float, list[SpeechSpan]]]:
    """Merge speech spans into padded, non-overlapping takes."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start)
    groups: list[list[SpeechSpan]] = [[ordered[0]]]
    for span in ordered[1:]:
        if span.start - groups[-1][-1].end < gap:
            groups[-1].append(span)
        else:
            groups.append([span])

    takes: list[tuple[float, float, list[SpeechSpan]]] = []
    for group in groups:
        t_in = max(0.0, group[0].start - pad)
        t_out = min(duration, group[-1].end + pad)
        takes.append((t_in, t_out, group))

    # Clamp any pad-induced overlap to the midpoint between neighbors.
    for i in range(1, len(takes)):
        prev_in, prev_out, prev_group = takes[i - 1]
        cur_in, cur_out, cur_group = takes[i]
        if cur_in < prev_out:
            mid = (prev_group[-1].end + cur_group[0].start) / 2.0
            takes[i - 1] = (prev_in, mid, prev_group)
            takes[i] = (mid, cur_out, cur_group)
    return takes


def _split_region(
    region_start: float, region_end: float, cuts: list[float]
) -> list[tuple[float, float]]:
    """Split one no-speech region at the scene cuts inside it, merging slivers."""
    if region_end - region_start < MIN_REGION_SECONDS:
        return []
    interior = [c for c in cuts if region_start < c < region_end]
    bounds = [region_start, *interior, region_end]
    pieces = [
        (a, b) for a, b in zip(bounds, bounds[1:]) if b - a > 1e-9
    ]
    merged: list[tuple[float, float]] = []
    for piece in pieces:
        if merged and piece[1] - piece[0] < MIN_BROLL_SECONDS:
            merged[-1] = (merged[-1][0], piece[1])
        else:
            merged.append(piece)
    if len(merged) >= 2 and merged[0][1] - merged[0][0] < MIN_BROLL_SECONDS:
        first, second = merged[0], merged[1]
        merged[:2] = [(first[0], second[1])]
    return merged


def _speech_overlap(spans: list[SpeechSpan], t_in: float, t_out: float) -> float:
    total = 0.0
    for span in spans:
        total += max(0.0, min(span.end, t_out) - max(span.start, t_in))
    return total


def build_segments(
    spans: list[SpeechSpan],
    cuts: list[float],
    duration: float,
    gap: float = GAP_SECONDS,
    pad: float = PAD_SECONDS,
) -> list[Segment]:
    """Assemble the full segment list for one source file."""
    takes = group_takes(spans, duration, gap=gap, pad=pad)

    segments: list[Segment] = []
    for t_in, t_out, group in takes:
        seg_duration = max(1e-9, t_out - t_in)
        ratio = min(1.0, _speech_overlap(group, t_in, t_out) / seg_duration)
        segments.append(
            Segment(
                index=0,
                t_in=round(t_in, 3),
                t_out=round(t_out, 3),
                klass="speech" if ratio >= 0.5 else "broll",
                speech_ratio=round(ratio, 3),
                transcript=" ".join(s.text.strip() for s in group).strip(),
                spans=list(group),
            )
        )

    # Fill the complement regions with scene-bounded b-roll segments.
    cursor = 0.0
    boundaries = [(t.t_in, t.t_out) for t in segments]
    regions: list[tuple[float, float]] = []
    for t_in, t_out in boundaries:
        if t_in - cursor > 0:
            regions.append((cursor, t_in))
        cursor = max(cursor, t_out)
    if duration - cursor > 0:
        regions.append((cursor, duration))

    for region_start, region_end in regions:
        for piece_in, piece_out in _split_region(region_start, region_end, cuts):
            segments.append(
                Segment(
                    index=0,
                    t_in=round(piece_in, 3),
                    t_out=round(piece_out, 3),
                    klass="broll",
                )
            )

    segments.sort(key=lambda s: s.t_in)
    for i, segment in enumerate(segments):
        segment.index = i
    return segments
