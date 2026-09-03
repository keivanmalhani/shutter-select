from __future__ import annotations

from shutter_select.segments import (
    Segment,
    _split_region,
    build_segments,
    group_takes,
)
from shutter_select.transcribe import SpeechSpan


def span(start: float, end: float, text: str = "words here") -> SpeechSpan:
    return SpeechSpan(start=start, end=end, text=text)


def test_group_takes_joins_short_gaps_and_splits_long_ones():
    spans = [span(1.0, 3.0), span(4.0, 6.0), span(9.0, 11.0)]  # gaps 1.0 and 3.0
    takes = group_takes(spans, duration=20.0)
    assert len(takes) == 2
    first, second = takes
    assert first[0] == 0.7 and abs(first[1] - 6.3) < 1e-9
    assert second[0] == 8.7 and abs(second[1] - 11.3) < 1e-9
    assert len(first[2]) == 2 and len(second[2]) == 1


def test_group_takes_clamps_to_bounds_and_neighbors():
    spans = [span(0.1, 2.0), span(2.2, 5.0)]
    takes = group_takes(spans, duration=5.1, gap=0.1)
    # Padding would overlap and overflow: clamp at 0, file end, and midpoint.
    assert takes[0][0] == 0.0
    assert takes[-1][1] == 5.1
    assert takes[0][1] == takes[1][0] == (2.0 + 2.2) / 2


def test_build_segments_speech_first_then_scene_fill():
    spans = [span(10.0, 20.0, "a great take about the canyon")]
    cuts = [4.0, 30.0, 30.6]  # one real cut early, one cut plus sliver later
    segs = build_segments(spans, cuts, duration=40.0)

    kinds = [(s.klass, round(s.t_in, 1), round(s.t_out, 1)) for s in segs]
    # Regions: 0-9.7 split at 4.0, take 9.7-20.3, then 20.3-40 split at 30.0
    # with the 0.6s sliver between the 30.0 and 30.6 cuts folded into its
    # left neighbor, so that boundary lands on 30.6.
    assert kinds == [
        ("broll", 0.0, 4.0),
        ("broll", 4.0, 9.7),
        ("speech", 9.7, 20.3),
        ("broll", 20.3, 30.6),
        ("broll", 30.6, 40.0),
    ]
    take = segs[2]
    assert take.transcript == "a great take about the canyon"
    assert take.speech_ratio > 0.9
    assert [s.index for s in segs] == [0, 1, 2, 3, 4]


def test_scene_cut_never_splits_a_speech_take():
    spans = [span(2.0, 30.0, "one long interview answer")]
    cuts = [10.0, 20.0]  # both inside the take
    segs = build_segments(spans, cuts, duration=32.0)
    speech = [s for s in segs if s.klass == "speech"]
    assert len(speech) == 1
    assert speech[0].duration > 27.0  # natural length kept, no cap


def test_no_speech_at_all_still_yields_broll_segments():
    segs = build_segments([], cuts=[5.0], duration=10.0)
    assert [s.klass for s in segs] == ["broll", "broll"]
    assert segs[0].transcript == ""


def test_tiny_gap_between_takes_produces_no_sliver_segment():
    spans = [span(0.0, 5.0), span(7.2, 12.0)]
    segs = build_segments(spans, [], duration=12.0)
    # 5.3-6.9 region is 1.6s -> kept; shrink the gap and it must vanish.
    spans_tight = [span(0.0, 5.0), span(5.8, 12.0)]
    segs_tight = build_segments(spans_tight, [], duration=12.0)
    assert sum(1 for s in segs if s.klass == "broll") == 1
    assert sum(1 for s in segs_tight if s.klass == "broll") == 0


def test_words_per_second_property():
    seg = Segment(index=0, t_in=0.0, t_out=10.0, klass="speech", transcript="one two three four")
    assert seg.words_per_second == 0.4


# ---------------------------------------------------------------------------
# Boundary cases added 2026-09-02 after a mutation sweep. Four defects in this module
# left all 97 tests green while changing behaviour a user would see.


def test_a_gap_of_exactly_gap_seconds_splits_the_take():
    """group_takes joins while `span.start - previous.end < gap`, so a gap of exactly
    GAP_SECONDS splits. The existing test uses gaps of 1.0 and 3.0, either side of the
    1.5 default and never on it, so widening the comparison to `<=` - which glues two
    separate takes into one - passed every test."""
    spans = [span(0.0, 1.0), span(2.5, 3.5)]  # gap is exactly 1.5
    takes = group_takes(spans, duration=10.0, gap=1.5)
    assert len(takes) == 2

    # A hair under and they do join: the negative half.
    spans = [span(0.0, 1.0), span(2.49, 3.5)]
    assert len(group_takes(spans, duration=10.0, gap=1.5)) == 1


def test_speech_covering_exactly_half_a_segment_is_a_speech_take():
    """The module docstring: speech when speech covers AT LEAST half the segment. The
    rule reads `ratio >= 0.5`. One 1.0 s span padded 0.5 s either side is a 2.0 s take
    that is exactly half speech; narrowing to `> 0.5` reclassifies it as b-roll, which
    silently swaps the whole weight table used to score it."""
    spans = [span(1.0, 2.0, "one line")]
    segments = build_segments(spans, [], duration=10.0, pad=0.5)
    take = [s for s in segments if s.transcript][0]
    assert take.speech_ratio == 0.5
    assert take.klass == "speech"

    # A shade less speech in the same take and it is b-roll: the negative half.
    spans = [span(1.0, 1.9, "one line")]
    segments = build_segments(spans, [], duration=10.0, pad=0.5)
    take = [s for s in segments if s.transcript][0]
    assert take.speech_ratio < 0.5 and take.klass == "broll"


def test_a_region_of_exactly_min_region_seconds_still_produces_a_segment():
    """_split_region drops a region shorter than MIN_REGION_SECONDS, so one of exactly
    0.5 s is kept. Widening that to `<=` throws it away and every test passed."""
    assert _split_region(0.0, 0.5, []) == [(0.0, 0.5)]
    # A hair shorter and it is dropped: the negative half.
    assert _split_region(0.0, 0.49, []) == []


def test_a_piece_of_exactly_min_broll_seconds_is_not_merged_away():
    """Slivers shorter than MIN_BROLL_SECONDS are folded into the piece before them.
    A piece of exactly 1.0 s is not a sliver and must stand on its own. Widening the
    test to `<=` merges it into its neighbour, and no test noticed because every
    existing case is well clear of 1.0 s."""
    # cuts at 2.0 and 3.0 carve 0..5 into 2.0 / 1.0 / 2.0
    assert _split_region(0.0, 5.0, [2.0, 3.0]) == [(0.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    # a 0.9 s middle piece IS a sliver and does get merged: the negative half.
    assert _split_region(0.0, 5.0, [2.0, 2.9]) == [(0.0, 2.9), (2.9, 5.0)]
