from __future__ import annotations

from shutter_select.segments import detect_scene_cuts
from shutter_select.visual import features_from_frames, sample_segment, sample_times
from tests.conftest import make_video, needs_ffmpeg

pytestmark = needs_ffmpeg


def test_sample_times_bounds_and_count():
    times = sample_times(10.0, 22.0)
    assert 3 <= len(times) <= 12
    assert times[0] >= 10.0 and times[-1] <= 22.0

    short = sample_times(1.0, 1.4)
    assert len(short) == 3


def test_sharpness_separates_sharp_from_blurred(tmp_path):
    sharp = make_video(tmp_path / "sharp.mp4", duration=2.0)
    soft = make_video(tmp_path / "soft.mp4", duration=2.0, blur=True)

    sharp_feats = features_from_frames(sample_segment(sharp, 0.2, 1.8))
    soft_feats = features_from_frames(sample_segment(soft, 0.2, 1.8))

    assert sharp_feats.frames_sampled >= 3
    assert sharp_feats.sharpness > soft_feats.sharpness * 3


def test_motion_detected_on_moving_pattern(fixture_clip):
    feats = features_from_frames(sample_segment(fixture_clip, 0.2, 2.8))
    assert feats.motion > 0.005  # testsrc2 animates constantly
    assert 0.0 < feats.mean_luma < 255.0


def test_scene_cut_found_in_two_scene_clip(fixture_two_scene):
    cuts = detect_scene_cuts(fixture_two_scene)
    assert len(cuts) == 1
    assert 1.5 < cuts[0] < 2.5


def test_no_cuts_in_continuous_clip(fixture_clip):
    assert detect_scene_cuts(fixture_clip) == []
