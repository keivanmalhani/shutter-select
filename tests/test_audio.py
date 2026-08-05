from __future__ import annotations

import numpy as np

from shutter_select import audio
from tests.conftest import sine, write_wav


def test_read_wav_roundtrip(tmp_path):
    samples = sine(1.0, amp=0.25)
    path = write_wav(tmp_path / "a.wav", samples)
    loaded, rate = audio.read_wav(path)
    assert rate == 16000
    assert abs(len(loaded) - len(samples)) <= 1
    assert np.allclose(loaded[:100], samples[:100], atol=1e-3)


def test_frame_rms_matches_sine_math():
    # RMS of a sine is amplitude / sqrt(2); at amp 0.5 that is about -9 dBFS.
    samples = sine(2.0, amp=0.5)
    rms_db, times = audio.frame_rms_db(samples, 16000)
    assert abs(float(np.median(rms_db)) - (-9.03)) < 0.5
    assert times[0] < times[-1] < 2.0


def test_silence_spans_found_between_tones():
    rate = 16000
    loud = sine(1.0, rate, amp=0.5)
    quiet = np.zeros(rate, dtype=np.float32)
    samples = np.concatenate([loud, quiet, loud])
    rms_db, times = audio.frame_rms_db(samples, rate)
    spans = audio.silence_spans(rms_db, times, threshold_db=-45.0, min_duration=0.4)
    assert len(spans) == 1
    start, end = spans[0]
    assert 0.8 < start < 1.2
    assert 1.8 < end < 2.2


def test_file_features_clipping_and_floor(tmp_path):
    rate = 16000
    clipped_square = np.ones(rate, dtype=np.float32)  # hard-clipped full scale
    samples = np.concatenate([clipped_square, sine(1.0, rate, amp=0.01)])
    feats = audio.file_features(samples, rate)
    assert feats.clipped_ratio > 0.1
    assert feats.peak_db > -0.5
    assert feats.noise_floor_db < -35.0


def test_segment_features_margin_and_silence():
    rate = 16000
    voice = sine(2.0, rate, amp=0.4)
    room_tone = sine(2.0, rate, amp=0.02)  # about -37 dBFS, above the -45 gate
    samples = np.concatenate([voice, room_tone])
    file_feats = audio.file_features(samples, rate)

    seg_voice = audio.segment_features(samples, rate, 0.0, 2.0, file_feats)
    seg_quiet = audio.segment_features(samples, rate, 2.0, 4.0, file_feats)

    assert not seg_voice.clipped
    assert seg_voice.noise_margin_db > 20.0
    assert seg_voice.rms_db > seg_quiet.rms_db + 20.0
    assert seg_quiet.silence_ratio == 0.0  # room tone sits above the silence gate


def test_segment_features_out_of_range_is_safe():
    rate = 16000
    samples = sine(1.0, rate)
    file_feats = audio.file_features(samples, rate)
    seg = audio.segment_features(samples, rate, 5.0, 6.0, file_feats)
    assert seg.silence_ratio == 1.0
    assert seg.rms_db <= audio.FLOOR_DB
