from __future__ import annotations

import pytest

from shutter_select.scoring import percentile_ranks, score_run
from tests.conftest import make_row


def _speech_ladder(n: int) -> list[dict]:
    """n speech rows with strictly increasing quality on every feature."""
    rows = []
    for i in range(n):
        rows.append(
            make_row(
                index=i,
                noise_margin_db=10.0 + i,
                silence_ratio=0.0,
                words_per_second=1.0 + 0.1 * i,
                sharpness=100.0 + 10 * i,
                crushed_frac=max(0.0, 0.2 - 0.02 * i),
                blown_frac=0.0,
            )
        )
    return rows


def test_percentile_ranks_edges():
    assert percentile_ranks([]) == []
    assert percentile_ranks([7.0]) == [0.5]
    ranks = percentile_ranks([30.0, 10.0, 20.0])
    assert ranks == [1.0, 0.0, 0.5]


def test_ladder_orders_composites_and_decides():
    scored = score_run(_speech_ladder(10))
    composites = [row["composite"] for row in scored]
    assert composites == sorted(composites)
    assert scored[-1]["decision"] == "select"
    assert scored[0]["decision"] == "reject"
    assert "bottom decile" in scored[0]["reasons"][0]
    middle = scored[3]
    assert middle["decision"] == "none" and middle["reasons"] == []


def test_small_group_selects_every_clean_take():
    # The shutter-cull singleton lesson: one clean take must be a Select.
    scored = score_run([make_row()])
    assert scored[0]["decision"] == "select"
    assert "small group" in scored[0]["reasons"][0]

    # Small group with one hard fail: the fail still rejects.
    scored = score_run([make_row(), make_row(clipped=True)])
    assert scored[0]["decision"] == "select"
    assert scored[1]["decision"] == "reject"


def test_hard_fails_beat_good_composites():
    rows = _speech_ladder(10)
    rows[-1]["clipped"] = True  # best composite in the run, but distorted
    scored = score_run(rows)
    assert scored[-1]["decision"] == "reject"
    assert any("clipped" in reason for reason in scored[-1]["reasons"])

    rows = _speech_ladder(10)
    rows[-1]["rms_db"] = -50.0
    scored = score_run(rows)
    assert scored[-1]["decision"] == "reject"
    assert any("quiet" in reason for reason in scored[-1]["reasons"])


def test_clipped_broll_is_not_an_audio_hard_fail():
    rows = [make_row(klass="broll", clipped=True, transcript="", words_per_second=0.0)]
    scored = score_run(rows)
    assert scored[0]["decision"] == "select"  # small group, clean visually


def test_undecodable_segment_rejects_in_any_class():
    scored = score_run([make_row(klass="broll", frames_sampled=0)])
    assert scored[0]["decision"] == "reject"
    assert any("decoded" in reason for reason in scored[0]["reasons"])


def test_classes_are_ranked_separately():
    speech = _speech_ladder(4)
    broll = [
        make_row(
            index=10 + i,
            klass="broll",
            transcript="",
            words_per_second=0.0,
            sharpness=50.0 + i,
            motion=0.02 * i,
            duration=8.0,
        )
        for i in range(4)
    ]
    scored = score_run(speech + broll)
    speech_scores = [r["composite_percentile"] for r in scored if r["klass"] == "speech"]
    broll_scores = [r["composite_percentile"] for r in scored if r["klass"] == "broll"]
    # Each class spans its own full 0..1 percentile range independently.
    assert max(speech_scores) == 1.0 and min(speech_scores) == 0.0
    assert max(broll_scores) == 1.0 and min(broll_scores) == 0.0


def test_faces_weight_redistributes_when_absent():
    rows = _speech_ladder(6)  # face_ratio None everywhere
    scored = score_run(rows)
    assert all(0.0 <= row["composite"] <= 1.0 for row in scored)
    # Same ladder with faces present on every row keeps the same ordering.
    for i, row in enumerate(rows):
        row["face_ratio"] = 0.1 * i
    rescored = score_run(rows)
    assert [r["index"] for r in sorted(rescored, key=lambda r: r["composite"])] == list(
        range(6)
    )


def test_broll_profile_ignores_speech_features():
    quiet_but_gorgeous = make_row(
        index=0, noise_margin_db=1.0, words_per_second=0.1, sharpness=900.0, motion=0.2
    )
    loud_but_soft = make_row(
        index=1, noise_margin_db=35.0, words_per_second=3.0, sharpness=20.0, motion=0.01
    )
    filler = [
        make_row(index=2 + i, noise_margin_db=15.0, sharpness=100.0 + i, motion=0.05)
        for i in range(4)
    ]
    rows = [quiet_but_gorgeous, loud_but_soft, *filler]

    default_scored = {r["index"]: r["composite"] for r in score_run(rows)}
    broll_scored = {r["index"]: r["composite"] for r in score_run(rows, profile="broll")}

    assert default_scored[1] > default_scored[0]  # audio wins normally
    assert broll_scored[0] > broll_scored[1]  # visuals win under broll profile


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        score_run([make_row()], profile="cinematic")


def test_custom_thresholds_are_respected():
    scored = score_run(_speech_ladder(10), select_percentile=0.95, reject_percentile=0.0)
    selects = [r for r in scored if r["decision"] == "select"]
    rejects = [r for r in scored if r["decision"] == "reject"]
    assert len(selects) == 1
    assert len(rejects) == 1  # only the exact bottom (percentile 0.0)
