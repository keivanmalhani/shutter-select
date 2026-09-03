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


def test_small_group_makes_no_percentile_calls():
    # The shutter-cull singleton lesson, applied with the right sign
    # (2026-08-05 review checkpoint): a tiny group has no comparative
    # evidence, so nothing is auto-Selected OR auto-Rejected. The old
    # select-every-clean-segment rule sent a two-file shoot back 100
    # percent Selected, terrible takes included.
    scored = score_run([make_row()])
    assert scored[0]["decision"] == "none"
    assert "too small" in scored[0]["reasons"][0]

    # Small group with one hard fail: the fail still rejects.
    scored = score_run([make_row(), make_row(clipped=True)])
    assert scored[0]["decision"] == "none"
    assert scored[1]["decision"] == "reject"


def test_terrible_lone_take_is_not_endorsed():
    awful = make_row(
        noise_margin_db=0.5,
        silence_ratio=0.9,
        words_per_second=0.1,
        sharpness=3.0,
    )
    scored = score_run([awful])
    assert scored[0]["decision"] == "none"


def test_identical_rows_share_identical_fate():
    # Duplicate card import: twelve byte-identical segments must not be
    # split into selects and rejects by list order (review probe A).
    scored = score_run([make_row(klass="broll", transcript="", words_per_second=0.0) for _ in range(12)])
    assert len({row["decision"] for row in scored}) == 1
    assert len({row["composite"] for row in scored}) == 1


def test_tie_ranks_are_averaged():
    assert percentile_ranks([5.0, 5.0, 9.0]) == [0.25, 0.25, 1.0]


def test_bottom_decile_is_a_decile():
    # 11 strictly ordered rows: exactly the bottom one rejects, not two.
    scored = score_run(_speech_ladder(11))
    rejects = [r for r in scored if r["decision"] == "reject"]
    assert len(rejects) == 1


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
    assert scored[0]["decision"] == "none"  # no audio fail, tiny group: undecided
    assert not any("clipped" in reason for reason in scored[0]["reasons"])


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
    scored = score_run(_speech_ladder(10), select_percentile=0.95, reject_percentile=0.05)
    selects = [r for r in scored if r["decision"] == "select"]
    rejects = [r for r in scored if r["decision"] == "reject"]
    assert len(selects) == 1
    assert len(rejects) == 1  # only the exact bottom sits below 0.05


def test_zero_reject_percentile_disables_relative_rejects():
    scored = score_run(_speech_ladder(10), reject_percentile=0.0)
    assert not any(r["decision"] == "reject" for r in scored)


# ---------------------------------------------------------------------------
# Boundary cases added 2026-09-02 after a mutation sweep. 26 defects were planted
# across scoring, segments, walk and cache; 15 left all 97 tests green, and 12 of
# those changed behaviour a user would see. The ones below are those 12, in scoring.


def _broll_ladder(n: int) -> list[dict]:
    """n b-roll rows, strictly increasing on the two features broll weights lean on."""
    return [
        make_row(index=i, klass="broll", sharpness=100.0 + 10 * i, motion=0.01 * (i + 1))
        for i in range(n)
    ]


def test_a_segment_exactly_on_the_select_percentile_is_selected():
    """SELECT_PERCENTILE is 0.60 and the rule reads `comp_p >= select_percentile`.

    Six segments rank at exactly 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, so the fourth sits on
    the threshold. Flipping the comparison to `>` demotes it to unmarked and every
    test passed: no test ever produced a composite percentile equal to the threshold.
    """
    scored = score_run(_broll_ladder(6))
    percentiles = [row["composite_percentile"] for row in scored]
    assert percentiles == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert scored[3]["decision"] == "select"
    assert scored[2]["decision"] == "none"  # the neighbour below must stay unmarked


def test_a_group_of_exactly_small_group_max_makes_no_percentile_calls():
    """SMALL_GROUP_MAX is 3 and the rule reads `group_size <= SMALL_GROUP_MAX`.

    The existing small-group test uses groups of 1 and 2. At exactly 3, narrowing the
    comparison to `<` turns a three-clip shoot into one reject, one unmarked and one
    select - the 2026-08-05 failure this rule exists to prevent, at the one size
    nothing covered.
    """
    scored = score_run(_broll_ladder(3))
    assert [row["decision"] for row in scored] == ["none", "none", "none"]
    assert all("too small" in row["reasons"][0] for row in scored)

    # One more segment and the percentile calls come back: the negative half.
    scored = score_run(_broll_ladder(4))
    assert [row["decision"] for row in scored] != ["none", "none", "none"]


def test_the_softest_frames_fail_in_a_group_of_exactly_the_minimum():
    """SHARPNESS_FAIL_MIN_GROUP is 50 and the rule reads `group_size >= ...`.

    A run of exactly 50 segments is an ordinary card. With `>` the softest segment
    stops being called out and nothing notices.
    """
    scored = score_run(_broll_ladder(50))
    failed = [row for row in scored if row["hard_fail"]]
    assert len(failed) == 1
    assert "softest frames" in failed[0]["reasons"][0]

    # 49 is below the minimum, so no sharpness fail: the negative half.
    assert not any(row["hard_fail"] for row in score_run(_broll_ladder(49)))


def test_a_sharpness_rank_exactly_on_the_fail_percentile_still_fails():
    """SHARPNESS_FAIL_PERCENTILE is 0.02 and the rule reads `<= ...`.

    With 51 segments the ranks are k/50, so the second-softest lands on exactly 0.02.
    Narrowing to `<` drops it from two hard fails to one and re-labels it as an
    ordinary bottom-decile reject.
    """
    scored = score_run(_broll_ladder(51))
    failed = [row for row in scored if row["hard_fail"]]
    assert len(failed) == 2
    assert all("softest frames" in row["reasons"][0] for row in failed)


def test_speech_exactly_at_the_quiet_threshold_is_not_a_hard_fail():
    """MIN_SPEECH_RMS_DB is -35.0 and the rule reads `rms_db < min_speech_rms_db`.

    A take sitting exactly on the floor is usable; only below it is not. Widening to
    `<=` fails all six takes here, and the suite could not see it because no test ever
    put rms_db on the threshold.
    """
    scored = score_run([make_row(index=i, rms_db=-35.0) for i in range(6)])
    assert not any(row["hard_fail"] for row in scored)

    # A tenth of a dB under and it does fail: the negative half.
    scored = score_run([make_row(index=i, rms_db=-35.1) for i in range(6)])
    assert all(row["hard_fail"] for row in scored)
    assert all("too quiet" in row["reasons"][0] for row in scored)


def test_blown_highlights_cost_more_than_crushed_shadows():
    """_exposure_quality weights crushed at 3x and blown at 4x: clipped highlights are
    unrecoverable, crushed shadows often are not. Swapping the two constants left every
    test green, because no test has both fractions non-zero and unequal, so nothing
    pins which one is worse."""
    from shutter_select.scoring import _exposure_quality

    crushed = _exposure_quality({"crushed_frac": 0.2, "blown_frac": 0.0})
    blown = _exposure_quality({"crushed_frac": 0.0, "blown_frac": 0.2})
    assert crushed == pytest.approx(0.4)
    assert blown == pytest.approx(0.2)
    assert blown < crushed

    # And it reaches the decision: same fraction of bad pixels, blown ranks lower.
    rows = [make_row(index=i, crushed_frac=0.2, blown_frac=0.0) for i in range(3)]
    rows += [make_row(index=3 + i, crushed_frac=0.0, blown_frac=0.2) for i in range(3)]
    scored = score_run(rows)
    assert scored[0]["composite"] > scored[3]["composite"]


def test_silence_costs_twenty_db_of_noise_margin():
    """_audio_quality_raw is `noise_margin_db - 20 * silence_ratio`. Halving that
    coefficient reversed the ranking of these two takes and every test passed,
    because no test has two rows differing on both terms at once."""
    quiet_but_full = make_row(index=0, noise_margin_db=12.0, silence_ratio=0.0)
    cleaner_but_half_silent = make_row(index=1, noise_margin_db=20.0, silence_ratio=0.5)
    filler = [make_row(index=i, noise_margin_db=30.0, silence_ratio=0.0) for i in range(2, 6)]
    scored = score_run([quiet_but_full, cleaner_but_half_silent, *filler])
    assert scored[0]["composite"] > scored[1]["composite"], (
        "half a take of silence must cost more than the 8 dB of noise margin it buys"
    )
