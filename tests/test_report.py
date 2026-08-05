from __future__ import annotations

from shutter_select.report import build_report
from tests.conftest import make_row


def _payload(name="interview.mp4", duration=120.0):
    return {
        "source": {
            "path": f"/shoot/{name}",
            "name": name,
            "duration": duration,
            "width": 3840,
            "height": 2160,
            "fps": [24, 1],
        },
        "transcript": {"language": "en"},
    }


def _row(decision, **overrides):
    row = make_row(**overrides)
    row["decision"] = decision
    row["composite"] = overrides.get("composite", 0.75)
    row["reasons"] = overrides.get("reasons", [])
    return row


def test_report_contains_decisions_reasons_and_totals():
    rows = [
        _row("select", t_in=10.0, t_out=40.0, reasons=["top 20 percent of speech segments"]),
        _row(
            "reject",
            t_in=50.0,
            t_out=60.0,
            reasons=["clipped audio, distorted at full scale"],
            transcript="ruined take",
        ),
        _row("none", t_in=70.0, t_out=80.0, transcript=""),
    ]
    text = build_report([(_payload(), rows)])

    assert "interview.mp4" in text
    assert "PICK" in text and "DROP" in text
    assert "clipped audio" in text
    assert '"we should film the canyon at sunrise"' in text
    assert "Footage scanned: 2:00" in text
    assert "Selects runtime: 0:30" in text
    assert "1 picked, 1 dropped, 1 left for your judgment" in text
    assert "review 25 percent of what you shot" in text


def test_report_handles_run_with_no_selects():
    rows = [_row("none", t_in=0.0, t_out=5.0)]
    text = build_report([(_payload(duration=5.0), rows)])
    assert "Selects runtime: 0:00" in text
