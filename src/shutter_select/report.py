"""Human-readable run report, same posture as shutter-cull's report.py.

Plain text, per file and per segment, every decision explained with its
reasons, plus run totals that answer the only question that matters: how
much footage do I actually have to watch now.
"""

from __future__ import annotations

from pathlib import Path


def _fmt_clock(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(minutes)}:{int(secs):02d}"


def _quote(row: dict, limit: int = 70) -> str:
    text = (row.get("transcript") or "").strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return f' "{text}"'


def build_report(analyzed: list[tuple[dict, list[dict]]]) -> str:
    lines: list[str] = []
    lines.append("shutter-select run report")
    lines.append("=" * 60)

    total_duration = 0.0
    selects_duration = 0.0
    counts = {"select": 0, "reject": 0, "none": 0}

    for payload, rows in analyzed:
        src = payload["source"]
        total_duration += src["duration"]
        language = payload.get("transcript", {}).get("language") or "-"
        speech_n = sum(1 for r in rows if r["klass"] == "speech")
        broll_n = len(rows) - speech_n
        lines.append("")
        take_word = "take" if speech_n == 1 else "takes"
        lines.append(
            f"{src['name']}  ({_fmt_clock(src['duration'])}, "
            f"{src['width']}x{src['height']}, lang {language}, "
            f"{speech_n} {take_word} / {broll_n} b-roll)"
        )
        for row in sorted(rows, key=lambda r: r["t_in"]):
            counts[row["decision"]] += 1
            if row["decision"] == "select":
                selects_duration += row["t_out"] - row["t_in"]
            mark = {"select": "PICK ", "reject": "DROP ", "none": "  -  "}[row["decision"]]
            reasons = "; ".join(row.get("reasons", []))
            lines.append(
                f"  {mark}{_fmt_clock(row['t_in'])}-{_fmt_clock(row['t_out'])} "
                f"{row['klass']:<6} score {row.get('composite', 0):.2f}"
                + (f"  [{reasons}]" if reasons else "")
                + _quote(row)
            )

    lines.append("")
    lines.append("-" * 60)
    lines.append(
        f"Footage scanned: {_fmt_clock(total_duration)}   "
        f"Selects runtime: {_fmt_clock(selects_duration)}"
    )
    lines.append(
        f"Segments: {sum(counts.values())} total, {counts['select']} picked, "
        f"{counts['reject']} dropped, {counts['none']} left for your judgment"
    )
    if total_duration > 0:
        ratio = 100.0 * selects_duration / total_duration
        lines.append(f"You now review {ratio:.0f} percent of what you shot.")
    lines.append("")
    return "\n".join(lines)


def write_report(out_dir: Path, analyzed: list[tuple[dict, list[dict]]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report.txt"
    target.write_text(build_report(analyzed), encoding="utf-8")
    return target
