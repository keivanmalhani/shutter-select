"""shutter-select CLI.

Subcommands:
  scan     inventory plus probe report, no analysis
  analyze  full engine, writes one cache JSON per source file
  emit     timelines, transcripts, report from cache, re-runnable freely
  run      analyze then emit, the one-shot

Sources are only ever opened read-only. Every write lands under the out
directory (default <root>/_selects). The only network access the tool can
ever perform is the explicit, opt-in --allow-download model fetch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shutter_select import __version__, engine
from shutter_select.faces import ModelNotAvailable
from shutter_select.probe import ProbeError, probe, require_binaries
from shutter_select.report import build_report, write_report
from shutter_select.scoring import PROFILES
from shutter_select.timeline import write_srt, write_timelines, write_transcript_json
from shutter_select.transcribe import ModelNotCachedError
from shutter_select.walk import PathAccessError, discover, resolve_root


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", help="Footage folder to work on")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <root>/_selects)",
    )


def _add_analysis_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="small", help="Whisper model size (default: small)")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit the one-time model download (the only network access)",
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help="Store word-level timestamps in the transcript (for captions later)",
    )
    parser.add_argument(
        "--faces",
        action="store_true",
        help="Enable face presence scoring (downloads YuNet once with --allow-download)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=27.0,
        help="PySceneDetect content threshold (default: 27)",
    )
    parser.add_argument(
        "--take-gap",
        type=float,
        default=1.5,
        help="Silence gap in seconds that splits two takes (default: 1.5)",
    )


def _add_emit_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default="auto",
        choices=sorted(PROFILES),
        help="Scoring weight profile (default: auto)",
    )
    parser.add_argument(
        "--select-percentile",
        type=float,
        default=None,
        help="Composite percentile a segment needs to be picked (default: 0.60)",
    )
    parser.add_argument(
        "--reject-percentile",
        type=float,
        default=None,
        help="Composite percentile at or below which segments drop (default: 0.10)",
    )


def _cmd_scan(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    sources = discover(root)
    if not sources:
        print(f"No video files found under {root}")
        return 0
    print(f"{len(sources)} video file(s) under {root}\n")
    total = 0.0
    for source in sources:
        try:
            info = probe(source)
        except ProbeError as exc:
            print(f"  !! {source.relative_to(root)}: {exc}")
            continue
        total += info.duration
        audio = info.acodec or "no audio"
        print(
            f"  {source.relative_to(root)}  "
            f"{info.width}x{info.height} {float(info.fps):.3g} fps "
            f"{info.duration:.1f}s  {info.vcodec}/{audio}"
        )
    minutes, secs = divmod(round(total), 60)
    print(f"\nTotal footage: {minutes}m {secs:02d}s")
    return 0


def _progress_printer(verbose: bool):
    def emit(msg: str) -> None:
        if verbose or not msg.startswith("  "):
            print(msg)

    return emit


def _cmd_analyze(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    out_dir = engine.resolve_out_dir(root, args.out)
    payloads = engine.analyze_root(
        root,
        out_dir,
        model=args.model,
        allow_download=args.allow_download,
        words=args.words,
        faces=args.faces,
        scene_threshold=args.scene_threshold,
        take_gap=args.take_gap,
        progress=_progress_printer(args.verbose),
    )
    if not payloads:
        print(f"No video files found under {root}")
        return 0
    segments = sum(len(p["segments"]) for p in payloads)
    print(f"Analyzed {len(payloads)} file(s), {segments} segments -> {out_dir / 'cache'}")
    return 0


def _cmd_emit(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    out_dir = engine.resolve_out_dir(root, args.out)
    payloads = engine.load_cached_payloads(root, out_dir)
    if not payloads:
        print(
            "No analysis cache found. Run 'shutter-select analyze' first "
            f"(looked in {out_dir / 'cache'})."
        )
        return 1

    analyzed = engine.decide(
        payloads,
        profile=args.profile,
        select_percentile=args.select_percentile,
        reject_percentile=args.reject_percentile,
    )

    written, warnings = write_timelines(out_dir, analyzed)
    for payload in payloads:
        srt = write_srt(out_dir / "transcripts", payload)
        if srt:
            written.append(srt)
    written.append(write_transcript_json(out_dir / "transcripts", payloads))
    written.append(write_report(out_dir, analyzed))

    print(build_report(analyzed))
    print("Written:")
    for path in written:
        print(f"  {path}")
    for warning in warnings:
        print(f"  warning: {warning}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    code = _cmd_analyze(args)
    if code != 0:
        return code
    return _cmd_emit(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shutter-select",
        description=(
            "Local-first video culling: find the good takes in raw footage "
            "and hand your editor a ready-made selects timeline."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-step progress output")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="List footage and formats, no analysis")
    _add_shared(p_scan)
    p_scan.set_defaults(func=_cmd_scan)

    p_analyze = sub.add_parser("analyze", help="Run the engine, cache raw features")
    _add_shared(p_analyze)
    _add_analysis_flags(p_analyze)
    p_analyze.set_defaults(func=_cmd_analyze)

    p_emit = sub.add_parser("emit", help="Write timelines, transcripts, report from cache")
    _add_shared(p_emit)
    _add_emit_flags(p_emit)
    p_emit.set_defaults(func=_cmd_emit)

    p_run = sub.add_parser("run", help="analyze then emit in one shot")
    _add_shared(p_run)
    _add_analysis_flags(p_run)
    _add_emit_flags(p_run)
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_binaries()
        return args.func(args)
    except (PathAccessError, ProbeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ModelNotCachedError, ModelNotAvailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
