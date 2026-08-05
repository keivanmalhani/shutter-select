"""Pre-flight checks, so `run` fails at the terminal instead of quietly
transcribing nothing because ffmpeg is missing or a footage drive is
mounted read-only.

The first support question for a tool like this is never "why is the
selects timeline wrong", it is "why did nothing happen": ffmpeg or ffprobe
is not on PATH so every source fails to probe, PySceneDetect is not
importable so b-roll segmentation dies on the first file, the whisper model
is not cached and --allow-download was not passed, or the footage folder is
on a read-only mount. `shutter-select doctor` asks those questions on
purpose, in one pass, before a real `run` does.

Modeled on shutter-farm's doctor.py (github.com/keivanmalhani/shutter-farm,
src/shutter_farm/doctor.py) and its sibling in shutter-cull
(src/shutter_cull/doctor.py): same Check/Context shape, same "never stop at
the first failure" posture, same version-probe fallback from --version to
--help, and the same reason permission checks go through an injected
access() rather than os.access directly - chmod-based tests silently no-op
under a root user, and CI containers usually run as root.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Matches pyproject.toml's requires-python and the README's stated minimum.
MIN_PYTHON = (3, 11)

# ffmpeg and ffprobe are both hard requirements: probe.require_binaries()
# checks both before any command runs, including a bare `scan`.
FFMPEG_BINARIES = ("ffmpeg", "ffprobe")

# Whisper model size -> the Hugging Face repo faster-whisper resolves it
# from. Mirrors MODEL_MANIFEST.json's "whisper.repos" at the repo root, kept
# here as a plain dict rather than read from that file at runtime, since a
# non-editable install would not carry a repo-root file alongside the
# installed package.
WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}
DEFAULT_WHISPER_MODEL = "small"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""

    @property
    def fatal(self) -> bool:
        return self.status == FAIL


@dataclass
class Context:
    """Everything a check is allowed to look at.

    The callables are the seam. Real runs get the real ones, tests get a
    machine with exactly one thing wrong. Permissions go through an injected
    access() rather than os.access directly, because chmod-based tests skip
    themselves under a root user and CI containers usually run as root: a
    permission check that only runs on a laptop is not a tested check.
    """

    root: Path | None = None
    model: str = DEFAULT_WHISPER_MODEL
    which: Callable[[str], str | None] = shutil.which
    access: Callable[[Path, int], bool] = os.access
    version_of: Callable[[str], tuple[int, str]] | None = None
    can_import: Callable[[str], tuple[bool, str]] | None = None
    whisper_cache_state: Callable[[str], tuple[bool, str]] | None = None
    python_version: tuple[int, int] = field(default_factory=lambda: sys.version_info[:2])

    def __post_init__(self) -> None:
        if self.version_of is None:
            self.version_of = _real_version_of
        if self.can_import is None:
            self.can_import = _real_can_import
        if self.whisper_cache_state is None:
            self.whisper_cache_state = _real_whisper_cache_state


def _real_version_of(binary: str) -> tuple[int, str]:
    """Answer one question: does this binary actually execute?

    Being on PATH is not the same as being runnable. A broken symlink, a
    binary built for the wrong architecture, and a container missing a
    shared library all pass a `which` check and fail on use.

    `--version` is the probe, but a non-zero exit from it does not mean the
    binary is broken. Plenty of CLIs, including this project's own, have a
    required subcommand and exit non-zero on a bare `--version`. So a
    failing `--version` falls back to `--help`: if that runs, the binary
    runs, and the only honest thing to report is that it does not announce
    a version. Reporting "it will not run" for a tool that runs fine is the
    worst failure a diagnostic can have, because it sends someone to
    reinstall a working install.
    """

    def attempt(flag: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                [binary, flag], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, str(exc)
        text = (proc.stdout or proc.stderr).strip().splitlines()
        return proc.returncode, text[0] if text else ""

    code, line = attempt("--version")
    if code == 0:
        return 0, line
    help_code, help_line = attempt("--help")
    if help_code == 0:
        return 0, f"{binary}, installed, does not report a version"
    return help_code or code, help_line or line


def _real_can_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # ImportError and the occasional ABI SystemError
        return False, str(exc)
    version = getattr(module, "__version__", "")
    return True, f"{module_name} {version}".strip()


def _real_whisper_cache_state(model_name: str) -> tuple[bool, str]:
    """Is faster-whisper's underlying Hugging Face repo for this model size
    already cached? Answered by looking at the cache layout directly, the
    same thing Transcriber(..., allow_download=False) (local_files_only=True
    under the hood) would find, without paying the cost of actually loading
    a model into memory just to run doctor.
    """
    repo_id = WHISPER_REPOS.get(model_name)
    if repo_id is None:
        return False, f"Unknown whisper model size '{model_name}'"

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError as exc:
        return False, f"huggingface_hub is not importable: {exc}"

    repo_folder = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_folder / "snapshots"
    if not snapshots.is_dir():
        return False, str(repo_folder)
    if not any(snapshots.iterdir()):
        return False, str(repo_folder)
    return True, str(repo_folder)


# ---------------------------------------------------------------- the checks


def check_python(ctx: Context) -> Check:
    have = ".".join(str(n) for n in ctx.python_version)
    want = ".".join(str(n) for n in MIN_PYTHON)
    if ctx.python_version >= MIN_PYTHON:
        return Check("python", OK, f"Python {have}")
    return Check(
        "python",
        FAIL,
        f"Python {have}, but this needs {want} or newer",
        fix=f"Install Python {want}+ and reinstall: pip install -e .",
    )


def check_ffmpeg(ctx: Context) -> list[Check]:
    """ffmpeg and ffprobe are both hard requirements: every command,
    including a bare `scan`, calls require_binaries() for both before doing
    anything else. Missing either is fatal, not a warning."""
    checks: list[Check] = []
    for binary in FFMPEG_BINARIES:
        path = ctx.which(binary)
        if path is None:
            checks.append(
                Check(
                    binary,
                    FAIL,
                    f"Not on PATH. Every command, including scan, needs {binary}.",
                    fix="brew install ffmpeg  (macOS), or "
                    "sudo apt install ffmpeg  (Debian/Ubuntu). ffmpeg and "
                    "ffprobe both ship in the same package.",
                )
            )
            continue
        code, line = ctx.version_of(binary)
        if code != 0:
            checks.append(
                Check(
                    binary,
                    FAIL,
                    f"Found at {path} but it will not run: {line[:120]}",
                    fix=f"Reinstall ffmpeg for this environment. Being on PATH "
                    f"is not the same as {binary} being runnable, and this is "
                    "what a broken install or an architecture mismatch looks "
                    "like.",
                )
            )
            continue
        checks.append(Check(binary, OK, line or path))
    return checks


def check_import(ctx: Context, module: str, package: str, purpose: str) -> Check:
    ok, detail = ctx.can_import(module)
    if not ok:
        return Check(
            module,
            FAIL,
            f"Not importable, needed for {purpose}: {detail}",
            fix=f'pip install -e ".[dev]" in this environment, or directly: '
            f"pip install {package}",
        )
    return Check(module, OK, detail)


def check_whisper_cache(ctx: Context) -> Check:
    """Reports whether the whisper model faster-whisper would load is
    already cached, never fetches it. Missing is a warning: `scan` and
    `emit` (from an existing cache) both work without it, and `analyze`/
    `run` only need it once, with --allow-download."""
    cached, where = ctx.whisper_cache_state(ctx.model)
    if cached:
        return Check(
            f"whisper model ({ctx.model})", OK, f"Cached at {where}"
        )
    return Check(
        f"whisper model ({ctx.model})",
        WARN,
        f"Not cached ({where}). `analyze`/`run` will download it once the "
        "first time they see this model size.",
        fix=f"shutter-select run <root> --model {ctx.model} --allow-download   "
        "(one-time fetch from Hugging Face, cached after)",
    )


def check_root(ctx: Context) -> Check:
    if ctx.root is None:
        return Check(
            "target root", OK, "No root given, checking tools and this machine only."
        )
    if not ctx.root.exists():
        return Check(
            "target root",
            FAIL,
            f"{ctx.root} does not exist",
            fix="Check the path you passed to scan/analyze/emit/run/doctor.",
        )
    if not ctx.root.is_dir():
        return Check(
            "target root",
            FAIL,
            f"{ctx.root} is not a directory",
            fix="Point at the footage folder itself, not a file inside it.",
        )
    if not ctx.access(ctx.root, os.R_OK | os.X_OK):
        return Check(
            "target root",
            FAIL,
            f"{ctx.root} is not readable",
            fix=f"chmod +rx {ctx.root}, or run as a user that can read it.",
        )
    return Check("target root", OK, f"{ctx.root} is readable")


def run_checks(ctx: Context) -> list[Check]:
    checks = [check_python(ctx)]
    checks.extend(check_ffmpeg(ctx))
    checks.append(
        check_import(ctx, "scenedetect", "scenedetect", "b-roll segmentation (PySceneDetect)")
    )
    checks.append(check_whisper_cache(ctx))
    checks.append(check_root(ctx))
    return checks


def verdict(checks: list[Check]) -> tuple[bool, str]:
    """The one place that decides whether a real run will work.

    Both the printed summary and the exit code come from here, so the two
    can never contradict each other."""
    fails = sum(1 for c in checks if c.status == FAIL)
    warns = sum(1 for c in checks if c.status == WARN)

    if fails:
        return False, (
            f"{fails} blocking problem(s), {warns} warning(s). A run will "
            "not work until the blocking ones are fixed."
        )
    if warns:
        return True, f"No blocking problems, {warns} warning(s). A run will work."
    return True, "Everything checks out. A run will work."


def render(checks: list[Check]) -> str:
    """Human first. This is the output someone pastes into a bug report, so
    it has to be readable without a JSON parser."""
    marks = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}
    width = max(len(c.name) for c in checks)
    indent = " " * (width + 9)
    lines = []
    for check in checks:
        lines.append(f"  [{marks[check.status]}] {check.name.ljust(width)}  {check.detail}")
        if check.fix and check.status != OK:
            for i, part in enumerate(_wrap(check.fix, 68)):
                lines.append(f"{indent}{'-> ' if i == 0 else '   '}{part}")
    lines.append("")
    for part in _wrap(verdict(checks)[1], 74):
        lines.append(f"  {part}")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
