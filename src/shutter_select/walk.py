"""Footage discovery: symlink-safe root validation plus family skip rules.

resolve_root is ported from the shutter-mcp / shutter-cull paths.py pattern:
the supplied root must exist, must be a directory, and is resolved through
symlinks before anything walks it.

Skip rules are ported from shutter-clip and apply at any depth:
- directories whose name contains "do not include" or "do not use"
  (case-insensitive), which encodes the DO NOT INCLUDE THESE (CLAUDE)
  folder convention on the footage drive
- dotfiles and dot-directories, including AppleDouble "._*" files
- directories starting with "_", which also makes this tool ignore its own
  _selects/ output tree and shutter-clip's _phone-ready/ tree on re-runs
"""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mxf",
    ".avi",
    ".mkv",
    ".mts",
    ".m2ts",
    ".webm",
}

_SKIP_DIR_MARKERS = ("do not include", "do not use")


class PathAccessError(Exception):
    """Raised when a supplied root path is missing, not a directory, or invalid."""


def resolve_root(raw_root: str) -> Path:
    """Resolve and validate a user-supplied root directory."""
    if not raw_root or not str(raw_root).strip():
        raise PathAccessError("Root path must not be empty.")

    candidate = Path(raw_root).expanduser()

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PathAccessError(f"Root path does not exist: {raw_root}") from exc
    except OSError as exc:
        raise PathAccessError(f"Could not resolve root path '{raw_root}': {exc}") from exc

    if not resolved.is_dir():
        raise PathAccessError(f"Root path is not a directory: {raw_root}")

    return resolved


def _skip_dir(name: str) -> bool:
    lowered = name.casefold()
    if name.startswith(".") or name.startswith("_"):
        return True
    return any(marker in lowered for marker in _SKIP_DIR_MARKERS)


def _skip_file(name: str) -> bool:
    if name.startswith("."):  # covers AppleDouble "._*" too
        return True
    if name.startswith("_"):
        return True
    return Path(name).suffix.lower() not in VIDEO_EXTS


def discover(root: Path) -> list[Path]:
    """Walk root and return sorted video file paths, honoring all skip rules.

    Symlinked directories are not followed, and any file whose resolved
    location escapes the root is skipped, so a stray symlink inside the
    shoot folder can never pull outside footage into the run.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        for name in sorted(filenames):
            if _skip_file(name):
                continue
            path = Path(dirpath) / name
            try:
                if not path.resolve().is_relative_to(root):
                    continue
            except OSError:
                continue
            found.append(path)
    return found
