from __future__ import annotations

from pathlib import Path

import pytest

from shutter_select.walk import PathAccessError, discover, resolve_root


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_resolve_root_rejects_missing_and_files(tmp_path):
    with pytest.raises(PathAccessError):
        resolve_root(str(tmp_path / "nope"))
    file_path = _touch(tmp_path / "a.mp4")
    with pytest.raises(PathAccessError):
        resolve_root(str(file_path))
    with pytest.raises(PathAccessError):
        resolve_root("   ")


def test_resolve_root_resolves_symlinked_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert resolve_root(str(link)) == real.resolve()


def test_discover_finds_videos_and_skips_everything_else(tmp_path):
    keep_a = _touch(tmp_path / "a.MP4")
    keep_b = _touch(tmp_path / "sub" / "b.mov")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "._a.MP4")  # AppleDouble
    _touch(tmp_path / ".hidden.mp4")
    _touch(tmp_path / "_render" / "c.mp4")  # underscore dir
    _touch(tmp_path / "_selects" / "cacheed.mp4")  # own output tree
    _touch(tmp_path / "DO NOT INCLUDE THESE (CLAUDE)" / "d.mp4")
    _touch(tmp_path / "sub" / "do not use these" / "e.mp4")

    found = discover(resolve_root(str(tmp_path)))
    assert found == [keep_a, keep_b]


def test_discover_does_not_follow_symlinks_outside_root(tmp_path):
    outside = tmp_path / "outside"
    inside = tmp_path / "root"
    _touch(outside / "leak.mp4")
    inside.mkdir()
    (inside / "linkdir").symlink_to(outside)
    _touch(inside / "ok.mp4")

    found = discover(resolve_root(str(inside)))
    assert [p.name for p in found] == ["ok.mp4"]


def test_a_symlink_escaping_the_root_is_not_walked_in(tmp_path):
    """The discover() docstring: "any file whose resolved location escapes the root is
    skipped, so a stray symlink inside the shoot folder can never pull outside footage
    into the run." Nothing tested it. Deleting the check left all 97 tests green while
    the walk started returning footage from outside the shoot.

    Added 2026-09-02 after a mutation sweep.
    """
    root = tmp_path / "shoot"
    outside = tmp_path / "elsewhere"
    _touch(root / "mine.mp4")
    _touch(outside / "not_mine.mp4")
    (root / "linked.mp4").symlink_to(outside / "not_mine.mp4")

    found = [p.name for p in discover(root.resolve())]
    assert found == ["mine.mp4"]
