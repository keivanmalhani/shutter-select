from __future__ import annotations

import json
import os
from fractions import Fraction

from shutter_select import cache
from shutter_select.probe import ProbeInfo


def _fake_info(path: str) -> ProbeInfo:
    return ProbeInfo(
        path=path,
        duration=12.5,
        fps=Fraction(30000, 1001),
        width=3840,
        height=2160,
        has_audio=True,
        vcodec="hevc",
        acodec="aac",
        rotation=0,
    )


def _make_source(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake video bytes")
    return source


def _payload_for(source):
    return cache.build_payload(
        source,
        _fake_info(str(source)),
        {"has_audio": True},
        {"language": "en", "spans": [], "words": None},
        [{"index": 0, "t_in": 0.0, "t_out": 1.0, "klass": "broll"}],
    )


def test_roundtrip_and_exact_rational_fps(tmp_path):
    source = _make_source(tmp_path)
    cache.write_cache(tmp_path / "cache", source, _payload_for(source))

    loaded = cache.load_cache(tmp_path / "cache", source)
    assert loaded is not None
    assert loaded["schema_version"] == cache.SCHEMA_VERSION
    assert loaded["source"]["fps"] == [30000, 1001]  # 29.97 never rounded
    info = cache.probe_from_payload(loaded)
    assert info.fps == Fraction(30000, 1001)


def test_stale_on_mtime_change(tmp_path):
    source = _make_source(tmp_path)
    cache.write_cache(tmp_path / "cache", source, _payload_for(source))
    os.utime(source, (source.stat().st_atime, source.stat().st_mtime + 5))
    assert cache.load_cache(tmp_path / "cache", source) is None


def test_stale_on_size_change(tmp_path):
    source = _make_source(tmp_path)
    cache.write_cache(tmp_path / "cache", source, _payload_for(source))
    mtime = source.stat().st_mtime
    source.write_bytes(b"different, longer fake video bytes")
    os.utime(source, (mtime, mtime))
    assert cache.load_cache(tmp_path / "cache", source) is None


def test_schema_version_gates_reads(tmp_path):
    source = _make_source(tmp_path)
    target = cache.write_cache(tmp_path / "cache", source, _payload_for(source))
    body = json.loads(target.read_text())
    body["schema_version"] = 999
    target.write_text(json.dumps(body))
    assert cache.load_cache(tmp_path / "cache", source) is None


def test_corrupt_cache_is_treated_as_missing(tmp_path):
    source = _make_source(tmp_path)
    target = cache.write_cache(tmp_path / "cache", source, _payload_for(source))
    target.write_text("{ not json")
    assert cache.load_cache(tmp_path / "cache", source) is None


def test_missing_source_invalidates(tmp_path):
    source = _make_source(tmp_path)
    cache.write_cache(tmp_path / "cache", source, _payload_for(source))
    payload_path = cache.cache_path(tmp_path / "cache", source)
    assert payload_path.exists()
    source.unlink()
    assert cache.load_cache(tmp_path / "cache", source) is None
