"""Tests for doctor, written as machines that are broken in one way each.

The point of doctor is that it is right about someone else's computer, so
every test builds the specific breakage rather than asserting on whatever
the test runner happens to be sitting on. The environment is injected
(which, access, version_of, can_import, whisper_cache_state,
python_version), so these run identically on a laptop with everything
installed and in a container missing ffmpeg, or running as root where
chmod-based permission tests would otherwise silently no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

from shutter_select import doctor
from shutter_select.cli import main


def ctx(tmp_path: Path, **overrides) -> doctor.Context:
    """A machine where everything works, then broken on purpose per test."""
    defaults = dict(
        root=None,
        model="small",
        which=lambda name: f"/usr/bin/{name}",
        access=lambda path, mode: True,
        version_of=lambda name: (0, f"{name} 1.2.3"),
        can_import=lambda name: (True, f"{name} 1.0.0"),
        whisper_cache_state=lambda model: (True, f"/cache/models--Systran--faster-whisper-{model}"),
        python_version=(3, 12),
    )
    defaults.update(overrides)
    return doctor.Context(**defaults)


def deny(*paths: Path):
    """An access() that refuses exactly these paths and allows everything else."""
    blocked = {str(p) for p in paths}
    return lambda path, mode: str(path) not in blocked


def by_name(checks, name) -> doctor.Check:
    return next(c for c in checks if c.name == name)


# ------------------------------------------------------------------ python


def test_old_python_is_fatal_and_says_the_version(tmp_path):
    check = doctor.check_python(ctx(tmp_path, python_version=(3, 9)))
    assert check.status == doctor.FAIL
    assert "3.9" in check.detail and "3.11" in check.detail
    assert check.fix


def test_current_python_passes(tmp_path):
    assert doctor.check_python(ctx(tmp_path)).status == doctor.OK


# -------------------------------------------------------------- ffmpeg/ffprobe


def test_missing_ffmpeg_is_fatal_since_every_command_needs_it(tmp_path):
    """Unlike shutter-cull's exiftool, ffmpeg/ffprobe are needed by every
    command including a bare scan, so their absence is blocking, not a
    warning."""
    checks = doctor.check_ffmpeg(ctx(tmp_path, which=lambda n: None))
    assert {c.name for c in checks} == {"ffmpeg", "ffprobe"}
    assert all(c.status == doctor.FAIL for c in checks)
    assert all("install" in c.fix.lower() for c in checks)


def test_one_binary_missing_is_still_reported_for_both(tmp_path):
    checks = doctor.check_ffmpeg(
        ctx(tmp_path, which=lambda n: None if n == "ffprobe" else f"/usr/bin/{n}")
    )
    ffmpeg_check = by_name(checks, "ffmpeg")
    ffprobe_check = by_name(checks, "ffprobe")
    assert ffmpeg_check.status == doctor.OK
    assert ffprobe_check.status == doctor.FAIL


def test_ffmpeg_on_path_but_not_runnable_is_fatal(tmp_path):
    """A broken install passes `which` and fails on use, a different fix
    from not being installed at all."""
    checks = doctor.check_ffmpeg(
        ctx(tmp_path, version_of=lambda n: (127, "libx264.so: cannot open shared object file"))
    )
    assert all(c.status == doctor.FAIL for c in checks)
    assert all("will not run" in c.detail for c in checks)


def test_healthy_ffmpeg_reports_its_version_line(tmp_path):
    checks = doctor.check_ffmpeg(ctx(tmp_path))
    assert all(c.status == doctor.OK for c in checks)
    assert all("1.2.3" in c.detail for c in checks)


# -------------------------------------------------------------- imports


def test_missing_scenedetect_is_fatal_since_it_is_a_hard_dependency(tmp_path):
    check = doctor.check_import(
        ctx(tmp_path, can_import=lambda n: (False, "No module named 'scenedetect'")),
        "scenedetect", "scenedetect", "b-roll segmentation (PySceneDetect)",
    )
    assert check.status == doctor.FAIL
    assert "scenedetect" in check.detail
    assert "pip install" in check.fix


def test_importable_scenedetect_passes(tmp_path):
    check = doctor.check_import(
        ctx(tmp_path), "scenedetect", "scenedetect", "b-roll segmentation (PySceneDetect)"
    )
    assert check.status == doctor.OK


# ---------------------------------------------------------- whisper model cache


def test_uncached_whisper_model_warns_and_points_at_allow_download(tmp_path):
    check = doctor.check_whisper_cache(
        ctx(tmp_path, model="small", whisper_cache_state=lambda m: (False, "/cache/nope"))
    )
    assert check.status == doctor.WARN
    assert "not cached" in check.detail.lower()
    assert "--allow-download" in check.fix
    assert "small" in check.fix


def test_cached_whisper_model_passes(tmp_path):
    check = doctor.check_whisper_cache(
        ctx(tmp_path, model="medium", whisper_cache_state=lambda m: (True, "/cache/hit"))
    )
    assert check.status == doctor.OK
    assert "/cache/hit" in check.detail
    assert "medium" in check.name


def test_real_whisper_cache_state_reports_not_cached_for_a_fresh_hf_home(tmp_path, monkeypatch):
    """No network call, no faster-whisper import: only a filesystem look at
    where huggingface_hub says its cache lives."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty-hf-cache"))
    cached, where = doctor._real_whisper_cache_state("small")
    assert cached is False
    assert "models--Systran--faster-whisper-small" in where


def test_real_whisper_cache_state_reports_cached_when_a_snapshot_exists(tmp_path, monkeypatch):
    import huggingface_hub.constants as hf_constants

    hf_cache = tmp_path / "hf-cache"
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hf_cache))
    snapshot_dir = hf_cache / "models--Systran--faster-whisper-small" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_bytes(b"fake")

    cached, where = doctor._real_whisper_cache_state("small")
    assert cached is True
    assert "models--Systran--faster-whisper-small" in where


def test_real_whisper_cache_state_rejects_unknown_model_size():
    cached, detail = doctor._real_whisper_cache_state("gigantic")
    assert cached is False
    assert "Unknown" in detail


# -------------------------------------------------------------- target root


def test_no_root_given_is_fine_and_says_so(tmp_path):
    check = doctor.check_root(ctx(tmp_path, root=None))
    assert check.status == doctor.OK
    assert "No root given" in check.detail


def test_missing_root_is_fatal(tmp_path):
    check = doctor.check_root(ctx(tmp_path, root=tmp_path / "nope"))
    assert check.status == doctor.FAIL
    assert "does not exist" in check.detail


def test_root_that_is_a_file_is_fatal(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x")
    check = doctor.check_root(ctx(tmp_path, root=target))
    assert check.status == doctor.FAIL
    assert "not a directory" in check.detail


def test_unreadable_root_is_fatal_and_names_the_fix(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    check = doctor.check_root(ctx(tmp_path, root=locked, access=deny(locked)))
    assert check.status == doctor.FAIL
    assert "chmod" in check.fix


def test_readable_root_passes(tmp_path):
    root = tmp_path / "footage"
    root.mkdir()
    check = doctor.check_root(ctx(tmp_path, root=root))
    assert check.status == doctor.OK
    assert "readable" in check.detail


# ---------------------------------------------------------------- rendering


def test_every_non_passing_check_carries_a_fix(tmp_path):
    """A check that cannot tell the user what to type is not finished."""
    broken = ctx(
        tmp_path,
        root=tmp_path / "nope",
        python_version=(3, 9),
        which=lambda n: None,
        can_import=lambda n: (False, "boom"),
        whisper_cache_state=lambda m: (False, "/cache/nope"),
    )
    for check in doctor.run_checks(broken):
        if check.status != doctor.OK:
            assert check.fix, f"{check.name} has no fix"


def test_run_checks_reports_every_problem_not_just_the_first(tmp_path):
    broken = ctx(tmp_path, root=tmp_path / "nope", python_version=(3, 9), which=lambda n: None)
    checks = doctor.run_checks(broken)
    assert sum(1 for c in checks if c.status == doctor.FAIL) >= 3


def test_render_is_readable_and_counts_the_blockers(tmp_path):
    checks = doctor.run_checks(ctx(tmp_path, root=tmp_path / "nope"))
    out = doctor.render(checks)
    assert "[FAIL]" in out
    assert "blocking problem" in out
    assert "->" in out  # the fix line is shown, not just the failure


def test_the_summary_never_contradicts_the_exit_code(tmp_path):
    for broken in [
        dict(whisper_cache_state=lambda m: (False, "/cache/nope")),  # only a warning
        dict(root=tmp_path / "nope"),  # a fatal check
        dict(),  # healthy
    ]:
        checks = doctor.run_checks(ctx(tmp_path, **broken))
        workable, summary = doctor.verdict(checks)
        assert ("will work" in summary) == workable, summary


def test_render_says_so_when_everything_is_fine(tmp_path):
    out = doctor.render(doctor.run_checks(ctx(tmp_path)))
    assert "Everything checks out" in out
    assert "[FAIL]" not in out
    assert "[warn]" not in out


def test_render_output_is_ascii(tmp_path):
    out = doctor.render(doctor.run_checks(ctx(tmp_path, which=lambda n: None)))
    assert out.encode("ascii")


# ------------------------------------------------------------------- cli


def test_cli_doctor_exits_one_when_the_root_is_missing(tmp_path, capsys):
    code = main(["doctor", str(tmp_path / "nope")])
    assert code == 1
    assert "does not exist" in capsys.readouterr().out


def test_cli_doctor_reports_every_check_not_just_the_first_failure(tmp_path, capsys):
    """Three round trips to fix three things is the support experience this
    command exists to avoid: every check name appears even though the root
    check fails immediately."""
    code = main(["doctor", str(tmp_path / "nope")])
    out = capsys.readouterr().out
    assert code == 1
    for name in ("python", "ffmpeg", "ffprobe", "scenedetect", "whisper model", "target root"):
        assert name in out
    assert out.count("FAIL") >= 1


def test_cli_doctor_bypasses_require_binaries(tmp_path, capsys, monkeypatch):
    """main() normally calls require_binaries() before dispatch and turns a
    missing-ffmpeg ProbeError into one line on stderr, exit 1, before any
    command body runs at all. doctor's whole job is to report that same
    problem in one pass alongside everything else, so it must bypass that
    early call rather than dying on it before doctor.render() ever prints."""
    from shutter_select import cli as cli_module
    from shutter_select.probe import ProbeError

    def boom():
        raise ProbeError("ffmpeg was not found on PATH")

    monkeypatch.setattr(cli_module, "require_binaries", boom)
    code = main(["doctor", str(tmp_path)])
    out = capsys.readouterr().out
    # If require_binaries() had run, main() would have caught the ProbeError
    # itself and returned before doctor.render() ever printed anything.
    assert "python" in out
    assert "target root" in out
    assert code in (0, 1)


def test_cli_doctor_json_mode_emits_one_object_per_check(tmp_path, capsys):
    code = main(["doctor", str(tmp_path), "--json"])
    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line.strip()]
    objects = [json.loads(line) for line in lines]

    checks = [o for o in objects if o["event"] == "doctor_check"]
    verdicts = [o for o in objects if o["event"] == "doctor_verdict"]
    assert checks and all({"check", "status", "detail"} <= set(o) for o in checks)
    assert len(verdicts) == 1, "a probe needs exactly one line to alert on"
    assert {"workable", "summary"} <= set(verdicts[0])
    assert objects[-1]["event"] == "doctor_verdict"
    assert code in (0, 1)


def test_cli_doctor_runs_with_no_root_at_all(capsys):
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "tools and this machine only" in out
    assert "python" in out
    assert code in (0, 1)


def test_cli_doctor_accepts_a_model_flag(tmp_path, capsys):
    code = main(["doctor", str(tmp_path), "--model", "medium", "--json"])
    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line.strip()]
    objects = [json.loads(line) for line in lines]
    whisper_checks = [o for o in objects if o.get("check", "").startswith("whisper model")]
    assert whisper_checks
    assert "medium" in whisper_checks[0]["check"]
    assert code in (0, 1)


def test_cli_doctor_exit_code_matches_verdict(tmp_path, capsys):
    root = tmp_path / "footage"
    root.mkdir()
    code = main(["doctor", str(root), "--json"])
    out = capsys.readouterr().out
    verdict_line = json.loads(out.strip().splitlines()[-1])
    assert (code == 0) == verdict_line["workable"]


# ------------------------------------------------- the real version probe


def test_version_probe_accepts_a_cli_that_has_no_version_flag(monkeypatch):
    calls = []

    class Result:
        def __init__(self, code, out):
            self.returncode, self.stdout, self.stderr = code, out, ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])
        if cmd[1] == "--version":
            return Result(2, "usage: shutter-select [-h] {scan,analyze,emit,run} ...")
        return Result(0, "usage: shutter-select [-h] {scan,analyze,emit,run} ...")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    code, line = doctor._real_version_of("some-cli-with-required-subcommand")
    assert code == 0
    assert "does not report a version" in line
    assert calls == ["--version", "--help"]


def test_version_probe_reports_a_binary_that_cannot_execute_at_all(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("libx264.so: cannot open shared object file")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    code, line = doctor._real_version_of("ffmpeg")
    assert code == 127
    assert "libx264" in line


def test_version_probe_prefers_a_real_version_line(monkeypatch):
    class Result:
        def __init__(self, code, out):
            self.returncode, self.stdout, self.stderr = code, out, ""

    monkeypatch.setattr(
        doctor.subprocess, "run", lambda cmd, **kw: Result(0, "ffmpeg version 6.1.1")
    )
    assert doctor._real_version_of("ffmpeg") == (0, "ffmpeg version 6.1.1")


# ------------------------------------------------------------- real imports


def test_real_can_import_reports_a_missing_module():
    ok, detail = doctor._real_can_import("definitely_not_a_real_module_xyz")
    assert ok is False
    assert detail


def test_real_can_import_reports_an_installed_module():
    ok, detail = doctor._real_can_import("json")
    assert ok is True
    assert "json" in detail
