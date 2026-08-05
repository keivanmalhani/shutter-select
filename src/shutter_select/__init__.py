"""shutter-select: local-first video culling into an editor-ready selects timeline.

Point it at a folder of raw footage. It transcribes speech, detects scenes,
flags silence and bad audio, scores visual quality, and emits selects
timelines (OTIO, FCP7 XML, EDL), SRT transcripts, and per-file analysis
JSON. Everything runs locally; source files are opened read-only.

Spec: _hq/specs/video-selects-spec.md (idea 2). The per-file analysis JSON
written by the cache module is the reuse contract consumed by shutter-clip
(idea 5) via subprocess.
"""

__version__ = "0.1.0"
