"""faster-whisper wrapper with the family model-download policy.

No silent network calls: a model that is not already cached locally is only
downloaded when the caller passed allow_download=True (the --allow-download
CLI flag). Intended model revisions are pinned in MODEL_MANIFEST.json at
the repo root; faster-whisper resolves models from the Hugging Face hub
into its own local cache, so the pin here is the model identifier plus the
manifest documentation rather than a per-file sha256 (whisper models are
multi-file repos).

Audio is extracted once per source file to 16 kHz mono PCM via ffmpeg. The
same wav feeds both transcription (this module) and waveform-based audio
features (audio.py), so each source is decoded for audio exactly once.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16000


class ModelNotCachedError(Exception):
    """The whisper model is not cached and download was not opted into."""


class AudioExtractError(Exception):
    """ffmpeg could not extract an audio track."""


@dataclass(frozen=True)
class SpeechSpan:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    spans: list[SpeechSpan] = field(default_factory=list)
    language: str | None = None
    language_probability: float = 0.0
    words: list[dict] | None = None

    @property
    def text(self) -> str:
        return " ".join(span.text.strip() for span in self.spans).strip()


def extract_audio(source: Path, out_wav: Path) -> None:
    """Extract 16 kHz mono pcm_s16le audio for transcription and analysis."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not out_wav.exists():
        raise AudioExtractError(
            f"Audio extraction failed for {source.name}: {result.stderr.strip()[:200]}"
        )


class Transcriber:
    """Lazy-loading transcription engine. Import cost is paid on first use."""

    def __init__(
        self,
        model: str = "small",
        allow_download: bool = False,
        download_root: str | None = None,
    ) -> None:
        self.model_name = model
        self.allow_download = allow_download
        self.download_root = download_root
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        kwargs = {"device": "cpu", "compute_type": "int8"}
        if self.download_root:
            kwargs["download_root"] = self.download_root
        try:
            self._model = WhisperModel(
                self.model_name, local_files_only=not self.allow_download, **kwargs
            )
        except Exception as exc:
            if self.allow_download:
                raise
            raise ModelNotCachedError(
                f"Whisper model '{self.model_name}' is not cached locally. "
                "Re-run with --allow-download to fetch it once (this is the "
                "only network access shutter-select ever performs)."
            ) from exc
        return self._model

    def transcribe(self, wav_path: Path, words: bool = False) -> TranscriptResult:
        """Transcribe one extracted wav.

        Word timestamps are always computed internally: whisper's raw
        segment ends run long across silence (a segment before a 5 second
        pause reports its end at the next segment's start), which would
        glue separate takes together. Snapping each span to its first and
        last word keeps take boundaries honest. The words flag only
        controls whether the word list is stored in the result.
        """
        model = self._load()
        segments, info = model.transcribe(
            str(wav_path), vad_filter=True, word_timestamps=True
        )
        result = TranscriptResult(
            language=info.language, language_probability=float(info.language_probability)
        )
        word_rows: list[dict] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            start = float(seg.start)
            end = float(seg.end)
            if seg.words:
                start = float(seg.words[0].start)
                end = float(seg.words[-1].end)
            result.spans.append(SpeechSpan(start=start, end=end, text=text))
            if words and seg.words:
                word_rows.extend(
                    {
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "word": w.word,
                    }
                    for w in seg.words
                )
        if words:
            result.words = word_rows
        return result
