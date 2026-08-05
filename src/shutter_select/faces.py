"""Opt-in face presence via YuNet, ported from shutter-cull's model policy.

Off by default (--faces enables it). Absent faces are neutral, never a
penalty: face_ratio feeds the speech-class composite only when the flag is
on, and scoring.py redistributes the faces weight when it is off or when a
segment sampled zero frames.

Model weights policy, family baseline: never vendored, never fetched
silently. ensure_model() only opens the network when allow_download=True,
and only caches bytes whose sha256 matches the pin below (which matches
MODEL_MANIFEST.json at the repo root and the pin already verified in the
shutter-cull build).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str
    license_note: str


MODEL_MANIFEST: dict[str, ModelSpec] = {
    "face_detector": ModelSpec(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license_note="MIT, Shiqi Yu, github.com/opencv/opencv_zoo",
    ),
}


class ModelNotAvailable(Exception):
    """Model not cached locally and download was not opted into."""


class ModelVerificationError(Exception):
    """Downloaded bytes did not match the pinned sha256. Never cached."""


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "shutter-select" / "models"


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(
    name: str, *, allow_download: bool = False, model_dir: Path | None = None
) -> Path:
    spec = MODEL_MANIFEST[name]
    directory = model_dir or cache_dir()
    target = directory / spec.filename

    if target.exists():
        if _sha256_of_file(target) == spec.sha256:
            return target
        target.unlink()

    if not allow_download:
        raise ModelNotAvailable(
            f"Face model '{spec.filename}' is not cached. Re-run with "
            "--allow-download to fetch and checksum-verify it once."
        )

    directory.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(spec.url, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if _sha256_of_file(tmp) != spec.sha256:
        tmp.unlink()
        raise ModelVerificationError(
            f"Downloaded {spec.filename} failed sha256 verification; refusing to cache it."
        )
    tmp.replace(target)
    return target


class FaceCounter:
    """Wraps cv2.FaceDetectorYN for frame batches from visual sampling."""

    def __init__(self, model_path: Path, score_threshold: float = 0.7) -> None:
        self._detector = cv2.FaceDetectorYN_create(
            str(model_path), "", (0, 0), score_threshold, 0.3, 5000
        )

    def count(self, frame_bgr: np.ndarray) -> int:
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame_bgr)
        return 0 if faces is None else len(faces)

    def face_ratio(self, frames: list[np.ndarray]) -> float | None:
        """Fraction of frames containing at least one face. None if no frames."""
        if not frames:
            return None
        hits = sum(1 for frame in frames if self.count(frame) > 0)
        return hits / len(frames)
