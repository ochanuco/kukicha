from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE

SPEAKER_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class SpeakerEncoder(Protocol):
    @property
    def name(self) -> str: ...

    def embed(self, audio: np.ndarray) -> np.ndarray: ...


class SherpaSpeakerEncoder:
    """sherpa-onnx の話者特徴量（CAM++、ONNX、CPU）。Whisper と VRAM を取り合わないよう CPU で動かす。"""

    def __init__(self, model_path: Path, num_threads: int = 2):
        if not model_path.is_file():
            raise FileNotFoundError(f"speaker model not found: {model_path} (run `kukicha fetch-models`)")
        self._model_path = model_path
        self._num_threads = num_threads
        self._extractor = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_path.name

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import sherpa_onnx

        with self._lock:
            if self._extractor is None:
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(self._model_path), num_threads=self._num_threads, provider="cpu")
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            stream = self._extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
            stream.input_finished()
            return np.asarray(self._extractor.compute(stream), dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


@dataclass(frozen=True)
class Voiceprint:
    speaker: str
    model: str
    embedding: np.ndarray
    seconds: float


class SpeakerStore:
    """speaker id ごとの声の特徴量を <data>/speakers/<id>.json に置く。"""

    def __init__(self, root: Path):
        self._root = root

    def _path(self, speaker: str) -> Path:
        if not SPEAKER_ID.match(speaker):
            raise ValueError(f"invalid speaker id: {speaker!r}")
        return self._root / f"{speaker}.json"

    def get(self, speaker: str) -> Voiceprint | None:
        path = self._path(speaker)
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Voiceprint(raw["speaker"], raw["model"], np.asarray(raw["embedding"], dtype=np.float32),
                          float(raw["seconds"]))

    def save(self, print_: Voiceprint, windows: int) -> Path:
        path = self._path(print_.speaker)
        self._root.mkdir(parents=True, exist_ok=True)
        body = {
            "speaker": print_.speaker,
            "model": print_.model,
            "seconds": round(print_.seconds, 2),
            "windows": windows,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "embedding": [round(float(x), 6) for x in print_.embedding],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def remove(self, speaker: str) -> bool:
        path = self._path(speaker)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def list(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(p.stem for p in self._root.glob("*.json"))
