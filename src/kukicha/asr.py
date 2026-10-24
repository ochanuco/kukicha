from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .errors import ToolError

log = logging.getLogger(__name__)

_OOM_MARKERS = ("out of memory", "cublas_status_alloc_failed", "cudaerrormemoryallocation")


@dataclass(frozen=True)
class Segment:
    """渡した音声の中での位置（秒）と文字。"""

    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0


class Asr(Protocol):
    @property
    def state(self) -> str: ...

    def transcribe(self, audio: np.ndarray, language: str, initial_prompt: str | None) -> list[Segment]: ...

    def unload_if_idle(self) -> bool: ...


class WhisperAsr:
    """faster-whisper のモデルを使う時に読み込み、使わない間が続けば捨てて VRAM を返す。"""

    def __init__(
        self,
        model: str,
        download_root: str,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        cpu_threads: int = 4,
        idle_unload_seconds: float = 300,
    ):
        self._model_name = model
        self._download_root = download_root
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._cpu_threads = cpu_threads
        self._idle_unload_seconds = idle_unload_seconds
        self._lock = threading.Lock()
        self._model = None
        self._last_used = 0.0
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    def _resolve(self) -> tuple[str, str]:
        from .cuda import cuda_available

        device = self._device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        compute_type = self._compute_type
        if compute_type == "auto":
            compute_type = "int8_float16" if device == "cuda" else "int8"
        return device, compute_type

    def _load(self):
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve()
        self._state = "loading"
        started = time.monotonic()
        model = WhisperModel(
            self._model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=self._cpu_threads,
            download_root=self._download_root,
        )
        log.info("loaded whisper model=%s device=%s compute_type=%s ms=%d",
                 self._model_name, device, compute_type, (time.monotonic() - started) * 1000)
        return model

    def _drop(self) -> None:
        self._model = None
        # ctranslate2 はモデルのオブジェクトが消えた時点で VRAM を返す
        gc.collect()

    def transcribe(self, audio: np.ndarray, language: str, initial_prompt: str | None) -> list[Segment]:
        with self._lock:
            try:
                if self._model is None:
                    self._model = self._load()
                segments, _ = self._model.transcribe(
                    audio,
                    language=language,
                    initial_prompt=initial_prompt,
                    beam_size=self._beam_size,
                    vad_filter=False,
                    # 前の区間の文字に引きずられて同じ定型文を繰り返すのを防ぐ
                    condition_on_previous_text=False,
                )
                # segments は generator で、ここで回して初めて推論が走る
                result = [
                    Segment(s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob, s.compression_ratio)
                    for s in segments
                ]
            except Exception as e:
                self._drop()
                if any(m in str(e).lower() for m in _OOM_MARKERS):
                    self._state = "idle"
                    log.warning("whisper ran out of memory: %s", e)
                    raise ToolError("out of GPU memory; try again later") from e
                self._state = "unhealthy"
                log.exception("whisper failed")
                raise ToolError(f"transcription failed: {type(e).__name__}") from e
            self._last_used = time.monotonic()
            self._state = "ready"
            return result

    def unload_if_idle(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._model is None or time.monotonic() - self._last_used < self._idle_unload_seconds:
                return False
            self._drop()
            self._state = "idle"
            log.info("unloaded whisper model after %ds idle", self._idle_unload_seconds)
            return True
        finally:
            self._lock.release()
