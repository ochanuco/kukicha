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


@dataclass(frozen=True)
class Runtime:
    device: str
    compute_type: str
    model: str
    beam_size: int
    idle_unload_seconds: float


def resolve_runtime(
    device: str = "auto",
    compute_type: str = "auto",
    model: str = "auto",
    beam_size: int = 0,
    idle_unload_seconds: float = -1,
) -> Runtime:
    """auto（beam_size は 0 以下、idle_unload_seconds は負）を、使う device に合わせた値にする。

    CPU では large-v3-turbo が発話ごとに数秒かかるので、small とビーム幅 1 にする。
    モデルを外すのは VRAM を他の GPU の仕事に返すためなので、CPU では外さない（0）。
    """
    if device == "auto":
        from .cuda import cuda_available

        device = "cuda" if cuda_available() else "cpu"
    cuda = device == "cuda"
    if compute_type == "auto":
        compute_type = "int8_float16" if cuda else "int8"
    if model == "auto":
        model = "large-v3-turbo" if cuda else "small"
    if beam_size <= 0:
        beam_size = 5 if cuda else 1
    if idle_unload_seconds < 0:
        idle_unload_seconds = 300 if cuda else 0
    return Runtime(device, compute_type, model, beam_size, idle_unload_seconds)


class WhisperAsr:
    """faster-whisper のモデルを使う時に読み込み、使わない間が続けば捨てて VRAM を返す（0 なら捨てない）。"""

    def __init__(
        self,
        model: str,
        download_root: str,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 0,
        cpu_threads: int = 4,
        idle_unload_seconds: float = -1,
    ):
        self._model_name = model
        self._download_root = download_root
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._cpu_threads = cpu_threads
        self._idle_unload_seconds = idle_unload_seconds
        self._runtime: Runtime | None = None
        self._lock = threading.Lock()
        self._model = None
        self._last_used = 0.0
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    def _resolve(self) -> Runtime:
        if self._runtime is None:
            self._runtime = resolve_runtime(
                self._device, self._compute_type, self._model_name, self._beam_size, self._idle_unload_seconds
            )
        return self._runtime

    def _load(self):
        from faster_whisper import WhisperModel

        rt = self._resolve()
        self._state = "loading"
        started = time.monotonic()
        model = WhisperModel(
            rt.model,
            device=rt.device,
            compute_type=rt.compute_type,
            cpu_threads=self._cpu_threads,
            download_root=self._download_root,
        )
        log.info("loaded whisper model=%s device=%s compute_type=%s beam_size=%d ms=%d",
                 rt.model, rt.device, rt.compute_type, rt.beam_size, (time.monotonic() - started) * 1000)
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
                    beam_size=self._resolve().beam_size,
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
            if self._model is None:
                return False
            idle = self._resolve().idle_unload_seconds
            if idle == 0 or time.monotonic() - self._last_used < idle:
                return False
            self._drop()
            self._state = "idle"
            log.info("unloaded whisper model after %ds idle", idle)
            return True
        finally:
            self._lock.release()
