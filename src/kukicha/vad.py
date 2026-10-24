from __future__ import annotations

from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE


class Vad(Protocol):
    def speech_spans(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """発話区間を [start, end) のサンプル位置で返す。"""


class SileroVad:
    """faster-whisper に同梱の Silero VAD（ONNX、CPU）。"""

    def __init__(self, threshold: float = 0.5):
        from faster_whisper.vad import VadOptions

        # 発話ごとに切られた短い音声が来るので、無音の待ちと前後の余白は faster-whisper の既定より短くする
        self._options = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=150,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        )

    def speech_spans(self, audio: np.ndarray) -> list[tuple[int, int]]:
        from faster_whisper.vad import get_speech_timestamps

        chunks = get_speech_timestamps(audio, self._options, sampling_rate=SAMPLE_RATE)
        return [(c["start"], c["end"]) for c in chunks]
