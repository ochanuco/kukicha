from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from kukicha.asr import Segment
from kukicha.audio import SAMPLE_RATE
from kukicha.speaker import SpeakerStore, Voiceprint

OWNER = "100000000000000001"
OWNER_VEC = np.array([1.0, 0.0, 0.0], dtype=np.float32)
OTHER_VEC = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_b64(seconds: float = 2.0) -> str:
    return base64.b64encode(wav_bytes(np.zeros(int(seconds * SAMPLE_RATE), np.float32))).decode()


class FakeVad:
    """音声全体を 1 つの発話区間とみなす。spans を渡せば、その区間を返す。"""

    def __init__(self, spans=None):
        self.spans = spans

    def speech_spans(self, audio):
        return self.spans if self.spans is not None else [(0, len(audio))]


class FakeAsr:
    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self.calls = []
        self.state = "idle"

    def transcribe(self, audio, language, initial_prompt):
        self.calls.append((len(audio), language, initial_prompt))
        self.state = "ready"
        return self.segments

    def unload_if_idle(self):
        return False


class FakeEncoder:
    """音声の中身の代わりに、区間の位置から特徴量を決める。voices は (開始秒, 終了秒, 特徴量)。"""

    name = "fake.onnx"

    def __init__(self, voices=()):
        self.voices = list(voices)
        self.calls = 0

    def embed(self, audio):
        self.calls += 1
        # 区間の位置は音声の値に埋めてある（make_marked_audio）
        t = float(np.median(audio)) * 100
        for start, end, vec in self.voices:
            if start <= t < end:
                return vec
        return OTHER_VEC


def make_marked_audio(seconds: float) -> np.ndarray:
    """各サンプルの値を「その位置の秒 / 100」にして、FakeEncoder が区間の位置を読めるようにする。"""
    return (np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE) / 100


@pytest.fixture
def store(tmp_path):
    s = SpeakerStore(tmp_path / "speakers")
    s.save(Voiceprint(OWNER, FakeEncoder.name, OWNER_VEC, 30.0), windows=10)
    return s
