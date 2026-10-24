from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioFormatError(ValueError):
    pass


def decode_wav(data: bytes) -> np.ndarray:
    """16kHz mono 16bit の WAV を float32 の [-1, 1] にする。それ以外は AudioFormatError。"""
    rate, channels, samples = _read_pcm16(data)
    if rate != SAMPLE_RATE or channels != 1:
        raise AudioFormatError(f"audio must be {SAMPLE_RATE} Hz mono, got {rate} Hz {channels} channel(s)")
    return samples[:, 0]


def load_wav_any(path: Path) -> np.ndarray:
    """登録用。16bit PCM の WAV なら、レートとチャンネル数を問わず 16kHz mono にする。"""
    rate, _, samples = _read_pcm16(path.read_bytes())
    mono = samples.mean(axis=1)
    if rate == SAMPLE_RATE:
        return mono.astype(np.float32)
    # 話者の特徴量を作るだけなので、線形補間で足りる
    n = int(round(len(mono) * SAMPLE_RATE / rate))
    t = np.arange(n) * (rate / SAMPLE_RATE)
    return np.interp(t, np.arange(len(mono)), mono).astype(np.float32)


def _read_pcm16(data: bytes) -> tuple[int, int, np.ndarray]:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                raise AudioFormatError("audio must be 16-bit PCM WAV")
            rate, channels = w.getframerate(), w.getnchannels()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise AudioFormatError(f"audio is not a readable WAV: {e}") from e
    pcm = np.frombuffer(frames, dtype="<i2")
    pcm = pcm[: len(pcm) - len(pcm) % channels].reshape(-1, channels)
    return rate, channels, pcm.astype(np.float32) / 32768.0
