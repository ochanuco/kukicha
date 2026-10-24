import json

import numpy as np
import pytest

from kukicha.asr import WhisperAsr
from kukicha.audio import SAMPLE_RATE, AudioFormatError, decode_wav, load_wav_any
from kukicha.auth import BearerAuthenticator, parse_tokens
from kukicha.config import Config, load_config
from kukicha.errors import ToolError
from kukicha.vad import SileroVad

from conftest import wav_bytes


def test_decode_wav():
    audio = decode_wav(wav_bytes(np.full(1600, 0.5, np.float32)))
    assert audio.dtype == np.float32 and len(audio) == 1600
    assert audio[0] == pytest.approx(0.5, abs=1e-3)


def test_decode_wav_rejects_stereo():
    with pytest.raises(AudioFormatError, match="mono"):
        decode_wav(wav_bytes(np.zeros(3200, np.float32), channels=2))


def test_load_wav_any_resamples_and_downmixes(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(wav_bytes(np.zeros(48000 * 2, np.float32), rate=48000, channels=2))
    assert len(load_wav_any(path)) == SAMPLE_RATE


def test_parse_tokens():
    assert parse_tokens('{"dev": "a", "prod": "b"}') == {"dev": "a", "prod": "b"}
    for bad in ('[]', '{"dev": ""}', '{"dev": 1}', '{"dev": "a", "prod": "a"}', "{"):
        with pytest.raises(ValueError):
            parse_tokens(bad)


def test_authenticator_maps_token_to_caller():
    auth = BearerAuthenticator({"dev": "a", "prod": "b"})
    assert auth.authenticate("Bearer b") == "prod"
    assert auth.authenticate("Bearer  a ") == "dev"
    assert auth.authenticate("Bearer c") is None
    assert auth.authenticate(None) is None


def test_config(tmp_path):
    assert load_config(tmp_path) == Config()
    (tmp_path / "config.json").write_text(json.dumps({"whisperModel": "small", "port": 9000}), encoding="utf-8")
    cfg = load_config(tmp_path)
    assert (cfg.whisperModel, cfg.port, cfg.computeType) == ("small", 9000, "auto")
    (tmp_path / "config.json").write_text(json.dumps({"wisperModel": "small"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(tmp_path)


def test_silero_vad_finds_no_speech_in_silence_and_noise():
    vad = SileroVad()
    assert vad.speech_spans(np.zeros(2 * SAMPLE_RATE, np.float32)) == []
    noise = np.random.default_rng(0).normal(0, 0.01, 2 * SAMPLE_RATE).astype(np.float32)
    assert vad.speech_spans(noise) == []


class _Model:
    def __init__(self, error=None):
        self.error = error

    def transcribe(self, audio, **kw):
        if self.error:
            raise self.error
        return iter([]), None


def _asr(monkeypatch, model, **kw):
    asr = WhisperAsr("tiny", "/nonexistent", device="cpu", **kw)
    monkeypatch.setattr(asr, "_load", lambda: model)
    return asr


def test_asr_loads_on_demand_and_unloads_when_idle(monkeypatch):
    asr = _asr(monkeypatch, _Model(), idle_unload_seconds=0)
    assert asr.state == "idle"
    assert asr.transcribe(np.zeros(16000, np.float32), "ja", None) == []
    assert asr.state == "ready"
    assert asr.unload_if_idle() is True
    assert asr.state == "idle" and asr._model is None


def test_asr_keeps_model_while_recently_used(monkeypatch):
    asr = _asr(monkeypatch, _Model(), idle_unload_seconds=3600)
    asr.transcribe(np.zeros(16000, np.float32), "ja", None)
    assert asr.unload_if_idle() is False and asr.state == "ready"


def test_asr_out_of_memory_is_a_tool_error_and_frees_the_model(monkeypatch):
    asr = _asr(monkeypatch, _Model(RuntimeError("CUDA failed with error out of memory")))
    with pytest.raises(ToolError, match="out of GPU memory"):
        asr.transcribe(np.zeros(16000, np.float32), "ja", None)
    assert asr._model is None and asr.state == "idle"


def test_asr_other_failure_marks_unhealthy(monkeypatch):
    asr = _asr(monkeypatch, _Model(ValueError("bad")))
    with pytest.raises(ToolError, match="transcription failed"):
        asr.transcribe(np.zeros(16000, np.float32), "ja", None)
    assert asr.state == "unhealthy"


def test_asr_load_failure_is_a_tool_error(monkeypatch):
    asr = WhisperAsr("tiny", "/nonexistent", device="cpu")

    def fail():
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(asr, "_load", fail)
    with pytest.raises(ToolError, match="out of GPU memory"):
        asr.transcribe(np.zeros(16000, np.float32), "ja", None)
