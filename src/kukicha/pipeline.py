from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from .asr import Asr, Segment
from .audio import SAMPLE_RATE
from .errors import EngineBusy, ToolError
from .speaker import SpeakerEncoder, SpeakerStore, Voiceprint, cosine, normalize
from .vad import Vad

log = logging.getLogger(__name__)

# 無音や雑音から Whisper が作りがちな文。字幕付きの動画で学習したことに由来する
DEFAULT_HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "ご視聴いただきありがとうございました",
    "最後までご視聴いただきありがとうございました",
    "最後までご視聴ありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録をお願いします",
    "チャンネル登録お願いいたします",
    "高評価とチャンネル登録お願いします",
    "次回もお楽しみに",
    "それではまた次回",
    "字幕視聴ありがとうございました",
    "thankyouforwatching",
    "thanksforwatching",
)

_PUNCT = re.compile(r"[\s\W_]+")
_ANNOTATION = re.compile(r"^[\[\(（【♪].*[\]\)）】♪]$")
_GAP = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
_NO_SPACE_LANGUAGES = {"ja", "zh", "yue"}
MIN_ENROLL_SECONDS = 10.0
ENROLL_WINDOW_SECONDS = 3.0


def normalize_text(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text)).lower()


def is_noise(seg: Segment, phrases: frozenset[str]) -> bool:
    text = unicodedata.normalize("NFKC", seg.text).strip()
    norm = normalize_text(text)
    if not norm or norm in phrases or _ANNOTATION.match(text):
        return True
    # 同じ語の繰り返し
    if seg.compression_ratio > 2.4:
        return True
    return seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0


@dataclass
class Result:
    text: str
    duration_ms: int
    detail: dict = field(default_factory=dict)


class Transcriber:
    def __init__(
        self,
        vad: Vad,
        asr: Asr,
        encoder: SpeakerEncoder,
        store: SpeakerStore,
        *,
        threshold: float = 0.4,
        min_utterance_ms: int = 400,
        min_embed_ms: int = 1000,
        busy_wait_seconds: float = 10.0,
        extra_hallucinations: tuple[str, ...] | list[str] = (),
    ):
        self.vad = vad
        self.asr = asr
        self.encoder = encoder
        self.store = store
        self.threshold = threshold
        self.min_utterance_ms = min_utterance_ms
        self.min_embed_samples = min_embed_ms * SAMPLE_RATE // 1000
        self.busy_wait_seconds = busy_wait_seconds
        self.phrases = frozenset(normalize_text(p) for p in (*DEFAULT_HALLUCINATIONS, *extra_hallucinations))
        # GPU を使う処理は 1 件ずつ
        self._lock = threading.Lock()

    def voiceprint(self, speaker: str) -> Voiceprint:
        vp = self.store.get(speaker)
        if vp is None:
            raise ToolError(f"speaker '{speaker}' is not enrolled")
        if vp.model != self.encoder.name:
            raise ToolError(f"speaker '{speaker}' was enrolled with {vp.model}; enroll again with {self.encoder.name}")
        return vp

    def transcribe(self, audio: np.ndarray, language: str, speaker: str, initial_prompt: str | None) -> Result:
        vp = self.voiceprint(speaker)
        if not self._lock.acquire(timeout=self.busy_wait_seconds):
            raise EngineBusy()
        try:
            return self._run(audio, language, vp, initial_prompt)
        finally:
            self._lock.release()

    def _run(self, audio: np.ndarray, language: str, vp: Voiceprint, initial_prompt: str | None) -> Result:
        spans = self.vad.speech_spans(audio)
        speech_ms = sum(e - s for s, e in spans) * 1000 // SAMPLE_RATE
        detail = {"audio_ms": len(audio) * 1000 // SAMPLE_RATE, "speech_ms": speech_ms, "segments": 0, "kept": 0}
        if speech_ms < self.min_utterance_ms:
            return Result("", 0, detail)

        parts: list[np.ndarray] = []
        for s, e in spans:
            parts += [audio[s:e], _GAP]
        speech = np.concatenate(parts)

        segments = self.asr.transcribe(speech, language, initial_prompt)
        detail["segments"] = len(segments)
        kept: list[Segment] = []
        scores: list[float] = []
        for seg in segments:
            if is_noise(seg, self.phrases):
                continue
            score = self._score(speech, seg, vp)
            scores.append(round(score, 3))
            if score >= self.threshold:
                kept.append(seg)
        detail["kept"] = len(kept)
        detail["scores"] = scores

        duration_ms = int(round(sum(s.end - s.start for s in kept) * 1000))
        if duration_ms < self.min_utterance_ms:
            return Result("", 0, detail)
        sep = "" if language in _NO_SPACE_LANGUAGES else " "
        text = sep.join(s.text.strip() for s in kept)
        return Result(text, duration_ms, detail)

    def _score(self, speech: np.ndarray, seg: Segment, vp: Voiceprint) -> float:
        start = max(0, int(seg.start * SAMPLE_RATE))
        end = min(len(speech), int(seg.end * SAMPLE_RATE))
        # 短い区間は特徴量が安定しないので、前後に広げる
        if end - start < self.min_embed_samples:
            center = (start + end) // 2
            start = max(0, center - self.min_embed_samples // 2)
            end = min(len(speech), start + self.min_embed_samples)
        if end <= start:
            return -1.0
        try:
            embedding = self.encoder.embed(speech[start:end])
        except Exception as e:
            log.exception("speaker embedding failed")
            raise ToolError(f"speaker embedding failed: {type(e).__name__}") from e
        return cosine(embedding, vp.embedding)


def enroll(vad: Vad, encoder: SpeakerEncoder, recordings: list[np.ndarray], speaker: str) -> tuple[Voiceprint, int, float]:
    """登録用の録音から特徴量を作る。(声の特徴量, 窓の数, 窓どうしのまとまり) を返す。"""
    speech = np.concatenate([audio[s:e] for audio in recordings for s, e in vad.speech_spans(audio)] or [np.zeros(0, np.float32)])
    seconds = len(speech) / SAMPLE_RATE
    if seconds < MIN_ENROLL_SECONDS:
        raise ValueError(f"only {seconds:.1f}s of speech found; at least {MIN_ENROLL_SECONDS:.0f}s is needed")
    window = int(ENROLL_WINDOW_SECONDS * SAMPLE_RATE)
    starts = range(0, len(speech) - window // 2, window)
    embeddings = [normalize(encoder.embed(speech[s:s + window])) for s in starts]
    centroid = normalize(np.mean(embeddings, axis=0))
    consistency = float(np.mean([np.dot(e, centroid) for e in embeddings]))
    return Voiceprint(speaker, encoder.name, centroid.astype(np.float32), seconds), len(embeddings), consistency
