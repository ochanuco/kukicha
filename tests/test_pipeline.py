import numpy as np
import pytest

from kukicha.asr import Segment
from kukicha.audio import SAMPLE_RATE
from kukicha.errors import EngineBusy, ToolError
from kukicha.pipeline import Transcriber, enroll, is_noise, normalize_text
from kukicha.speaker import SpeakerStore, Voiceprint

from conftest import OWNER, OWNER_VEC, FakeAsr, FakeEncoder, FakeVad, make_marked_audio


def transcriber(store, segments, voices, vad=None, **kw):
    return Transcriber(vad or FakeVad(), FakeAsr(segments), FakeEncoder(voices), store, **kw)


def test_keeps_only_the_enrolled_speaker(store):
    t = transcriber(
        store,
        [Segment(0.0, 1.5, "こんにちは"), Segment(1.5, 3.0, "テレビの声"), Segment(3.0, 4.5, "元気です")],
        voices=[(0.0, 1.5, OWNER_VEC), (3.0, 4.5, OWNER_VEC)],
    )
    r = t.transcribe(make_marked_audio(5), "ja", OWNER, None)
    assert r.text == "こんにちは元気です"
    assert r.duration_ms == 3000
    assert r.detail["segments"] == 3 and r.detail["kept"] == 2


def test_joins_with_spaces_for_spaced_languages(store):
    t = transcriber(store, [Segment(0.0, 1.0, " hello"), Segment(1.0, 2.0, " world")], voices=[(0, 3, OWNER_VEC)])
    assert t.transcribe(make_marked_audio(3), "en", OWNER, None).text == "hello world"


def test_other_speaker_only_is_empty(store):
    t = transcriber(store, [Segment(0.0, 2.0, "周りの人の声")], voices=[])
    r = t.transcribe(make_marked_audio(3), "ja", OWNER, None)
    assert (r.text, r.duration_ms) == ("", 0)


def test_short_segment_is_widened_before_embedding(store):
    # 0.2 秒の区間は前後に広げて 1 秒にしてから比べる
    t = transcriber(store, [Segment(2.0, 2.2, "はい")], voices=[(1.5, 2.7, OWNER_VEC)], min_utterance_ms=100)
    r = t.transcribe(make_marked_audio(4), "ja", OWNER, None)
    assert r.text == "はい"
    assert r.duration_ms == 200


def test_hallucinations_are_dropped_without_embedding(store):
    t = transcriber(
        store,
        [Segment(0.0, 2.0, "ご視聴ありがとうございました。"), Segment(2.0, 3.0, "(音楽)"), Segment(3.0, 4.0, "♪～♪")],
        voices=[(0, 5, OWNER_VEC)],
    )
    r = t.transcribe(make_marked_audio(5), "ja", OWNER, None)
    assert r.text == ""
    assert t.encoder.calls == 0


def test_extra_hallucinations_from_config(store):
    t = transcriber(store, [Segment(0.0, 2.0, "定型の文")], voices=[(0, 3, OWNER_VEC)], extra_hallucinations=["定型の文"])
    assert t.transcribe(make_marked_audio(3), "ja", OWNER, None).text == ""


def test_no_speech_skips_whisper(store):
    t = transcriber(store, [Segment(0.0, 1.0, "x")], voices=[], vad=FakeVad(spans=[]))
    r = t.transcribe(np.zeros(SAMPLE_RATE, np.float32), "ja", OWNER, None)
    assert (r.text, r.duration_ms) == ("", 0)
    assert t.asr.calls == []


def test_extremely_short_speech_is_empty(store):
    t = transcriber(store, [Segment(0.0, 0.2, "あ")], voices=[(0, 1, OWNER_VEC)], vad=FakeVad(spans=[(0, 3200)]))
    assert t.transcribe(make_marked_audio(1), "ja", OWNER, None).text == ""
    assert t.asr.calls == []


def test_extremely_short_kept_speech_is_empty(store):
    t = transcriber(store, [Segment(0.0, 0.3, "あ"), Segment(0.3, 2.0, "他の人")], voices=[(0, 0.5, OWNER_VEC)],
                    min_embed_ms=300)
    assert t.transcribe(make_marked_audio(3), "ja", OWNER, None).text == ""


def test_speech_spans_are_concatenated_for_whisper(store):
    vad = FakeVad(spans=[(0, SAMPLE_RATE), (2 * SAMPLE_RATE, 3 * SAMPLE_RATE)])
    t = transcriber(store, [], voices=[], vad=vad)
    t.transcribe(make_marked_audio(4), "ja", OWNER, "固有名詞")
    length, language, prompt = t.asr.calls[0]
    assert length == 2 * SAMPLE_RATE + 2 * (SAMPLE_RATE // 10)
    assert (language, prompt) == ("ja", "固有名詞")


def test_unknown_speaker_is_a_tool_error(store):
    t = transcriber(store, [], voices=[])
    with pytest.raises(ToolError, match="not enrolled"):
        t.transcribe(make_marked_audio(1), "ja", "999", None)


def test_voiceprint_from_another_model_is_a_tool_error(store):
    store.save(Voiceprint("200", "old.onnx", OWNER_VEC, 30.0), windows=10)
    t = transcriber(store, [], voices=[])
    with pytest.raises(ToolError, match="enroll again"):
        t.transcribe(make_marked_audio(1), "ja", "200", None)


def test_busy_while_another_call_runs(store):
    t = transcriber(store, [], voices=[], busy_wait_seconds=0.05)
    t._lock.acquire()
    try:
        with pytest.raises(EngineBusy):
            t.transcribe(make_marked_audio(1), "ja", OWNER, None)
    finally:
        t._lock.release()


def test_encoder_failure_is_a_tool_error(store):
    class Broken(FakeEncoder):
        def embed(self, audio):
            raise RuntimeError("boom")

    t = Transcriber(FakeVad(), FakeAsr([Segment(0.0, 2.0, "こんにちは")]), Broken(), store)
    with pytest.raises(ToolError, match="speaker embedding failed"):
        t.transcribe(make_marked_audio(3), "ja", OWNER, None)


@pytest.mark.parametrize("seg, noisy", [
    (Segment(0, 1, "  "), True),
    (Segment(0, 1, "。"), True),
    (Segment(0, 1, "チャンネル登録よろしくお願いします！"), True),
    (Segment(0, 1, "Thank you for watching."), True),
    (Segment(0, 1, "あああああ", compression_ratio=3.0), True),
    (Segment(0, 1, "えっと", avg_logprob=-1.5, no_speech_prob=0.9), True),
    (Segment(0, 1, "えっと", avg_logprob=-0.3, no_speech_prob=0.9), False),
    (Segment(0, 1, "ありがとうございました"), False),
])
def test_is_noise(seg, noisy):
    from kukicha.pipeline import DEFAULT_HALLUCINATIONS

    assert is_noise(seg, frozenset(normalize_text(p) for p in DEFAULT_HALLUCINATIONS)) is noisy


def test_enroll_averages_windows(tmp_path):
    enc = FakeEncoder(voices=[(0, 1000, OWNER_VEC)])
    vp, windows, consistency = enroll(FakeVad(), enc, [make_marked_audio(12), make_marked_audio(9)], OWNER)
    assert windows == 7
    assert vp.model == "fake.onnx" and vp.seconds == pytest.approx(21.0)
    assert np.allclose(vp.embedding, OWNER_VEC) and consistency == pytest.approx(1.0)
    store = SpeakerStore(tmp_path)
    store.save(vp, windows)
    assert store.list() == [OWNER]
    assert np.allclose(store.get(OWNER).embedding, OWNER_VEC)


def test_enroll_needs_enough_speech():
    with pytest.raises(ValueError, match="at least"):
        enroll(FakeVad(), FakeEncoder(), [make_marked_audio(5)], OWNER)


def test_store_rejects_path_like_ids(tmp_path):
    with pytest.raises(ValueError):
        SpeakerStore(tmp_path).get("../x")
