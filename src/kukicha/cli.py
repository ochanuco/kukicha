from __future__ import annotations

import argparse
import hashlib
import logging
import logging.handlers
import shutil
import sys
import urllib.request
from pathlib import Path

from .config import (
    SPEAKER_MODEL_SHA256,
    SPEAKER_MODEL_URL,
    Config,
    default_config_dir,
    load_config,
)

log = logging.getLogger("kukicha")


def _setup_logging(config: Config, to_file: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if to_file:
        config.logs_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.TimedRotatingFileHandler(
            config.logs_path / "kukicha.log", when="midnight", backupCount=14, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=handlers)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def _encoder(config: Config):
    from .speaker import SherpaSpeakerEncoder

    return SherpaSpeakerEncoder(config.speaker_model_path)


def cmd_serve(config: Config, config_dir: Path) -> int:
    import uvicorn

    from .app import create_app, release_version
    from .asr import WhisperAsr
    from .auth import load_tokens, BearerAuthenticator
    from .cuda import add_nvidia_dll_dirs
    from .mcp import McpServer
    from .pipeline import Transcriber
    from .speaker import SpeakerStore
    from .vad import SileroVad

    _setup_logging(config, to_file=True)
    try:
        auth = BearerAuthenticator(load_tokens(config_dir / "tokens.json"))
        if len(auth) == 0:
            raise ValueError("tokens.json has no callers")
        add_nvidia_dll_dirs()
        asr = WhisperAsr(
            config.whisperModel,
            str(config.whisper_download_root),
            device=config.device,
            compute_type=config.computeType,
            beam_size=config.beamSize,
            cpu_threads=config.cpuThreads,
            idle_unload_seconds=config.idleUnloadSeconds,
        )
        transcriber = Transcriber(
            SileroVad(config.vadThreshold),
            asr,
            _encoder(config),
            SpeakerStore(config.speakers_path),
            threshold=config.speakerThreshold,
            pass_unenrolled=config.passUnenrolled,
            min_utterance_ms=config.minUtteranceMs,
            min_embed_ms=config.minEmbedMs,
            busy_wait_seconds=config.busyWaitSeconds,
            extra_hallucinations=config.extraHallucinations,
        )
    except (OSError, ValueError) as e:
        log.error("startup failed: %s", e)
        return 2
    version = release_version()
    app = create_app(McpServer(transcriber, version, config.maxAudioSeconds), auth, asr, version, config.maxAudioSeconds)
    log.info("kukicha %s listening on %s:%d callers=%d", version, config.host, config.port, len(auth))
    uvicorn.run(app, host=config.host, port=config.port, log_config=None, access_log=False)
    return 0


def cmd_enroll(config: Config, speaker: str, wavs: list[Path]) -> int:
    from .audio import load_wav_any
    from .pipeline import enroll
    from .speaker import SPEAKER_ID, SpeakerStore
    from .vad import SileroVad

    if not SPEAKER_ID.match(speaker):
        print(f"speaker id must match {SPEAKER_ID.pattern}", file=sys.stderr)
        return 1
    try:
        recordings = [load_wav_any(p) for p in wavs]
        vp, windows, consistency = enroll(SileroVad(config.vadThreshold), _encoder(config), recordings, speaker)
    except (OSError, ValueError) as e:
        print(f"enroll failed: {e}", file=sys.stderr)
        return 1
    path = SpeakerStore(config.speakers_path).save(vp, windows)
    print(f"enrolled {speaker}: {vp.seconds:.1f}s of speech, {windows} windows, consistency {consistency:.3f} -> {path}")
    return 0


def cmd_speakers(config: Config) -> int:
    from .speaker import SpeakerStore

    for speaker in SpeakerStore(config.speakers_path).list():
        print(speaker)
    return 0


def cmd_unenroll(config: Config, speaker: str) -> int:
    from .speaker import SpeakerStore

    if not SpeakerStore(config.speakers_path).remove(speaker):
        print(f"{speaker} is not enrolled", file=sys.stderr)
        return 1
    return 0


def cmd_fetch_models(config: Config) -> int:
    from faster_whisper.utils import download_model

    _setup_logging(config, to_file=False)
    target = config.speaker_model_path
    if not (target.is_file() and _sha256(target) == SPEAKER_MODEL_SHA256):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        log.info("downloading %s", SPEAKER_MODEL_URL)
        with urllib.request.urlopen(SPEAKER_MODEL_URL, timeout=60) as res, tmp.open("wb") as f:
            shutil.copyfileobj(res, f)
        if _sha256(tmp) != SPEAKER_MODEL_SHA256:
            tmp.unlink()
            log.error("speaker model checksum mismatch")
            return 1
        tmp.replace(target)
    log.info("speaker model: %s", target)
    log.info("whisper model %s: %s", config.whisperModel,
             download_model(config.whisperModel, cache_dir=str(config.whisper_download_root)))
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kukicha")
    parser.add_argument("--config-dir", type=Path, default=None,
                        help="directory holding config.json and tokens.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the MCP server")
    p = sub.add_parser("enroll", help="enroll a speaker from WAV recordings")
    p.add_argument("speaker")
    p.add_argument("wav", nargs="+", type=Path)
    sub.add_parser("speakers", help="list enrolled speakers")
    p = sub.add_parser("unenroll", help="remove an enrolled speaker")
    p.add_argument("speaker")
    sub.add_parser("fetch-models", help="download the Whisper and speaker models into the data directory")
    args = parser.parse_args(argv)

    config_dir = args.config_dir or default_config_dir()
    try:
        config = load_config(config_dir)
    except (OSError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.command == "serve":
        code = cmd_serve(config, config_dir)
    elif args.command == "enroll":
        code = cmd_enroll(config, args.speaker, args.wav)
    elif args.command == "speakers":
        code = cmd_speakers(config)
    elif args.command == "unenroll":
        code = cmd_unenroll(config, args.speaker)
    else:
        code = cmd_fetch_models(config)
    sys.exit(code)
