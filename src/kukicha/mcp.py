from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from faster_whisper.tokenizer import _LANGUAGE_CODES

from .audio import SAMPLE_RATE, AudioFormatError, decode_wav
from .errors import InvalidParams, ToolError
from .pipeline import Result
from .speaker import SPEAKER_ID

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

TRANSCRIBE = "transcribe"
MAX_INITIAL_PROMPT = 1000

TOOLS = [
    {
        "name": TRANSCRIBE,
        "title": "Transcribe an enrolled speaker",
        "description": (
            "Transcribe a 16 kHz mono 16-bit WAV, keeping only the parts spoken by the enrolled `speaker`. "
            "An empty text means the speaker did not speak."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio": {"type": "string", "contentEncoding": "base64", "description": "16 kHz mono 16-bit PCM WAV"},
                "language": {"type": "string", "minLength": 2, "maxLength": 8},
                "speaker": {"type": "string", "pattern": SPEAKER_ID.pattern},
                "initial_prompt": {"type": "string", "maxLength": MAX_INITIAL_PROMPT},
            },
            "required": ["audio", "language", "speaker"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "duration_ms": {"type": "integer"}},
            "required": ["text", "duration_ms"],
        },
    }
]


class Engine(Protocol):
    def transcribe(self, audio: np.ndarray, language: str, speaker: str, initial_prompt: str | None) -> Result: ...


@dataclass(frozen=True)
class Response:
    status: int
    body: dict | None


@dataclass(frozen=True)
class TranscribeArgs:
    audio: np.ndarray
    language: str
    speaker: str
    initial_prompt: str | None


def parse_transcribe(arguments: Any, max_audio_seconds: int) -> TranscribeArgs:
    if not isinstance(arguments, dict):
        raise InvalidParams("arguments must be an object")
    unknown = sorted(set(arguments) - {"audio", "language", "speaker", "initial_prompt"})
    if unknown:
        raise InvalidParams(f"unknown arguments: {', '.join(unknown)}")

    audio_b64 = arguments.get("audio")
    if not isinstance(audio_b64, str) or not audio_b64:
        raise InvalidParams("audio must be a base64 string")
    try:
        wav = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidParams("audio is not valid base64") from e
    try:
        audio = decode_wav(wav)
    except AudioFormatError as e:
        raise InvalidParams(str(e)) from e
    if len(audio) > max_audio_seconds * SAMPLE_RATE:
        raise InvalidParams(f"audio must be at most {max_audio_seconds} s")

    language = arguments.get("language")
    if not isinstance(language, str) or language not in _LANGUAGE_CODES:
        raise InvalidParams("language must be a Whisper language code such as \"ja\"")

    speaker = arguments.get("speaker")
    if not isinstance(speaker, str) or not SPEAKER_ID.match(speaker):
        raise InvalidParams(f"speaker must match {SPEAKER_ID.pattern}")

    prompt = arguments.get("initial_prompt")
    if prompt is not None and (not isinstance(prompt, str) or len(prompt) > MAX_INITIAL_PROMPT):
        raise InvalidParams(f"initial_prompt must be a string of at most {MAX_INITIAL_PROMPT} characters")

    return TranscribeArgs(audio, language, speaker, prompt or None)


class McpServer:
    """Streamable HTTP の POST 1 件分を処理する。セッションは持たない。"""

    def __init__(self, engine: Engine, version: str, max_audio_seconds: int = 120):
        self._engine = engine
        self._version = version
        self._max_audio_seconds = max_audio_seconds

    def handle(self, body: bytes, caller: str, protocol_version: str | None = None) -> Response:
        try:
            msg = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(None, PARSE_ERROR, "Parse error", 400)
        if isinstance(msg, list):
            return _error(None, INVALID_REQUEST, "Batch requests are not supported", 400)
        if not isinstance(msg, dict):
            return _error(None, INVALID_REQUEST, "Invalid Request", 400)

        has_id = "id" in msg
        msg_id = msg.get("id")
        if has_id and (isinstance(msg_id, bool) or not isinstance(msg_id, (str, int, float))):
            return _error(None, INVALID_REQUEST, "id must be a string or a number", 400)

        method = msg.get("method")
        if not isinstance(method, str):
            # クライアントからの応答(result / error)は受け取るだけ
            if has_id and ("result" in msg or "error" in msg):
                return Response(202, None)
            return _error(msg_id, INVALID_REQUEST, "Invalid Request", 400)

        # ヘッダーがなければ仕様どおり PROTOCOL_VERSION とみなす。initialize は交渉前なので見ない
        if method != "initialize" and protocol_version is not None and protocol_version != PROTOCOL_VERSION:
            return _error(msg_id, INVALID_REQUEST, f"Unsupported MCP-Protocol-Version: {protocol_version}", 400)

        if not has_id:
            return Response(202, None)

        params = msg.get("params")
        try:
            if method == "initialize":
                return _result(msg_id, self._initialize())
            if method == "ping":
                return _result(msg_id, {})
            if method == "tools/list":
                return _result(msg_id, {"tools": TOOLS})
            if method == "tools/call":
                return _result(msg_id, self._call_tool(params, caller))
            return _error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}", 200)
        except InvalidParams as e:
            return _error(msg_id, INVALID_PARAMS, str(e), 200)
        except Exception:
            log.exception("internal error")
            return _error(msg_id, INTERNAL_ERROR, "Internal error", 200)

    def _initialize(self) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "kukicha", "version": self._version},
        }

    def _call_tool(self, params: Any, caller: str) -> dict:
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            raise InvalidParams("params.name is required")
        name = params["name"]
        if name != TRANSCRIBE:
            raise InvalidParams(f"Unknown tool: {name}")
        args = parse_transcribe(params.get("arguments"), self._max_audio_seconds)
        started = time.monotonic()
        try:
            result = self._engine.transcribe(args.audio, args.language, args.speaker, args.initial_prompt)
        except ToolError as e:
            log.warning("caller=%s tool=%s speaker=%s error=%s ms=%d",
                        caller, name, args.speaker, e, (time.monotonic() - started) * 1000)
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}
        # 文字の中身は記録しない
        log.info("caller=%s tool=%s speaker=%s text_len=%d duration_ms=%d detail=%s ms=%d",
                 caller, name, args.speaker, len(result.text), result.duration_ms,
                 json.dumps(result.detail), (time.monotonic() - started) * 1000)
        return {
            "content": [{"type": "text", "text": result.text}],
            "structuredContent": {"text": result.text, "duration_ms": result.duration_ms},
        }


def _result(msg_id: Any, result: dict) -> Response:
    return Response(200, {"jsonrpc": "2.0", "id": msg_id, "result": result})


def _error(msg_id: Any, code: int, message: str, status: int) -> Response:
    return Response(status, {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})
