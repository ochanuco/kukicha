from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import __version__
from .asr import Asr
from .audio import SAMPLE_RATE
from .auth import BearerAuthenticator
from .mcp import McpServer


def release_version() -> str:
    """release.json の version。なければパッケージの version。"""
    for path in (Path.cwd() / "release.json", Path(__file__).resolve().parents[2] / "release.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))["version"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return __version__


def create_app(
    mcp: McpServer,
    auth: BearerAuthenticator,
    asr: Asr,
    version: str,
    max_audio_seconds: int = 120,
    idle_check_seconds: float = 15,
) -> Starlette:
    # base64 の WAV が max_audio_seconds 分と、JSON の包み
    max_body = max_audio_seconds * SAMPLE_RATE * 2 * 4 // 3 + 64 * 1024

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok", "version": version, "engine": asr.state})

    async def mcp_endpoint(request: Request) -> Response:
        # 呼び出し元はサーバー間なので Origin を送らない。付いていればブラウザ経由とみなして拒否する
        if "origin" in request.headers:
            return Response(status_code=403)
        if request.method != "POST":
            return Response(status_code=405, headers={"Allow": "POST"})
        caller = auth.authenticate(request.headers.get("authorization"))
        if caller is None:
            return Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > max_body:
            return Response(status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > max_body:
                return Response(status_code=413)
        result = await run_in_threadpool(mcp.handle, bytes(body), caller, request.headers.get("mcp-protocol-version"))
        if result.body is None:
            return Response(status_code=result.status)
        return JSONResponse(result.body, status_code=result.status)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async def unload_loop() -> None:
            while True:
                await asyncio.sleep(idle_check_seconds)
                await run_in_threadpool(asr.unload_if_idle)

        task = asyncio.create_task(unload_loop())
        try:
            yield
        finally:
            task.cancel()

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE", "PUT", "PATCH"]),
        ],
        lifespan=lifespan,
    )
