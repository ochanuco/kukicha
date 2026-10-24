import base64

import numpy as np
import pytest
from starlette.testclient import TestClient

from kukicha.app import create_app
from kukicha.asr import Segment
from kukicha.auth import BearerAuthenticator
from kukicha.mcp import PROTOCOL_VERSION, McpServer
from kukicha.pipeline import Transcriber

from conftest import OWNER, OWNER_VEC, FakeAsr, FakeEncoder, FakeVad, wav_b64, wav_bytes

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(store):
    # 無音の WAV に対して、持ち主の声と判定される区間を 1 つ返す
    asr = FakeAsr([Segment(0.0, 1.5, "こんにちは")])
    transcriber = Transcriber(FakeVad(), asr, FakeEncoder(voices=[(-1, 1000, OWNER_VEC)]), store)
    mcp = McpServer(transcriber, "r1-abcdef0", max_audio_seconds=10)
    app = create_app(mcp, BearerAuthenticator({"dev": TOKEN}), asr, "r1-abcdef0", max_audio_seconds=10)
    with TestClient(app) as c:
        yield c


def rpc(client, method, params=None, id_=1, headers=None):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return client.post("/mcp", json=msg, headers={**AUTH, **(headers or {})})


def call(client, arguments):
    return rpc(client, "tools/call", {"name": "transcribe", "arguments": arguments})


def args(**over):
    return {"audio": wav_b64(2.0), "language": "ja", "speaker": OWNER, **over}


def test_healthz_needs_no_auth(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "version": "r1-abcdef0", "engine": "idle"}


def test_initialize(client):
    res = rpc(client, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
              headers={"MCP-Protocol-Version": "2024-11-05"})
    assert res.status_code == 200
    body = res.json()
    assert body["jsonrpc"] == "2.0" and body["id"] == 1
    assert body["result"] == {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "kukicha", "version": "r1-abcdef0"},
    }


def test_tools_list(client):
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    assert [t["name"] for t in tools] == ["transcribe"]
    schema = tools[0]["inputSchema"]
    assert schema["required"] == ["audio", "language", "speaker"]
    assert set(schema["properties"]) == {"audio", "language", "speaker", "initial_prompt"}
    assert tools[0]["outputSchema"]["required"] == ["text", "duration_ms"]


def test_transcribe_result_shape(client):
    res = call(client, args(initial_prompt="固有名詞"))
    assert res.status_code == 200
    result = res.json()["result"]
    assert result == {
        "content": [{"type": "text", "text": "こんにちは"}],
        "structuredContent": {"text": "こんにちは", "duration_ms": 1500},
    }


def test_transcribe_unknown_speaker_is_tool_error(client):
    result = call(client, args(speaker="42")).json()["result"]
    assert result["isError"] is True
    assert "not enrolled" in result["content"][0]["text"]
    assert "structuredContent" not in result


@pytest.mark.parametrize("bad, message", [
    ({"audio": "not base64!"}, "base64"),
    ({"audio": base64.b64encode(b"RIFFjunk").decode()}, "WAV"),
    ({"audio": base64.b64encode(wav_bytes(np.zeros(8000, np.float32), rate=8000)).decode()}, "16000 Hz"),
    ({"audio": base64.b64encode(wav_bytes(np.zeros(16000 * 11, np.float32))).decode()}, "at most 10 s"),
    ({"language": "xx"}, "language"),
    ({"language": 1}, "language"),
    ({"speaker": "../etc"}, "speaker"),
    ({"initial_prompt": 3}, "initial_prompt"),
    ({"extra": True}, "unknown arguments: extra"),
])
def test_transcribe_invalid_params(client, bad, message):
    body = call(client, args(**bad)).json()
    assert body["error"]["code"] == -32602
    assert message in body["error"]["message"]


def test_missing_arguments(client):
    body = call(client, {"language": "ja", "speaker": OWNER}).json()
    assert body["error"]["code"] == -32602


def test_unknown_tool(client):
    body = rpc(client, "tools/call", {"name": "synthesize", "arguments": {}}).json()
    assert body["error"]["code"] == -32602


def test_method_not_found(client):
    res = rpc(client, "resources/list")
    assert res.status_code == 200 and res.json()["error"]["code"] == -32601


def test_ping(client):
    assert rpc(client, "ping", id_="a").json() == {"jsonrpc": "2.0", "id": "a", "result": {}}


def test_notification_and_response_are_accepted(client):
    res = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=AUTH)
    assert res.status_code == 202 and res.content == b""
    res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 5, "result": {}}, headers=AUTH)
    assert res.status_code == 202


@pytest.mark.parametrize("body, code", [
    (b"{not json", -32700),
    (b"[]", -32600),
    (b"3", -32600),
    (b'{"jsonrpc":"2.0","id":{},"method":"ping"}', -32600),
])
def test_malformed_requests(client, body, code):
    res = client.post("/mcp", content=body, headers={**AUTH, "Content-Type": "application/json"})
    assert res.status_code == 400 and res.json()["error"]["code"] == code


def test_protocol_version_header(client):
    assert rpc(client, "tools/list", headers={"MCP-Protocol-Version": "2024-11-05"}).status_code == 400
    assert rpc(client, "tools/list", headers={"MCP-Protocol-Version": PROTOCOL_VERSION}).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic dGVzdA=="},
                                     {"Authorization": "Bearer "}])
def test_auth_required(client, headers):
    res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=headers)
    assert res.status_code == 401
    assert res.headers["WWW-Authenticate"] == "Bearer"


def test_scheme_is_case_insensitive(client):
    res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers={"Authorization": f"bearer {TOKEN}"})
    assert res.status_code == 200


def test_origin_is_rejected(client):
    res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                      headers={**AUTH, "Origin": "https://example.com"})
    assert res.status_code == 403


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_only_post(client, method):
    res = client.request(method, "/mcp", headers=AUTH)
    assert res.status_code == 405 and res.headers["Allow"] == "POST"


def test_body_too_large(client):
    res = client.post("/mcp", content=b"x" * (2 * 1024 * 1024), headers=AUTH)
    assert res.status_code == 413


def test_unknown_path(client):
    assert client.get("/").status_code == 404
