from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

_SCHEME = "bearer "


class BearerAuthenticator:
    def __init__(self, tokens: dict[str, str]):
        self._entries = [(caller, _hash(token)) for caller, token in tokens.items()]

    def __len__(self) -> int:
        return len(self._entries)

    def authenticate(self, header: str | None) -> str | None:
        """トークンに対応する呼び出し元の名前。認証できなければ None。"""
        if header is None or not header.lower().startswith(_SCHEME):
            return None
        token = header[len(_SCHEME):].strip()
        if not token:
            return None
        presented = _hash(token)
        caller = None
        # 一致した時点で打ち切らず、全件を同じ手間で比べる
        for name, digest in self._entries:
            if hmac.compare_digest(presented, digest):
                caller = name
        return caller


# 長さの違いが比較時間に出ないよう、ハッシュ同士を比べる
def _hash(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def parse_tokens(text: str) -> dict[str, str]:
    """{"<caller>": "<token>"} を読む。形が違えば ValueError。"""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"tokens.json is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError("tokens.json must be a JSON object of caller -> token")
    seen: set[str] = set()
    for caller, token in raw.items():
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"tokens.json: token of '{caller}' must be a non-empty string")
        if token in seen:
            raise ValueError(f"tokens.json: '{caller}' shares a token with another caller")
        seen.add(token)
    return raw


def load_tokens(path: Path) -> dict[str, str]:
    return parse_tokens(path.read_text(encoding="utf-8-sig"))
