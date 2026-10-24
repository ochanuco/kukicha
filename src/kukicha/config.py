from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
SPEAKER_MODEL_SHA256 = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
SPEAKER_MODEL_FILE = "campplus.onnx"


def default_config_dir() -> Path:
    if env := os.environ.get("KUKICHA_CONFIG_DIR"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "kukicha"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "kukicha"


def default_data_dir() -> Path:
    if env := os.environ.get("KUKICHA_DATA_DIR"):
        return Path(env)
    if sys.platform == "win32":
        return Path(r"C:\apps\kukicha\data")
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "kukicha"


@dataclass
class Config:
    """config.json の値。キーは ochoko と同じく camelCase で書く。"""

    host: str = "0.0.0.0"
    port: int = 8191
    dataDir: str = ""
    whisperModel: str = "large-v3-turbo"
    # auto: CUDA が見えれば cuda、なければ cpu
    device: str = "auto"
    # auto: cuda なら int8_float16、cpu なら int8
    computeType: str = "auto"
    beamSize: int = 5
    cpuThreads: int = 4
    speakerModel: str = ""
    speakerThreshold: float = 0.4
    # 声の登録の前でも動かすため、未登録の speaker の発話を声で絞らずに返す。他の人の声も通る
    passUnenrolled: bool = False
    vadThreshold: float = 0.5
    minUtteranceMs: int = 400
    minEmbedMs: int = 1000
    maxAudioSeconds: int = 120
    idleUnloadSeconds: int = 300
    busyWaitSeconds: float = 10.0
    extraHallucinations: list[str] = field(default_factory=list)

    @property
    def data_path(self) -> Path:
        return Path(self.dataDir) if self.dataDir else default_data_dir()

    @property
    def models_path(self) -> Path:
        return self.data_path / "models"

    @property
    def whisper_download_root(self) -> Path:
        return self.models_path / "whisper"

    @property
    def speaker_model_path(self) -> Path:
        return Path(self.speakerModel) if self.speakerModel else self.models_path / "speaker" / SPEAKER_MODEL_FILE

    @property
    def speakers_path(self) -> Path:
        return self.data_path / "speakers"

    @property
    def logs_path(self) -> Path:
        return self.data_path / "logs"


def load_config(config_dir: Path) -> Config:
    path = config_dir / "config.json"
    if not path.exists():
        return Config()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object")
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    return Config(**raw)
