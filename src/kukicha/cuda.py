from __future__ import annotations

import os
import sys
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    """nvidia-*-cu12 の wheel が置く DLL を、ctranslate2 が見つけられるようにする。

    ctranslate2 は cuDNN を使う時点で名前だけで読み込むので、add_dll_directory に加えて PATH にも足す。
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    dirs = [p / "bin" for root in nvidia.__path__ for p in Path(root).iterdir() if (p / "bin").is_dir()]
    for d in dirs:
        os.add_dll_directory(str(d))
    os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])


def cuda_available() -> bool:
    import ctranslate2

    try:
        return ctranslate2.get_cuda_device_count() > 0
    except RuntimeError:
        return False
