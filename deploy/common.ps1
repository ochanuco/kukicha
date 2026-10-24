# prepare.ps1 と run.ps1 が読む。版の venv の場所と、uv / Python の置き場所を決める。
$App = "kukicha"
$AppRoot = "C:\apps\$App"
$DataDir = Join-Path $AppRoot "data"
$ReleaseDir = Split-Path -Parent $PSScriptRoot

# uv.lock が同じ版どうしは venv を使い回す
function Get-LockHash([string]$Dir) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Dir "uv.lock")).Hash.Substring(0, 16).ToLowerInvariant()
}

$LockHash = Get-LockHash $ReleaseDir
$VenvDir = Join-Path $DataDir "venv\$LockHash"
$Python = Join-Path $VenvDir "Scripts\python.exe"

# 実行ユーザーのプロファイルに依らないよう、uv の Python とキャッシュ、モデルの取得先も data に置く
$env:UV_PROJECT_ENVIRONMENT = $VenvDir
$env:UV_PYTHON_INSTALL_DIR = Join-Path $DataDir "python"
$env:UV_CACHE_DIR = Join-Path $DataDir "uv-cache"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:KUKICHA_DATA_DIR = $DataDir
$env:HF_HOME = Join-Path $DataDir "hf"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
# 版のコードは venv に入れず、版のフォルダから読む。venv を版の間で使い回せるようにするため
$env:PYTHONPATH = Join-Path $ReleaseDir "src"
$env:PYTHONUTF8 = "1"
