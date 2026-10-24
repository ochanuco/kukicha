# タスク kukicha が起動する。版の venv の python.exe を前面で動かし、その終了コードで終わる。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

if (-not (Test-Path -LiteralPath $Python)) {
    throw "venv $LockHash がない。先に deploy\prepare.ps1 を実行する"
}
# 再起動のたびにモデルを取りに行かない。モデルは prepare.ps1 が置く
$env:HF_HUB_OFFLINE = "1"
& $Python -m kukicha serve
exit $LASTEXITCODE
