# 版を配備するとき、配備ワークフローが stop.ps1 より先に実行する(作業フォルダは新しい版、引数なし)。
# 動いている版を止めずに、新しい版の venv とモデルを data に用意する。
#   - data\venv\<uv.lock の hash> に uv sync する(CUDA の DLL は nvidia-*-cu12 の wheel で入る)
#   - Whisper と話者のモデルを data\models に置く
#   - 動いている版と新しい版のどちらも使わない venv を消す
# uv は PATH にあるか、環境変数 UV に uv.exe の場所を入れておく。終了コード 0 で成功。
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
. (Join-Path $PSScriptRoot "common.ps1")

$uv = if ($env:UV) { $env:UV } else { (Get-Command uv -ErrorAction Stop).Source }

# 5.1 は外部コマンドの stderr をエラーとして扱うので、終了コードで判断する
function Invoke-Native([scriptblock]$Command, [string]$What) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command 2>&1 | Out-Host
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw "$What に失敗した（終了コード $code）" }
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

$marker = Join-Path $VenvDir ".kukicha-ready"
if (Test-Path -LiteralPath $marker) {
    Write-Host "==> venv $LockHash は用意済み"
} else {
    Write-Host "==> venv $LockHash を作る"
    Push-Location $ReleaseDir
    try {
        Invoke-Native { & $uv sync --frozen --no-dev --extra cuda --no-install-project } "uv sync"
    } finally {
        Pop-Location
    }
    New-Item -ItemType File -Force -Path $marker | Out-Null
}

Write-Host "==> モデルを用意する"
Invoke-Native { & $Python -m kukicha fetch-models } "モデルの取得"

$keep = @($LockHash)
$currentLock = Join-Path $AppRoot "current\uv.lock"
if (Test-Path -LiteralPath $currentLock) {
    $keep += Get-LockHash (Split-Path -Parent $currentLock)
}
foreach ($dir in @(Get-ChildItem -LiteralPath (Join-Path $DataDir "venv") -Directory -ErrorAction SilentlyContinue)) {
    if ($keep -notcontains $dir.Name) {
        Write-Host "==> 使わない venv $($dir.Name) を消す"
        Remove-Item -LiteralPath $dir.FullName -Recurse -Force
    }
}
Write-Host "prepare ok: venv=$LockHash"
exit 0
