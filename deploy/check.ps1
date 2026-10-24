# 配備した版が応答するまで /healthz を待つ。Whisper のモデルが読み込まれているか(engine)は問わない。
# 配備ワークフローが、タスクを起こした後に実行する。
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$expected = $null
$releaseJson = Join-Path $PSScriptRoot "..\release.json"
if (Test-Path -LiteralPath $releaseJson) {
    $expected = (Get-Content -LiteralPath $releaseJson -Raw -Encoding UTF8 | ConvertFrom-Json).version
}

$deadline = (Get-Date).AddSeconds(60)
$last = ""
while ((Get-Date) -lt $deadline) {
    try {
        $res = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8191/healthz" -TimeoutSec 5
        if ($res.StatusCode -eq 200) {
            $health = $res.Content | ConvertFrom-Json
            # 古い版のプロセスが残って応答している場合を成功にしない
            if ($expected -and $health.version -ne $expected) {
                $last = "version が違う: $($health.version)（期待 $expected）"
            } else {
                Write-Host "healthz ok: version=$($health.version) engine=$($health.engine)"
                exit 0
            }
        }
    } catch {
        $last = $_.Exception.Message
    }
    Start-Sleep -Seconds 2
}
Write-Host "healthz が 60 秒以内に応答しなかった: $last"
exit 1
