# 版を止めるとき、配備ワークフローがタスクより先に実行する(作業フォルダは止める版、引数なし)。
# タスクを止めても、run.ps1 が起こした python.exe や手で起動したものが :8191 を握り続けるので、プロセスツリーごと終わらせる。
# 終了コード 0 で成功。プロセスが残れば 1 を返し、配備は失敗する。
$ErrorActionPreference = "Stop"

$App = if ($env:APP_NAME) { $env:APP_NAME } else { "kukicha" }
$Port = 8191

if (Get-ScheduledTask -TaskName $App -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $App -ErrorAction SilentlyContinue
}

function Get-Leftover {
    $ids = @()
    # venv の python.exe は本体の python.exe を子として起こすので、コマンドラインと待ち受けの両方で探す
    $ids += @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
        Where-Object { $_.CommandLine -match "-m\s+$App\s+serve" -or $_.CommandLine -match "\\deploy\\run\.ps1" } |
        ForEach-Object { [int]$_.ProcessId })
    $ids += @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { [int]$_.OwningProcess })
    # 自分自身は対象にしない
    @($ids | Where-Object { $_ -ne $PID -and $_ -ne 0 } | Sort-Object -Unique)
}

foreach ($id in Get-Leftover) {
    Write-Host "==> pid $id をプロセスツリーごと終了する"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & taskkill.exe /PID $id /T /F 2>&1 | Out-Host
    $ErrorActionPreference = $previous
}

$deadline = (Get-Date).AddSeconds(15)
while ($true) {
    $left = @(Get-Leftover)
    if ($left.Count -eq 0) { break }
    if ((Get-Date) -gt $deadline) {
        Write-Host "$App のプロセスが終了しない: pid $($left -join ', ')"
        exit 1
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "$App は止まっている"
exit 0
