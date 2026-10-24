# kukicha を Windows ホストに初めて入れるときに、管理者権限で一度だけ手で実行する。何度実行しても同じ状態になる。
#   - C:\apps\kukicha\{releases,data} と C:\ProgramData\kukicha を作り、権限を付ける
#   - ログオン時に実行ユーザーの非昇格で動くタスク kukicha を登録する
#   - Gateway から TCP 8191 への受信を許す
# アカウントは名前の解決が ssh 越しに失敗するので、SID で扱う。
param(
    # kukicha を動かすユーザーの SID。省略すると、このスクリプトを実行しているユーザー
    [string]$UserSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value,
    # TCP 8191 への受信を許すアドレス。カンマ区切りで複数、または CIDR も書ける
    [Parameter(Mandatory)]
    [string[]]$Gateway
)
$ErrorActionPreference = "Stop"

$App = "kukicha"
$Port = 8191
$AppRoot = "C:\apps\$App"
$Config = Join-Path $env:ProgramData $App
$Admins = "*S-1-5-32-544"
$System = "*S-1-5-18"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal $identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "管理者権限で実行する"
}
$null = New-Object Security.Principal.SecurityIdentifier $UserSid
$User = "*$UserSid"
Write-Host "==> 実行ユーザーの SID: $UserSid"

# 5.1 は外部コマンドの stderr をエラーとして扱うので、終了コードで判断する
function Invoke-Native([scriptblock]$Command, [string]$What) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command 2>&1 | Out-Host
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw "$What に失敗した（終了コード $code）" }
}

# C:\apps は事前に作り、実行ユーザーに変更を許しておく(docs/setup.md)。releases と data はそこから継承する
if (-not (Test-Path -LiteralPath (Split-Path $AppRoot) -PathType Container)) {
    throw "$(Split-Path $AppRoot) がない。先に作って実行ユーザーに変更を許す(docs/setup.md)"
}
New-Item -ItemType Directory -Force -Path "$AppRoot\releases", "$AppRoot\data" | Out-Null

# 秘密を含む設定。管理者と SYSTEM は全権、実行ユーザーは読むだけ
New-Item -ItemType Directory -Force -Path $Config | Out-Null
Invoke-Native { icacls $Config /inheritance:r /grant:r "${Admins}:(OI)(CI)F" "${System}:(OI)(CI)F" "${User}:(OI)(CI)R" } "$Config の ACL 設定"

Write-Host "==> タスク $App を登録する"
$run = "$AppRoot\current\deploy\run.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$run`"" `
    -WorkingDirectory "$AppRoot\current"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserSid
# ochoko と同じく、ログオンしている実行ユーザーの非昇格の権限で動かす
$principal = New-ScheduledTaskPrincipal -UserId $UserSid -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName $App -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "==> ファイアウォール規則を入れる"
Get-NetFirewallRule -DisplayName $App -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $App -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -RemoteAddress $Gateway -Profile Any | Out-Null

Get-ScheduledTask -TaskName $App | Select-Object TaskName, State
if (-not (Test-Path -LiteralPath (Join-Path $Config "tokens.json"))) {
    Write-Host "次に $Config\tokens.json を置く（{`"dev`": `"<token>`", `"prod`": `"<token>`"}）。呼び出し元ごとに別のトークンにする"
}
