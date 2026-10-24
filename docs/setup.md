# Setting up kukicha

This installs kukicha by hand on one Windows machine. A deployer following
[deployment.md](deployment.md) repeats steps 3, 4 and 8 for each release; the
rest is done once.

## Requirements

- Windows 10 or 11, x64, with an NVIDIA driver that supports CUDA 12. The CUDA
  Toolkit and cuDNN are not needed; they come as Python wheels.
- [uv](https://docs.astral.sh/uv/) on `PATH` for the user that runs
  `prepare.ps1` (or its full path in the environment variable `UV`). uv
  downloads its own Python into `C:\apps\kukicha\data\python`.
- Network access to PyPI, Hugging Face, and GitHub Releases while
  `prepare.ps1` runs. The running server does not download anything.
- The machine that calls kukicha (the gateway) has a fixed address reachable
  from this machine.

## 1. Prepare `C:\apps`

`register.ps1` expects `C:\apps` to exist and to be writable by the user that
runs kukicha. If ochoko is already installed this is done. Otherwise, in an
elevated PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\apps | Out-Null
icacls C:\apps /grant "${env:USERNAME}:(OI)(CI)M"
```

## 2. Get a build

Download `kukicha-win-x64.zip` from the repository's Releases, or make one
from a checkout: copy `src`, `deploy`, `pyproject.toml`, `uv.lock`,
`.python-version`, `README.md` and `LICENSE` into a folder. Without
`release.json`, `/healthz` reports the package version.

## 3. Unpack it

```powershell
$version = "r1-abcdef0"   # the folder you extract into
New-Item -ItemType Directory -Force C:\apps\kukicha\releases | Out-Null
Expand-Archive kukicha-win-x64.zip "C:\apps\kukicha\releases\$version"
```

## 4. Prepare it

```powershell
cd "C:\apps\kukicha\releases\$version"
.\deploy\prepare.ps1
```

It creates `C:\apps\kukicha\data\venv\<hash of uv.lock>` with `uv sync`
(including the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels), and puts
the Whisper and speaker models under `C:\apps\kukicha\data\models`. Releases
with the same `uv.lock` share the venv, and venvs used by neither the running
release nor the new one are deleted. The release's own code is not installed
into the venv; `run.ps1` puts the release's `src` on `PYTHONPATH`.

Then point `current` at the release:

```powershell
New-Item -ItemType Junction -Path C:\apps\kukicha\current -Target "C:\apps\kukicha\releases\$version"
```

## 5. Register

Run once, in an elevated PowerShell, as the user that will run kukicha:

```powershell
C:\apps\kukicha\current\deploy\register.ps1 -Gateway 192.0.2.10
# other user:      -UserSid S-1-5-21-...
# several callers: -Gateway 192.0.2.10,192.0.2.11  (CIDR also works)
```

It is idempotent. It creates `C:\apps\kukicha\{releases,data}` and
`C:\ProgramData\kukicha`, registers the logon task `kukicha` (non-elevated,
interactive, restarted on failure) that runs `current\deploy\run.ps1`, and
opens inbound TCP 8191 from the gateway addresses. Users are handled by SID
because name lookup fails over SSH sessions.

The first time Python listens on the port, Windows may ask whether to allow
it through the firewall. Allow it, or dismiss the dialog; the port rule above
admits the gateway either way. Choosing to block creates a block rule for
`python.exe` that overrides the port rule.

## 6. Configure

Settings and secrets live in `C:\ProgramData\kukicha\`; persistent data lives
in `C:\apps\kukicha\data\` (`venv\`, `python\`, `uv-cache\`, `models\`,
`speakers\`, `logs\`). An unreadable `config.json` or `tokens.json` makes the
process exit with code 2.

`tokens.json` (required) maps a caller name to its token. Give each caller its
own token; the caller name appears in logs.

```json
{ "dev": "<token>", "prod": "<token>" }
```

`config.json` (optional) overrides the defaults listed in the
[README](../README.md#configuration). Keep `busyWaitSeconds` plus the time a
call takes below the caller's request timeout.

## 7. Enroll the speaker

Copy a few tens of seconds of the speaker's recordings to the machine, then:

```powershell
cd C:\apps\kukicha\current
. .\deploy\common.ps1
& $Python -m kukicha enroll <speaker_id> C:\path\to\one.wav C:\path\to\two.wav
```

The voiceprint is written to `C:\apps\kukicha\data\speakers\` and survives
upgrades. Delete the recordings afterwards if they are not needed.

## 8. Start and verify

```powershell
Start-ScheduledTask kukicha
C:\apps\kukicha\current\deploy\check.ps1   # waits up to 60 s for /healthz
```

`/healthz` answers with `engine: idle` until the first call loads Whisper.
From the gateway:

```sh
curl -s http://<box>:8191/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Upgrading and rolling back

1. Extract the new build into its own `releases\<version>` folder.
2. Run the new version's `deploy\prepare.ps1` with that folder as the working
   directory. The running version keeps serving meanwhile.
3. Run the running version's `deploy\stop.ps1`. It stops the task and every
   kukicha process tree (the venv's `python.exe` starts the interpreter as a
   child, and a hand-started one keeps holding the port), and exits 1 if a
   process survives.
4. Point the `current` junction at the new folder
   (`Remove-Item C:\apps\kukicha\current; New-Item -ItemType Junction ...`).
5. `Start-ScheduledTask kukicha`, then run the new version's
   `deploy\check.ps1`, which also fails if an older version is still answering.

Rolling back is the same with an older folder; its venv is recreated by
`prepare.ps1` if it was deleted.
