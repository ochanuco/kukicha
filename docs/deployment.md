# Releases and deployment

`.github/workflows/release.yml` builds a release zip and publishes it as a
GitHub Release. It does not deploy: a deployer outside this repository picks
up Releases and installs them on the Windows machine, as it does for ochoko.
Without a deployer, install a Release by hand as in [setup.md](setup.md).

## What the pipeline does

A push to `main` (or `workflow_dispatch` with a `ref` on `main`, to republish
an older commit):

1. `build` runs the tests and zips `kukicha-win-x64.zip`: `src/`, `deploy/`,
   `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `LICENSE`, and
   `release.json` with `version` and `commit`. The zip holds no venv and no
   models; `prepare.ps1` creates them on the machine.
2. `publish` waits for approval in the `release` Environment, then publishes a
   GitHub Release `r<run_number>-<sha7>` with the zip, reusing an existing
   Release of the same commit. Add required reviewers to the Environment to
   gate releases; without them `publish` runs at once.
3. Releases older than the newest 3 are deleted with their tags.
4. `notify` starts the deployer's workflow, if the secrets below are set, and
   does not wait for it.

Release publishing uses the workflow's own `GITHUB_TOKEN`. Without the
secrets below, `notify` does nothing and the deployer has to find new Releases
itself.

## Settings the repository needs

These are not in the repository and must be set on GitHub before releases
reach the deployer:

| Setting | Value |
|---|---|
| Environment `release` | Created in the repository settings; add required reviewers to gate releases |
| Secret `DEPLOYER_REPO` | `owner/name` of the deployer repository |
| Secret `DEPLOYER_WORKFLOW` | file name of the deployer workflow; it is started on `main` by `workflow_dispatch` with no inputs, so it must look up the latest Release itself |
| Secret `DEPLOYER_APP_CLIENT_ID` | Client ID of a GitHub App installed on the deployer repository with the Actions: write permission |
| Secret `DEPLOYER_APP_PRIVATE_KEY` | private key of that App |

## The deployer contract

A deployer is expected to do what
[setup.md § Upgrading](setup.md#upgrading-and-rolling-back) does by hand:

1. Download `kukicha-win-x64.zip` from the Release and extract it to its own
   folder, for example `C:\apps\kukicha\releases\<version>`.
2. Run the new version's `deploy\prepare.ps1` with that version's folder as
   the working directory and no arguments. It needs `uv` on `PATH` (or `UV`
   set to `uv.exe`) and network access, and may take several minutes the
   first time a `uv.lock` or model is seen. Any non-zero exit code fails the
   deploy; the running version is untouched.
3. Run the current version's `deploy\stop.ps1` with that version's folder as
   the working directory and no arguments (`APP_NAME` may override the task
   name). Exit code 1 means a process survived; fail the deploy.
4. Switch the `C:\apps\kukicha\current` junction to the new folder.
5. Start the `kukicha` task.
6. Run the new version's `deploy\check.ps1`; exit code 1 means `/healthz` did
   not report the new version within 60 s.

`register.ps1` must have been run once on the machine, with `-Gateway` set to
the caller's address, before the first deploy. All scripts are UTF-8 with BOM
so that Windows PowerShell 5.1 reads them correctly.
