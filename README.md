# kukicha

A thin speech-to-text MCP server that transcribes only an enrolled speaker's
voice. It receives a short WAV and a speaker id and returns the text that
speaker said, dropping silence, noise, and other voices picked up by the same
microphone. It has no character logic. It was written as the hearing backend
for [yunomi](https://github.com/ochanuco/yunomi), next to
[ochoko](https://github.com/ochanuco/ochoko), its voice.

It runs on a Windows machine with an NVIDIA GPU as a logon scheduled task and
serves `http://0.0.0.0:8191/`. The firewall rule admits only the gateway
addresses given at registration. It also runs on CPU, which is how it is
developed and tested on macOS.

```text
src/kukicha     MCP JSON-RPC, Bearer auth, the transcription pipeline, enrollment CLI
tests/          pytest; the models are replaced with fakes
deploy/         hooks run by the deployer, register.ps1, and run.ps1 (the task's entry point)
```

## How `transcribe` works

1. Silero VAD (the ONNX model bundled with faster-whisper) finds the speech
   spans. If there is less than `minUtteranceMs` of speech, the result is empty
   and Whisper is not run.
2. The speech spans are joined and transcribed by faster-whisper.
3. Segments that are typical Whisper hallucinations on silence or noise
   (fixed sign-off phrases, bracketed annotations such as `(音楽)`, highly
   repetitive text, or a high no-speech probability with a low log
   probability) are dropped.
4. For each remaining segment, a speaker embedding is computed with CAM++
   (through sherpa-onnx, on CPU) and compared by cosine similarity with the
   enrolled voice. Segments at or above `speakerThreshold` are kept. Segments
   shorter than `minEmbedMs` are widened to that length before embedding.
5. The kept segments are joined. If they add up to less than
   `minUtteranceMs`, the result is empty.

Accuracy is deliberately modest: a short part of the speaker's speech may be
lost, which the caller tolerates.

## MCP

`POST /mcp` speaks MCP Streamable HTTP in stateless JSON mode (protocol
version `2025-06-18` only), the same way ochoko does. There are no sessions,
no batches, and no SSE: `GET /mcp` and `DELETE /mcp` return 405. A request
other than `initialize` whose `MCP-Protocol-Version` header is present and not
`2025-06-18` gets 400; an absent header is treated as `2025-06-18`.

Every request needs `Authorization: Bearer <token>`. The token maps to a
caller name (`dev`, `prod`, ...) that appears in the logs. A request that
carries an `Origin` header is rejected with 403. A body larger than
`maxAudioSeconds` of base64 WAV gets 413.

### `transcribe`

| Input | Type | Notes |
|---|---|---|
| `audio` | string | base64 of a 16 kHz mono 16-bit PCM WAV, at most `maxAudioSeconds` (120 s) long |
| `language` | string | A Whisper language code, for example `ja` |
| `speaker` | string | An enrolled speaker id (`[A-Za-z0-9_.-]{1,64}`) |
| `initial_prompt` | string, optional | Vocabulary hint passed to Whisper, at most 1000 characters |

Unknown keys, wrong types, undecodable audio, and audio in another format are
rejected with JSON-RPC error -32602.

Result: `content: [{type: "text", text}]` and
`structuredContent: {text, duration_ms}`. An empty `text` means the speaker did
not speak. `duration_ms` is the length of the kept segments (0 when `text` is
empty).

A speaker that is not enrolled, or was enrolled with a different speaker
model, returns a tool error (`isError: true`) with the reason in the text.
Calls run one at a time; a call that cannot start within `busyWaitSeconds`
returns the tool error `busy`. A GPU out-of-memory error unloads the Whisper
model and returns a tool error; the next call loads it again.

### `GET /healthz`

No auth. `200 {"status":"ok","version":"...","engine":"idle|loading|ready|unhealthy"}`.
`idle` means the Whisper model is not loaded and will be loaded by the next
call; `unhealthy` means the last call failed for a reason other than memory,
until a call succeeds. `version` comes from `release.json` in the working
directory or the project root, falling back to the package version.

## GPU memory

The machine is shared with other GPU workloads, so the Whisper model is loaded
on the first call and unloaded after `idleUnloadSeconds` (300 s) without a
call. The default is `large-v3-turbo` with `int8_float16` on CUDA (`int8` on
CPU). The speaker model and the VAD run on CPU.

## Enrolling a speaker

Record a few tens of seconds of the speaker talking normally (16-bit PCM WAV;
any sample rate or channel count is converted), then:

```sh
kukicha enroll <speaker_id> one.wav two.wav
kukicha speakers             # list enrolled ids
kukicha unenroll <speaker_id>
```

At least 10 s of speech is required. The embedding is saved to
`<data>/speakers/<speaker_id>.json`; recordings are not kept. Enrollment data
never belongs in this repository. Enrolling again replaces the voiceprint,
which is also needed after changing the speaker model.

## Configuration

`config.json` and `tokens.json` live in the config directory
(`%ProgramData%\kukicha` on Windows, `~/.config/kukicha` elsewhere, or
`KUKICHA_CONFIG_DIR`, or `--config-dir`). The data directory holds models,
voiceprints, and logs (`C:\apps\kukicha\data` on Windows,
`~/.local/share/kukicha` elsewhere, or `KUKICHA_DATA_DIR`, or `dataDir`).

`tokens.json` (required for `serve`) maps a caller name to its token:

```json
{ "dev": "<token>", "prod": "<token>" }
```

`config.json` (optional) overrides these defaults:

| Key | Default | Notes |
|---|---|---|
| `host`, `port` | `0.0.0.0`, `8191` | `register.ps1` only opens 8191 |
| `dataDir` | see above | |
| `whisperModel` | `large-v3-turbo` | A faster-whisper model name or a local CTranslate2 model directory |
| `device` | `auto` | `cuda` if a CUDA device is visible, else `cpu` |
| `computeType` | `auto` | `int8_float16` on CUDA, `int8` on CPU |
| `beamSize` | `5` | |
| `cpuThreads` | `4` | |
| `speakerModel` | `<data>/models/speaker/campplus.onnx` | |
| `speakerThreshold` | `0.4` | Cosine similarity needed to keep a segment |
| `vadThreshold` | `0.5` | |
| `minUtteranceMs` | `400` | |
| `minEmbedMs` | `1000` | |
| `maxAudioSeconds` | `120` | |
| `idleUnloadSeconds` | `300` | |
| `busyWaitSeconds` | `10` | |
| `extraHallucinations` | `[]` | More phrases to drop, compared after removing spaces and punctuation |

Logs go to `<data>/logs/kukicha.log` (rotated daily, 14 days) and stderr. They
record the length of the text and the similarity scores, never the text.

## Running it

- [docs/setup.md](docs/setup.md): install, configure, and verify kukicha on a
  Windows machine by hand.
- [docs/deployment.md](docs/deployment.md): the GitHub Actions pipeline that
  publishes Releases, and what a deployer must do to install them.

## Development

```sh
uv sync
uv run pytest
uv run kukicha fetch-models     # Whisper and speaker models into the data directory
uv run kukicha enroll <speaker_id> <wav...>
uv run kukicha serve
```

The tests replace the models with fakes and finish in a few seconds on CPU.
On macOS, `large-v3-turbo` on CPU takes a few seconds per short utterance;
set `"whisperModel": "small"` in `config.json` for faster iteration.

## License

[MIT](LICENSE)
