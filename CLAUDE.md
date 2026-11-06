# Auvren — Voice AI Assistant for Windows

## Overview
Auvren is a voice-first AI assistant for **Windows**, in the style of JARVIS.
It began as `ethanplusai/jarvis` (macOS) and is being rebuilt for Windows and
merged with an offline design (local Ollama brain, Piper voice, wake word —
the owner's "Auvren Build Plan"). Today it runs locally, thinks with a local
model under Ollama (or, with `JARVIS_BRAIN=claude`, a Claude Code process),
speaks with Piper, and hands development tasks to Claude Code — every
execution recorded as a *run* and watchable at `/dashboard`.

Internal identifiers still say jarvis — module names (`jarvis_mcp.py`,
`jarvis_memory.py`), `JARVIS_*` environment variables, the data layout — so
an existing `.env` and data folder keep working. User-facing text says Auvren.

**Windows port status.** Phase 1 (done): startup, the run pipeline, session
watching, voice, the dashboard, opening Windows Terminal / Chrome / VS Code,
toast notifications. Phase 2 (not yet): `screen.py` (screenshots, window
list), `dialog.py` (answering a prompt by keystroke) and `session_steer.py`
(messaging another session) refuse on Windows with a spoken sentence. Their
macOS internals are still in those files, unreachable on Windows, until
phase 2 replaces them. Nothing here has been run on real Windows by the
author of the port; the suite runs on Linux with Win32 faked at the seams.

## Quick Start
Read `skills/jarvis-setup/SKILL.md` first — it carries setup and debugging
facts (mic-in-Chrome-only, the per-port permission trap, subscription vs. API
key, what an expired login sounds like) that were only learned by hitting
them live, and this walkthrough assumes them. It was written on macOS: its
Accessibility and Terminal.app sections do not apply on Windows.

When a user clones this repo and starts Claude Code, help them:
1. Copy .env.example to .env
2. Install Ollama (ollama.com) and pull the brain's model:
   `ollama pull qwen3:4b` (or set `OLLAMA_MODEL`). Then install Claude Code
   (2.1.224 or newer) and log in with `claude` — coding runs, and the brain
   too with `JARVIS_BRAIN=claude`, use your Claude subscription, not an API
   key
3. Install the voice (Piper, offline, free):
   `powershell -ExecutionPolicy Bypass -File scripts\get_piper.ps1`
   — it downloads piper.exe and the en_GB-alan-medium voice into
   `data\piper\` and speaks a test sentence. Fish Audio is optional
   (`TTS_ENGINE=fish` + `FISH_API_KEY`).
   On Windows npm's `claude.cmd` is refused as a launcher (see `winplat.py`);
   use the native installer (`irm https://claude.ai/install.ps1 | iex`) or set
   `JARVIS_CLAUDE_PATH` to `claude.exe`.
4. Install Python dependencies: pip install -r requirements.txt — and, for
   speech recognition on the laptop, `pip install -r requirements-stt.txt`
   then `python stt.py --setup` (downloads base.en, lists mics)
5. Install the Playwright browser: python -m playwright install chromium
   (`read_page` / `look_at_page` need it)
6. Install frontend dependencies: cd frontend && npm install
7. Generate the SSL certs — see below, they are NOT optional (Git for
   Windows ships `openssl`, in Git Bash):
   `openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'`
8. Run the backend: python server.py --host 127.0.0.1
9. Run the frontend: cd frontend && npm run dev
10. Open Chrome (mic only works in Chrome) to http://localhost:5173
11. Click to enable audio, then say "Hey Auvren" (OV-ren) and speak

Do not pass `--reload` on Windows: uvicorn's reloader installs an event loop
that cannot start subprocesses, so the brain and every run would fail.
`server.py` ignores the flag there and says so.

**The certs are required for the dev-server workflow**, despite `server.py`
treating them as optional. `server.py` serves HTTPS only when `cert.pem` and
`key.pem` are both beside it, but `frontend/vite.config.ts` hard-codes its
proxy target as `https://localhost:8340`. Measured with a stand-in backend on
8340 and `npx vite`: with a plain-HTTP backend the Vite dev server itself
answers 200 while every proxied `/api` request returns **500**; with an HTTPS
backend the same request returns 200. The page loads either way, which is why
this presents as "the UI is up but nothing works" rather than as an error.

## Architecture
- **Backend**: FastAPI + Python (server.py, ~5800 lines)
- **Frontend**: Vite + TypeScript + Three.js (audio-reactive orb)
- **Communication**: WebSocket (JSON messages + binary audio)
- **AI**: one of two brains, chosen by `JARVIS_BRAIN`:
  - `local` (default) — `local_brain.py`: a local model under Ollama
    (`qwen3:4b` by default), streamed from `/api/chat` over `httpx`. Its
    tools run IN-PROCESS through `server.run_tool_call`, the same gate
    `/internal/tool` uses, so origin/taint/memory-writer rules apply
    unchanged. It is offered only `local_brain.LOCAL_TOOLS` (enforced in
    the brain, not just the prompt), and `CONFIRM_TOOLS` (`spawn_run`,
    `cancel_run`, `create_project`) are never run on the turn the model
    asks: code reads the exact arguments back and the next utterance must
    be a plain yes. Coding work goes to Claude Code via `spawn_run`.
  - `claude` — `brain.py`: a long-lived Claude Code process (Sonnet by
    default) on the user's subscription; no Anthropic API calls on the voice
    path
- **Speech recognition**: Chrome's Web Speech API (`STT_ENGINE=browser`) or
  ON THE LAPTOP (`stt.py`, used automatically once installed): `mic.MicHub`
  opens the microphone once and fans 80 ms frames out to the wake engine and
  `LocalEars` (energy VAD -> `Segmenter` -> faster-whisper base.en int8 on a
  worker thread). Its transcripts enter `_on_final_transcript` — the same
  path, gate, echo filter and brain a browser transcript takes; browser
  transcripts are ignored while it runs. The ears PAUSE while Auvren speaks
  (no echo cancellation on a raw mic), driven by `_voice_emit`'s status
  frames; the wake engine (if any) keeps listening and barges in. Whisper
  is primed (`hotwords`) with "Auvren" and the project names. Optional deps
  in `requirements-stt.txt`; the suite pins `STT_ENGINE=browser`
- **Wake word** (`wakeword.py`, on by default): "Hey Auvren" (OV-ren).
  Auvren is asleep until addressed. `WakeGate` is consulted where transcripts arrive in
  `/ws/voice` — asleep, a final is ignored and an interim never barges in.
  Awake = the wake word's window, OR held while he thinks/speaks (driven by
  the status frames `_voice_emit` sends), OR the follow-up window after he
  finishes, so a confirmation "yes" needs no second wake word. No
  pretrained model exists for "hey auvren", so it is matched in the
  TRANSCRIPT (`strip_wake_phrase`, with the measured mishearings "overn",
  "oven" accepted only after a greeting); the voice says the name via
  `tts.say_as` ("Ovren"). The audio engine (`WakeListener`: openWakeWord on
  the shared mic) runs only for a pretrained name (`WAKE_WORD=hey_jarvis`)
  or a trained `<data>/wakeword/hey_auvren*.onnx`, and is OPTIONAL — `requirements-wakeword.txt`,
  kept out of requirements.txt because it pulls onnxruntime/scipy/sklearn
  (and tflite on Linux). Without it, a transcript that opens with the wake
  phrase still wakes him. The browser gets `wake` frames for the orb
- **TTS**: Piper by default (offline, one `piper.exe` per sentence chunk, text
  on stdin, WAV out); Fish Audio optional (`tts.py`)
- **System**: Windows Terminal, Chrome/Firefox, VS Code exec'd directly —
  never through `cmd.exe` (`actions.py`); every Win32 seam in `winplat.py`
- **Runs**: every Claude Code execution goes through one recorded pipeline —
  `run_store.py` (SQLite) + `run_executor.py`, surfaced at `/api/runs`,
  `/ws/runs`, and the `/dashboard` UI
- **Internal tool channel**: the brain reaches JARVIS's tools (session
  listing, steering, etc.) via a stdio MCP child that forwards
  `tools/call` to `POST /internal/tool` over loopback HTTP, bearing the
  token from `data_paths.ensure_tool_token()`.
- **The web boundary** (`web_auth.py`): one ASGI gate above the router.
  Every WebSocket handshake and every state-changing request must carry
  either an `Origin` JARVIS serves from (localhost/127.0.0.1/[::1] on the
  API port or Vite's) or that same tool token. The browser needs no setup —
  it is same-origin through Vite's proxy, so it sends the `Origin` itself.
  There is no CORS: same-origin needs none, and sending none is what stops
  a hostile page reading the GETs that cannot be gated (a same-origin GET
  carries no `Origin` to check). The `Host` header IS checked on every
  request including reads, because DNS rebinding is the way around "no
  CORS". `--host` defaults to `127.0.0.1`; `0.0.0.0` still works and warns,
  and then needs `JARVIS_ALLOWED_ORIGINS`.

## The Run Pipeline
Nothing spawns Claude Code outside `RunExecutor`. Two invariants govern it:

1. **A run always reaches a terminal state.** `succeeded`, `failed`,
   `timed_out`, or `cancelled` — never left stuck in `running` or `queued`.
   Every exit path out of `_drive` writes a terminal status, and callers that
   create a run then hand it off must fail it if they throw before the
   executor takes ownership.
2. **Every state transition is a DB write BEFORE it is a notification.** The
   WebSocket is a cache-invalidation hint, never a source of truth; clients
   reconcile against `/api/runs`.

Two more rules for this area: no new npm or Python dependencies, and the
dashboard never uses `innerHTML` / `insertAdjacentHTML` — it renders
arbitrary LLM and file content, so everything goes through
`createElement` / `textContent`.

## Key Files
- `server.py` — Main server, WebSocket handler, HTTP API, action system
- `local_brain.py` — The default brain: a local Ollama model, the tool
  allowlist, spoken confirmation for consequential tools, auto-recovery
  when Ollama comes up late. Drop-in for `Brain` as far as server.py is
  concerned (same turn/ready/origin/taint surface)
- `brain.py` — The Claude brain (`JARVIS_BRAIN=claude`): one long-lived
  `claude -p` process on the user's Claude subscription, fed over stdin as
  stream-json
- `jarvis_mcp.py` — Stdio MCP server exposing JARVIS's tools to the brain;
  forwards `tools/call` to `POST /internal/tool`
- `speech.py` — Sentence splitting and the scheduler that owns every
  utterance JARVIS speaks
- `mic.py` — `MicHub`: one microphone capture, fanned out to every listener
- `stt.py` — On-device speech recognition: `EnergyVAD`, `Segmenter`,
  `Transcriber` (faster-whisper), `LocalEars`, `python stt.py --setup`
- `wakeword.py` — The wake word: `WakeGate` (pure window/hold/follow-up
  state), `strip_wake_phrase`, `WakeListener` (openWakeWord on the mic,
  optional deps), `python wakeword.py --setup`
- `tts.py` — Speech synthesis: Piper (default) or Fish Audio, one call per
  sentence chunk; `engine()` says which
- `winplat.py` — Every Win32 seam: `pid_alive` (never `os.kill(pid, 0)`,
  which TERMINATES on Windows), `kill_tree` (`taskkill /T /F`),
  `claude_argv` (resolves npm's `claude.cmd` to the real program, or
  refuses — cmd.exe would re-parse the brain's system prompt), reparse
  points, System32 paths
- `frontend/src/orb.ts` — Three.js particle orb visualization
- `frontend/src/voice.ts` — Web Speech API + audio playback
- `frontend/src/main.ts` — Frontend state machine
- `frontend/src/dashboard/` — The `/dashboard` run monitor (vanilla TS)
- `run_store.py` — SQLite `runs` / `run_events` tables, six-value status enum
- `run_executor.py` — Spawns `claude -p --output-format stream-json` and drives
  each run to a terminal state, streaming its events into the store
- `stream_parser.py` — Pure parsing of the stream-json output (no I/O)
- `builds.py` — The spec/brief/plan pipeline behind real, phased,
  multi-session builds (see "The Run Pipeline" above)
- `specs.py` — The review surface: reads back what JARVIS proposed and what a
  build produced
- `projects_view.py` — The Projects tab: a read-only JOIN over runs,
  sessions, plans and the repo, by project
- `session_watch.py` — Watches every Claude Code session on the machine
  (process / conversation / project)
- `session_steer.py` — Sends a message into a running session's inbox socket
  (Unix socket; returns `failed` on Windows until phase 2)
- `dialog.py` — Answers a permission prompt by sending one keystroke
  (macOS Terminal.app only; returns `not_found` on Windows until phase 2)
- `notifier.py` — Windows toast fallback (fixed PowerShell script, untrusted
  text only in environment variables) when no browser tab is
  connected to speak through
- `jarvis_memory.py` — Long-term memory: a folder of plain Markdown files the
  user can read and edit directly, not a database
- `usage_store.py` — Tracks the subscription's five-hour / seven-day
  rate-limit usage (there is no spend to report — see brain.py)
- `preflight.py` — First-run environment checks: Ollama and its model (local
  brain), `claude` CLI/login, the voice, cross-session steering (and the
  macOS-only Accessibility/Screen Recording checks off Windows)
- `claude_env.py` — The environment every spawned Claude Code child gets,
  including the `ANTHROPIC_*` scrub
- `actions.py` — System actions: Windows Terminal (PowerShell fallback),
  Chrome/Firefox by their own exe, VS Code as `Code.exe` (Notepad/Explorer
  fallback). Every program exec'd directly with argv, never via `cmd.exe`
- `browser.py` — Playwright. Only the headless half is live (`read_page`,
  `capture_page`, behind `read_page` / `look_at_page`); the headful
  `JarvisBrowser` search/research class is reachable from nothing but
  `tests/test_browser_integration.py`
- `screen.py` — Seeing the screen (refuses on Windows until phase 2; the
  macOS path below is still in the file): the window list (`osascript`) and one
  downscaled `screencapture` the brain sees as an MCP image block. Captured
  only on a turn the user drove, never persisted, never on a timer
- `repo_read.py` — Cheap, model-free reading of a repository (no `claude`
  subprocess)
- `project_maker.py` — Creates a new project directory from a spoken name,
  path-validated against the projects root
- `work_mode.py` — Vestigial. `is_casual_question` is imported by `server.py`
  and called by nothing; the Haiku-vs-`claude -p` routing it classified for is
  gone. Left in place only because a test pins it
- `data_paths.py` — Single source of truth for where data is written

## Where the author's own notes live
This repository ships nothing personal. Research notes, milestone
verification checklists and this project's own superpowers specs and plans
were moved to `.agents/`, which is ignored in full — do not re-add them, and
do not treat `git log` as their backup.

Note that `docs/superpowers/specs` and `docs/superpowers/plans` are still a
live convention: `builds.py` and `specs.py` create them **inside the projects
JARVIS builds**. It is only this repository's own copies that are gone.

## Environment Variables
- `ANTHROPIC_API_KEY` — **read by nothing.** Not asked for in setup, not read
  by any module, and the `anthropic` SDK is not a dependency: JARVIS's brain
  and every spawned run go through your Claude Code subscription login.
  `claude_env.child_env()` scrubs every `ANTHROPIC_*` variable from every
  spawned Claude Code child (brain and run pipeline alike) so the CLI can
  never bill an API key instead, and `preflight.py`'s `anthropic_key_leftover`
  check warns you if one is sitting in your `.env` — that is the only place
  the name appears in live code.
- `JARVIS_BRAIN` (optional, default `local`) — `local` (Ollama) or `claude`.
  The test suite pins `claude` in `tests/conftest.py`, so no test talks to a
  real Ollama; local-brain tests build `LocalBrain` against a fake one
- `OLLAMA_MODEL` (default `qwen3:4b`), `OLLAMA_HOST` (default
  `127.0.0.1:11434`; a bind-all `0.0.0.0` is dialled as loopback),
  `OLLAMA_NUM_CTX` (8192), `OLLAMA_KEEP_ALIVE` (`30m`) — the local brain
- `HONORIFIC` (optional, default `sir`) — how Auvren addresses the user
- `STT_ENGINE` (`auto` default: on-device when installed; `local`;
  `browser`), `STT_MODEL` (`base.en`), `STT_LISTEN_WHILE_SPEAKING`,
  `STT_VAD_MIN_RMS` (400), `STT_BEAM_SIZE` (1), `STT_THREADS`,
  `STT_LANGUAGE` (`en`), `MIC_DEVICE` (index or name; `WAKE_DEVICE` is the
  older spelling)
- `WAKE_WORD` (default `hey_auvren`; `off` disables; `hey_jarvis`; or a path to a custom
  `.onnx`), `WAKE_THRESHOLD` (0.5), `WAKE_WINDOW_SEC` (8),
  `WAKE_FOLLOW_UP_SEC` (8), `WAKE_DEVICE` (mic index or name). The suite
  pins `WAKE_WORD=off` in `tests/conftest.py` (the voice tests predate it,
  and no test may open the real microphone)
- `JARVIS_BRAIN_MODEL` (optional, default `sonnet`) — model for the Claude
  brain, always passed explicitly as `--model`
- `JARVIS_BRAIN_AUTOSTART` (optional, default `1`) — `0` builds the brain but
  never spawns it (every test sets this)
- `JARVIS_MUTE_MIC_DURING_SPEECH` (optional, default false) — fallback if echo
  rejection is not enough with a given microphone
- `TTS_ENGINE` (optional) — `piper` or `fish`. Unset: Piper if installed,
  else Fish if `FISH_API_KEY` is set, else Piper (and preflight says what is
  missing)
- `PIPER_PATH` / `PIPER_MODEL` (optional) — piper.exe and the voice's
  `.onnx` (its `.onnx.json` must sit beside it). Unset, they are found in
  `<data>\piper\` where `scripts\get_piper.ps1` puts them. Deliberately
  NOT writable through the settings API: `PIPER_PATH` names a program
- `PIPER_LENGTH_SCALE`, `PIPER_SPEAKER`, `PIPER_MAX_PARALLEL` (optional) —
  speaking rate, speaker of a multi-speaker voice, models loaded at once (2)
- `FISH_API_KEY` (optional) — Fish Audio TTS, only with `TTS_ENGINE=fish`
- `FISH_VOICE_ID` (optional) — Fish voice model ID
- `USER_NAME` (optional) — Your name for JARVIS to use
- `JARVIS_DATA_DIR` (optional) — Where JARVIS writes everything: the SQLite
  database, memory Markdown, usage.json, the tool token. Defaults to `data/`
  (`data_paths.py`). Set it to run an isolated instance without touching live
  data — the test suite gives every test a fresh one this way.
- `JARVIS_SKIP_PERMISSIONS` (optional) — Defaults to true; passes
  `--dangerously-skip-permissions` to spawned runs, which have no TTY to
  answer a permission prompt
- `WEATHER_LOCATION_LABEL` / `WEATHER_LATITUDE` / `WEATHER_LONGITUDE` /
  `WEATHER_UNIT` (all optional) — override the auto-detected (public-IP)
  weather location and units
- `JARVIS_ALLOWED_ORIGINS` (optional) — extra origins the web boundary
  accepts, comma-separated. Only needed when the page is opened at an
  address JARVIS does not serve from itself (a LAN IP, a tunnel). Their
  host names are also what the `Host` check accepts, so a `.local` or
  tailscale name has to be listed here to work at all
- `JARVIS_DEBUG_DOCS` (optional, default off) — serve `/docs`, `/redoc` and
  `/openapi.json`. That console has a "Try it out" button on every route
- `JARVIS_ENV_FILE` (optional) — where the settings endpoints read and write
  `.env`. Defaults to the repo's own; the test suite redirects it so no test
  can rewrite the developer's real configuration

## Testing
Run the suite as:

```bash
pytest
```

No flags. `pytest.ini` sets `testpaths` and deselects `-m "not browser"`, so a
bare run is the whole suite and touches neither the network nor the screen.

`tests/test_browser_integration.py` is marked `browser` (one `pytestmark` for
the file) because it drives `browser.py`, which launches Chromium with
`headless=False` on purpose and runs live searches. Run those with
`pytest -m browser`; they skip themselves if there is no network.

Everything else uses fakes at the subprocess seam — `piper`, `powershell`,
`taskkill`, `wt`, `claude` (and the macOS `screencapture`/`sips`/`osascript`
still in phase-2 files) are never really invoked; Win32 calls are faked in
`winplat`. The suite runs on Linux or Windows. If you add a test that
needs the network, a real window, or a real `claude`, mark it.

## Conventions
- JARVIS personality: British butler, dry wit, economy of language
- Max 1-2 sentences per voice response
- The brain reaches JARVIS's capabilities as MCP tools (`jarvis_mcp.py` ->
  `/internal/tool`), not by parsing `[ACTION:X]` tags out of its reply — that
  tag machinery is gone in full, including the last handler (`_execute_browse`),
  which lost its caller with the voice dispatch chain
- Programs are exec'd directly with argv — never `shell=True`, `start`,
  `os.startfile` on a file that could be a script, or a `.cmd`/`.bat` shim
- SQLite for runs, run events and usage (`run_store.py`, `usage_store.py`);
  long-term memory is plain Markdown files instead (`jarvis_memory.py`) —
  the user edits it directly, so it is never a database
