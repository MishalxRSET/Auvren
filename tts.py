"""
tts.py — speech synthesis, one call per sentence chunk.

Two engines, chosen by `TTS_ENGINE`:

* **piper** (the default) — offline and free. `piper.exe` from the Piper
  release, plus one voice: an `.onnx` model with its `.onnx.json` config
  beside it. Each chunk is one Piper process: the text goes in on STDIN,
  never on the command line, and a WAV comes back through a temp file. The
  browser decodes WAV as readily as MP3 (`decodeAudioData`), so nothing
  downstream changes.
* **fish** — Fish Audio's hosted voices: online, paid, needs `FISH_API_KEY`.
  The response is streamed so time-to-first-byte can be measured; the chunk
  is returned whole because the browser decodes one complete file per chunk.

Unset, the engine is Piper when Piper is installed, else Fish when a Fish key
is set (so an existing setup keeps its voice), else Piper — whose preflight
check then says exactly what is missing.

Why one process per chunk rather than one long-lived Piper: it is the
documented, stable interface, and a wedged or crashed process costs one
sentence instead of the voice. The price is loading the model each time
(a medium voice is ~60 MB); `PIPER_MAX_PARALLEL` bounds how many load at
once so a long reply cannot fill a 16 GB laptop's memory with copies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

import winplat

log = logging.getLogger("jarvis.tts")

FISH_TTS_URL = "https://api.fish.audio/v1/tts"

ENGINE_PIPER = "piper"
ENGINE_FISH = "fish"
ENGINES = (ENGINE_PIPER, ENGINE_FISH)

# A voice chunk is a sentence or two. Anything longer is a bug upstream, and
# Piper's time grows with it; refusing keeps one runaway from holding the mouth.
PIPER_MAX_CHARS = 2000
PIPER_TIMEOUT_SEC = 30.0


@dataclass
class SynthResult:
    audio: bytes
    first_byte_sec: float
    total_sec: float


# --- engine selection --------------------------------------------------------

def piper_dir() -> Path:
    """Where `scripts/get_piper.ps1` installs Piper and its voices."""
    import data_paths
    return data_paths.data_dir() / "piper"


def piper_exe() -> Optional[str]:
    """PIPER_PATH, else `piper` on PATH, else the installed copy. None if absent."""
    configured = (os.getenv("PIPER_PATH") or "").strip().strip('"')
    if configured:
        return configured if Path(configured).is_file() else None
    found = shutil.which("piper")
    if found:
        return found
    for name in ("piper.exe", "piper"):
        local = piper_dir() / "piper" / name
        if local.is_file():
            return str(local)
    return None


def piper_model() -> Optional[str]:
    """PIPER_MODEL, else the one `.onnx` voice installed. None if unresolvable.

    More than one installed voice and no PIPER_MODEL is None, not a guess:
    which voice speaks is the user's choice.
    """
    configured = (os.getenv("PIPER_MODEL") or "").strip().strip('"')
    if configured:
        return configured if Path(configured).is_file() else None
    voices = piper_dir() / "voices"
    found = sorted(voices.glob("*.onnx")) if voices.is_dir() else []
    return str(found[0]) if len(found) == 1 else None


def piper_problem() -> Optional[str]:
    """None if Piper can speak, else what is missing, in a sentence."""
    exe = piper_exe()
    if exe is None:
        if os.getenv("PIPER_PATH"):
            return f"PIPER_PATH points at {os.getenv('PIPER_PATH')}, which is not there."
        return "Piper is not installed (no piper.exe on PATH, PIPER_PATH unset)."
    model = piper_model()
    if model is None:
        if os.getenv("PIPER_MODEL"):
            return f"PIPER_MODEL points at {os.getenv('PIPER_MODEL')}, which is not there."
        return ("No Piper voice chosen: set PIPER_MODEL to a voice's .onnx file, "
                f"or install exactly one into {piper_dir() / 'voices'}.")
    if not Path(model + ".json").is_file():
        return f"The voice's config is missing: {Path(model).name}.json must sit beside it."
    return None


def engine() -> str:
    """The engine this process speaks with. See the module docstring."""
    chosen = (os.getenv("TTS_ENGINE") or "").strip().lower()
    if chosen in ENGINES:
        return chosen
    if piper_problem() is None:
        return ENGINE_PIPER
    if (os.getenv("FISH_API_KEY") or "").strip():
        return ENGINE_FISH
    return ENGINE_PIPER


def voice_problem() -> Optional[str]:
    """None if the selected engine can speak, else why not."""
    if engine() == ENGINE_FISH:
        return None if (os.getenv("FISH_API_KEY") or "").strip() else \
            "TTS_ENGINE is fish but FISH_API_KEY is not set."
    return piper_problem()


# --- Piper -------------------------------------------------------------------

_piper_slots: dict = {}


def _slots() -> asyncio.Semaphore:
    """One semaphore per event loop: a semaphore that has had waiters is
    bound to its loop, and the server and the test suite each run several."""
    loop = asyncio.get_running_loop()
    sem = _piper_slots.get(loop)
    if sem is None:
        try:
            n = max(1, int(os.getenv("PIPER_MAX_PARALLEL", "2")))
        except ValueError:
            n = 2
        _piper_slots.clear()          # drop semaphores of loops that are gone
        sem = _piper_slots[loop] = asyncio.Semaphore(n)
    return sem


def piper_argv(exe: str, model: str, out_path: str) -> list[str]:
    """The command line. Fixed flags and paths only — never the text."""
    argv = [exe, "--model", model, "--output_file", out_path]
    speaker = (os.getenv("PIPER_SPEAKER") or "").strip()
    if speaker.isdigit():
        argv += ["--speaker", speaker]
    rate = (os.getenv("PIPER_LENGTH_SCALE") or "").strip()
    try:
        if rate and 0.3 <= float(rate) <= 3.0:
            argv += ["--length_scale", rate]
    except ValueError:
        pass
    return argv


def _one_line(text: str) -> bytes:
    """Piper reads STDIN a line at a time and treats each line as its own
    utterance; with --output_file every line overwrites the last. So the
    chunk goes in as exactly one line."""
    return (" ".join(text.split()) + "\n").encode("utf-8")


async def synthesize_piper(text: str, *, exe: Optional[str] = None,
                           model: Optional[str] = None,
                           timeout: float = PIPER_TIMEOUT_SEC) -> Optional[SynthResult]:
    """One chunk through Piper, as WAV bytes. None on any failure; never raises."""
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > PIPER_MAX_CHARS:
        log.warning(f"piper: refusing a {len(text)}-char chunk")
        return None
    exe = exe or piper_exe()
    model = model or piper_model()
    if not exe or not model:
        log.error(f"piper: cannot speak: {piper_problem()}")
        return None

    t0 = time.monotonic()
    workdir = Path(tempfile.mkdtemp(prefix="auvren-tts-"))
    out = workdir / "chunk.wav"
    try:
        async with _slots():
            try:
                proc = await asyncio.create_subprocess_exec(
                    *piper_argv(exe, model, str(out)),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    # Piper finds espeak-ng-data and its DLLs beside itself.
                    cwd=str(Path(exe).parent),
                    creationflags=winplat.no_window_flag(),
                )
            except OSError as e:
                log.error(f"piper: could not start {exe}: {e}")
                return None
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(_one_line(text)),
                                                   timeout=timeout)
            except asyncio.TimeoutError:
                log.error(f"piper: no audio after {timeout:.0f}s, killing it")
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
                return None
        if proc.returncode != 0:
            log.error(f"piper exited {proc.returncode}: "
                      f"{(stderr or b'').decode(errors='replace').strip()[-300:]}")
            return None
        try:
            audio = out.read_bytes()
        except OSError:
            log.error("piper exited cleanly but wrote no audio")
            return None
        if audio[:4] != b"RIFF" or len(audio) <= 44:
            log.error("piper wrote something that is not a WAV with audio in it")
            return None
        elapsed = time.monotonic() - t0
        return SynthResult(audio, elapsed, elapsed)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def piper_sample_rate(model: str) -> Optional[int]:
    """The voice's sample rate, from its .onnx.json, for diagnostics."""
    try:
        return int(json.loads(Path(model + ".json").read_text("utf-8"))["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


# --- Fish Audio --------------------------------------------------------------

async def synthesize_chunk(text: str, *, api_key: str, voice_id: str,
                           client: Optional[httpx.AsyncClient] = None,
                           latency: str = "balanced", timeout: float = 15.0) -> Optional[SynthResult]:
    """One chunk through Fish Audio, as MP3 bytes."""
    text = (text or "").strip()
    if not text or not api_key:
        return None
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    t0 = time.monotonic()
    first: Optional[float] = None
    buf = bytearray()
    try:
        async with client.stream(
            "POST", FISH_TTS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"text": text, "reference_id": voice_id, "format": "mp3", "mp3_bitrate": 128, "latency": latency},
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                log.error(f"TTS {resp.status_code} for {text[:40]!r}")
                return None
            async for part in resp.aiter_bytes():
                if first is None:
                    first = time.monotonic() - t0
                buf.extend(part)
    except (httpx.HTTPError, OSError) as e:
        log.error(f"TTS error: {e}")
        return None
    finally:
        if own:
            try:
                await client.aclose()
            except Exception as e:      # never turn a clean None into an exception
                log.debug(f"TTS client close failed: {e}")
    if not buf:
        return None
    return SynthResult(bytes(buf), first if first is not None else 0.0, time.monotonic() - t0)


# --- the one entry point -----------------------------------------------------

# How names are SAID. "Auvren" is OV-ren, /ˈɒvrən/; Piper's phonemizer reads
# the spelling as "oven" (heard back through Whisper as "Hey Oven"), so the
# voice is handed a spelling it says right. Only the audio changes: the text
# the browser shows, the transcript and the brain all keep "Auvren".
_SAY_AS = ((re.compile(r"\bAuvren\b", re.IGNORECASE), "Ovren"),)


def say_as(text: str) -> str:
    for pattern, spoken in _SAY_AS:
        text = pattern.sub(spoken, text)
    return text


async def synthesize(text: str, *, fish_api_key: str = "", fish_voice_id: str = "",
                     client: Optional[httpx.AsyncClient] = None) -> Optional[SynthResult]:
    """Speak one chunk with whichever engine is selected."""
    text = say_as(text)
    if engine() == ENGINE_FISH:
        return await synthesize_chunk(text, api_key=fish_api_key, voice_id=fish_voice_id,
                                      client=client)
    return await synthesize_piper(text)
