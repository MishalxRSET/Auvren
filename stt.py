"""stt.py — speech recognition on this machine (faster-whisper), not in Chrome.

Until now the browser did the listening: Chrome's Web Speech API, which sends
the microphone to Google, works only in Chrome, and is the one part of the
voice path that needs the internet. With this module the server listens for
itself:

    microphone (mic.MicHub, shared with the wake word)
      -> EnergyVAD     is this 80 ms frame speech?
      -> Segmenter     one utterance: from the first words to a pause
      -> Transcriber   faster-whisper, on the CPU, int8
      -> server        the SAME path a browser transcript takes: the wake
                       gate, the echo filter, the brain

The audio never leaves the machine, and nothing is transcribed while Auvren is
asleep if the wake-word engine is running — a segment with no wake word in it
is dropped unheard (`should_transcribe`).

Echo. A browser microphone came with Chrome's echo cancellation; a raw
capture does not, and over laptop speakers Auvren would hear himself. So
while he speaks, the ears are PAUSED (server.py drives `set_paused` from the
status frames it already sends) and resume a moment after he stops. To cut
him off, say the wake word — the wake engine keeps listening, and "Hey
Jarvis" while he talks stops him — or press Esc / Stop. With headphones,
STT_LISTEN_WHILE_SPEAKING=1 keeps the ears open and the echo filter in
speech.py does the rest.

Measured on the build box with real speech (Piper's en_GB-alan voice):
base.en, int8, beam 1 transcribed "Hey Jarvis, what time is it?" exactly in
~0.5 s; an unknown project name ("sitearc") came out as "sitock" — which is
why the server passes its project names as `hotwords`.

Optional packages: requirements-stt.txt. Model files: `python stt.py --setup`.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from array import array
from collections import deque
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("jarvis.stt")

ENGINES = ("auto", "local", "browser")
DEFAULT_MODEL = "base.en"
FRAME_SEC = 0.08


# --- settings ----------------------------------------------------------------------

def setting() -> str:
    """STT_ENGINE: `local`, `browser`, or `auto` (local when installed)."""
    raw = (os.getenv("STT_ENGINE") or "").strip().lower()
    return raw if raw in ENGINES else "auto"


def model_name() -> str:
    return (os.getenv("STT_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def models_dir() -> Path:
    import data_paths
    return data_paths.data_dir() / "whisper"


def model_path() -> Path:
    # A name, never a path: it becomes a directory under the data folder.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_name())
    return models_dir() / safe


def engine_problem() -> Optional[str]:
    """None if on-device recognition can run, else what is missing."""
    try:
        import faster_whisper  # noqa: F401
        import numpy  # noqa: F401
        import sounddevice  # noqa: F401
    except ImportError as e:
        return (f"On-device speech recognition is not installed ({e.name}): "
                "pip install -r requirements-stt.txt")
    except OSError as e:
        return f"The microphone library would not load: {e}"
    if not (model_path() / "model.bin").is_file():
        return f"The speech model {model_name()} is not downloaded: python stt.py --setup"
    return None


def mode() -> str:
    """'local' or 'browser' — who turns speech into text this run."""
    chosen = setting()
    if chosen == "auto":
        return "local" if engine_problem() is None else "browser"
    return chosen


def listen_while_speaking() -> bool:
    return (os.getenv("STT_LISTEN_WHILE_SPEAKING") or "").strip().lower() in ("1", "true", "yes")


# --- voice activity ----------------------------------------------------------------

def frame_rms(frame: bytes) -> float:
    """Root-mean-square of 16-bit little-endian PCM. Pure Python: 1,280
    samples twelve times a second is nothing, and it keeps numpy out of the
    path the tests drive."""
    samples = array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class EnergyVAD:
    """Speech is louder than the room. The room is learned as it goes.

    The floor follows the quiet frames quickly and the loud ones very slowly,
    so a fan that comes on is absorbed into the floor in a few seconds
    instead of being an endless utterance. `min_rms` is the absolute bar —
    nothing quieter is ever speech, however silent the room.
    """

    def __init__(self, ratio: float = 3.0, min_rms: float = 400.0,
                 initial_floor: float = 150.0, adapt: float = 0.05):
        self.ratio = ratio
        self.min_rms = min_rms
        self.floor = initial_floor
        self.adapt = adapt

    @property
    def threshold(self) -> float:
        return max(self.floor * self.ratio, self.min_rms)

    def __call__(self, rms: float) -> bool:
        speech = rms >= self.threshold
        self.floor += (self.adapt / 40 if speech else self.adapt) * (rms - self.floor)
        return speech


class Segmenter:
    """Frames in, whole utterances out.

    An utterance starts after `start_frames` speech frames in a row (a click
    is one frame) and carries `preroll` frames from before that, so its first
    syllable is not clipped. It ends after `end_frames` of silence (~0.7 s) —
    a pause in a sentence is shorter — or at `max_frames` (30 s, Whisper's
    window). Fewer than `min_speech_frames` of speech is a cough, not a
    sentence, and is dropped.
    """

    def __init__(self, start_frames: int = 2, end_frames: int = 9, preroll: int = 4,
                 max_frames: int = 375, min_speech_frames: int = 4):
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.max_frames = max_frames
        self.min_speech_frames = min_speech_frames
        self._pre: deque = deque(maxlen=max(preroll, start_frames))
        self._frames: list[bytes] = []
        self._active = False
        self._run = 0
        self._silence = 0
        self._speech = 0

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._pre.clear()
        self._frames = []
        self._active = False
        self._run = self._silence = self._speech = 0

    def push(self, frame: bytes, speech: bool) -> Optional[bytes]:
        if not self._active:
            self._run = self._run + 1 if speech else 0
            self._pre.append(frame)
            if self._run >= self.start_frames:
                self._active = True
                self._frames = list(self._pre)
                self._pre.clear()
                self._speech = self._run
                self._silence = 0
            return None
        self._frames.append(frame)
        if speech:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self.end_frames or len(self._frames) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self) -> Optional[bytes]:
        frames, speech = self._frames, self._speech
        self.reset()
        if speech < self.min_speech_frames:
            return None
        return b"".join(frames)


# --- recognition ---------------------------------------------------------------------

_HOTWORD_CHARS = re.compile(r"[^A-Za-z0-9' ]+")


def hotwords(vocabulary, limit: int = 240) -> str:
    """Words Whisper would not otherwise guess — project names, his name —
    as a hint string. Bounded, de-duplicated, and stripped to letters and
    digits: it biases decoding and must never be able to carry anything
    else into the model's prompt."""
    seen, out, size = set(), [], 0
    for word in vocabulary or ():
        w = " ".join(_HOTWORD_CHARS.sub(" ", str(word).replace("-", " ").replace("_", " ")).split())
        key = w.lower()
        if not w or key in seen or size + len(w) + 1 > limit:
            continue
        seen.add(key)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)


def _hallucinated(segment) -> bool:
    """Whisper's well-known failure is inventing text for noise ("Thank you.").
    Its own rule — likely silence AND low confidence — plus runaway
    repetition (a high compression ratio)."""
    no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
    logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
    ratio = float(getattr(segment, "compression_ratio", 0.0) or 0.0)
    return (no_speech > 0.6 and logprob < -1.0) or ratio > 2.4


class Transcriber:
    def __init__(self, model, beam_size: int = 1, language: Optional[str] = "en"):
        self.model = model
        self.beam_size = beam_size
        self.language = language

    def __call__(self, pcm: bytes, vocabulary=()) -> str:
        import numpy
        audio = numpy.frombuffer(pcm, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        kwargs = dict(language=self.language, beam_size=self.beam_size, vad_filter=True,
                      condition_on_previous_text=False, without_timestamps=True)
        hint = hotwords(vocabulary)
        if hint:
            kwargs["hotwords"] = hint
        segments, _info = self.model.transcribe(audio, **kwargs)
        parts = [s.text.strip() for s in segments if not _hallucinated(s)]
        return " ".join(p for p in parts if p).strip()


def load_transcriber() -> Transcriber:
    from faster_whisper import WhisperModel

    def num(name, default):
        try:
            return int(os.getenv(name, default))
        except ValueError:
            return default
    model = WhisperModel(str(model_path()), device="cpu", compute_type="int8",
                         cpu_threads=num("STT_THREADS", 0), local_files_only=True)
    language = (os.getenv("STT_LANGUAGE") or "en").strip() or None
    return Transcriber(model, beam_size=max(1, num("STT_BEAM_SIZE", 1)), language=language)


# --- the ears -----------------------------------------------------------------------

class LocalEars:
    """VAD -> segments -> transcripts, on two background threads.

    The frame thread does the cheap work (energy, segmenting) at the
    microphone's pace; the transcription thread does Whisper, one segment at
    a time, so a slow transcription never makes the microphone drop audio.
    `deliver(text)` and `on_speech()` are called FROM THOSE THREADS; the
    server hands them to its event loop.
    """

    def __init__(self, deliver: Callable[[str], None], *, subscribe,
                 transcriber_factory: Callable[[], Callable] = load_transcriber,
                 should_transcribe: Callable[[], bool] = lambda: True,
                 on_speech: Optional[Callable[[], None]] = None,
                 vocabulary: Callable[[], list] = lambda: [],
                 vad: Optional[EnergyVAD] = None, segmenter: Optional[Segmenter] = None,
                 rms: Callable[[bytes], float] = frame_rms,
                 clock: Callable[[], float] = time.monotonic,
                 resume_guard: float = 0.35, speech_ping: float = 0.5):
        self._deliver = deliver
        self._subscribe = subscribe
        self._transcriber_factory = transcriber_factory
        self._should_transcribe = should_transcribe
        self._on_speech = on_speech
        self._vocabulary = vocabulary
        self.vad = vad or EnergyVAD(min_rms=float(os.getenv("STT_VAD_MIN_RMS", "400") or 400))
        self.segmenter = segmenter or Segmenter()
        self._rms = rms
        self._clock = clock
        self._resume_guard = resume_guard
        self._speech_ping = speech_ping
        self._frames: "queue.Queue[bytes]" = queue.Queue(maxsize=128)
        self._segments: "queue.Queue[bytes]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sub = None
        self._transcriber = None
        self._paused = False
        self._ignore_until = 0.0
        self._reset = False
        self.transcribed = 0
        self.skipped = 0
        self.dropped = 0

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def set_paused(self, paused: bool) -> None:
        """Paused: every frame is dropped and any half-heard utterance is
        discarded. Resuming waits `resume_guard` for the speaker to fall
        silent, so the last of his own voice is not the start of a 'request'."""
        if paused:
            self._paused = True
            self._reset = True
        elif self._paused:
            self._ignore_until = self._clock() + self._resume_guard
            self._paused = False

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._paused:
            return
        try:
            self._frames.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped += 1

    def start(self) -> Optional[str]:
        """Load the model and open the microphone. None, or why not.
        Loading takes a second or two: call it off the event loop."""
        if self.running:
            return None
        try:
            self._transcriber = self._transcriber_factory()
        except Exception as e:
            return f"The speech model would not load: {e}"
        try:
            self._sub = self._subscribe(self._callback)
            self._sub.start()
        except Exception as e:
            self._sub = None
            return f"The microphone would not open for speech recognition: {e}"
        self._stop.clear()
        self._threads = [threading.Thread(target=self._frames_loop, name="stt-frames", daemon=True),
                         threading.Thread(target=self._transcribe_loop, name="stt-whisper",
                                          daemon=True)]
        for t in self._threads:
            t.start()
        log.info("on-device speech recognition listening (%s)", model_name())
        return None

    def _frames_loop(self) -> None:
        last_ping = -float("inf")
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._reset:
                self._reset = False
                self.segmenter.reset()
            if self._paused or self._clock() < self._ignore_until:
                continue
            speech = self.vad(self._rms(frame))
            now = self._clock()
            if speech and self._on_speech is not None and now - last_ping >= self._speech_ping:
                last_ping = now
                try:
                    self._on_speech()
                except Exception:
                    pass
            segment = self.segmenter.push(frame, speech)
            if segment is None:
                continue
            try:
                wanted = self._should_transcribe()
            except Exception:
                wanted = True
            if not wanted:
                self.skipped += 1          # asleep: never transcribed, never kept
                continue
            try:
                self._segments.put_nowait(segment)
            except queue.Full:
                try:
                    self._segments.get_nowait()   # keep the newest: it is what he just said
                except queue.Empty:
                    pass
                self._segments.put_nowait(segment)
                self.dropped += 1

    def _transcribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                segment = self._segments.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                vocabulary = self._vocabulary()
            except Exception:
                vocabulary = []
            t0 = self._clock()
            try:
                text = self._transcriber(segment, vocabulary)
            except Exception as e:
                log.warning(f"speech recognition failed: {e}")
                continue
            self.transcribed += 1
            log.info("stt: %.1fs of audio in %.2fs -> %r",
                     len(segment) / 2 / 16000, self._clock() - t0, text[:80])
            if text:
                try:
                    self._deliver(text)
                except Exception as e:
                    log.warning(f"stt: delivering a transcript failed: {e}")

    def stop(self) -> None:
        self._stop.set()
        sub, self._sub = self._sub, None
        if sub is not None:
            try:
                sub.stop()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []


# --- setup ---------------------------------------------------------------------------

def setup() -> int:
    """Download the model into <data>/whisper and list the microphones."""
    try:
        from faster_whisper import download_model
        import sounddevice
    except ImportError as e:
        print(f"Missing {e.name}. First run:  pip install -r requirements-stt.txt")
        return 1
    target = model_path()
    print(f"Downloading the {model_name()} speech model into {target} ...")
    download_model(model_name(), output_dir=str(target))
    print("Microphones (set MIC_DEVICE to a number or name to choose one):")
    for i, dev in enumerate(sounddevice.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  {i}: {dev.get('name')}")
    problem = engine_problem()
    if problem:
        print(problem)
        return 1
    print("Ready. Auvren will listen on this machine from the next start.")
    return 0


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--setup"]:
        raise SystemExit(setup())
    print("usage: python stt.py --setup")
    raise SystemExit(2)
