"""wakeword.py — Auvren answers only when he is addressed.

Without this, everything the microphone hears is a request: a roommate, a
lecture video, the user thinking aloud. With it (the default), Auvren is
asleep until he hears his wake word, then listens for one request — and stays
awake while he thinks and speaks, and for a short follow-up after, so "yes" to
"Shall I start Claude Code…?" needs no second wake word.

Two halves, deliberately separable:

* `WakeGate` — pure bookkeeping, no audio. It decides, for every transcript
  the browser sends, whether Auvren is awake to hear it. server.py consults it
  at the one point transcripts arrive.

* `WakeListener` — the wake-word ENGINE: openWakeWord's "hey jarvis" model
  (or a custom one, e.g. a trained "hey auvren" .onnx) run on the microphone
  in a background thread, 16 kHz in 80 ms frames. Measured on a Linux build
  box: ~3 ms of CPU per frame, i.e. a few percent of one core. It needs three
  optional packages (requirements-wakeword.txt) and the model files
  (`python wakeword.py --setup`).

The engine is optional on purpose. Without it the gate still works from the
TRANSCRIPT: a request that begins "Hey Jarvis…", "Jarvis…" or "Auvren…" wakes
him. That is less good — the browser's recogniser (Chrome's, online) is still
hearing everything, and an unusual word like "Auvren" is often misheard — but
it is never deaf. With the engine running, the wake word is caught locally,
before the recogniser has even finished the sentence.

The wake word is "Hey Auvren" — said OV-ren, /ˈɒvrən/. openWakeWord ships no
model for it (only "hey jarvis", "alexa", "hey mycroft", "hey rhasspy"), so
until one is trained (the build plan's phase 7: Piper-generated samples of
"hey ovren", trained on Colab, dropped into <data>/wakeword/hey_auvren.onnx)
"Hey Auvren" is heard from the TRANSCRIPT — which, with on-device recognition
(stt.py), never leaves the machine. Measured with real speech: Whisper hears
OV-ren as "Overn" unprimed and as "Auvren" exactly when primed with the name,
so the server always passes it as a hotword, and the common mishearings
("hey overn", "hey oven") count when they follow a greeting.

`WAKE_WORD=hey_jarvis` opts back into the pretrained audio model.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("jarvis.wakeword")

OFF_VALUES = {"off", "0", "false", "no", "none", "disabled"}
DEFAULT_MODEL = "hey_auvren"

# openWakeWord's own models: downloadable by name with `--setup`. Any other
# name needs its own trained .onnx in <data>/wakeword.
PRETRAINED = {"hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy"}

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280           # 80 ms: openWakeWord's native frame

# How a recogniser spells the wake word's name. The first spellings are safe
# on their own ("Auvren, open my notes"); the rest are what OV-ren is heard
# as, and count only straight after a greeting — "hey oven" is the wake word,
# "the oven is on" is not.
_SPELLINGS = {
    "auvren": (("auvren", "ovren"),
               ("overn", "ovrin", "auvern", "avren", "ovaran", "ovan", "oven")),
    "jarvis": (("jarvis",), ()),
}
_GREETING = r"(?:hey|hi|hello|ok|okay|oi|yo)"


def setting() -> str:
    """WAKE_WORD: `off`, a pretrained model name, or a path to an .onnx."""
    raw = (os.getenv("WAKE_WORD") or "").strip()
    return raw or DEFAULT_MODEL


def enabled() -> bool:
    return setting().lower() not in OFF_VALUES


def wake_name() -> str:
    """The wake word as a model name: `hey_auvren`, `hey_jarvis`, or the
    stem of a custom .onnx (`hey_auvren_v0.1.onnx` -> `hey_auvren`)."""
    value = setting()
    if value.lower().endswith(".onnx"):
        # Either separator: a Windows path must name the same phrase anywhere.
        return re.split(r"[\\/]", value)[-1][:-len(".onnx")].split("_v")[0].lower()
    return value.lower()


def spoken_phrase() -> str:
    """What the user should say, for the UI hint."""
    return wake_name().replace("_", " ").title()


_pattern_cache: dict = {}


def _wake_pattern() -> "re.Pattern":
    name = wake_name()
    pat = _pattern_cache.get(name)
    if pat is None:
        word = name.split("_")[-1]
        alone, after_greeting = _SPELLINGS.get(word, ((word,), ()))
        alone_re = "|".join(map(re.escape, alone))
        any_re = "|".join(map(re.escape, alone + after_greeting))
        pat = re.compile(rf"\W*(?:{_GREETING}\W+(?:{any_re})|(?:{alone_re}))\b[\W_]*",
                         re.IGNORECASE)
        _pattern_cache[name] = pat
    return pat


def strip_wake_phrase(text: str) -> tuple[bool, str]:
    """(True, rest) if `text` opens with the wake word, else (False, text)."""
    m = _wake_pattern().match(text or "")
    if not m:
        return False, text
    return True, text[m.end():].strip()


def vocabulary() -> list[str]:
    """Words to prime the recogniser with, so the name is spelled right."""
    word = wake_name().split("_")[-1]
    return [word.title()] if word else []


def models_dir() -> Path:
    import data_paths
    return data_paths.data_dir() / "wakeword"


# Models that ship with Auvren: the trained hey_auvren (scripts/train_wakeword.py).
# A copy in the data folder (a retrained one, say) wins.
BUNDLED_DIR = Path(__file__).parent / "models" / "wakeword"


def engine_trusted() -> bool:
    """Whether the audio engine alone may decide that Auvren was called.

    openWakeWord's pretrained models, yes. Our own hey_auvren was trained on
    synthetic voices only, so by default speech recognition keeps listening
    for "Hey Auvren" too, and a model that misses your voice can never make
    Auvren deaf. WAKE_ENGINE_ONLY=1 trusts it (and saves the CPU Whisper
    spends on everything said in the room)."""
    if (os.getenv("WAKE_ENGINE_ONLY") or "").strip().lower() in ("1", "true", "yes"):
        return True
    return wake_name() in PRETRAINED and not setting().lower().endswith(".onnx")


def model_files() -> tuple[Path, Path, Path]:
    """(wake model, melspectrogram, embedding) — all ONNX, all local files."""
    d = models_dir()
    value = setting()
    if value.lower().endswith(".onnx"):
        wake = Path(value)
    else:
        name = value.lower()
        found = sorted(d.glob(f"{name}*.onnx")) or sorted(BUNDLED_DIR.glob(f"{name}*.onnx"))
        wake = found[-1] if found else d / f"{name}_v0.1.onnx"
    return wake, d / "melspectrogram.onnx", d / "embedding_model.onnx"


def engine_problem() -> Optional[str]:
    """None if the audio engine can run, else what is missing, in a sentence."""
    try:
        import numpy  # noqa: F401
        import openwakeword  # noqa: F401
        import sounddevice  # noqa: F401
    except ImportError as e:
        return (f"The wake-word engine is not installed ({e.name}): "
                "pip install -r requirements-wakeword.txt")
    except OSError as e:          # sounddevice without PortAudio
        return f"The microphone library would not load: {e}"
    wake, *features = model_files()
    if not wake.is_file():
        if wake_name() not in PRETRAINED and not setting().lower().endswith(".onnx"):
            return (f"There is no trained \"{spoken_phrase()}\" model yet, so the "
                    "wake word is heard from what you say instead")
        return f"The wake-word model {wake.name} is missing: python wakeword.py --setup"
    for f in features:
        if not f.is_file():
            return f"The wake-word model {f.name} is missing: python wakeword.py --setup"
    return None


# --- the gate ------------------------------------------------------------------

class WakeGate:
    """Is Auvren awake to hear this? Pure state; the clock is injectable.

    Awake means any of: inside the window a wake word opened; Auvren is
    thinking or speaking (`hold`); inside the follow-up after he spoke
    (`release`). A disabled gate is always awake and changes nothing.
    """

    def __init__(self, enabled: bool = True, window: float = 8.0,
                 follow_up: float = 8.0, interim_grace: float = 4.0,
                 clock: Callable[[], float] = time.monotonic):
        self.enabled = enabled
        self.window = window
        self.follow_up = follow_up
        self.interim_grace = interim_grace
        self._clock = clock
        self._until = 0.0
        self._held = False

    @property
    def awake(self) -> bool:
        return (not self.enabled) or self._held or self._clock() < self._until

    @property
    def seconds_left(self) -> float:
        if not self.enabled or self._held:
            return float("inf")
        return max(0.0, self._until - self._clock())

    def open(self, seconds: Optional[float] = None) -> None:
        """Awake for at least `seconds` from now (never shortens a window)."""
        self._until = max(self._until, self._clock() + (self.window if seconds is None else seconds))

    def close(self) -> None:
        self._until = 0.0
        self._held = False

    def hold(self) -> None:
        """Auvren is thinking or speaking: awake until `release`."""
        self._held = True

    def release(self) -> None:
        """He has finished: awake for the follow-up, then asleep."""
        if self._held:
            self._held = False
            self.open(self.follow_up)

    def on_interim(self, text: str) -> str:
        """'wake' (it opened with the wake word), 'pass', or 'ignore'."""
        if not self.enabled:
            return "pass"
        woke, _ = strip_wake_phrase(text)
        if woke:
            self.open()
            return "wake"
        if self.awake:
            # Still talking: do not let the window close mid-sentence.
            self.open(self.interim_grace)
            return "pass"
        return "ignore"

    def on_final(self, text: str) -> tuple[str, str]:
        """('accept', request) / ('wake', '') for the wake word alone /
        ('ignore', text) while asleep. The wake word is stripped."""
        if not self.enabled:
            return "accept", text
        woke, rest = strip_wake_phrase(text)
        if woke:
            self.open()
            return ("accept", rest) if rest else ("wake", "")
        if self.awake:
            return "accept", text
        return "ignore", text


# --- the engine ----------------------------------------------------------------

def _default_model():
    from openwakeword.model import Model
    wake, melspec, embedding = model_files()
    return Model(wakeword_models=[str(wake)], inference_framework="onnx",
                 melspec_model_path=str(melspec), embedding_model_path=str(embedding))


def _default_stream(callback, device):
    import sounddevice
    return sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SAMPLES, device=device, callback=callback)


def _int16_frame(data: bytes):
    import numpy
    return numpy.frombuffer(data, dtype=numpy.int16)


def _device_setting():
    raw = (os.getenv("WAKE_DEVICE") or "").strip()
    if not raw:
        return None
    return int(raw) if raw.isdigit() else raw


class WakeListener:
    """openWakeWord on the microphone, in a background thread.

    `on_wake(score)` is called FROM THAT THREAD; the caller hands it to its
    event loop (server.py uses `call_soon_threadsafe`). Never raises out of
    `start`: a missing package, model or microphone comes back as a sentence,
    and Auvren carries on with the transcript wake alone.
    """

    def __init__(self, on_wake: Callable[[float], None], *, threshold: float = 0.5,
                 cooldown: float = 2.0, device=None,
                 model_factory=None, stream_factory=None, to_frame=None,
                 clock: Callable[[], float] = time.monotonic):
        self._on_wake = on_wake
        self.threshold = threshold
        self.cooldown = cooldown
        self._device = device if device is not None else _device_setting()
        self._model_factory = model_factory or _default_model
        self._stream_factory = stream_factory or _default_stream
        self._to_frame = to_frame or _int16_frame
        self._clock = clock
        self._frames: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._model = None
        self.dropped = 0
        self.detections = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _callback(self, indata, frames, time_info, status) -> None:
        # Audio thread: copy and hand off, nothing else. A full queue means
        # the model is behind; drop the frame rather than block the driver.
        try:
            self._frames.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped += 1

    def start(self) -> Optional[str]:
        """Start listening. None on success, else why not."""
        if self.running:
            return None
        try:
            self._model = self._model_factory()
        except Exception as e:
            return f"The wake-word model would not load: {e}"
        try:
            self._stream = self._stream_factory(self._callback, self._device)
            self._stream.start()
        except Exception as e:
            self._stream = None
            return f"The microphone would not open for the wake word: {e}"
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="wakeword", daemon=True)
        self._thread.start()
        log.info("wake word: listening for %r (threshold %.2f)", spoken_phrase(), self.threshold)
        return None

    def _loop(self) -> None:
        last = -float("inf")
        failures = 0
        while not self._stop.is_set():
            try:
                data = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                scores = self._model.predict(self._to_frame(data))
            except Exception as e:
                failures += 1
                if failures in (1, 100):
                    log.warning(f"wake word: prediction failed ({failures}x): {e}")
                continue
            score = max((float(v) for v in (scores or {}).values()), default=0.0)
            now = self._clock()
            if score >= self.threshold and now - last >= self.cooldown:
                last = now
                self.detections += 1
                try:
                    self._model.reset()      # one utterance, one detection
                except Exception:
                    pass
                try:
                    self._on_wake(score)
                except Exception as e:
                    log.warning(f"wake word: handler failed: {e}")

    def stop(self) -> None:
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            for step in ("stop", "close"):
                try:
                    getattr(stream, step)()
                except Exception:
                    pass
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)


# --- setup ---------------------------------------------------------------------

def setup() -> int:
    """Download the pretrained model files into <data>/wakeword and list
    the microphones, so WAKE_DEVICE can be set if the default is wrong."""
    try:
        import openwakeword.utils
        import sounddevice
    except ImportError as e:
        print(f"Missing {e.name}. First run:  pip install -r requirements-wakeword.txt")
        return 1
    target = models_dir()
    target.mkdir(parents=True, exist_ok=True)
    name = wake_name()
    names = [name] if name in PRETRAINED and not setting().lower().endswith(".onnx") else []
    print(f"Downloading wake-word models into {target} ...")
    # Only the feature models when the wake word is not a pretrained one: a
    # trained hey_auvren.onnx dropped in this folder then just works.
    openwakeword.utils.download_models(model_names=names or ["__none__"],
                                       target_directory=str(target))
    problem = engine_problem()
    print("Microphones (set WAKE_DEVICE to a number or name to choose one):")
    for i, dev in enumerate(sounddevice.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  {i}: {dev.get('name')}")
    if problem:
        print(problem)
        return 1
    print(f"Ready. Say \"{spoken_phrase()}\" to wake Auvren.")
    return 0


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--setup"]:
        raise SystemExit(setup())
    print("usage: python wakeword.py --setup")
    raise SystemExit(2)
