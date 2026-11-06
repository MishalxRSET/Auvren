"""mic.py — one microphone capture, shared by everything that listens.

The wake-word engine and on-device speech recognition both need the same
audio: 16 kHz, mono, 16-bit, in 80 ms frames. Opening the device twice would
work on some Windows audio stacks and fail on others, and would double the
driver work, so `MicHub` opens it ONCE and fans each frame out to every
subscriber.

A subscription looks like a stream — `start()`, `stop()`, `close()` — so a
consumer written against a stream factory (`wakeword.WakeListener`) takes
one unchanged. The device is opened when the first subscriber starts and
closed when the last one stops.

Subscriber callbacks run on PortAudio's audio thread. They must copy and
hand off, never compute: `WakeListener` and `stt.LocalEars` each put the
frame on their own bounded queue and return.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

log = logging.getLogger("jarvis.mic")

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280            # 80 ms — openWakeWord's frame, and ours

Callback = Callable[[bytes, int, object, object], None]


def device_setting():
    """MIC_DEVICE (or the older WAKE_DEVICE): an index or a name; None = default."""
    raw = (os.getenv("MIC_DEVICE") or os.getenv("WAKE_DEVICE") or "").strip()
    if not raw:
        return None
    return int(raw) if raw.isdigit() else raw


def _default_stream(callback, device):
    import sounddevice
    return sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SAMPLES, device=device, callback=callback)


class Subscription:
    """One listener's view of the hub, shaped like a stream."""

    def __init__(self, hub: "MicHub", callback: Callback):
        self._hub = hub
        self.callback = callback
        self.active = False

    def start(self) -> None:
        self._hub._start(self)

    def stop(self) -> None:
        self._hub._stop(self)

    def close(self) -> None:
        self._hub._stop(self)


class MicHub:
    def __init__(self, device=None, stream_factory=None):
        self.device = device if device is not None else device_setting()
        self._stream_factory = stream_factory or _default_stream
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []
        self._stream = None
        self.status_errors = 0

    def subscribe(self, callback: Callback, device=None) -> Subscription:
        """A stream-shaped handle. `device` is accepted and ignored, so this
        can be passed straight in as a `stream_factory(callback, device)`."""
        return Subscription(self, callback)

    @property
    def open(self) -> bool:
        return self._stream is not None

    def _fan_out(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_errors += 1
        data = bytes(indata)
        for sub in list(self._subs):
            try:
                sub.callback(data, frames, time_info, status)
            except Exception:       # one bad listener must not deafen the rest
                pass

    def _start(self, sub: Subscription) -> None:
        with self._lock:
            if sub.active:
                return
            if self._stream is None:
                stream = self._stream_factory(self._fan_out, self.device)
                stream.start()          # raises: the caller reports it
                self._stream = stream
                log.info("microphone open (device %s)", self.device if self.device is not None
                         else "default")
            sub.active = True
            self._subs.append(sub)

    def _stop(self, sub: Subscription) -> None:
        with self._lock:
            if not sub.active:
                return
            sub.active = False
            self._subs = [s for s in self._subs if s is not sub]
            if self._subs or self._stream is None:
                return
            stream, self._stream = self._stream, None
        for step in ("stop", "close"):
            try:
                getattr(stream, step)()
            except Exception:
                pass
        log.info("microphone closed")

    def close(self) -> None:
        for sub in list(self._subs):
            self._stop(sub)
