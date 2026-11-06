"""The wake word: the gate (pure), the engine (fake microphone and model), and
the whole thing end to end through the real /ws/voice socket.

No test here opens a real microphone or loads a real model. The engine's real
openWakeWord path was exercised by hand against the pretrained hey_jarvis
model (noise never fired it; ~3 ms per 80 ms frame) — see wakeword.py.
"""
import importlib
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import wakeword as w


@pytest.fixture(autouse=True)
def _the_default_wake_word(monkeypatch):
    """conftest switches the wake word off for the rest of the suite; every
    test here is about it, so they start from the default ("Hey Auvren")
    and set WAKE_WORD themselves when they mean something else."""
    monkeypatch.delenv("WAKE_WORD", raising=False)


# --- the phrase ------------------------------------------------------------------

@pytest.mark.parametrize("text,woke,rest", [
    ("Hey Auvren, what time is it?", True, "what time is it?"),
    ("hey auvren what time is it", True, "what time is it"),
    ("Auvren. Open my notes", True, "Open my notes"),
    ("ok Auvren start the build", True, "start the build"),
    ("Hey, Auvren!", True, ""),
    ("auvren", True, ""),
    # How OV-ren is actually heard: Whisper unprimed says "Overn", Piper's
    # own reading of the spelling is heard as "Oven". After a greeting only.
    ("Hey Overn, what time is it?", True, "what time is it?"),
    ("hey oven stop", True, "stop"),
    ("Ovren, play some music", True, "play some music"),
    ("the oven is on", False, "the oven is on"),
    ("oven timer for ten minutes", False, "oven timer for ten minutes"),
    ("Owen said hi", False, "Owen said hi"),
    ("Hey Jarvis what time is it", False, "Hey Jarvis what time is it"),
    ("tell auvren I said hi", False, "tell auvren I said hi"),
    ("auvrenify the thing", False, "auvrenify the thing"),
    ("what time is it", False, "what time is it"),
    ("hey there", False, "hey there"),
    ("", False, ""),
])
def test_strip_wake_phrase(text, woke, rest, monkeypatch):
    monkeypatch.delenv("WAKE_WORD", raising=False)
    assert w.strip_wake_phrase(text) == (woke, rest)


def test_hey_jarvis_is_still_there_if_chosen(monkeypatch):
    monkeypatch.setenv("WAKE_WORD", "hey_jarvis")
    assert w.strip_wake_phrase("Hey Jarvis, what time is it?") == (True, "what time is it?")
    assert w.strip_wake_phrase("Hey Auvren, what time")[0] is False
    assert w.spoken_phrase() == "Hey Jarvis" and w.vocabulary() == ["Jarvis"]


def test_settings(monkeypatch):
    monkeypatch.delenv("WAKE_WORD", raising=False)
    assert w.enabled() and w.setting() == "hey_auvren"
    assert w.spoken_phrase() == "Hey Auvren" and w.vocabulary() == ["Auvren"]
    for off in ("off", "OFF", "0", "false", "none"):
        monkeypatch.setenv("WAKE_WORD", off)
        assert not w.enabled()
    monkeypatch.setenv("WAKE_WORD", r"C:\models\hey_auvren_v0.1.onnx")
    assert w.enabled() and w.spoken_phrase() == "Hey Auvren"
    assert w.model_files()[0] == Path(r"C:\models\hey_auvren_v0.1.onnx")


def test_model_files_live_in_the_data_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WAKE_WORD", raising=False)
    wake, mel, emb = w.model_files()
    assert wake == tmp_path / "wakeword" / "hey_auvren_v0.1.onnx"
    assert mel.parent == emb.parent == tmp_path / "wakeword"


def test_engine_problem_names_the_missing_piece(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "openwakeword", None)    # import raises
    assert "requirements-wakeword.txt" in w.engine_problem()


def test_until_hey_auvren_is_trained_the_engine_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WAKE_WORD", raising=False)
    for name in ("numpy", "openwakeword", "sounddevice"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    problem = w.engine_problem()
    assert "no trained \"Hey Auvren\" model" in problem
    d = tmp_path / "wakeword"
    d.mkdir()
    for f in ("hey_auvren_v0.1.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
        (d / f).write_bytes(b"x")
    assert w.engine_problem() is None, "a trained model dropped in the folder just works"


def test_engine_problem_names_missing_model_files(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WAKE_WORD", "hey_jarvis")
    for name in ("numpy", "openwakeword", "sounddevice"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    assert "--setup" in w.engine_problem()
    d = tmp_path / "wakeword"
    d.mkdir()
    for f in ("hey_jarvis_v0.1.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
        (d / f).write_bytes(b"x")
    assert w.engine_problem() is None


# --- the gate --------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def gate(**kw):
    clock = Clock()
    return w.WakeGate(clock=clock, **kw), clock


def test_a_disabled_gate_changes_nothing():
    g, _ = gate(enabled=False)
    assert g.awake
    assert g.on_final("Hey Auvren what time") == ("accept", "Hey Auvren what time")
    assert g.on_interim("anything") == "pass"


def test_asleep_it_ignores_what_it_hears():
    g, _ = gate()
    assert not g.awake
    assert g.on_final("what time is it") == ("ignore", "what time is it")
    assert g.on_interim("what time") == "ignore"


def test_the_wake_word_with_a_request_is_accepted_without_it():
    g, _ = gate()
    assert g.on_final("Hey Auvren, what time is it?") == ("accept", "what time is it?")
    assert g.awake


def test_the_wake_word_alone_opens_a_window_for_the_request():
    g, clock = gate(window=8.0)
    assert g.on_final("Hey Auvren") == ("wake", "")
    clock.t += 5
    assert g.on_final("what time is it") == ("accept", "what time is it")
    clock.t += 3.1
    assert g.on_final("and the date") == ("ignore", "and the date")


def test_speaking_keeps_the_window_open_then_a_follow_up_remains():
    g, clock = gate(window=8.0, follow_up=6.0)
    g.on_final("hey auvren start the build")
    g.hold()                           # thinking, then speaking
    clock.t += 60
    assert g.awake, "never falls asleep mid-answer"
    g.release()                        # finished speaking
    clock.t += 5.9
    assert g.on_final("yes") == ("accept", "yes"), "a yes needs no second wake word"
    clock.t += 0.2
    assert not g.awake


def test_talking_keeps_the_window_open_mid_sentence():
    g, clock = gate(window=2.0, interim_grace=4.0)
    g.on_final("hey auvren")
    clock.t += 1.9
    assert g.on_interim("I need you to") == "pass"
    clock.t += 3.5
    assert g.awake


def test_an_interim_wake_word_wakes():
    g, _ = gate()
    assert g.on_interim("hey auvren what") == "wake"
    assert g.awake


def test_open_never_shortens_and_close_closes():
    g, clock = gate(window=8.0)
    g.open(30)
    g.open(1)
    clock.t += 20
    assert g.awake
    g.close()
    assert not g.awake


# --- the engine ------------------------------------------------------------------

class FakeModel:
    def __init__(self, scores):
        self.scores = list(scores)
        self.resets = 0

    def predict(self, frame):
        s = self.scores.pop(0) if self.scores else 0.0
        if isinstance(s, Exception):
            raise s
        return {"hey_jarvis_v0.1": s}

    def reset(self):
        self.resets += 1


class FakeStream:
    def __init__(self, callback, device, frames=0, fail=False):
        self.callback, self.device, self.frames = callback, device, frames
        self.fail = fail
        self.started = self.stopped = self.closed = False

    def start(self):
        if self.fail:
            raise OSError("no input device")
        self.started = True
        for _ in range(self.frames):
            self.callback(b"\x00\x00" * w.FRAME_SAMPLES, w.FRAME_SAMPLES, None, None)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def run_listener(scores, frames, **kw):
    hits = []
    model = FakeModel(scores)
    streams = []

    def stream_factory(cb, device):
        streams.append(FakeStream(cb, device, frames=frames))
        return streams[-1]

    clock = kw.pop("clock", None) or (lambda: time.monotonic())
    L = w.WakeListener(hits.append, model_factory=lambda: model,
                       stream_factory=stream_factory, to_frame=lambda b: b,
                       clock=clock, **kw)
    assert L.start() is None
    deadline = time.time() + 2
    while L._frames.qsize() and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    L.stop()
    return L, hits, model, streams[0]


def test_it_wakes_once_per_utterance():
    L, hits, model, stream = run_listener([0.1, 0.2, 0.9, 0.95, 0.97, 0.1], frames=6,
                                          cooldown=2.0)
    assert hits == [0.9], "the frames after a detection are the same utterance"
    assert model.resets == 1 and L.detections == 1
    assert stream.started and stream.stopped and stream.closed
    assert not L.running


def test_below_the_threshold_nothing_happens():
    L, hits, _, _ = run_listener([0.3, 0.49, 0.2], frames=3, threshold=0.5)
    assert hits == []


def test_after_the_cooldown_it_wakes_again():
    ticks = iter([0.0, 0.1, 5.0, 5.1])
    L, hits, _, _ = run_listener([0.9, 0.9, 0.9, 0.9], frames=4, cooldown=2.0,
                                 clock=lambda: next(ticks))
    assert hits == [0.9, 0.9]


def test_a_failing_frame_or_handler_does_not_stop_it():
    hits = []

    def handler(score):
        hits.append(score)
        raise RuntimeError("the loop was gone")

    model = FakeModel([RuntimeError("bad frame"), 0.9])
    L = w.WakeListener(handler, model_factory=lambda: model,
                       stream_factory=lambda cb, d: FakeStream(cb, d, frames=2),
                       to_frame=lambda b: b)
    assert L.start() is None
    time.sleep(0.3)
    assert L.running
    L.stop()
    assert hits == [0.9]


def test_a_full_queue_drops_frames_instead_of_blocking_the_audio_thread():
    L = w.WakeListener(lambda s: None)
    for _ in range(70):
        L._callback(b"\x00\x00", 1, None, None)
    assert L.dropped == 70 - 64


def test_start_says_why_it_cannot_run():
    def boom():
        raise FileNotFoundError("hey_jarvis_v0.1.onnx")
    L = w.WakeListener(lambda s: None, model_factory=boom)
    assert "model would not load" in L.start()

    L = w.WakeListener(lambda s: None, model_factory=lambda: FakeModel([]),
                       stream_factory=lambda cb, d: FakeStream(cb, d, fail=True))
    assert "microphone would not open" in L.start()
    assert not L.running


def test_wake_device_setting(monkeypatch):
    monkeypatch.setenv("WAKE_DEVICE", "3")
    assert w.WakeListener(lambda s: None)._device == 3
    monkeypatch.setenv("WAKE_DEVICE", "USB Microphone")
    assert w.WakeListener(lambda s: None)._device == "USB Microphone"


# --- end to end through /ws/voice ---------------------------------------------------

class FakeBrain:
    def __init__(self):
        self.ready = True
        self.failed = False
        self.turns = []

    async def stop(self):
        pass

    async def turn(self, text, origin="user", on_delta=None, on_tool=None):
        from brain import TurnResult
        self.turns.append(text)
        if on_delta:
            on_delta(f"Echo: {text}.")
        return TurnResult(origin=origin, text=f"Echo: {text}.", stop_reason="result")


@pytest.fixture
def awake_client(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WAKE_WORD", raising=False)      # the default: Hey Auvren
    monkeypatch.setenv("WAKE_WINDOW_SEC", "8")
    import data_paths
    importlib.reload(data_paths)
    import run_store
    importlib.reload(run_store)
    import server
    importlib.reload(server)
    run_store.init_db()
    # Never the real microphone, whatever is installed on this machine.
    monkeypatch.setattr(server.wakeword, "engine_problem", lambda: "not under test")

    # Nor the startup checks: a failing one is SPOKEN, and while Auvren is
    # speaking the gate (correctly) stays awake — so a test that never acks
    # that audio would wait for ever for the window to close.
    async def no_preflight():
        return None
    monkeypatch.setattr(server, "_run_preflight", no_preflight)

    async def fake_synth(text):
        return b"MP3:" + text.encode()

    monkeypatch.setattr(server, "_synth_for_speech", fake_synth)
    server._last_greeting_time = time.time()
    with TestClient(server.app, headers={"Origin": "http://localhost:5173"}) as c:
        server.brain_instance = FakeBrain()
        assert server.wake_gate.enabled and not server.wake_gate.awake
        yield c, server


def _until(ws, pred, limit=12):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if pred(m):
            return seen
    raise AssertionError(seen)


def _open(c):
    ws = c.websocket_connect("/ws/voice")
    sock = ws.__enter__()
    config = sock.receive_json()
    sock.receive_json()                       # status idle
    return ws, sock, config


def test_the_page_is_told_it_is_asleep_and_what_to_say(awake_client):
    c, server = awake_client
    ws, sock, config = _open(c)
    try:
        assert config["wake"] == {"type": "wake", "enabled": True, "awake": False,
                                  "phrase": "Hey Auvren"}
        assert config["stt"] == "browser"
    finally:
        ws.__exit__(None, None, None)


def test_asleep_a_request_is_ignored(awake_client):
    c, server = awake_client
    ws, sock, _ = _open(c)
    try:
        sock.send_json({"type": "transcript", "text": "what time is it", "isFinal": True})
        sock.send_json({"type": "transcript", "text": "hey auvren what time is it",
                        "isFinal": True})
        seen = _until(sock, lambda m: m.get("type") == "audio")
        assert server.brain_instance.turns == ["what time is it"], \
            "only the addressed one, and without the wake word"
        wake = [m for m in seen if m.get("type") == "wake"]
        assert wake and wake[0]["awake"] is True and wake[0]["cue"] is True
    finally:
        ws.__exit__(None, None, None)


def test_the_wake_word_alone_then_the_request(awake_client):
    c, server = awake_client
    ws, sock, _ = _open(c)
    try:
        sock.send_json({"type": "transcript", "text": "Hey Auvren", "isFinal": True})
        wake = _until(sock, lambda m: m.get("type") == "wake")[-1]
        assert wake["awake"] and wake["cue"]
        sock.send_json({"type": "transcript", "text": "open my notes", "isFinal": True})
        _until(sock, lambda m: m.get("type") == "audio")
        assert server.brain_instance.turns == ["open my notes"]
    finally:
        ws.__exit__(None, None, None)


def test_after_he_answers_a_plain_reply_is_heard(awake_client):
    """The confirmation flow: "Shall I…?" — "yes" must not need the wake word."""
    c, server = awake_client
    server.wake_gate.window = 0.05           # so only the follow-up can hold it open
    ws, sock, _ = _open(c)
    try:
        sock.send_json({"type": "transcript", "text": "hey auvren start the build",
                        "isFinal": True})
        audio = _until(sock, lambda m: m.get("type") == "audio")[-1]
        time.sleep(0.1)                      # the wake word's own window is long gone
        sock.send_json({"type": "played", "utt": audio["utt"], "idx": audio["idx"]})
        _until(sock, lambda m: m == {"type": "status", "state": "idle"})
        assert server.wake_gate.awake and server.wake_gate.seconds_left > 7
        sock.send_json({"type": "transcript", "text": "yes", "isFinal": True})
        _until(sock, lambda m: m.get("type") == "audio")
        assert server.brain_instance.turns == ["start the build", "yes"]
    finally:
        ws.__exit__(None, None, None)


def test_when_the_window_closes_the_page_is_told(awake_client):
    c, server = awake_client
    server.wake_gate.window = 0.2
    ws, sock, _ = _open(c)
    try:
        sock.send_json({"type": "transcript", "text": "Hey Auvren", "isFinal": True})
        assert _until(sock, lambda m: m.get("type") == "wake")[-1]["awake"] is True
        asleep = _until(sock, lambda m: m.get("type") == "wake")[-1]
        assert asleep["awake"] is False and "cue" not in asleep
    finally:
        ws.__exit__(None, None, None)


def test_asleep_somebody_elses_voice_cannot_cut_him_off(awake_client, monkeypatch):
    c, server = awake_client
    heard = []

    async def interim(text):
        heard.append(text)
    monkeypatch.setattr(server.speech, "user_interim", interim)
    ws, sock, _ = _open(c)
    try:
        sock.send_json({"type": "interim", "text": "so anyway I told him"})
        sock.send_json({"type": "interim", "text": "hey auvren stop"})
        _until(sock, lambda m: m.get("type") == "wake")
        assert heard == ["hey auvren stop"]
    finally:
        ws.__exit__(None, None, None)


def test_the_engine_wakes_him(awake_client):
    c, server = awake_client
    ws, sock, _ = _open(c)
    try:
        c.portal.call(server._wake_up, "audio", 0.93)
        wake = _until(sock, lambda m: m.get("type") == "wake")[-1]
        assert wake["awake"] and wake["cue"]
        sock.send_json({"type": "transcript", "text": "what's on today", "isFinal": True})
        _until(sock, lambda m: m.get("type") == "audio")
        assert server.brain_instance.turns == ["what's on today"]
    finally:
        ws.__exit__(None, None, None)


def test_the_engine_is_started_when_installed_and_stopped_at_shutdown(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WAKE_WORD", raising=False)
    import server
    importlib.reload(server)
    started, stopped = [], []

    class Listener:
        def __init__(self, on_wake, threshold, stream_factory=None):
            self.on_wake, self.threshold = on_wake, threshold

        def start(self):
            started.append(self.threshold)
            return None

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(server.wakeword, "engine_problem", lambda: None)
    monkeypatch.setattr(server.wakeword, "WakeListener", Listener)
    with TestClient(server.app):
        assert started == [0.5] and server.wake_listener is not None
    assert stopped == [True] and server.wake_listener is None


# --- the trained hey_auvren model --------------------------------------------------------------

def test_the_bundled_model_is_found_and_a_retrained_one_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "hey_auvren_v0.1.onnx").write_bytes(b"x")
    monkeypatch.setattr(w, "BUNDLED_DIR", bundled)
    assert w.model_files()[0] == bundled / "hey_auvren_v0.1.onnx"
    mine = tmp_path / "data" / "wakeword"
    mine.mkdir(parents=True)
    (mine / "hey_auvren_v0.2.onnx").write_bytes(b"x")
    assert w.model_files()[0] == mine / "hey_auvren_v0.2.onnx", "a retrained copy wins"


def test_a_self_trained_model_is_not_trusted_alone(monkeypatch):
    monkeypatch.delenv("WAKE_ENGINE_ONLY", raising=False)
    assert w.engine_trusted() is False, "synthetic voices only: Whisper keeps listening too"
    monkeypatch.setenv("WAKE_ENGINE_ONLY", "1")
    assert w.engine_trusted() is True
    monkeypatch.delenv("WAKE_ENGINE_ONLY")
    monkeypatch.setenv("WAKE_WORD", "hey_jarvis")
    assert w.engine_trusted() is True, "openWakeWord's own models are"
