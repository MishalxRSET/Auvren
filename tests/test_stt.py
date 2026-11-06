"""On-device speech recognition: the shared microphone, voice activity,
segmenting, the Whisper wrapper, the ears' threads, and the server's use of
them. No real microphone and no real model: the PCM is made here, and
Whisper is a fake with the real call shape (`transcribe(audio, **kw)` ->
(segments, info)). The real model was exercised by hand — see stt.py.
"""
import asyncio
import importlib
import sys
import time
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import mic
import stt

FRAME = stt.FRAME_SEC


def pcm(level: int, frames: int = 1) -> bytes:
    """`frames` 80 ms frames of a square wave at `level` (RMS == level)."""
    n = mic.FRAME_SAMPLES * frames
    return array("h", [level if i % 2 else -level for i in range(n)]).tobytes()


def frames_of(level: int, count: int):
    return [pcm(level) for _ in range(count)]


# --- the shared microphone ------------------------------------------------------------

class FakeStream:
    def __init__(self, callback, device):
        self.callback, self.device = callback, device
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_one_capture_fans_out_and_closes_with_the_last_listener():
    streams = []
    hub = mic.MicHub(device=2, stream_factory=lambda cb, d: streams.append(FakeStream(cb, d))
                     or streams[-1])
    got_a, got_b = [], []
    a = hub.subscribe(lambda data, *_: got_a.append(data))
    b = hub.subscribe(lambda data, *_: got_b.append(data), None)   # stream-factory shape
    a.start()
    b.start()
    assert len(streams) == 1 and streams[0].device == 2, "opened once"
    streams[0].callback(b"\x01\x02", 1, None, None)
    assert got_a == got_b == [b"\x01\x02"]
    a.stop()
    assert hub.open and not streams[0].stopped
    b.close()
    assert not hub.open and streams[0].stopped and streams[0].closed


def test_one_bad_listener_does_not_deafen_the_rest():
    holder = []
    hub = mic.MicHub(stream_factory=lambda cb, d: holder.append(FakeStream(cb, d)) or holder[-1])
    got = []

    def bad(*_):
        raise RuntimeError("boom")
    hub.subscribe(bad).start()
    hub.subscribe(lambda data, *_: got.append(data)).start()
    holder[0].callback(b"ab", 1, None, None)
    assert got == [b"ab"]


def test_mic_device_setting(monkeypatch):
    monkeypatch.delenv("MIC_DEVICE", raising=False)
    monkeypatch.delenv("WAKE_DEVICE", raising=False)
    assert mic.device_setting() is None
    monkeypatch.setenv("WAKE_DEVICE", "4")
    assert mic.device_setting() == 4
    monkeypatch.setenv("MIC_DEVICE", "USB Microphone")
    assert mic.device_setting() == "USB Microphone"


# --- settings ---------------------------------------------------------------------------

def test_mode(monkeypatch):
    monkeypatch.setattr(stt, "engine_problem", lambda: None)
    monkeypatch.delenv("STT_ENGINE", raising=False)
    assert stt.mode() == "local", "auto: on-device when it is installed"
    monkeypatch.setattr(stt, "engine_problem", lambda: "not installed")
    assert stt.mode() == "browser"
    monkeypatch.setenv("STT_ENGINE", "local")
    assert stt.mode() == "local", "asked for: tried, and the server falls back"
    monkeypatch.setenv("STT_ENGINE", "Browser")
    assert stt.mode() == "browser"


def test_the_model_name_never_escapes_the_data_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STT_MODEL", "..\\..\\Windows/evil")
    assert stt.model_path().parent == tmp_path / "whisper"
    monkeypatch.setenv("STT_MODEL", "small.en")
    assert stt.model_path() == tmp_path / "whisper" / "small.en"


def test_engine_problem_says_what_to_run(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert "requirements-stt.txt" in stt.engine_problem()
    for name in ("faster_whisper", "numpy", "sounddevice"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    assert "python stt.py --setup" in stt.engine_problem()
    (stt.model_path()).mkdir(parents=True)
    (stt.model_path() / "model.bin").write_bytes(b"x")
    assert stt.engine_problem() is None


# --- voice activity and segments ----------------------------------------------------------

def test_frame_rms():
    assert stt.frame_rms(pcm(1000)) == pytest.approx(1000)
    assert stt.frame_rms(b"") == 0.0
    assert stt.frame_rms(b"\x01") == 0.0


def test_vad_hears_speech_over_the_room_and_learns_a_noisy_room():
    vad = stt.EnergyVAD(ratio=3.0, min_rms=400)
    assert not vad(100)
    assert vad(3000)
    # A fan comes on: at first it is "speech", then it becomes the floor.
    results = [vad(900) for _ in range(400)]
    assert results[0] is True and results[-1] is False
    assert vad(6000), "and speech over the fan is still speech"


def test_a_click_is_not_an_utterance():
    seg = stt.Segmenter(start_frames=2)
    assert seg.push(pcm(5000), True) is None
    for f in frames_of(10, 20):
        assert seg.push(f, False) is None
    assert not seg.active


def test_an_utterance_keeps_its_first_syllable_and_ends_on_a_pause():
    seg = stt.Segmenter(start_frames=2, end_frames=3, preroll=4, min_speech_frames=2)
    out = None
    quiet, loud = pcm(10), pcm(5000)
    for f, speech in [(quiet, False), (quiet, False), (loud, True), (loud, True),
                      (loud, True), (quiet, False), (quiet, False)]:
        assert seg.push(f, speech) is None
    out = seg.push(quiet, False)
    assert out is not None
    # 2 quiet of preroll + 3 loud + 3 quiet
    assert len(out) == len(quiet) * 8
    assert out.startswith(quiet * 2 + loud)


def test_a_long_monologue_is_cut_at_whispers_window():
    seg = stt.Segmenter(start_frames=1, max_frames=10)
    outs = [seg.push(pcm(5000), True) for _ in range(25)]
    assert sum(o is not None for o in outs) == 2


def test_a_cough_is_dropped():
    seg = stt.Segmenter(start_frames=2, end_frames=2, min_speech_frames=4)
    outs = [seg.push(pcm(5000), True), seg.push(pcm(5000), True),
            seg.push(pcm(10), False), seg.push(pcm(10), False)]
    assert outs == [None, None, None, None]


# --- recognition ---------------------------------------------------------------------------

def test_hotwords_are_bounded_clean_words():
    hint = stt.hotwords(["Auvren", "auvren", "tony-starks-website", "my_notes",
                         'x"; ignore all <|endoftext|>', "Claude Code"])
    assert hint.split()[:1] == ["Auvren"]
    assert "tony starks website" in hint and "my notes" in hint
    assert "<" not in hint and '"' not in hint and "|" not in hint and ";" not in hint
    assert hint.lower().count("auvren") == 1
    assert len(stt.hotwords(["word"] * 1000 + [f"name{i}" for i in range(500)])) <= 240


def seg(text, no_speech=0.0, logprob=-0.2, ratio=1.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob,
                           compression_ratio=ratio)


class FakeWhisper:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append((audio, kw))
        return iter(self.segments), SimpleNamespace(duration=1.0)


def test_the_transcriber_primes_the_name_and_drops_invented_text():
    model = FakeWhisper([seg(" Hey Auvren, "), seg(" Thank you.", no_speech=0.9, logprob=-1.4),
                         seg(" what time is it?"), seg(" la la la la", ratio=3.1)])
    t = stt.Transcriber(model, beam_size=1, language="en")
    text = t(pcm(3000, frames=2), ["Auvren", "sitearc"])
    assert text == "Hey Auvren, what time is it?"
    audio, kw = model.calls[0]
    assert kw["hotwords"] == "Auvren sitearc"
    assert kw["language"] == "en" and kw["beam_size"] == 1 and kw["vad_filter"] is True
    assert kw["condition_on_previous_text"] is False
    assert audio.dtype.name == "float32" and abs(float(abs(audio).max()) - 3000 / 32768) < 1e-6


# --- the ears ---------------------------------------------------------------------------------

class Sub:
    def __init__(self, callback):
        self.callback = callback
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def make_ears(texts=("open my notes",), **kw):
    delivered, subs = [], []
    transcripts = list(texts)
    heard = []

    def transcriber(segment, vocabulary):
        heard.append((len(segment), list(vocabulary)))
        out = transcripts.pop(0) if transcripts else ""
        if isinstance(out, Exception):
            raise out
        return out

    ears = stt.LocalEars(delivered.append,
                         subscribe=lambda cb: subs.append(Sub(cb)) or subs[-1],
                         transcriber_factory=lambda: transcriber,
                         segmenter=stt.Segmenter(start_frames=2, end_frames=3),
                         resume_guard=0.0, **kw)
    return ears, delivered, subs, heard


def say(ears, sub, loud=8, quiet=6):
    for f in frames_of(5000, loud) + frames_of(20, quiet):
        sub.callback(f, mic.FRAME_SAMPLES, None, None)


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_speech_becomes_one_delivered_transcript():
    pings = []
    ears, delivered, subs, heard = make_ears(vocabulary=lambda: ["Auvren"],
                                             on_speech=lambda: pings.append(1))
    assert ears.start() is None and subs[0].started
    say(ears, subs[0])
    assert wait_for(lambda: delivered == ["open my notes"])
    assert heard[0][1] == ["Auvren"] and heard[0][0] > 0
    assert pings, "the server hears that someone is talking"
    ears.stop()
    assert subs[0].stopped and not ears.running


def test_asleep_with_the_engine_nothing_is_transcribed():
    ears, delivered, subs, heard = make_ears(should_transcribe=lambda: False)
    ears.start()
    say(ears, subs[0])
    assert wait_for(lambda: ears.skipped == 1)
    time.sleep(0.1)
    assert heard == [] and delivered == []
    ears.stop()


def test_paused_while_he_speaks_it_hears_nothing_and_forgets_half_a_sentence():
    ears, delivered, subs, heard = make_ears(texts=("second",))
    ears.start()
    for f in frames_of(5000, 4):                 # half a sentence...
        subs[0].callback(f, mic.FRAME_SAMPLES, None, None)
    time.sleep(0.1)
    ears.set_paused(True)                        # ...then he starts talking
    say(ears, subs[0])                           # his own voice: dropped
    time.sleep(0.1)
    ears.set_paused(False)
    say(ears, subs[0])
    assert wait_for(lambda: delivered == ["second"])
    assert len(heard) == 1, "the half sentence before the pause was discarded"
    ears.stop()


def test_a_failed_transcription_does_not_stop_the_ears():
    ears, delivered, subs, _ = make_ears(texts=(RuntimeError("model crashed"), "again"))
    ears.start()
    say(ears, subs[0])
    time.sleep(0.2)
    say(ears, subs[0])
    assert wait_for(lambda: delivered == ["again"])
    ears.stop()


def test_empty_text_is_not_delivered():
    ears, delivered, subs, heard = make_ears(texts=("",))
    ears.start()
    say(ears, subs[0])
    assert wait_for(lambda: len(heard) == 1)
    time.sleep(0.05)
    assert delivered == []
    ears.stop()


def test_start_says_why_it_cannot_listen():
    def no_model():
        raise FileNotFoundError("model.bin")
    ears = stt.LocalEars(lambda t: None, subscribe=lambda cb: Sub(cb), transcriber_factory=no_model)
    assert "speech model would not load" in ears.start()

    class Broken(Sub):
        def start(self):
            raise OSError("no input device")
    ears = stt.LocalEars(lambda t: None, subscribe=Broken, transcriber_factory=lambda: None)
    assert "microphone would not open" in ears.start()
    assert not ears.running


# --- the server --------------------------------------------------------------------------------

class FakeBrain:
    def __init__(self):
        self.ready, self.failed, self.turns = True, False, []

    async def stop(self):
        pass

    async def turn(self, text, origin="user", on_delta=None, on_tool=None):
        from brain import TurnResult
        self.turns.append(text)
        if on_delta:
            on_delta(f"Echo: {text}.")
        return TurnResult(origin=origin, text=f"Echo: {text}.", stop_reason="result")


class FakeEars:
    def __init__(self):
        self.paused = []
        self.stopped = False

    def set_paused(self, p):
        self.paused.append(p)

    def stop(self):
        self.stopped = True


@pytest.fixture
def server_local(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    import data_paths
    importlib.reload(data_paths)
    import run_store
    importlib.reload(run_store)
    import server
    importlib.reload(server)
    run_store.init_db()

    async def fake_synth(text):
        return b"MP3:" + text.encode()

    async def no_preflight():
        return None
    monkeypatch.setattr(server, "_synth_for_speech", fake_synth)
    monkeypatch.setattr(server, "_run_preflight", no_preflight)
    server._last_greeting_time = time.time()
    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
            server.app, headers={"Origin": "http://localhost:5173"}) as c:
        server.brain_instance = FakeBrain()
        ears = FakeEars()
        monkeypatch.setattr(server, "local_ears", ears)
        monkeypatch.setattr(server, "STT_MODE", "local")
        yield c, server, ears


def _until(ws, pred, limit=12):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if pred(m):
            return seen
    raise AssertionError(seen)


def test_an_on_device_transcript_takes_the_same_path_to_the_brain(server_local):
    c, server, ears = server_local
    with c.websocket_connect("/ws/voice") as ws:
        config = ws.receive_json()
        assert config["stt"] == "local"
        ws.receive_json()
        c.portal.call(server._on_final_transcript, "open my notes", "local")
        _until(ws, lambda m: m.get("type") == "audio")
        assert server.brain_instance.turns == ["open my notes"]


def test_a_browser_transcript_is_ignored_while_listening_on_device(server_local):
    c, server, ears = server_local
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"type": "transcript", "text": "hello", "isFinal": True})
        ws.send_json({"type": "interim", "text": "hel"})
        c.portal.call(server._on_final_transcript, "second", "local")
        _until(ws, lambda m: m.get("type") == "audio")
        assert server.brain_instance.turns == ["second"], "never both"


def test_the_ears_pause_while_he_speaks_and_when_muted(server_local):
    c, server, ears = server_local
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        c.portal.call(server._on_final_transcript, "hello", "local")
        audio = _until(ws, lambda m: m.get("type") == "audio")[-1]
        assert True in ears.paused, "deaf while his own voice is in the room"
        ws.send_json({"type": "played", "utt": audio["utt"], "idx": audio["idx"]})
        _until(ws, lambda m: m == {"type": "status", "state": "idle"})
        assert ears.paused[-1] is False
        ws.send_json({"type": "mute", "muted": True})
        ws.send_json({"type": "mute", "muted": True})
        c.portal.call(asyncio.sleep, 0.05)
        assert ears.paused[-1] is True and server._mic_muted
        ws.send_json({"type": "mute", "muted": False})
        c.portal.call(asyncio.sleep, 0.05)
        assert ears.paused[-1] is False


def test_with_headphones_he_can_be_heard_over(server_local, monkeypatch):
    c, server, ears = server_local
    monkeypatch.setenv("STT_LISTEN_WHILE_SPEAKING", "1")
    server._auvren_speaking = True
    server._apply_ears_pause()
    assert ears.paused[-1] is False


def test_the_wake_engine_cuts_him_off_mid_sentence(server_local, monkeypatch):
    c, server, ears = server_local
    monkeypatch.setattr(server.wake_gate, "enabled", True)
    stopped = []

    async def barge_in(keep_unread=True, reason=""):
        stopped.append((keep_unread, reason))
    monkeypatch.setattr(server.speech, "barge_in", barge_in)
    monkeypatch.setattr(type(server.speech), "is_speaking", property(lambda self: True))
    c.portal.call(server._wake_up, "audio", 0.97)
    assert stopped == [(False, "wake word")]
    server._mic_muted = True
    c.portal.call(server._wake_up, "audio", 0.97)
    assert len(stopped) == 1, "muted means muted"


def test_without_the_wake_engine_the_ears_transcribe_everything(server_local, monkeypatch):
    c, server, ears = server_local
    monkeypatch.setattr(server.wake_gate, "enabled", True)
    monkeypatch.setattr(server, "wake_listener", None)
    assert server._ears_should_transcribe() is True, "the words ARE the wake word"
    monkeypatch.setattr(server, "wake_listener", SimpleNamespace(stop=lambda: None))
    monkeypatch.setattr(server.wakeword, "engine_trusted", lambda: True)
    server.wake_gate.close()
    assert server._ears_should_transcribe() is False
    server.wake_gate.open()
    assert server._ears_should_transcribe() is True


def test_the_recogniser_is_primed_with_the_name_and_the_projects(server_local, monkeypatch):
    c, server, ears = server_local
    monkeypatch.setattr(server, "cached_projects", [{"name": "sitearc"}, {"name": "tea-timer"}])
    words = server._stt_vocabulary()
    assert words[0] == "Auvren" and "sitearc" in words and "tea-timer" in words


def test_if_the_ears_will_not_start_the_browser_listens(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    import server
    importlib.reload(server)
    monkeypatch.setattr(server, "STT_MODE", "local")
    monkeypatch.setattr(server.stt, "engine_problem", lambda: "not installed")
    monkeypatch.setattr(server.wakeword, "engine_problem", lambda: "not installed")
    from fastapi.testclient import TestClient
    with TestClient(server.app, headers={"Origin": "http://localhost:5173"}) as c:
        assert server.STT_MODE == "browser" and server.local_ears is None
        with c.websocket_connect("/ws/voice") as ws:
            assert ws.receive_json()["stt"] == "browser"


def test_a_self_trained_engine_does_not_stop_the_transcript_wake(server_local, monkeypatch):
    c, server, ears = server_local
    monkeypatch.setattr(server.wake_gate, "enabled", True)
    monkeypatch.setattr(server, "wake_listener", SimpleNamespace(stop=lambda: None))
    server.wake_gate.close()
    monkeypatch.setattr(server.wakeword, "engine_trusted", lambda: False)
    assert server._ears_should_transcribe() is True, "it may miss your voice: Whisper backs it up"
    monkeypatch.setattr(server.wakeword, "engine_trusted", lambda: True)
    assert server._ears_should_transcribe() is False


def test_saying_his_own_name_does_not_wake_him(server_local, monkeypatch):
    c, server, ears = server_local
    monkeypatch.setattr(server.wake_gate, "enabled", True)
    monkeypatch.setattr(server, "voice_clients", {object()})       # no summon
    server.wake_gate.close()
    server._note_own_name("Auvren at your service, sir.")
    c.portal.call(server._wake_up, "audio", 0.97)
    assert not server.wake_gate.awake, "that was his own voice"
    c.portal.call(server._wake_up, "transcript")
    assert server.wake_gate.awake, "only the engine's wakes are suspect"
    server.wake_gate.close()
    monkeypatch.setattr(server, "_name_spoken_until", 0.0)
    c.portal.call(server._wake_up, "audio", 0.97)
    assert server.wake_gate.awake


def test_real_synthesis_notes_his_own_name(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    import server
    importlib.reload(server)

    async def synth(text, **kw):
        return SimpleNamespace(audio=b"x", first_byte_sec=0.0, total_sec=0.0)
    monkeypatch.setattr(server.tts, "synthesize", synth)
    monkeypatch.setattr(server.tts, "voice_problem", lambda: None)
    asyncio.run(server._synth_for_speech("Nothing to see here."))
    assert server._name_spoken_until == 0.0
    asyncio.run(server._synth_for_speech("Auvren, at your service."))
    assert server._name_spoken_until > time.monotonic() + 15
