"""Being summoned: the wake word with no window open opens Auvren's own
window, which plays its intro and asks — or answers what was said in the same
breath. No browser is launched here: actions.open_app_window is faked, and
its argv is tested on its own."""
import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import actions


# --- the window itself ----------------------------------------------------------------

@pytest.fixture
def chrome(monkeypatch):
    found = {"chrome": r"C:\Chrome\chrome.exe", "edge": r"C:\Edge\msedge.exe"}
    monkeypatch.setattr(actions, "browser_exe", lambda b: found.get(b))
    return found


def test_the_app_window_is_its_own_chromium_profile(chrome):
    argv = actions.app_window_argv("http://127.0.0.1:8340/?summoned=1", r"C:\data\app-window")
    assert argv[0] == chrome["chrome"]
    assert "--app=http://127.0.0.1:8340/?summoned=1" in argv
    assert r"--user-data-dir=C:\data\app-window" in argv
    assert "--autoplay-policy=no-user-gesture-required" in argv, "so it can ask out loud"


def test_edge_when_there_is_no_chrome(chrome):
    chrome["chrome"] = None
    argv = actions.app_window_argv("http://localhost:5173/?summoned=1", "p")
    assert argv[0] == chrome["edge"]


def test_neither_browser_means_no_window(chrome):
    chrome["chrome"] = chrome["edge"] = None
    assert actions.app_window_argv("http://localhost:5173/", "p") is None


@pytest.mark.parametrize("url", [
    "https://example.com/",
    "http://127.0.0.1.evil.com/",
    "javascript:alert(1)",
    "file:///C:/Windows/system32/calc.exe",
    "http://localhost/ --renderer-cmd-prefix=calc",
    "http://localhost/\n--evil",
    "",
])
def test_only_a_loopback_page_is_ever_opened(chrome, url):
    assert actions.app_window_argv(url, "p") is None


# --- the server ----------------------------------------------------------------------------

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


@pytest.fixture
def summoner(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WAKE_WORD", raising=False)
    monkeypatch.delenv("AUVREN_SUMMON", raising=False)
    import data_paths
    importlib.reload(data_paths)
    import run_store
    importlib.reload(run_store)
    import server
    importlib.reload(server)
    run_store.init_db()
    monkeypatch.setattr(server.wakeword, "engine_problem", lambda: "not under test")
    opened = []

    async def open_app_window(url, profile_dir):
        opened.append((url, profile_dir))
        return {"success": True, "confirmation": "Here I am, sir."}

    async def fake_synth(text):
        return b"MP3:" + text.encode()

    async def no_preflight():
        return None
    monkeypatch.setattr(server.actions, "open_app_window", open_app_window)
    monkeypatch.setattr(server, "_synth_for_speech", fake_synth)
    monkeypatch.setattr(server, "_run_preflight", no_preflight)
    monkeypatch.setattr(server, "SUMMON_HELLO_GRACE_SEC", 30.0)
    server._last_greeting_time = time.time()
    with TestClient(server.app, headers={"Origin": "http://localhost:5173"}) as c:
        server.brain_instance = FakeBrain()
        assert server.wake_gate.enabled and not server.voice_clients
        yield c, server, opened


def _until(ws, pred, limit=12):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if pred(m):
            return seen
    raise AssertionError(seen)


def test_the_wake_word_with_no_window_opens_one(summoner, tmp_path):
    c, server, opened = summoner
    c.portal.call(server._wake_up, "audio", 0.95)
    c.portal.call(server._wake_up, "audio", 0.95)
    assert len(opened) == 1, "one window, however often it is said"
    url, profile = opened[0]
    assert url.endswith("?summoned=1")
    assert Path(profile) == tmp_path / "app-window"
    assert server.wake_gate.awake and server.wake_gate.seconds_left > 30, \
        "still listening while the window loads"


def test_the_window_plays_its_intro_then_asks(summoner):
    c, server, opened = summoner
    c.portal.call(server._on_final_transcript, "Hey Auvren", "local")
    assert len(opened) == 1
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"type": "hello", "summoned": True})
        audio = _until(ws, lambda m: m.get("type") == "audio")[-1]
        assert audio["text"] == "Yes, sir?"
    assert server.brain_instance.turns == []


def test_a_request_in_the_same_breath_is_answered_once_the_window_is_up(summoner):
    c, server, opened = summoner
    c.portal.call(server._on_final_transcript, "Hey Auvren, what time is it?", "local")
    assert len(opened) == 1, "one window, carrying the request"
    assert server.brain_instance.turns == [], "nobody to say it to yet"
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"type": "hello", "summoned": True})
        audio = _until(ws, lambda m: m.get("type") == "audio")[-1]
        assert audio["text"] == "Echo: what time is it?."
    assert server.brain_instance.turns == ["what time is it?"]


def test_a_page_that_never_says_hello_is_asked_anyway(summoner, monkeypatch):
    c, server, opened = summoner
    monkeypatch.setattr(server, "SUMMON_HELLO_GRACE_SEC", 0.05)
    c.portal.call(server._wake_up, "audio", 0.9)
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        audio = _until(ws, lambda m: m.get("type") == "audio")[-1]
        assert audio["text"] == "Yes, sir?"


def test_with_a_window_open_nothing_is_launched(summoner):
    c, server, opened = summoner
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        c.portal.call(server._wake_up, "audio", 0.9)
        _until(ws, lambda m: m.get("type") == "wake")
    assert opened == []


def test_it_can_be_turned_off(summoner, monkeypatch):
    c, server, opened = summoner
    monkeypatch.setenv("AUVREN_SUMMON", "0")
    c.portal.call(server._wake_up, "audio", 0.9)
    assert opened == []


def test_a_window_that_will_not_open_is_not_waited_for(summoner, monkeypatch):
    c, server, opened = summoner

    async def broken(url, profile_dir):
        return {"success": False, "confirmation": "no browser"}
    monkeypatch.setattr(server.actions, "open_app_window", broken)
    c.portal.call(server._on_final_transcript, "Hey Auvren, open my notes", "local")
    assert not server._summon_pending() and server._summon["request"] is None
    tried = []

    async def works(url, profile_dir):
        tried.append(url)
        return {"success": True, "confirmation": "ok"}
    monkeypatch.setattr(server.actions, "open_app_window", works)
    c.portal.call(server._wake_up, "audio", 0.9)
    assert tried, "a failure does not hold the next wake word off"


def test_where_the_window_points(summoner, monkeypatch):
    c, server, _ = summoner
    monkeypatch.setenv("AUVREN_APP_URL", "http://127.0.0.1:9000/")
    assert server._app_url() == "http://127.0.0.1:9000/?summoned=1"
    monkeypatch.delenv("AUVREN_APP_URL")
    monkeypatch.setattr(server, "FRONTEND_DIST", Path("/nonexistent"))
    assert server._app_url() == "http://localhost:5173/?summoned=1"
    monkeypatch.setattr(server, "FRONTEND_DIST", Path(__file__))   # "exists"
    monkeypatch.setenv("JARVIS_SCHEME", "http")
    monkeypatch.setenv("JARVIS_PORT", "8340")
    assert server._app_url() == "http://127.0.0.1:8340/?summoned=1"


def test_background_logging_catches_a_missing_console(monkeypatch, tmp_path):
    import logging
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    import server
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        server._start_background_logging()
        print("stray print under pythonw")
        logging.getLogger("jarvis").warning("logged in the background")
        assert (tmp_path / "logs" / "auvren.log").read_text().count("logged in the background") == 1
        sys.stdout.flush()
        assert "stray print" in (tmp_path / "logs" / "auvren.out").read_text()
    finally:
        for h in root.handlers[:]:
            if h not in before:
                root.removeHandler(h)
                h.close()
        out = sys.stdout
        if out is not None:
            out.close()
