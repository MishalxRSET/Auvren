"""The tray icon: tray.py against a fake pystray with the real API shape
(Icon(name, image, title, menu), run_detached(setup), MenuItem(text, action,
checked=, default=), Menu.SEPARATOR, notify, update_menu, stop), and the
server's menu actions, called from another thread as pystray calls them."""
import asyncio
import importlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import tray


# --- a fake pystray ----------------------------------------------------------------------

class FakeItem:
    def __init__(self, text, action, checked=None, default=False, **kw):
        self.text, self.action, self.checked, self.default = text, action, checked, default


class FakeMenu:
    SEPARATOR = FakeItem("- - - -", None)

    def __init__(self, *items):
        self.items = list(items)


class FakeIcon:
    HAS_NOTIFICATION = True
    made = []

    def __init__(self, name, icon=None, title=None, menu=None):
        self.name, self.icon, self.title, self.menu = name, icon, title, menu
        self.visible = False
        self.notes, self.menu_updates, self.stopped = [], 0, False
        FakeIcon.made.append(self)

    def run_detached(self, setup=None):
        setup(self)

    def notify(self, message, title=None):
        self.notes.append((title, message))

    def update_menu(self):
        self.menu_updates += 1

    def stop(self):
        self.stopped = True


fake_pystray = SimpleNamespace(Icon=FakeIcon, Menu=FakeMenu, MenuItem=FakeItem)


def actions(**over):
    calls = []
    state = {"muted": False, "auto": True}

    def rec(name):
        return lambda: calls.append(name)
    a = tray.TrayActions(
        open=rec("open"), dashboard=rec("dashboard"), listen=rec("listen"),
        toggle_mute=rec("mute"), toggle_autostart=rec("autostart"),
        show_log=rec("log"), quit=rec("quit"),
        is_muted=lambda: state["muted"], is_autostart=lambda: state["auto"])
    for k, v in over.items():
        setattr(a, k, v)
    return a, calls, state


def items(icon):
    return [i for i in icon.menu.items if i is not FakeMenu.SEPARATOR]


# --- the module -------------------------------------------------------------------------

def test_tooltips_say_what_to_say():
    assert tray.tooltip("asleep", "Hey Auvren") == "Auvren — asleep. Say “Hey Auvren”"
    assert tray.tooltip("down", detail="Ollama is not running").endswith("Ollama is not running")
    assert len(tray.tooltip("down", detail="x" * 500)) <= 120, "Windows refuses > 127"


def test_every_state_draws_an_icon():
    pytest.importorskip("PIL")
    seen = set()
    for state in tray.STATES:
        img = tray.draw_icon(state)
        assert img.size == (64, 64) and img.mode == "RGBA"
        seen.add(img.tobytes())
    assert len(seen) == len(tray.STATES), "each state looks different"


def test_tray_problem(monkeypatch):
    monkeypatch.setattr(tray.winplat, "IS_WINDOWS", False)
    assert "Windows-only" in tray.tray_problem()
    monkeypatch.setattr(tray.winplat, "IS_WINDOWS", True)
    monkeypatch.setitem(sys.modules, "pystray", None)
    assert "requirements-tray.txt" in tray.tray_problem()


def test_start_shows_the_icon_with_its_menu():
    a, calls, state = actions()
    t = tray.Tray(a, backend=fake_pystray, draw=lambda s: f"img:{s}")
    assert t.start() is None
    icon = FakeIcon.made[-1]
    assert icon.visible and icon.icon == "img:asleep" and "asleep" in icon.title
    texts = [i.text for i in items(icon)]
    assert texts == ["Open Auvren", "Dashboard", "Listen now", "Mute microphone",
                     "Start with Windows", "Show log", "Quit Auvren"]
    assert [i.text for i in items(icon) if i.default] == ["Open Auvren"], "double-click opens"
    for item in items(icon):
        item.action(icon, item)
    assert calls == ["open", "dashboard", "listen", "mute", "autostart", "log", "quit"]
    by = {i.text: i for i in items(icon)}
    assert by["Mute microphone"].checked(None) is False
    state["muted"] = True
    assert by["Mute microphone"].checked(None) is True
    assert by["Start with Windows"].checked(None) is True


def test_a_failing_menu_action_does_not_take_the_tray_down():
    def boom():
        raise RuntimeError("loop gone")
    a, calls, _ = actions(open=boom)
    t = tray.Tray(a, backend=fake_pystray, draw=lambda s: s)
    t.start()
    item = items(FakeIcon.made[-1])[0]
    item.action(None, item)          # no exception


def test_repaint_only_on_change_and_stop():
    a, _, _ = actions()
    drawn = []
    t = tray.Tray(a, backend=fake_pystray, draw=lambda s: drawn.append(s) or s)
    t.start()
    icon = FakeIcon.made[-1]
    t.set_state("awake")
    t.set_state("awake")
    t.set_state("down", "Ollama is not running")
    assert drawn == ["asleep", "awake", "down"]
    assert icon.title.endswith("Ollama is not running")
    t.notify("hello")
    assert icon.notes == [("Auvren", "hello")]
    t.stop()
    assert icon.stopped
    t.set_state("asleep")             # after stop: no-op, no error


def test_start_says_why_it_cannot():
    class Broken(FakeIcon):
        def run_detached(self, setup=None):
            raise OSError("no shell")
    t = tray.Tray(actions()[0], backend=SimpleNamespace(Icon=Broken, Menu=FakeMenu,
                                                        MenuItem=FakeItem), draw=lambda s: s)
    assert "would not start" in t.start()


# --- Start with Windows -------------------------------------------------------------------

def test_the_shortcut_lives_in_the_startup_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    link = tray.startup_shortcut()
    assert link == tmp_path / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Auvren.lnk"
    assert not tray.autostart_enabled()
    link.parent.mkdir(parents=True)
    link.write_bytes(b"lnk")
    assert tray.autostart_enabled()
    ok, _ = tray.set_autostart(False, repo=tmp_path)
    assert ok and not link.exists()


def test_making_the_shortcut_passes_paths_as_data(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    pythonw = tmp_path / "pythonw.exe"
    pythonw.write_bytes(b"")
    repo = tmp_path / "repo with spaces & stuff"
    seen = {}

    def run(argv, input, env, **kw):
        seen.update(argv=argv, script=input.decode(), env=env)
        return SimpleNamespace(returncode=0, stderr=b"")
    ok, sentence = tray.set_autostart(True, repo=repo, pythonw=str(pythonw), run=run)
    assert ok and "sign in" in sentence
    assert seen["argv"][-2:] == ["-Command", "-"]
    assert str(repo) not in seen["script"], "never spliced into the script"
    assert seen["env"]["AUVREN_TARGET"] == str(pythonw)
    assert seen["env"]["AUVREN_ARGS"] == f'"{repo / "server.py"}" --background'
    assert seen["env"]["AUVREN_LNK"].endswith("Auvren.lnk")


def test_no_pythonw_no_shortcut(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    ok, sentence = tray.set_autostart(True, repo=tmp_path, pythonw=str(tmp_path / "nope.exe"))
    assert not ok and "pythonw" in sentence


# --- the server ------------------------------------------------------------------------------

@pytest.fixture
def server_tray(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUVREN_TRAY", "1")
    monkeypatch.delenv("WAKE_WORD", raising=False)
    import data_paths
    importlib.reload(data_paths)
    import server
    importlib.reload(server)
    made = []

    class FakeTray:
        def __init__(self, actions, phrase=""):
            self.actions, self.phrase = actions, phrase
            self.states, self.notes, self.menu_refreshes, self.stopped = [], [], 0, False
            made.append(self)

        def start(self):
            return None

        def set_state(self, state, detail=""):
            self.states.append((state, detail))

        def notify(self, msg):
            self.notes.append(msg)

        def refresh_menu(self):
            self.menu_refreshes += 1

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(server.tray, "tray_problem", lambda: None)
    monkeypatch.setattr(server.tray, "Tray", FakeTray)
    monkeypatch.setattr(server.wakeword, "engine_problem", lambda: "not under test")

    async def no_preflight():
        return None
    monkeypatch.setattr(server, "_run_preflight", no_preflight)
    opened = []

    async def open_app_window(url, profile):
        opened.append(url)
        return {"success": True}
    monkeypatch.setattr(server.actions, "open_app_window", open_app_window)
    from fastapi.testclient import TestClient
    with TestClient(server.app, headers={"Origin": "http://localhost:5173"}) as c:
        assert made, "the tray started with the server"
        yield c, server, made[0], opened
    assert made[0].stopped, "and stopped with it"


def from_tray_thread(c, fn, settle=0.1):
    """Menu actions run on pystray's thread, not the event loop."""
    t = threading.Thread(target=fn)
    t.start()
    t.join()
    c.portal.call(asyncio.sleep, settle)


def test_the_icon_follows_the_state(server_tray):
    c, server, t, _ = server_tray
    assert t.states[-1] == ("asleep", "")
    c.portal.call(server._wake_up, "transcript")
    assert t.states[-1] == ("awake", "")
    c.portal.call(server._voice_emit, {"type": "status", "state": "thinking"})
    assert t.states[-1] == ("thinking", "")
    c.portal.call(server._on_brain_state, "failed", {"failure_reason": "ollama_down"})
    assert t.states[-1] == ("down", "Ollama is not running")
    c.portal.call(server._on_brain_state, "ready", {})
    assert t.states[-1][0] != "down"


def test_mute_from_the_tray_mutes_and_tells_the_pages(server_tray):
    c, server, t, _ = server_tray
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        from_tray_thread(c, t.actions.toggle_mute)
        msg = ws.receive_json()
        while msg.get("type") != "mute":
            msg = ws.receive_json()
        assert msg == {"type": "mute", "muted": True}
    assert server._mic_muted and t.states[-1] == ("muted", "") and t.menu_refreshes
    assert t.actions.is_muted() is True


def test_open_dashboard_and_listen(server_tray):
    c, server, t, opened = server_tray
    from_tray_thread(c, t.actions.open)
    from_tray_thread(c, t.actions.dashboard)
    assert opened[0].endswith("/") and "summoned" not in opened[0]
    assert opened[1].endswith("/dashboard") or opened[1].endswith("/dashboard.html")
    from_tray_thread(c, t.actions.listen)
    assert opened[-1].endswith("?summoned=1"), "no window open: listen summons one"


def test_show_log_says_when_there_is_none(server_tray):
    c, server, t, _ = server_tray
    from_tray_thread(c, t.actions.show_log)
    assert t.notes and "No log yet" in t.notes[-1]


def test_quit_is_a_clean_shutdown(server_tray, monkeypatch):
    c, server, t, _ = server_tray
    raised = []
    monkeypatch.setattr(server.signal, "raise_signal", lambda sig: raised.append(sig))
    from_tray_thread(c, t.actions.quit)
    assert raised == [server.signal.SIGINT], "the same path as Ctrl+C"
