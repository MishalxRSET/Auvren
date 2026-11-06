"""tray.py — Auvren in the Windows notification area.

Running in the background (`server.py --background`, started at sign-in),
Auvren has no window until he is called. The tray icon is how he can be seen
and reached in between:

* the icon itself says his state at a glance — asleep (a dim ring), awake
  and listening (bright), thinking or speaking (a white core), microphone
  muted (grey, struck through), or down (red: Ollama not running, say) —
  and its tooltip says what to say to wake him;
* the menu: Open Auvren (also a double-click), Dashboard, Listen now (no wake
  word needed), Mute microphone, Start with Windows, Show log, Quit.

pystray draws it (its Windows backend is Shell_NotifyIcon over ctypes) and
Pillow paints the icon, in code, so there is no image file to ship. Both are
optional: requirements-tray.txt. `Icon.run_detached()` runs the tray's own
message loop on its own thread on Windows; menu callbacks run THERE, and the
server hands them to its event loop.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import winplat

log = logging.getLogger("jarvis.tray")

STATES = ("asleep", "awake", "thinking", "speaking", "muted", "down")

# (ring, core) colours, RGBA. The accent is the orb's own #4ca8e8.
_PALETTE = {
    "asleep":   ((76, 168, 232, 120), (76, 168, 232, 60)),
    "awake":    ((76, 168, 232, 255), (140, 205, 250, 220)),
    "thinking": ((76, 168, 232, 255), (255, 255, 255, 200)),
    "speaking": ((140, 205, 250, 255), (255, 255, 255, 255)),
    "muted":    ((130, 130, 140, 220), (90, 90, 100, 160)),
    "down":     ((225, 72, 72, 255), (225, 72, 72, 120)),
}


def setting_enabled() -> bool:
    return (os.getenv("AUVREN_TRAY") or "1").strip().lower() not in ("0", "false", "no", "off")


def tray_problem() -> Optional[str]:
    """None if a tray icon can be shown, else why not."""
    if not winplat.IS_WINDOWS:
        return "the tray icon is Windows-only"
    try:
        import PIL.Image  # noqa: F401
        import pystray  # noqa: F401
    except ImportError as e:
        return f"the tray icon is not installed ({e.name}): pip install -r requirements-tray.txt"
    return None


def tooltip(state: str, phrase: str = "Hey Auvren", detail: str = "") -> str:
    text = {
        "asleep": f"Auvren — asleep. Say “{phrase}”",
        "awake": "Auvren — listening",
        "thinking": "Auvren — thinking",
        "speaking": "Auvren — speaking",
        "muted": "Auvren — microphone muted",
        "down": f"Auvren — {detail or 'not working'}",
    }.get(state, "Auvren")
    # A Windows tooltip holds 127 characters; a longer one is refused.
    return text if len(text) <= 120 else text[:119] + "…"


def draw_icon(state: str, size: int = 64):
    """The orb, as a tray icon: a ring and a core, coloured by state.

    Drawn at four times the size and scaled down: Pillow does not
    anti-alias an ellipse, and a tray icon is mostly edge."""
    from PIL import Image
    big = _draw(state, size * 4)
    return big.resize((size, size), Image.LANCZOS)


def _draw(state: str, size: int):
    from PIL import Image, ImageDraw
    ring, core = _PALETTE.get(state, _PALETTE["asleep"])
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = size // 16
    width = max(2, size // 9)
    d.ellipse((pad, pad, size - pad - 1, size - pad - 1), outline=ring, width=width)
    c = size // 2
    r = size // 5 if state not in ("thinking", "speaking") else size // 4
    d.ellipse((c - r, c - r, c + r, c + r), fill=core)
    if state == "muted":
        d.line((pad + width, size - pad - width, size - pad - width, pad + width),
               fill=(230, 230, 235, 255), width=max(2, size // 12))
    return img


# --- Start with Windows ----------------------------------------------------------------

def startup_shortcut() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return (Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
            / "Startup" / "Auvren.lnk")


def autostart_enabled() -> bool:
    return startup_shortcut().is_file()


# The same shortcut scripts/install_autostart.ps1 makes. A fixed script; the
# paths reach it only as environment variables, never as script text.
_SHORTCUT_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:AUVREN_LNK)
$s.TargetPath = $env:AUVREN_TARGET
$s.Arguments = $env:AUVREN_ARGS
$s.WorkingDirectory = $env:AUVREN_DIR
$s.Description = 'Auvren voice assistant (background)'
$s.WindowStyle = 7
$s.Save()
"""


def set_autostart(enabled: bool, repo: Path, pythonw: Optional[str] = None,
                  run=subprocess.run) -> tuple[bool, str]:
    """Add or remove the Startup-folder shortcut. (ok, sentence)."""
    link = startup_shortcut()
    if not enabled:
        try:
            link.unlink(missing_ok=True)
        except OSError as e:
            return False, f"Could not remove {link.name}: {e}"
        return True, "Auvren will no longer start with Windows."
    pythonw = pythonw or str(Path(sys.executable).with_name("pythonw.exe"))
    if not Path(pythonw).is_file():
        return False, f"pythonw.exe was not found next to {sys.executable}."
    link.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AUVREN_LNK=str(link), AUVREN_TARGET=pythonw,
               AUVREN_ARGS=f'"{repo / "server.py"}" --background', AUVREN_DIR=str(repo))
    try:
        proc = run([winplat.powershell_exe(), "-NoProfile", "-NonInteractive",
                    "-ExecutionPolicy", "Bypass", "-Command", "-"],
                   input=_SHORTCUT_SCRIPT.encode("utf-8"), env=env, capture_output=True,
                   timeout=20, creationflags=winplat.no_window_flag())
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Could not create the shortcut: {e}"
    if proc.returncode != 0:
        return False, f"Could not create the shortcut: {proc.stderr.decode(errors='replace')[:200]}"
    return True, "Auvren will start in the background when you sign in."


# --- the icon ---------------------------------------------------------------------------

@dataclass
class TrayActions:
    """What the menu does. Called on the TRAY's thread."""
    open: Callable[[], None]
    dashboard: Callable[[], None]
    listen: Callable[[], None]
    toggle_mute: Callable[[], None]
    toggle_autostart: Callable[[], None]
    show_log: Callable[[], None]
    quit: Callable[[], None]
    is_muted: Callable[[], bool]
    is_autostart: Callable[[], bool]


class Tray:
    def __init__(self, actions: TrayActions, *, phrase: str = "Hey Auvren",
                 backend=None, draw: Callable = draw_icon):
        self.actions = actions
        self.phrase = phrase
        self._backend = backend
        self._draw = draw
        self._icon = None
        self.state = "asleep"
        self.detail = ""

    def _pystray(self):
        if self._backend is None:
            import pystray
            self._backend = pystray
        return self._backend

    def menu(self):
        ps = self._pystray()
        a = self.actions

        def call(fn):
            def handler(icon=None, item=None):
                try:
                    fn()
                except Exception as e:
                    log.warning(f"tray: menu action failed: {e}")
            return handler

        return ps.Menu(
            ps.MenuItem("Open Auvren", call(a.open), default=True),
            ps.MenuItem("Dashboard", call(a.dashboard)),
            ps.MenuItem("Listen now", call(a.listen)),
            ps.Menu.SEPARATOR,
            ps.MenuItem("Mute microphone", call(a.toggle_mute),
                        checked=lambda item: bool(a.is_muted())),
            ps.MenuItem("Start with Windows", call(a.toggle_autostart),
                        checked=lambda item: bool(a.is_autostart())),
            ps.MenuItem("Show log", call(a.show_log)),
            ps.Menu.SEPARATOR,
            ps.MenuItem("Quit Auvren", call(a.quit)),
        )

    def start(self) -> Optional[str]:
        """Show the icon. None on success, else why not."""
        if self._icon is not None:
            return None
        try:
            ps = self._pystray()
            self._icon = ps.Icon("auvren", self._draw(self.state),
                                 tooltip(self.state, self.phrase, self.detail), self.menu())

            def setup(icon):
                icon.visible = True
            self._icon.run_detached(setup)
        except Exception as e:
            self._icon = None
            return f"the tray icon would not start: {e}"
        log.info("tray icon shown")
        return None

    def set_state(self, state: str, detail: str = "") -> None:
        """Repaint for `state`. Safe from any thread; a no-op if unchanged."""
        if state == self.state and detail == self.detail:
            return
        self.state, self.detail = state, detail
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = self._draw(state)
            icon.title = tooltip(state, self.phrase, detail)
        except Exception as e:
            log.warning(f"tray: repaint failed: {e}")

    def refresh_menu(self) -> None:
        if self._icon is not None:
            try:
                self._icon.update_menu()
            except Exception:
                pass

    def notify(self, message: str) -> None:
        icon = self._icon
        if icon is None or not getattr(icon, "HAS_NOTIFICATION", True):
            return
        try:
            icon.notify(message, "Auvren")
        except Exception as e:
            log.warning(f"tray: notification failed: {e}")

    def stop(self) -> None:
        icon, self._icon = self._icon, None
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
