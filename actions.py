"""
Auvren Action Executor — Windows system actions.

Opens Windows Terminal, the browser and the editor. Each function returns
{"success": bool, "confirmation": str}.

The one rule for this file: **every program is exec'd directly, with its
arguments as separate argv entries, and never through `cmd.exe`.** The text
reaching these functions comes from a model, out of speech, possibly echoing a
web page or a README. `cmd.exe` re-parses a command line with its own
metacharacters (`&`, `|`, `%VAR%`) and quoting rules that Python's
`list2cmdline` does not model, so anything routed through it — a `.cmd` shim,
`start`, `shell=True` — is a path from that text to a shell. That is also why
VS Code is launched as `Code.exe`, not through its `code.cmd` shim, and why
the fallbacks are `notepad.exe` / `explorer.exe` rather than `os.startfile`
(which would RUN a `.py` or `.bat` file associated with an interpreter instead
of opening it for reading).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import winplat

log = logging.getLogger("jarvis.actions")


async def _spawn(argv: list[str], *, cwd: str | None = None,
                 new_console: bool = False) -> tuple[bool, str]:
    """Start a GUI/console program and return (started, error).

    Launchers like Code.exe and chrome.exe hand off to an already-running
    instance and exit, so "started" means the exec succeeded and the process
    did not exit non-zero within a short window — not that a window is now
    visible. Output goes to DEVNULL, never a pipe: a long-lived window
    writing into a pipe nobody drains would eventually block on it.
    """
    flags = 0
    if winplat.IS_WINDOWS:
        flags = (subprocess.CREATE_NEW_CONSOLE if new_console
                 else subprocess.DETACHED_PROCESS)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=flags,
        )
    except OSError as e:
        return False, str(e)
    try:
        code = await asyncio.wait_for(proc.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        return True, ""          # still running: a long-lived window, fine
    if code != 0:
        return False, f"{Path(argv[0]).name} exited {code}"
    return True, ""


# --- Windows Terminal --------------------------------------------------------

def _windows_terminal() -> str | None:
    """`wt.exe`, the Windows Terminal app-execution alias, or None."""
    found = shutil.which("wt.exe") or shutil.which("wt")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        alias = Path(local) / "Microsoft" / "WindowsApps" / "wt.exe"
        if alias.exists():
            return str(alias)
    return None


def _wt_escape(arg: str) -> str:
    """`wt` splits its own command line on `;` into subcommands. Escape it."""
    return arg.replace(";", r"\;")


def terminal_argv(cwd: str | None, command: str | None) -> tuple[list[str], bool]:
    """The argv that opens a terminal in `cwd` running `command`.

    Returns (argv, needs_new_console). Windows Terminal when installed,
    otherwise a plain PowerShell console. The shell is always PowerShell with
    -NoExit, so the window stays open and the user can see what ran and stop
    it. `command` has already passed `builds.command_problem`, whose
    allowlist admits none of PowerShell's metacharacters (`$`, backtick,
    `;`, `&`, `|`, parentheses, braces).
    """
    shell = [winplat.powershell_exe(), "-NoLogo", "-NoExit"]
    if command:
        shell += ["-Command", command]
    wt = _windows_terminal()
    if wt:
        argv = [wt, "new-tab"]
        if cwd:
            argv += ["-d", _wt_escape(cwd)]
        argv += [_wt_escape(a) for a in shell]
        return argv, False
    return shell, True


async def open_terminal(cwd: str | None = None, command: str | None = None) -> dict:
    """Open a terminal window in `cwd`, optionally running one vetted command."""
    if cwd and not Path(cwd).is_dir():
        return {"success": False,
                "confirmation": "That folder isn't there any more, sir."}
    argv, new_console = terminal_argv(cwd, command)
    ok, err = await _spawn(argv, cwd=cwd or None, new_console=new_console)
    if not ok:
        log.error(f"open_terminal failed: {err}")
    return {
        "success": ok,
        "confirmation": "Terminal is open, sir." if ok
        else "I had trouble opening a terminal, sir.",
    }


# --- browsers ----------------------------------------------------------------

_BROWSER_EXES = {
    "chrome": (r"Google\Chrome\Application\chrome.exe", "Chrome"),
    "firefox": (r"Mozilla Firefox\firefox.exe", "Firefox"),
}

# A URL becomes ONE argv entry of chrome.exe/firefox.exe. Both treat an entry
# starting with `-` as a switch, and Chrome has switches that run programs
# (`--renderer-cmd-prefix`, `--utility-cmd-prefix`). Requiring a scheme
# prefix makes that impossible: nothing starting `http` or `file:` is a switch.
_OPENABLE = re.compile(r"^(https?://|file:///)", re.IGNORECASE)


def browser_exe(browser: str) -> str | None:
    """The browser's own executable, from the standard install locations."""
    rel, _name = _BROWSER_EXES.get(browser, _BROWSER_EXES["chrome"])
    roots = [os.environ.get(v) for v in
             ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    for root in roots:
        if root:
            exe = Path(root) / rel
            if exe.is_file():
                return str(exe)
    return shutil.which(Path(rel).name)


async def open_browser(url: str, browser: str = "chrome") -> dict:
    """Open a web or file URL in Chrome or Firefox."""
    browser = "firefox" if browser.lower() == "firefox" else "chrome"
    app_name = _BROWSER_EXES[browser][1]
    if not _OPENABLE.match(url or "") or any(c in url for c in "\r\n\x00"):
        return {"success": False,
                "confirmation": "I only open web addresses and files, sir."}
    exe = browser_exe(browser)
    if exe is None:
        return {"success": False,
                "confirmation": f"I can't find {app_name} on this machine, sir."}
    ok, err = await _spawn([exe, url])
    if not ok:
        log.error(f"open_browser ({app_name}) failed: {err}")
    return {
        "success": ok,
        "confirmation": f"Pulled that up in {app_name}, sir." if ok
        else f"{app_name} ran into a problem, sir.",
    }


async def open_chrome(url: str) -> dict:
    return await open_browser(url, "chrome")


# --- Opening code where the user actually reads it ---------------------------
#
# VS Code first when it is installed, a plain reader otherwise. Containment and
# the sensitive-file wall are the CALLER's job and have already run by the time
# this is reached — see server.tool_open_in_editor.

def _vscode_exe() -> str | None:
    """VS Code's `Code.exe`, never its `bin\\code.cmd` shim (see module doc)."""
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "Microsoft VS Code" / "Code.exe")
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(var)
        if root:
            candidates.append(Path(root) / "Microsoft VS Code" / "Code.exe")
    shim = shutil.which("code")
    if shim:
        # <install>\bin\code.cmd -> <install>\Code.exe
        candidates.append(Path(shim).resolve().parent.parent / "Code.exe")
    for exe in candidates:
        if exe.is_file():
            return str(exe)
    return None


def _vscode_command(path: str) -> list[str] | None:
    """The argv that opens `path` in VS Code, or None if it is not installed."""
    exe = _vscode_exe()
    return [exe, str(path)] if exe else None


def _fallback_command(path: str) -> tuple[list[str], str]:
    """Explorer for a folder, Notepad for a file — both only display."""
    if Path(path).is_dir():
        return [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"),
                str(path)], "Explorer"
    return [winplat.system32_exe("notepad.exe"), str(path)], "Notepad"


async def open_in_editor(path: str) -> dict:
    """Open a file or directory in VS Code, else Notepad / Explorer."""
    path = str(path)
    if path.startswith("-"):
        # An argv entry starting `-` is a switch to Code.exe. The caller hands
        # us resolved absolute paths, so this never fires; it stays as a wall.
        return {"success": False, "editor": "your editor",
                "confirmation": "I couldn't open an editor, sir."}
    argv = _vscode_command(path)
    editor = "VS Code"
    if argv is None:
        argv, editor = _fallback_command(path)

    ok, err = await _spawn(argv)
    if not ok:
        log.error(f"open_in_editor failed: {err}")
    return {
        "success": ok,
        "editor": editor,
        "confirmation": f"Opened that in {editor}, sir." if ok
        else f"{editor} wouldn't open that, sir.",
    }


def _generate_project_name(prompt: str) -> str:
    """Generate a kebab-case project folder name from the prompt."""
    # First: check for a quoted name like "tiktok-analytics-dashboard"
    quoted = re.search(r'"([^"]+)"', prompt)
    if quoted:
        name = quoted.group(1).strip()
        # Already kebab-case or close to it
        name = re.sub(r"[^a-zA-Z0-9\s-]", "", name).strip()
        if name:
            return re.sub(r"[\s]+", "-", name.lower())

    # Second: check for "called X" or "named X" pattern
    called = re.search(r'(?:called|named)\s+(\S+(?:[-_]\S+)*)', prompt, re.IGNORECASE)
    if called:
        name = re.sub(r"[^a-zA-Z0-9-]", "", called.group(1))
        if len(name) > 3:
            return name.lower()

    # Fallback: extract meaningful words
    words = re.sub(r"[^a-zA-Z0-9\s]", "", prompt.lower()).split()
    skip = {"a", "the", "an", "me", "build", "create", "make", "for", "with", "and",
            "to", "of", "i", "want", "need", "new", "project", "directory", "called",
            "on", "desktop", "that", "application", "app", "full", "stack", "simple",
            "web", "page", "site", "named"}
    meaningful = [w for w in words if w not in skip and len(w) > 2][:4]
    return "-".join(meaningful) if meaningful else "auvren-project"
