"""The Windows seams: winplat.py and actions.py.

Runs on any OS. Win32 calls are faked at the seam; what is tested is the
decision each function makes — which program is exec'd with which argv, and
what is refused.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import actions
import winplat


# --- pid_alive --------------------------------------------------------------

def test_pid_alive_reads_without_signalling(monkeypatch):
    """On Windows `os.kill(pid, 0)` is TerminateProcess. It must never run."""
    monkeypatch.setattr(winplat, "IS_WINDOWS", True)
    seen = []
    monkeypatch.setattr(winplat, "_win_pid_alive", lambda pid: seen.append(pid) or True)

    def _boom(*a):
        raise AssertionError("os.kill called on Windows")
    monkeypatch.setattr(winplat.os, "kill", _boom)

    assert winplat.pid_alive(1234) is True
    assert seen == [1234]


@pytest.mark.parametrize("bad", [None, 0, -1, "x"])
def test_pid_alive_rejects_non_pids(bad):
    assert winplat.pid_alive(bad) is False


def test_session_watch_goes_through_winplat(monkeypatch):
    import session_watch as sw
    monkeypatch.setattr(winplat, "pid_alive", lambda pid: pid == 7)
    assert sw.pid_alive(7) is True
    assert sw.pid_alive(8) is False


def test_kill_tree_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(winplat, "IS_WINDOWS", False)
    assert winplat.kill_tree(1234) is False


def test_kill_tree_uses_system32_taskkill(monkeypatch):
    monkeypatch.setattr(winplat, "IS_WINDOWS", True)
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    calls = []
    monkeypatch.setattr(winplat.subprocess, "run",
                        lambda argv, **kw: calls.append(argv))
    assert winplat.kill_tree(42) is True
    argv = calls[0]
    assert Path(argv[0]).name == "taskkill.exe"
    assert "System32" in argv[0]
    assert argv[1:] == ["/T", "/F", "/PID", "42"]


# --- the claude launcher ----------------------------------------------------

def _npm_prefix(tmp_path, *, cli_js=False, native=False, node=True):
    prefix = tmp_path / "npm"
    pkg = prefix / "node_modules" / "@anthropic-ai" / "claude-code"
    pkg.mkdir(parents=True)
    (prefix / "claude.cmd").write_text("@echo off")
    if cli_js:
        (pkg / "cli.js").write_text("")
    if native:
        (pkg / "bin").mkdir()
        (pkg / "bin" / "claude.exe").write_text("")
    if node:
        (prefix / "node.exe").write_text("")
    return prefix


def test_a_cmd_shim_is_resolved_to_node_and_cli_js(monkeypatch, tmp_path):
    monkeypatch.setattr(winplat, "IS_WINDOWS", True)
    prefix = _npm_prefix(tmp_path, cli_js=True)
    argv = winplat.claude_argv(str(prefix / "claude.cmd"))
    assert argv[0] == str(prefix / "node.exe")
    assert argv[1].endswith("cli.js")


def test_a_cmd_shim_prefers_the_native_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(winplat, "IS_WINDOWS", True)
    prefix = _npm_prefix(tmp_path, cli_js=True, native=True)
    argv = winplat.claude_argv(str(prefix / "claude.cmd"))
    assert argv == [str(prefix / "node_modules" / "@anthropic-ai" / "claude-code"
                        / "bin" / "claude.exe")]


def test_an_opaque_batch_file_is_refused(monkeypatch, tmp_path):
    """Never exec a .cmd: cmd.exe would re-parse the system prompt."""
    monkeypatch.setattr(winplat, "IS_WINDOWS", True)
    shim = tmp_path / "claude.cmd"
    shim.write_text("@echo off")
    with pytest.raises(winplat.ClaudeLaunchError):
        winplat.claude_argv(str(shim))


def test_a_launch_error_is_an_oserror():
    """Every spawn site already handles OSError as "could not start"."""
    assert issubclass(winplat.ClaudeLaunchError, OSError)


def test_a_windows_path_override_keeps_its_backslashes(monkeypatch, tmp_path):
    monkeypatch.setattr(winplat, "IS_WINDOWS", True)
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    assert winplat.claude_argv(str(exe)) == [str(exe)]


def test_posix_override_still_splits(monkeypatch):
    monkeypatch.setattr(winplat, "IS_WINDOWS", False)
    assert winplat.claude_argv("python fake_claude.py") == ["python", "fake_claude.py"]


# --- reparse points ---------------------------------------------------------

def test_a_symlink_is_a_reparse_point(tmp_path):
    target = tmp_path / "t"
    target.write_text("x")
    link = tmp_path / "l"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks not permitted here")
    assert winplat.is_reparse_point(link) is True
    assert winplat.is_reparse_point(target) is False


# --- terminal ---------------------------------------------------------------

def test_terminal_uses_windows_terminal_with_powershell(monkeypatch):
    monkeypatch.setattr(actions, "_windows_terminal", lambda: r"C:\wt.exe")
    monkeypatch.setattr(winplat, "powershell_exe", lambda: r"C:\ps.exe")
    argv, new_console = actions.terminal_argv(r"C:\proj", "npm run dev")
    assert new_console is False
    assert argv == [r"C:\wt.exe", "new-tab", "-d", r"C:\proj",
                    r"C:\ps.exe", "-NoLogo", "-NoExit", "-Command", "npm run dev"]


def test_terminal_escapes_wt_subcommand_separator(monkeypatch):
    monkeypatch.setattr(actions, "_windows_terminal", lambda: r"C:\wt.exe")
    argv, _ = actions.terminal_argv(r"C:\a;b", None)
    assert r"C:\a\;b" in argv


def test_terminal_falls_back_to_a_powershell_console(monkeypatch):
    monkeypatch.setattr(actions, "_windows_terminal", lambda: None)
    monkeypatch.setattr(winplat, "powershell_exe", lambda: r"C:\ps.exe")
    argv, new_console = actions.terminal_argv(r"C:\proj", None)
    assert new_console is True
    assert argv == [r"C:\ps.exe", "-NoLogo", "-NoExit"]


@pytest.mark.asyncio
async def test_open_terminal_refuses_a_missing_folder(tmp_path):
    out = await actions.open_terminal(str(tmp_path / "gone"))
    assert out["success"] is False


# --- browser ----------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "--renderer-cmd-prefix=calc.exe",
    "javascript:alert(1)",
    "file://server/share",
    "http://ok\n--evil",
    "",
])
async def test_browser_refuses_anything_that_could_be_a_switch(monkeypatch, url):
    spawned = []

    async def _spawn(argv, **kw):
        spawned.append(argv)
        return True, ""
    monkeypatch.setattr(actions, "_spawn", _spawn)
    monkeypatch.setattr(actions, "browser_exe", lambda b: r"C:\chrome.exe")
    out = await actions.open_browser(url)
    assert out["success"] is False
    assert spawned == []


@pytest.mark.asyncio
async def test_browser_execs_the_browser_with_the_url_as_one_arg(monkeypatch):
    spawned = []

    async def _spawn(argv, **kw):
        spawned.append(argv)
        return True, ""
    monkeypatch.setattr(actions, "_spawn", _spawn)
    monkeypatch.setattr(actions, "browser_exe", lambda b: r"C:\chrome.exe")
    url = "https://example.com/a?b=1&c=%PATH%"
    out = await actions.open_browser(url)
    assert out["success"] is True
    assert spawned == [[r"C:\chrome.exe", url]]


@pytest.mark.asyncio
async def test_browser_not_installed_is_said(monkeypatch):
    monkeypatch.setattr(actions, "browser_exe", lambda b: None)
    out = await actions.open_browser("https://example.com", "firefox")
    assert out["success"] is False
    assert "Firefox" in out["confirmation"]


# --- editor -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_editor_falls_back_to_notepad_for_a_file(monkeypatch, tmp_path):
    f = tmp_path / "script.py"
    f.write_text("print(1)")
    spawned = []

    async def _spawn(argv, **kw):
        spawned.append(argv)
        return True, ""
    monkeypatch.setattr(actions, "_spawn", _spawn)
    monkeypatch.setattr(actions, "_vscode_exe", lambda: None)
    out = await actions.open_in_editor(str(f))
    assert out["editor"] == "Notepad"
    assert Path(spawned[0][0]).name == "notepad.exe"
    assert spawned[0][1] == str(f)


@pytest.mark.asyncio
async def test_editor_falls_back_to_explorer_for_a_folder(monkeypatch, tmp_path):
    spawned = []

    async def _spawn(argv, **kw):
        spawned.append(argv)
        return True, ""
    monkeypatch.setattr(actions, "_spawn", _spawn)
    monkeypatch.setattr(actions, "_vscode_exe", lambda: None)
    out = await actions.open_in_editor(str(tmp_path))
    assert out["editor"] == "Explorer"
    assert Path(spawned[0][0]).name == "explorer.exe"


@pytest.mark.asyncio
async def test_editor_refuses_a_switch_shaped_path(monkeypatch):
    monkeypatch.setattr(actions, "_vscode_exe", lambda: r"C:\Code.exe")
    out = await actions.open_in_editor("--install-extension=evil")
    assert out["success"] is False
