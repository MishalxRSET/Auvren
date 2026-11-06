"""Windows platform seams: process liveness, process-tree kill, the `claude`
launcher, reparse points, and the handful of Windows binaries Auvren drives.

Everything that touches Win32 lives here, behind plain functions, so the rest
of the code never imports `ctypes.windll` and the test suite (which runs on
Linux CI as well as Windows) can fake one seam instead of many.

Three facts this module exists to encode, each of which is a trap:

1. **`os.kill(pid, 0)` on Windows is not a liveness probe — it KILLS.**
   CPython maps every signal except CTRL_C_EVENT / CTRL_BREAK_EVENT to
   `TerminateProcess(handle, sig)`. The macOS code asked "is this session
   alive?" once a second with signal 0; on Windows the identical call
   terminates every Claude Code session it looks at. `pid_alive` here uses
   `OpenProcess` + `GetExitCodeProcess`, which only read.

2. **`claude.cmd` must never be the program we exec.** npm installs `claude`
   on Windows as a batch shim. CreateProcess runs a `.cmd` through
   `cmd.exe`, which re-parses the whole command line with its own rules
   (`&`, `|`, `%VAR%`, and quote toggling that `list2cmdline` does not
   model — the "BatBadBut" class of bug). The brain passes its system prompt,
   which carries user/LLM-written memory text, on that command line. So
   `claude_argv` resolves the shim to the real program — `claude.exe`, or
   `node.exe <cli.js>` — and refuses a batch file it cannot see through.

3. **`proc.terminate()` on Windows kills one process, not the tree.** A
   `claude` run spawns shells, `node`, test runners; TerminateProcess on the
   parent orphans every one of them. `kill_tree` uses `taskkill /T /F`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"

# --- process liveness -------------------------------------------------------

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _win_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE,
                                            ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means the process EXISTS (it belongs to someone more
        # privileged); anything else (ERROR_INVALID_PARAMETER) means it does not.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid) -> bool:
    """True if a process with this pid exists. Never signals it.

    `pid` must be a positive integer. On Windows this only reads (see module
    docstring, fact 1). Off Windows — the test suite on Linux CI — signal 0
    is the read-only probe it is on POSIX.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if IS_WINDOWS:
        try:
            return _win_pid_alive(pid)
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# --- killing a process tree -------------------------------------------------

def kill_tree(pid) -> bool:
    """Forcefully end `pid` and every process it started. True if handed off.

    Windows only does anything here: `taskkill /T /F`. Elsewhere it returns
    False and the caller falls back to `proc.kill()`.
    """
    if not IS_WINDOWS:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    taskkill = system32_exe("taskkill.exe")
    try:
        subprocess.run([taskkill, "/T", "/F", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10, creationflags=no_window_flag())
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def no_window_flag() -> int:
    """CREATE_NO_WINDOW on Windows, so a helper never flashes a console."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def system32_exe(name: str) -> str:
    """The absolute path of a System32 binary, never a PATH lookup.

    A PATH lookup for `taskkill` or `powershell` would run whatever a
    project directory earlier on PATH happens to contain.
    """
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    return str(Path(root) / "System32" / name)


def powershell_exe() -> str:
    """Windows PowerShell 5.1 — present on every supported Windows."""
    return str(Path(system32_exe("WindowsPowerShell")) / "v1.0" / "powershell.exe")


# --- reparse points (the Windows symlink / junction) ------------------------

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def is_reparse_point(path) -> bool:
    """True for a symlink, junction or other reparse point. Does not follow it."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs & _FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    return Path(path).is_symlink()


# --- the claude launcher ----------------------------------------------------

class ClaudeLaunchError(OSError):
    """`claude` could not be resolved to a program that is safe to exec.

    An OSError so every spawn site's existing "could not start" handling
    catches it without a new except clause."""


_BATCH_SUFFIXES = (".cmd", ".bat")
_PACKAGE = Path("node_modules") / "@anthropic-ai" / "claude-code"


def _split_override(value: str) -> list[str]:
    """JARVIS_CLAUDE_PATH may be a bare path or a quoted path plus args.

    POSIX `shlex.split` eats backslashes, which turns `C:\\Users\\me\\claude.exe`
    into `C:Usersmeclaude.exe`. On Windows a value that names an existing
    file is taken whole; otherwise it is split in non-POSIX mode and the
    surrounding quotes stripped.
    """
    value = value.strip()
    if not IS_WINDOWS:
        return shlex.split(value)
    if Path(value.strip('"')).is_file():
        return [value.strip('"')]
    return [p.strip('"') for p in shlex.split(value, posix=False)]


def _see_through_shim(shim: Path) -> list[str] | None:
    """The real program behind an npm `claude.cmd`, or None.

    npm puts the shim at `<prefix>\\claude.cmd` and the package at
    `<prefix>\\node_modules\\@anthropic-ai\\claude-code`. The package carries
    either a native `claude.exe` or a `cli.js` run by node.
    """
    pkg = shim.parent / _PACKAGE
    for exe in (pkg / "bin" / "claude.exe", pkg / "claude.exe"):
        if exe.is_file():
            return [str(exe)]
    for script in ("cli.js", "cli.mjs"):
        js = pkg / script
        if js.is_file():
            node = (shim.parent / "node.exe")
            node_path = str(node) if node.is_file() else shutil.which("node")
            if node_path and not node_path.lower().endswith(_BATCH_SUFFIXES):
                return [node_path, str(js)]
    return None


def claude_argv(override: str | None = None) -> list[str]:
    """The argv prefix that starts Claude Code. Raises ClaudeLaunchError.

    Order: an explicit override (JARVIS_CLAUDE_PATH), then `claude` on PATH.
    On Windows a batch shim is resolved through to `claude.exe` or
    `node cli.js`, and refused if it cannot be — never exec'd (fact 2).
    """
    if override:
        argv = _split_override(override)
    else:
        found = shutil.which("claude")
        if not found and IS_WINDOWS:
            native = Path.home() / ".local" / "bin" / "claude.exe"
            if native.is_file():
                found = str(native)
        argv = [found] if found else ["claude"]
    if not argv:
        raise ClaudeLaunchError("JARVIS_CLAUDE_PATH is empty")

    if IS_WINDOWS and argv[0].lower().endswith(_BATCH_SUFFIXES):
        real = _see_through_shim(Path(argv[0]))
        if real is None:
            raise ClaudeLaunchError(
                f"{argv[0]} is a batch file, and Auvren will not pass prompts "
                "through cmd.exe. Install the native Claude Code build "
                "(claude.exe), or set JARVIS_CLAUDE_PATH to claude.exe."
            )
        argv = real + argv[1:]
    return argv
