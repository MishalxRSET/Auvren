"""Unit tests for notifier.py -- the Windows toast-notification fallback.

These tests never post a real notification: `asyncio.create_subprocess_exec`
is mocked at the boundary so the developer running the suite is not spammed.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import notifier


class _FakeProcess:
    """Stand-in for the object asyncio.create_subprocess_exec() returns."""

    def __init__(self, returncode=0, stdout=b"", stderr=b"", hang=False):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self.killed = False

    async def communicate(self, input=None):
        self.sent_stdin = input
        if self._hang and not self.killed:
            # Simulate a wedged PowerShell: never resolves on its own until
            # killed -- like a real process, communicate() after kill()
            # returns promptly instead of hanging forever.
            await asyncio.sleep(999)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True


def _patch_subprocess(fake_proc, capture=None):
    """Patch create_subprocess_exec to return fake_proc, recording call args."""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        if capture is not None:
            capture["args"] = args
            capture["kwargs"] = kwargs
        return fake_proc

    return patch("notifier.asyncio.create_subprocess_exec", side_effect=_fake_create_subprocess_exec)


@pytest.mark.asyncio
async def test_successful_post_returns_true():
    fake_proc = _FakeProcess(returncode=0)
    with patch("notifier.available", return_value=True):
        with _patch_subprocess(fake_proc):
            result = await notifier.notify("Title", "Message")
    assert result is True


@pytest.mark.asyncio
async def test_nonzero_exit_returns_false_without_raising():
    fake_proc = _FakeProcess(returncode=1, stderr=b"some powershell error")
    with patch("notifier.available", return_value=True):
        with _patch_subprocess(fake_proc):
            result = await notifier.notify("Title", "Message")
    assert result is False


@pytest.mark.asyncio
async def test_unavailable_returns_false():
    with patch("notifier.available", return_value=False):
        result = await notifier.notify("Title", "Message")
    assert result is False


@pytest.mark.asyncio
async def test_timeout_returns_false_without_raising():
    fake_proc = _FakeProcess(hang=True)
    with patch("notifier.available", return_value=True):
        with patch("notifier._TIMEOUT_SECONDS", 0.05):
            with _patch_subprocess(fake_proc):
                result = await notifier.notify("Title", "Message")
    assert result is False
    assert fake_proc.killed is True


@pytest.mark.asyncio
async def test_spawn_failure_returns_false_without_raising():
    async def _raise(*args, **kwargs):
        raise OSError("no such file: powershell.exe")

    with patch("notifier.available", return_value=True):
        with patch("notifier.asyncio.create_subprocess_exec", side_effect=_raise):
            result = await notifier.notify("Title", "Message")
    assert result is False


def test_available_false_off_windows(monkeypatch):
    monkeypatch.setattr(notifier.winplat, "IS_WINDOWS", False)
    assert notifier.available() is False


def test_available_false_when_powershell_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(notifier.winplat, "IS_WINDOWS", True)
    monkeypatch.setattr(notifier.winplat, "powershell_exe",
                        lambda: str(tmp_path / "missing.exe"))
    assert notifier.available() is False


def test_available_true_on_windows_with_powershell(monkeypatch, tmp_path):
    exe = tmp_path / "powershell.exe"
    exe.write_text("")
    monkeypatch.setattr(notifier.winplat, "IS_WINDOWS", True)
    monkeypatch.setattr(notifier.winplat, "powershell_exe", lambda: str(exe))
    assert notifier.available() is True


# --- Escaping / injection tests -------------------------------------------
#
# The notification text is attacker-influenced (it can carry a session
# title or last message from someone else's Claude Code transcript). The
# module's defense is to never put that text into the PowerShell *source*
# at all -- it travels as environment variables, which the fixed script reads
# as inert data and assigns to XML InnerText. These tests assert that the
# boundary call receives the raw, unmodified text in the child's environment,
# that nothing untrusted is on the command line, and that the script piped to
# PowerShell's stdin never contains the payload.

INJECTION_PAYLOAD = '"; Remove-Item -Recurse ~ ; $(calc) `n </text><audio src="x"/> \\ backslash " quote \n newline'


@pytest.mark.asyncio
async def test_injection_payload_reaches_boundary_as_literal_env():
    fake_proc = _FakeProcess(returncode=0)
    capture = {}
    with patch("notifier.available", return_value=True):
        with _patch_subprocess(fake_proc, capture=capture):
            result = await notifier.notify(INJECTION_PAYLOAD, "a normal message")

    assert result is True
    args = capture["args"]
    env = capture["kwargs"]["env"]
    assert args[0].lower().endswith("powershell.exe")
    assert args[-2:] == ("-Command", "-")
    # Nothing untrusted on the command line at all.
    assert all(INJECTION_PAYLOAD not in a for a in args)
    # The env entry is the exact, byte-identical payload.
    assert env["AUVREN_NOTIFY_TITLE"] == INJECTION_PAYLOAD

    stdin_sent = fake_proc.sent_stdin.decode("utf-8")
    assert INJECTION_PAYLOAD not in stdin_sent
    assert "Remove-Item" not in stdin_sent
    assert "$env:AUVREN_NOTIFY_TITLE" in stdin_sent
    assert ".InnerText" in stdin_sent


@pytest.mark.asyncio
async def test_quotes_and_backslashes_pass_through_unescaped_in_message():
    fake_proc = _FakeProcess(returncode=0)
    capture = {}
    payload = 'She said \\"hello\\" and left \\ trailing backslash'
    with patch("notifier.available", return_value=True):
        with _patch_subprocess(fake_proc, capture=capture):
            await notifier.notify("Title", payload, subtitle="sub")

    env = capture["kwargs"]["env"]
    assert env["AUVREN_NOTIFY_MESSAGE"] == payload
    assert env["AUVREN_NOTIFY_SUBTITLE"] == "sub"


@pytest.mark.asyncio
async def test_long_text_is_truncated():
    fake_proc = _FakeProcess(returncode=0)
    capture = {}
    long_title = "T" * 500
    long_message = "M" * 500
    with patch("notifier.available", return_value=True):
        with _patch_subprocess(fake_proc, capture=capture):
            await notifier.notify(long_title, long_message)

    env = capture["kwargs"]["env"]
    sent_title, sent_message = env["AUVREN_NOTIFY_TITLE"], env["AUVREN_NOTIFY_MESSAGE"]
    assert len(sent_title) <= notifier._TITLE_MAX
    assert len(sent_message) <= notifier._MESSAGE_MAX
    assert sent_title.endswith("…")
    assert sent_message.endswith("…")
