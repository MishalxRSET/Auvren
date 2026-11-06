"""Tests for preflight.py -- first-run environment checks.

NO TEST IN THIS FILE MAY INVOKE THE REAL `claude` BINARY, THE REAL
`osascript`, OR READ THE DEVELOPER'S REAL ~/.claude/settings.json.

Every subprocess call goes through the single seam `preflight._run_subprocess`
(the pattern `dialog.py` uses for `_osascript`), so tests monkeypatch that one
function rather than `asyncio.create_subprocess_exec` per call site. The
settings-file check goes through `preflight._settings_path`, which tests
point at a tmp_path fixture instead of the real home directory.
"""

import asyncio
import json
import time

import pytest

import preflight
from preflight import Check, STATUS_FAIL, STATUS_OK, STATUS_WARN


def _fake_run_subprocess(responses):
    """Build a fake `_run_subprocess` keyed by (args tuple) -> (rc, stdout, stderr).

    Matching is by the first two positional args (e.g. ("claude", "--version")
    or ("osascript", "-e")) so callers don't need to know exact remaining
    arguments (like the literal AppleScript source). Accepts (and ignores)
    the `env` kwarg that `claude_login` now passes.
    """
    async def fake(*args, timeout, env=None):
        key = tuple(args[:2])
        if key in responses:
            return responses[key]
        raise AssertionError(f"unexpected subprocess call: {args}")
    return fake


# --- claude_cli: PATH + version -------------------------------------------

@pytest.mark.asyncio
async def test_claude_missing_from_path(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    check = await preflight._check_claude_cli()
    assert check.status == STATUS_FAIL
    assert "PATH" in check.message
    assert check.remedy


@pytest.mark.asyncio
async def test_claude_present_but_too_old(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "--version"): (0, "2.1.9 (Claude Code)\n", "")}),
    )
    check = await preflight._check_claude_cli()
    assert check.status == STATUS_FAIL
    assert "2.1.9" in check.message
    assert check.remedy


@pytest.mark.asyncio
async def test_claude_present_and_new_enough(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "--version"): (0, "2.1.258 (Claude Code)\n", "")}),
    )
    check = await preflight._check_claude_cli()
    assert check.status == STATUS_OK
    assert check.remedy is None


@pytest.mark.asyncio
async def test_claude_exactly_at_minimum_version_is_ok(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "--version"): (0, "2.1.224\n", "")}),
    )
    check = await preflight._check_claude_cli()
    assert check.status == STATUS_OK


def test_version_comparison_is_numeric_not_lexicographic():
    """The whole point of parsing into a tuple of ints: '2.1.9' < '2.1.224'
    numerically even though it is greater lexicographically as a string."""
    assert preflight._parse_version("2.1.9") < preflight.MIN_CLAUDE_VERSION
    assert preflight._parse_version("2.1.224") == preflight.MIN_CLAUDE_VERSION
    assert preflight._parse_version("2.1.258") > preflight.MIN_CLAUDE_VERSION
    assert "2.1.9" > "2.1.224"  # the naive (wrong) string comparison, for contrast


@pytest.mark.asyncio
async def test_claude_version_unparseable_output_is_warn(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "--version"): (0, "garbage\n", "")}),
    )
    check = await preflight._check_claude_cli()
    assert check.status == STATUS_WARN


# --- claude_login -----------------------------------------------------------

def _fake_child_env(monkeypatch, config_dir):
    """Point claude_login's `claude_env.child_env()` call at a fixed,
    test-controlled environment instead of this process's real one, so
    results (and the config dir named in messages) are deterministic."""
    monkeypatch.setattr(
        preflight.claude_env, "child_env",
        lambda: {"CLAUDE_CONFIG_DIR": str(config_dir)},
    )


@pytest.mark.asyncio
async def test_claude_not_logged_in(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    _fake_child_env(monkeypatch, tmp_path / ".claude")
    body = json.dumps({"loggedIn": False})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "auth"): (0, body, "")}),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_FAIL
    assert "not logged in" in check.message
    assert check.remedy


@pytest.mark.asyncio
async def test_claude_logged_in(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    _fake_child_env(monkeypatch, tmp_path / ".claude")
    body = json.dumps({"loggedIn": True, "email": "you@example.com"})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "auth"): (0, body, "")}),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_OK
    assert check.remedy is None
    assert "you@example.com" in check.message


@pytest.mark.asyncio
async def test_claude_login_check_without_claude_on_path(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    check = await preflight._check_claude_login()
    assert check.status == STATUS_WARN
    assert check.remedy


@pytest.mark.asyncio
async def test_claude_login_nonzero_exit_is_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    _fake_child_env(monkeypatch, tmp_path / ".claude")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "auth"): (1, "", "not logged in at all")}),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_FAIL


@pytest.mark.asyncio
async def test_claude_login_malformed_json_is_warn(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    _fake_child_env(monkeypatch, tmp_path / ".claude")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "auth"): (0, "not json{{{", "")}),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_WARN


@pytest.mark.asyncio
async def test_claude_login_names_the_config_dir(monkeypatch, tmp_path):
    """A mismatched CLAUDE_CONFIG_DIR must be visible at a glance in the result."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    config_dir = tmp_path / "some-other-config-dir"
    _fake_child_env(monkeypatch, config_dir)
    body = json.dumps({"loggedIn": True, "email": "you@example.com"})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "auth"): (0, body, "")}),
    )
    check = await preflight._check_claude_login()
    assert str(config_dir) in check.message


@pytest.mark.asyncio
async def test_claude_login_expired_oauth_session_is_fail_not_ok(monkeypatch, tmp_path):
    """The real incident: `claude auth status` reports loggedIn: true, but the
    stored OAuth refresh token has already expired -- a real turn would fail
    with 'OAuth session expired and could not be refreshed'. A status probe
    alone can't see this; the check must go further and catch it."""
    monkeypatch.setattr(preflight.shutil, "which",
                        lambda name: {"claude": "/usr/local/bin/claude", "security": "/usr/bin/security"}.get(name))
    monkeypatch.setattr(preflight.sys, "platform", "darwin")
    config_dir = tmp_path / ".claude"
    _fake_child_env(monkeypatch, config_dir)
    status_body = json.dumps({"loggedIn": True, "authMethod": "claude.ai", "email": "you@example.com"})
    expired_at_ms = (time.time() - 3600) * 1000  # expired an hour ago
    keychain_body = json.dumps({"claudeAiOauth": {"refreshTokenExpiresAt": expired_at_ms}})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({
            ("/usr/local/bin/claude", "auth"): (0, status_body, ""),
            ("security", "find-generic-password"): (0, keychain_body, ""),
        }),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_FAIL
    assert "expired" in check.message.lower()
    assert check.remedy


@pytest.mark.asyncio
async def test_claude_login_valid_oauth_session_is_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which",
                        lambda name: {"claude": "/usr/local/bin/claude", "security": "/usr/bin/security"}.get(name))
    monkeypatch.setattr(preflight.sys, "platform", "darwin")
    config_dir = tmp_path / ".claude"
    _fake_child_env(monkeypatch, config_dir)
    status_body = json.dumps({"loggedIn": True, "authMethod": "claude.ai", "email": "you@example.com"})
    future_at_ms = (time.time() + 3600 * 24 * 30) * 1000  # valid for another 30 days
    keychain_body = json.dumps({"claudeAiOauth": {"refreshTokenExpiresAt": future_at_ms}})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({
            ("/usr/local/bin/claude", "auth"): (0, status_body, ""),
            ("security", "find-generic-password"): (0, keychain_body, ""),
        }),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_OK
    assert check.remedy is None


@pytest.mark.asyncio
async def test_claude_login_oauth_probe_unavailable_stays_ok_but_says_so(monkeypatch, tmp_path):
    """When the secondary Keychain probe can't be attempted (not macOS here),
    the check must not silently claim more certainty than it has."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(preflight.sys, "platform", "linux")
    config_dir = tmp_path / ".claude"
    _fake_child_env(monkeypatch, config_dir)
    status_body = json.dumps({"loggedIn": True, "authMethod": "claude.ai", "email": "you@example.com"})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("/usr/local/bin/claude", "auth"): (0, status_body, "")}),
    )
    check = await preflight._check_claude_login()
    assert check.status == STATUS_OK
    assert "could not independently verify" in check.message.lower()


@pytest.mark.asyncio
async def test_claude_login_keychain_probe_never_raises_on_garbage(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which",
                        lambda name: {"claude": "/usr/local/bin/claude", "security": "/usr/bin/security"}.get(name))
    monkeypatch.setattr(preflight.sys, "platform", "darwin")
    config_dir = tmp_path / ".claude"
    _fake_child_env(monkeypatch, config_dir)
    status_body = json.dumps({"loggedIn": True, "authMethod": "claude.ai", "email": "you@example.com"})
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({
            ("/usr/local/bin/claude", "auth"): (0, status_body, ""),
            ("security", "find-generic-password"): (0, "not json at all {{{", ""),
        }),
    )
    check = await preflight._check_claude_login()  # must not raise
    assert check.status == STATUS_OK
    assert "could not independently verify" in check.message.lower()


# --- accessibility -----------------------------------------------------------

@pytest.mark.asyncio
async def test_accessibility_granted(monkeypatch):
    monkeypatch.setattr(preflight.sys, "platform", "darwin")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("osascript", "-e"): (0, "window 1\n", "")}),
    )
    check = await preflight._check_accessibility()
    assert check.status == STATUS_OK


@pytest.mark.asyncio
async def test_accessibility_not_granted(monkeypatch):
    monkeypatch.setattr(preflight.sys, "platform", "darwin")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/osascript")
    err = '65:69: execution error: System Events got an error: osascript is not allowed assistive access. (-1728)'
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("osascript", "-e"): (1, "", err)}),
    )
    check = await preflight._check_accessibility()
    assert check.status == STATUS_FAIL
    assert check.remedy
    assert "Accessibility" in check.remedy


@pytest.mark.asyncio
async def test_accessibility_unknown_error_is_warn_not_ok(monkeypatch):
    monkeypatch.setattr(preflight.sys, "platform", "darwin")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(
        preflight, "_run_subprocess",
        _fake_run_subprocess({("osascript", "-e"): (1, "", "some other applescript error")}),
    )
    check = await preflight._check_accessibility()
    assert check.status == STATUS_WARN


@pytest.mark.asyncio
async def test_accessibility_skipped_off_darwin(monkeypatch):
    monkeypatch.setattr(preflight.sys, "platform", "linux")
    check = await preflight._check_accessibility()
    assert check.status == STATUS_WARN


# --- Screen Recording ----------------------------------------------------------
#
# The same lesson as Accessibility, one permission along: macOS attributes it
# to the app that LAUNCHED JARVIS, not to python. And unlike Accessibility it
# fails silently -- `screencapture` exits 0 and hands back a black frame -- so
# it is worth saying at startup rather than at the moment he asks.

def test_screen_recording_granted(monkeypatch):
    monkeypatch.setattr(preflight.screen, "screen_recording_granted", lambda: True)
    check = preflight._check_screen_recording_sync()
    assert check.status == STATUS_OK
    assert check.remedy is None


def test_screen_recording_not_granted_is_fail_with_the_launching_app_remedy(monkeypatch):
    monkeypatch.setattr(preflight.screen, "screen_recording_granted", lambda: False)
    check = preflight._check_screen_recording_sync()
    assert check.status == STATUS_FAIL
    assert check.remedy
    assert "Screen Recording" in check.remedy
    assert "launched" in check.remedy.lower()


def test_screen_recording_undeterminable_is_warn_not_fail(monkeypatch):
    """None means the probe could not run -- off macOS, or a macOS that moved
    the symbol. Reporting that as a missing permission would send the user to
    a settings pane over nothing."""
    monkeypatch.setattr(preflight.screen, "screen_recording_granted", lambda: None)
    check = preflight._check_screen_recording_sync()
    assert check.status == STATUS_WARN


def test_screen_recording_check_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("CoreGraphics went sideways")

    monkeypatch.setattr(preflight.screen, "screen_recording_granted", boom)
    check = preflight._check_screen_recording_sync()
    assert check.status == STATUS_WARN


def test_the_startup_check_never_takes_a_picture(monkeypatch):
    """Preflight asks the OS a question. It does NOT capture the screen to
    find out -- that would be a screenshot the user never asked for, at every
    boot."""
    monkeypatch.setattr(preflight.screen, "screen_recording_granted", lambda: True)

    def forbidden(*a, **k):
        raise AssertionError("preflight captured the screen")

    monkeypatch.setattr(preflight.screen, "capture_screen", forbidden)
    assert preflight._check_screen_recording_sync().status == STATUS_OK


# --- the voice (Piper by default, Fish optional) -------------------------------

def _piper_installed(tmp_path, monkeypatch):
    exe = tmp_path / "piper.exe"
    exe.write_text("")
    model = tmp_path / "en_GB-alan-medium.onnx"
    model.write_text("")
    (tmp_path / "en_GB-alan-medium.onnx.json").write_text("{}")
    monkeypatch.setenv("PIPER_PATH", str(exe))
    monkeypatch.setenv("PIPER_MODEL", str(model))


def test_voice_ok_with_piper_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    _piper_installed(tmp_path, monkeypatch)
    check = preflight._check_voice_sync()
    assert check.status == STATUS_OK
    assert "en_GB-alan-medium" in check.message


def test_voice_fails_with_nothing_installed_and_says_how(monkeypatch, tmp_path):
    for v in ("TTS_ENGINE", "FISH_API_KEY", "PIPER_PATH", "PIPER_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(preflight.tts.shutil, "which", lambda name: None)
    check = preflight._check_voice_sync()
    assert check.status == STATUS_FAIL
    assert "get_piper.ps1" in check.remedy


def test_voice_with_fish_selected_needs_the_key(monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "fish")
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    check = preflight._check_voice_sync()
    assert check.status == STATUS_FAIL
    assert "FISH_API_KEY" in check.message
    monkeypatch.setenv("FISH_API_KEY", "sk-fish-abc123")
    assert preflight._check_voice_sync().status == STATUS_OK


# --- leftover ANTHROPIC_* ------------------------------------------------------

def test_leftover_anthropic_key_warns(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    check = preflight._check_anthropic_key_leftover_sync()
    assert check.status == STATUS_WARN
    assert "ANTHROPIC_API_KEY" in check.message
    assert check.remedy


def test_no_anthropic_key_is_ok(monkeypatch):
    for key in list(preflight.os.environ):
        if key.startswith("ANTHROPIC_"):
            monkeypatch.delenv(key, raising=False)
    check = preflight._check_anthropic_key_leftover_sync()
    assert check.status == STATUS_OK
    assert check.remedy is None


# --- crossSessionInbound --------------------------------------------------------

def test_cross_session_inbound_missing_settings_file(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "_settings_path", lambda: tmp_path / "does-not-exist.json")
    check = preflight._check_cross_session_inbound_sync()
    assert check.status == STATUS_WARN
    assert "No settings file" in check.message


def test_cross_session_inbound_malformed_json(monkeypatch, tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{ not valid json ][")
    monkeypatch.setattr(preflight, "_settings_path", lambda: p)
    check = preflight._check_cross_session_inbound_sync()
    assert check.status == STATUS_WARN
    assert "Could not read/parse" in check.message


def test_cross_session_inbound_absent_key(monkeypatch, tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "sonnet"}))
    monkeypatch.setattr(preflight, "_settings_path", lambda: p)
    check = preflight._check_cross_session_inbound_sync()
    assert check.status == STATUS_WARN
    assert "not set" in check.message
    # Read-only: the check must never have written to the file.
    assert json.loads(p.read_text()) == {"model": "sonnet"}


def test_cross_session_inbound_set_to_accept(monkeypatch, tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"crossSessionInbound": "accept"}))
    monkeypatch.setattr(preflight, "_settings_path", lambda: p)
    check = preflight._check_cross_session_inbound_sync()
    assert check.status == STATUS_OK
    assert check.remedy is None


def test_cross_session_inbound_set_to_something_else(monkeypatch, tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"crossSessionInbound": "deny"}))
    monkeypatch.setattr(preflight, "_settings_path", lambda: p)
    check = preflight._check_cross_session_inbound_sync()
    assert check.status == STATUS_WARN
    assert "'deny'" in check.message or '"deny"' in check.message


def test_cross_session_inbound_never_writes_when_file_missing(monkeypatch, tmp_path):
    target = tmp_path / "does-not-exist.json"
    monkeypatch.setattr(preflight, "_settings_path", lambda: target)
    preflight._check_cross_session_inbound_sync()
    assert not target.exists()


# --- run_checks(): timeouts, internal exceptions, concurrency -----------------

@pytest.mark.asyncio
async def test_a_hung_check_times_out_as_warn(monkeypatch):
    async def hang(*, timeout):
        await asyncio.sleep(999)
        return Check(name="claude_cli", status=STATUS_OK, message="unreachable")

    monkeypatch.setattr(preflight, "_ASYNC_CHECKS", (hang,))
    monkeypatch.setattr(preflight, "_SYNC_CHECKS", ())

    results = await preflight.run_checks(timeout=0.05)
    assert len(results) == 1
    assert results[0].status == STATUS_WARN
    assert "timed out" in results[0].message


@pytest.mark.asyncio
async def test_a_raising_check_becomes_warn_not_an_exception(monkeypatch):
    def explode():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(preflight, "_ASYNC_CHECKS", ())
    monkeypatch.setattr(preflight, "_SYNC_CHECKS", (explode,))

    results = await preflight.run_checks(timeout=1.0)
    assert len(results) == 1
    assert results[0].status == STATUS_WARN
    assert "disk on fire" in results[0].message


@pytest.mark.asyncio
async def test_run_checks_never_raises_even_with_a_broken_check(monkeypatch):
    def explode():
        raise ValueError("boom")

    monkeypatch.setattr(preflight, "_SYNC_CHECKS", (explode,))
    # Should not raise.
    results = await preflight.run_checks(timeout=1.0)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_run_checks_runs_all_registered_checks(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)  # claude missing -> fast fail
    monkeypatch.setattr(preflight.sys, "platform", "linux")  # accessibility skipped -> fast warn
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "_settings_path", lambda: preflight.Path("/nonexistent/settings.json"))

    results = await preflight.run_checks(timeout=1.0)
    names = {c.name for c in results}
    assert names == {
        "claude_cli", "claude_login", "accessibility", "screen_recording",
        "voice", "local_brain", "wake_word", "speech_recognition", "notes",
        "anthropic_key_leftover", "cross_session_inbound",
    }


# --- spoken_summary -------------------------------------------------------------

def test_spoken_summary_empty_when_all_ok():
    checks = [
        Check(name="claude_cli", status=STATUS_OK, message="ok"),
        Check(name="voice", status=STATUS_OK, message="ok"),
    ]
    assert preflight.spoken_summary(checks) == ""


def test_spoken_summary_never_says_all_checks_passed():
    checks = [Check(name="claude_cli", status=STATUS_OK, message="ok")]
    summary = preflight.spoken_summary(checks)
    assert summary == ""
    assert "all checks passed" not in summary.lower()


def test_spoken_summary_names_a_single_failure():
    checks = [
        Check(name="claude_cli", status=STATUS_OK, message="ok"),
        Check(name="claude_login", status=STATUS_FAIL, message="Claude Code is not logged in."),
    ]
    summary = preflight.spoken_summary(checks)
    assert "One thing needs attention" in summary
    assert "isn't logged in" in summary
    assert "claude_cli" not in summary  # only the failing check is named


def test_spoken_summary_says_screen_recording_in_words_a_person_would_use():
    checks = [Check(name="screen_recording", status=STATUS_FAIL,
                    message="JARVIS has not been granted Screen Recording.")]
    summary = preflight.spoken_summary(checks)
    assert "Screen Recording permission" in summary
    assert "screen_recording" not in summary


def test_spoken_summary_names_two_failures_and_matches_the_example():
    checks = [
        Check(name="claude_login", status=STATUS_FAIL, message="Claude Code is not logged in."),
        Check(name="voice", status=STATUS_FAIL,
              message="TTS_ENGINE is fish but FISH_API_KEY is not set."),
    ]
    summary = preflight.spoken_summary(checks)
    assert summary == (
        "Two things need attention, sir: Claude Code isn't logged in, "
        "and I have no Fish Audio key."
    )


def test_spoken_summary_names_only_the_failures_among_a_mix():
    checks = [
        Check(name="claude_cli", status=STATUS_OK, message="ok"),
        Check(name="claude_login", status=STATUS_OK, message="ok"),
        Check(name="accessibility", status=STATUS_FAIL,
              message="osascript is not granted Accessibility (assistive access); answer_dialog's keystroke will fail."),
        Check(name="cross_session_inbound", status=STATUS_WARN, message="crossSessionInbound is not set"),
        Check(name="anthropic_key_leftover", status=STATUS_WARN, message="ANTHROPIC_API_KEY set"),
    ]
    summary = preflight.spoken_summary(checks)
    assert summary.startswith("3 things need attention, sir:")
    assert "Accessibility permission" in summary
    assert "cross-session steering" in summary
    assert "leftover Anthropic API key" in summary


def test_spoken_summary_warn_counts_as_something_wrong():
    checks = [Check(name="cross_session_inbound", status=STATUS_WARN, message="crossSessionInbound is not set")]
    summary = preflight.spoken_summary(checks)
    assert summary != ""
    assert "One thing needs attention" in summary


def test_spoken_summary_says_a_missing_piper_voice_plainly():
    checks = [Check(name="voice", status=STATUS_FAIL,
                    message="Piper is not installed (no piper.exe on PATH, PIPER_PATH unset).")]
    assert "my voice isn't installed" in preflight.spoken_summary(checks)


# --- the local brain (Ollama) -------------------------------------------------

class _Resp:
    def __init__(self, models):
        self._models = models

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": m} for m in self._models]}


def test_local_brain_ok_when_ollama_has_the_model(monkeypatch):
    import httpx
    monkeypatch.setenv("JARVIS_BRAIN", "local")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Resp(["llama3.2:latest"]))
    assert preflight._check_local_brain_sync().status == STATUS_OK


def test_local_brain_says_to_pull_a_missing_model(monkeypatch):
    import httpx
    monkeypatch.setenv("JARVIS_BRAIN", "local")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Resp([]))
    check = preflight._check_local_brain_sync()
    assert check.status == STATUS_FAIL and check.remedy == "Run: ollama pull qwen3:4b"
    assert "my local model isn't downloaded" in preflight.spoken_summary([check])


def test_local_brain_says_when_ollama_is_not_running(monkeypatch):
    import httpx

    def refused(url, timeout):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setenv("JARVIS_BRAIN", "local")
    monkeypatch.setattr(httpx, "get", refused)
    check = preflight._check_local_brain_sync()
    assert check.status == STATUS_FAIL and "ollama.com" in check.remedy
    assert "Ollama isn't running" in preflight.spoken_summary([check])


def test_local_brain_check_is_quiet_with_the_claude_brain(monkeypatch):
    import httpx

    def never(*a, **k):
        raise AssertionError("no Ollama probe with the Claude brain")
    monkeypatch.setenv("JARVIS_BRAIN", "claude")
    monkeypatch.setattr(httpx, "get", never)
    assert preflight._check_local_brain_sync().status == STATUS_OK


# --- the wake word ----------------------------------------------------------------

def test_wake_word_off_is_fine(monkeypatch):
    monkeypatch.setenv("WAKE_WORD", "off")
    assert preflight._check_wake_word_sync().status == STATUS_OK


def test_wake_word_without_the_engine_warns_but_does_not_fail(monkeypatch):
    import stt
    import wakeword
    monkeypatch.setenv("WAKE_WORD", "hey_jarvis")
    monkeypatch.setattr(stt, "mode", lambda: "browser")
    monkeypatch.setattr(wakeword, "engine_problem",
                        lambda: "The wake-word engine is not installed (openwakeword)")
    check = preflight._check_wake_word_sync()
    assert check.status == preflight.STATUS_WARN
    assert "requirements-wakeword.txt" in check.remedy


def test_wake_word_heard_by_on_device_recognition_is_fine(monkeypatch):
    import stt
    import wakeword
    monkeypatch.delenv("WAKE_WORD", raising=False)
    monkeypatch.setattr(wakeword, "engine_problem", lambda: "no trained model yet")
    monkeypatch.setattr(stt, "mode", lambda: "local")
    check = preflight._check_wake_word_sync()
    assert check.status == STATUS_OK and "Hey Auvren" in check.message


def test_speech_recognition_states(monkeypatch):
    import stt
    monkeypatch.setattr(stt, "engine_problem", lambda: None)
    monkeypatch.setenv("STT_ENGINE", "auto")
    assert preflight._check_speech_recognition_sync().status == STATUS_OK
    monkeypatch.setattr(stt, "engine_problem", lambda: "not installed")
    assert preflight._check_speech_recognition_sync().status == STATUS_OK, \
        "not installed and not asked for: the browser listens"
    monkeypatch.setenv("STT_ENGINE", "local")
    check = preflight._check_speech_recognition_sync()
    assert check.status == preflight.STATUS_WARN and "requirements-stt.txt" in check.remedy


def test_wake_word_ready(monkeypatch):
    import wakeword
    monkeypatch.setenv("WAKE_WORD", "hey_jarvis")
    monkeypatch.setattr(wakeword, "engine_problem", lambda: None)
    check = preflight._check_wake_word_sync()
    assert check.status == STATUS_OK and "Hey Jarvis" in check.message
