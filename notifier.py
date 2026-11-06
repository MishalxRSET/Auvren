"""
Auvren Notifier -- Windows toast-notification fallback.

When Auvren needs the user's attention but no browser tab is connected to
speak through, this posts a native Windows toast instead.

Notification titles/messages surface a Claude Code session's title or its
last message -- text that originates in someone else's transcript, not text
Auvren wrote itself. So the script never contains the untrusted text at all.
It is one fixed PowerShell script, fed on stdin, and the title/message/
subtitle reach it only as ENVIRONMENT VARIABLES, which PowerShell hands to the
script as plain data -- never parsed as code. Inside the script they are
assigned to `InnerText` of the toast's XML nodes, so they are not parsed as
XML either. There is no escaping step to get wrong because there is nothing
to escape: a value containing `"; Remove-Item ...` or `</text><audio ...>` is
just a string with those characters in it.

The toast is posted under PowerShell's own AppUserModelID, which every
Windows install registers, so no shortcut or app registration is needed.
"""

import asyncio
import logging
import os
from pathlib import Path

import winplat

log = logging.getLogger("jarvis.notifier")

# A notification is a glance, not an essay -- keep it short.
_TITLE_MAX = 120
_SUBTITLE_MAX = 120
_MESSAGE_MAX = 300

# PowerShell takes a moment to load the WinRT types; a wedged one must not
# hang the caller.
_TIMEOUT_SECONDS = 15.0

_ENV_TITLE = "AUVREN_NOTIFY_TITLE"
_ENV_MESSAGE = "AUVREN_NOTIFY_MESSAGE"
_ENV_SUBTITLE = "AUVREN_NOTIFY_SUBTITLE"

# Fixed script, no untrusted text ever enters this string (see module doc).
_NOTIFY_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text/><text/><text/></binding></visual></toast>')
$nodes = $xml.GetElementsByTagName('text')
$nodes.Item(0).InnerText = [string]$env:AUVREN_NOTIFY_TITLE
$nodes.Item(1).InnerText = [string]$env:AUVREN_NOTIFY_SUBTITLE
$nodes.Item(2).InnerText = [string]$env:AUVREN_NOTIFY_MESSAGE
if ([string]::IsNullOrEmpty($env:AUVREN_NOTIFY_SUBTITLE)) {
    [void]$nodes.Item(1).ParentNode.RemoveChild($nodes.Item(1))
}
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show($toast)
"""


def _truncate(text: str, limit: int) -> str:
    """Bound text length for a glanceable notification, marking any cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"  # ellipsis


def available() -> bool:
    """Whether posting a notification is plausible right now.

    Cheap and side-effect free: Windows + PowerShell present. A precondition
    check, not a delivery guarantee -- Focus Assist / Do Not Disturb or
    per-app notification settings can still silently drop the toast.
    """
    return winplat.IS_WINDOWS and Path(winplat.powershell_exe()).is_file()


async def notify(title: str, message: str, *, subtitle: str = "") -> bool:
    """Post a Windows toast. Returns whether it was handed off successfully.

    A fallback path for when nobody is listening on the voice channel, so it
    must never raise: any failure (not Windows, no PowerShell, non-zero exit,
    timeout, spawn error) is logged as a warning and reported back as False.
    """
    try:
        if not available():
            log.warning("notifier: notifications unavailable on this platform")
            return False

        env = dict(os.environ)
        env[_ENV_TITLE] = _truncate(str(title or ""), _TITLE_MAX)
        env[_ENV_MESSAGE] = _truncate(str(message or ""), _MESSAGE_MAX)
        env[_ENV_SUBTITLE] = _truncate(str(subtitle or ""), _SUBTITLE_MAX)

        try:
            proc = await asyncio.create_subprocess_exec(
                winplat.powershell_exe(), "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-Command", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                creationflags=winplat.no_window_flag(),
            )
        except OSError as e:
            log.warning(f"notifier: failed to spawn PowerShell: {e}")
            return False

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(_NOTIFY_SCRIPT.encode("utf-8")),
                timeout=_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.warning("notifier: PowerShell timed out, killing it")
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            return False

        if proc.returncode != 0:
            log.warning(
                f"notifier: PowerShell exited {proc.returncode}: "
                f"{stderr.decode(errors='replace').strip()[:300]}"
            )
            return False

        return True
    except Exception as e:
        # Belt and suspenders: this path must never raise into the caller.
        log.warning(f"notifier: unexpected error posting notification: {e}")
        return False
