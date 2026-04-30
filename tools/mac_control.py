"""
Mac Control Tool
Controls macOS via osascript (AppleScript) and subprocess.
Capabilities: open apps, volume, brightness, notifications,
clipboard, currently playing app info.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Dict, Optional


def _run_applescript(script: str) -> Dict[str, Any]:
    """Run an AppleScript and return result."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()}
        return {"success": True, "output": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Script timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "osascript not found (not running on macOS)"}


class MacControlTool:
    """
    macOS system control via AppleScript and subprocess.
    All methods are async (run in executor to avoid blocking).
    """

    async def _async_script(self, script: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_applescript, script)

    # ── Volume ─────────────────────────────────────────────────────────────

    async def set_volume(self, level: int) -> Dict[str, Any]:
        """Set system volume (0–100)."""
        level = max(0, min(100, level))
        script = f"set volume output volume {level}"
        result = await self._async_script(script)
        if result["success"]:
            result["volume"] = level
        return result

    async def get_volume(self) -> Dict[str, Any]:
        result = await self._async_script(
            "output volume of (get volume settings)"
        )
        if result["success"]:
            try:
                result["volume"] = int(result["output"])
            except ValueError:
                pass
        return result

    async def mute(self) -> Dict[str, Any]:
        return await self._async_script("set volume with output muted")

    async def unmute(self) -> Dict[str, Any]:
        return await self._async_script("set volume without output muted")

    # ── Brightness ─────────────────────────────────────────────────────────

    async def set_brightness(self, level: float) -> Dict[str, Any]:
        """Set display brightness (0.0–1.0)."""
        level = max(0.0, min(1.0, float(level)))
        script = f'tell application "System Events" to set brightness of display 1 to {level}'
        return await self._async_script(script)

    # ── Apps ───────────────────────────────────────────────────────────────

    async def open_app(self, app_name: str) -> Dict[str, Any]:
        """Open an application by name."""
        script = f'tell application "{app_name}" to activate'
        return await self._async_script(script)

    async def quit_app(self, app_name: str) -> Dict[str, Any]:
        """Quit an application."""
        script = f'tell application "{app_name}" to quit'
        return await self._async_script(script)

    async def get_running_apps(self) -> Dict[str, Any]:
        """List all running applications."""
        script = 'tell application "System Events" to get name of every process whose background only is false'
        result = await self._async_script(script)
        if result["success"]:
            result["apps"] = [a.strip() for a in result["output"].split(",")]
        return result

    # ── Clipboard ──────────────────────────────────────────────────────────

    async def get_clipboard(self) -> Dict[str, Any]:
        result = await self._async_script("the clipboard")
        if result["success"]:
            result["text"] = result["output"]
        return result

    async def set_clipboard(self, text: str) -> Dict[str, Any]:
        safe = text.replace('"', '\\"')
        return await self._async_script(f'set the clipboard to "{safe}"')

    # ── Notifications ──────────────────────────────────────────────────────

    async def send_notification(
        self,
        message: str,
        title: str = "Jarvis",
        subtitle: str = "",
    ) -> Dict[str, Any]:
        """Send a macOS notification."""
        safe_msg = message.replace('"', '\\"')
        safe_title = title.replace('"', '\\"')
        safe_sub = subtitle.replace('"', '\\"')
        script = (
            f'display notification "{safe_msg}" '
            f'with title "{safe_title}"'
            + (f' subtitle "{safe_sub}"' if subtitle else "")
        )
        return await self._async_script(script)

    # ── System info ────────────────────────────────────────────────────────

    async def get_battery(self) -> Dict[str, Any]:
        """Get battery percentage (MacBooks only)."""
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True, text=True, timeout=5
            )
            import re
            match = re.search(r"(\d+)%", result.stdout)
            if match:
                return {"success": True, "battery_pct": int(match.group(1))}
        except Exception:
            pass
        return {"success": False, "error": "Could not read battery"}

    async def lock_screen(self) -> Dict[str, Any]:
        return await self._async_script(
            'tell application "System Events" to keystroke "q" using {control down, command down}'
        )

    async def sleep(self) -> Dict[str, Any]:
        return await self._async_script('tell application "System Events" to sleep')
