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

from tools.platform_guard import is_mac, mac_only_response


def _run_applescript(script: str) -> Dict[str, Any]:
    """Run an AppleScript and return result."""
    # Cloud-deployment guard — every Mac-only method in this file ultimately
    # funnels through here, so one check disables them all cleanly.
    if not is_mac():
        return mac_only_response("mac_control")
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

    def __init__(self):
        # If we're not on macOS (e.g. inside a Fly.io Linux container),
        # replace every public async method with a stub that returns a
        # friendly "feature disabled in cloud" payload. This avoids
        # editing each individual method body.
        if not is_mac():
            for _name in dir(self):
                if _name.startswith("_"):
                    continue
                _attr = getattr(self.__class__, _name, None)
                if _attr is None or not asyncio.iscoroutinefunction(_attr):
                    continue
                # Capture _name in default arg to dodge late-binding bug.
                async def _stub(*_a, __feature=_name, **_kw):
                    return mac_only_response(f"mac_control.{__feature}")
                setattr(self, _name, _stub)

    async def _async_script(self, script: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_applescript, script)

    # ── Volume ─────────────────────────────────────────────────────────────

    async def set_volume(self, level: int) -> Dict[str, Any]:
        """
        Set system volume (0–100).

        AppleScript's `set volume output volume N` is unprivileged and works
        on every macOS version without permission prompts, but it silently
        no-ops when output is routed to certain Bluetooth devices. We
        verify the result by reading the volume back.
        """
        level = max(0, min(100, level))
        script = f"set volume output volume {level}"
        result = await self._async_script(script)
        if not result.get("success"):
            return result
        # Verify it actually took effect
        check = await self.get_volume()
        actual = check.get("volume")
        if actual is None:
            # Couldn't read back — report optimistic success
            result["volume"] = level
            result["message"] = f"Volume set to {level}%."
            return result
        if abs(actual - level) <= 2:
            return {
                "success": True,
                "volume": actual,
                "message": f"Volume set to {actual}%.",
            }
        return {
            "success": False,
            "volume": actual,
            "error": (
                f"Volume change didn't take effect (still {actual}%). "
                "This sometimes happens with Bluetooth output devices — "
                "adjust the volume directly on the device."
            ),
        }

    async def get_volume(self) -> Dict[str, Any]:
        """Return current system volume (0–100)."""
        result = await self._async_script(
            "output volume of (get volume settings)"
        )
        if result.get("success"):
            try:
                # AppleScript returns the raw integer as text
                result["volume"] = int(result["output"].strip())
                result["message"] = f"Volume is at {result['volume']}%."
            except (ValueError, AttributeError):
                # Couldn't parse — surface the problem rather than silently
                # leaving result["volume"] missing.
                result["success"] = False
                result["error"] = f"Unexpected volume reading: {result.get('output')!r}"
        return result

    async def mute(self) -> Dict[str, Any]:
        return await self._async_script("set volume with output muted")

    async def unmute(self) -> Dict[str, Any]:
        return await self._async_script("set volume without output muted")

    # ── Brightness ─────────────────────────────────────────────────────────

    async def set_brightness(self, level: float) -> Dict[str, Any]:
        """
        Set display brightness (0.0–1.0).

        Apple removed direct AppleScript brightness control years ago. The
        only reliable way on a stock macOS is the `brightness` Homebrew CLI
        (https://github.com/nriley/brightness). When it's missing we fall
        back to nudging the brightness keys, but we cannot read the current
        level so the result is approximate — the user gets a clear note
        explaining the limitation rather than a silently-wrong setting.
        """
        import subprocess
        level = max(0.0, min(1.0, float(level)))

        # Preferred: the `brightness` CLI from Homebrew is the only path that
        # sets an exact level deterministically.
        try:
            result = subprocess.run(
                ["brightness", str(level)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return {
                    "success": True,
                    "level": level,
                    "message": f"Brightness set to {int(level * 100)}%.",
                }
        except FileNotFoundError:
            pass

        # Fallback: tell the user, don't lie about success. Simulating F1/F2
        # without knowing the current level results in drift — better to be
        # honest about the prerequisite.
        return {
            "success": False,
            "error": (
                "Precise brightness control needs the `brightness` CLI. "
                "Install it once with: brew install brightness"
            ),
        }

    # ── Apps ───────────────────────────────────────────────────────────────

    async def open_app(self, app_name: str, new_window: bool = False) -> Dict[str, Any]:
        """
        Open any application using `open -a`. The Launch Services database
        handles aliasing, so case variations and partial names like "chrome"
        vs "Google Chrome" tend to resolve correctly.

        `open` returns exit code 0 even when the named app doesn't exist on
        disk — its error goes to stderr. We check stderr for the tell-tale
        "Unable to find application" string and surface a useful error.
        """
        import asyncio
        import subprocess as _sp
        loop = asyncio.get_event_loop()

        if new_window:
            # Try AppleScript new window first (preserves a single instance
            # making a new window). Fall back to `open -na` (always spawns).
            script = (
                f'tell application "{app_name}"\n'
                f'    activate\n'
                f'    make new window\n'
                f'end tell'
            )
            result = await self._async_script(script)
            if result.get("success"):
                result["message"] = f"Opened a new window in {app_name}."
                return result
            cmd = ["open", "-na", app_name]
        else:
            cmd = ["open", "-a", app_name]

        try:
            proc = await loop.run_in_executor(
                None,
                lambda: _sp.run(cmd, capture_output=True, text=True, timeout=10),
            )
            stderr = (proc.stderr or "").strip()
            # `open` exits 0 even on "Unable to find application" — check stderr
            if "Unable to find application" in stderr or "does not exist" in stderr:
                return {
                    "success": False,
                    "error": f"{app_name} isn't installed on this Mac.",
                }
            if proc.returncode == 0:
                return {
                    "success": True,
                    "message": f"Opened {app_name}.",
                }
            return {
                "success": False,
                "error": stderr or f"Could not open {app_name}.",
            }
        except _sp.TimeoutExpired:
            return {"success": False, "error": f"Opening {app_name} timed out."}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
        """
        Lock the screen.

        Try three approaches in order of reliability:
          1. `pmset displaysleepnow` — works without any permissions on macOS 11+
          2. `CGSession -suspend` — works on most macOS versions
          3. AppleScript ⌃⌘Q keystroke — needs Accessibility permission
        """
        import subprocess, asyncio
        loop = asyncio.get_event_loop()

        async def _run(cmd: list) -> bool:
            try:
                proc = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=5),
                )
                return proc.returncode == 0
            except Exception:
                return False

        if await _run(["pmset", "displaysleepnow"]):
            return {"success": True, "message": "Screen locked."}
        if await _run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"]):
            return {"success": True, "message": "Screen locked."}

        # Last resort — keystroke, only works with Accessibility permission
        result = await self._async_script(
            'tell application "System Events" to keystroke "q" using {control down, command down}'
        )
        if result.get("success"):
            result["message"] = "Screen locked."
        else:
            result["error"] = (
                "Could not lock the screen. Grant your terminal Accessibility "
                "permission in System Settings → Privacy & Security → Accessibility."
            )
        return result

    async def sleep(self) -> Dict[str, Any]:
        """Put the Mac to sleep. Uses pmset which needs no permissions."""
        import subprocess, asyncio
        loop = asyncio.get_event_loop()
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(["pmset", "sleepnow"], capture_output=True, text=True, timeout=5),
            )
            if proc.returncode == 0:
                return {"success": True, "message": "Sleeping the Mac."}
        except Exception:
            pass
        # Fallback to AppleScript (needs Automation permission)
        result = await self._async_script('tell application "System Events" to sleep')
        if not result.get("success"):
            result["error"] = "Could not put the Mac to sleep."
        return result

    # (quit_app is defined above in the Apps section — removed the duplicate
    # that previously lived here, which silently shadowed the original.)

    async def hide_app(self, app_name: str) -> Dict[str, Any]:
        """Hide an application."""
        script = f'tell application "System Events" to set visible of process "{app_name}" to false'
        return await self._async_script(script)

    async def take_screenshot(self, save_path: str = "~/Desktop/screenshot.png") -> Dict[str, Any]:
        """Take a screenshot and save to file."""
        import subprocess, os
        path = os.path.expanduser(save_path)
        try:
            result = subprocess.run(
                ["screencapture", "-x", path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return {"success": True, "path": path, "message": f"Screenshot saved to {path}"}
            return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_dark_mode(self) -> Dict[str, Any]:
        """
        Check if dark mode is enabled.

        Uses `defaults read -g AppleInterfaceStyle`. macOS sets this key to
        "Dark" when dark mode is on and *deletes* the key entirely when it's
        off — so a non-zero exit code means "off" rather than "error".
        """
        import subprocess
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=5,
            )
            is_dark = (
                result.returncode == 0
                and result.stdout.strip().lower() == "dark"
            )
            return {
                "success": True,
                "dark_mode": is_dark,
                "message": f"Dark mode is {'on' if is_dark else 'off'}.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def toggle_dark_mode(self) -> Dict[str, Any]:
        """
        Toggle between dark and light mode.

        Primary path: AppleScript against System Events — instant and clean
        when the app has Automation permission.
        Fallback path: the AppleScript silently no-ops without permission, so
        we double-check the actual state afterwards and report success based
        on whether the value flipped (rather than the AppleScript exit code,
        which lies).
        """
        # Read current state so we know what "toggled" means
        before = await self.get_dark_mode()
        was_dark = before.get("dark_mode", False)

        script = (
            'tell application "System Events"\n'
            '    tell appearance preferences\n'
            '        set dark mode to not dark mode\n'
            '    end tell\n'
            'end tell'
        )
        await self._async_script(script)

        # Verify it actually flipped (AppleScript returns 0 even when blocked
        # by missing Automation permission, so we trust the state read).
        after = await self.get_dark_mode()
        now_dark = after.get("dark_mode", was_dark)

        if now_dark != was_dark:
            return {
                "success": True,
                "dark_mode": now_dark,
                "message": f"Dark mode toggled {'on' if now_dark else 'off'}.",
            }
        return {
            "success": False,
            "dark_mode": now_dark,
            "error": (
                "Could not toggle dark mode. Grant Automation permission to "
                "your terminal in System Settings → Privacy & Security → "
                "Automation → (your terminal) → System Events."
            ),
        }

    async def get_system_info(self) -> Dict[str, Any]:
        """Get CPU, RAM, disk usage."""
        import subprocess
        try:
            # CPU usage
            cpu = subprocess.run(
                ["bash", "-c", "top -l 1 | grep 'CPU usage' | awk '{print $3}'"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()

            # RAM
            ram = subprocess.run(
                ["bash", "-c", "vm_stat | grep 'Pages active' | awk '{print $3}' | tr -d '.'"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()

            # Disk
            disk = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True, text=True, timeout=5
            ).stdout.split("\n")[1].split() if True else []

            info = {
                "success": True,
                "cpu_user": cpu or "unknown",
                "disk_used": disk[2] if len(disk) > 2 else "unknown",
                "disk_available": disk[3] if len(disk) > 3 else "unknown",
                "disk_percent": disk[4] if len(disk) > 4 else "unknown",
            }
            info["message"] = (
                f"CPU: {info['cpu_user']} | "
                f"Disk: {info['disk_used']} used, {info['disk_available']} free ({info['disk_percent']})"
            )
            return info
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def adjust_volume(self, direction: str, amount: int = 10) -> Dict[str, Any]:
        """Increase or decrease volume by an amount."""
        current = await self.get_volume()
        current_vol = current.get("volume", 50)
        if direction == "up":
            new_vol = min(100, current_vol + amount)
        else:
            new_vol = max(0, current_vol - amount)
        result = await self.set_volume(new_vol)
        result["volume"] = new_vol
        return result

    async def empty_trash(self) -> Dict[str, Any]:
        """Empty the trash."""
        script = 'tell application "Finder" to empty trash'
        return await self._async_script(script)

    async def get_wifi_info(self) -> Dict[str, Any]:
        """Get current WiFi network info — works on all macOS versions."""
        import subprocess
        try:
            # Primary: networksetup (works on all macOS)
            result = subprocess.run(
                ["networksetup", "-getairportnetwork", "en0"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout.strip()
            # Output: "Current Wi-Fi Network: NetworkName"
            if ":" in output:
                ssid = output.split(":", 1)[1].strip()
                if ssid and "not associated" not in ssid.lower():
                    return {
                        "success": True,
                        "ssid": ssid,
                        "message": f"Connected to WiFi: {ssid}",
                    }

            # Fallback: try en1
            result2 = subprocess.run(
                ["networksetup", "-getairportnetwork", "en1"],
                capture_output=True, text=True, timeout=5
            )
            output2 = result2.stdout.strip()
            if ":" in output2:
                ssid2 = output2.split(":", 1)[1].strip()
                if ssid2 and "not associated" not in ssid2.lower():
                    return {
                        "success": True,
                        "ssid": ssid2,
                        "message": f"Connected to WiFi: {ssid2}",
                    }

            return {"success": True, "ssid": "Not connected", "message": "Not connected to any WiFi network."}
        except Exception as e:
            return {"success": False, "error": str(e)}