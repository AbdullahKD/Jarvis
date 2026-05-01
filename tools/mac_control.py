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
        """
        Set display brightness (0.0–1.0).
        Uses keyboard simulation since direct AppleScript brightness
        control is not supported on modern macOS.
        """
        import subprocess
        level = max(0.0, min(1.0, float(level)))

        # Try 'brightness' CLI tool (brew install brightness)
        try:
            result = subprocess.run(
                ["brightness", str(level)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return {"success": True, "level": level}
        except FileNotFoundError:
            pass

        # Fallback: use keyboard F1/F2 simulation to approximate level
        # Calculate steps needed (16 steps from 0 to 100%)
        current_steps = 8  # assume middle
        target_steps = round(level * 16)
        diff = target_steps - current_steps

        if diff > 0:
            key = "F2"  # brightness up
            steps = diff
        else:
            key = "F1"  # brightness down
            steps = abs(diff)

        script = f'tell application "System Events"\n'
        for _ in range(min(steps, 16)):
            script += f'    key code {"144" if key == "F2" else "145"}\n'
        script += 'end tell'

        result = await self._async_script(script)
        result["note"] = f"Brightness adjusted {'up' if diff > 0 else 'down'} ({steps} steps). For precise control, install: brew install brightness"
        return result

    # ── Apps ───────────────────────────────────────────────────────────────

    async def open_app(self, app_name: str, new_window: bool = False) -> Dict[str, Any]:
        """
        Open any application using 'open -a' — works universally on macOS
        for every app without special cases. Always brings window to front.
        """
        import asyncio
        loop = asyncio.get_event_loop()

        if new_window:
            # Try AppleScript new window first, fallback to just opening
            script = (
                f'tell application "{app_name}"\n'
                f'    activate\n'
                f'    make new window\n'
                f'end tell'
            )
            result = await self._async_script(script)
            if result.get("success"):
                return result
            # Fallback for apps that don't support make new window
            cmd = f'open -na "{app_name}"'
        else:
            cmd = f'open -a "{app_name}"'

        # Use subprocess directly — faster and more reliable than osascript for opening apps
        import subprocess as _sp
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: _sp.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10
                )
            )
            if proc.returncode == 0:
                return {"success": True, "output": f"Opened {app_name}"}
            else:
                return {"success": False, "error": proc.stderr.strip() or f"Could not open {app_name}"}
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
        return await self._async_script(
            'tell application "System Events" to keystroke "q" using {control down, command down}'
        )

    async def sleep(self) -> Dict[str, Any]:
        return await self._async_script('tell application "System Events" to sleep')

    async def quit_app(self, app_name: str) -> Dict[str, Any]:
        """Quit an application gracefully."""
        script = f'tell application "{app_name}" to quit'
        return await self._async_script(script)

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

    async def toggle_dark_mode(self) -> Dict[str, Any]:
        """Toggle between dark and light mode."""
        script = (
            'tell application "System Events"\n'
            '    tell appearance preferences\n'
            '        set dark mode to not dark mode\n'
            '    end tell\n'
            'end tell'
        )
        return await self._async_script(script)

    async def get_dark_mode(self) -> Dict[str, Any]:
        """Check if dark mode is enabled."""
        script = (
            'tell application "System Events"\n'
            '    tell appearance preferences\n'
            '        return dark mode\n'
            '    end tell\n'
            'end tell'
        )
        result = await self._async_script(script)
        if result.get("success"):
            is_dark = result.get("output", "").strip().lower() == "true"
            result["dark_mode"] = is_dark
            result["message"] = f"Dark mode is {'on' if is_dark else 'off'}."
        return result

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