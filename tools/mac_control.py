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


def _as_quote(text: str) -> str:
    """Escape a Python string for safe embedding inside an AppleScript
    double-quoted literal. Backslashes MUST be escaped before quotes,
    otherwise crafted input like `Safari" \\n do shell script "..."` can
    break out of the literal and run arbitrary AppleScript (which can in
    turn run arbitrary shell commands)."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


# ── Off-loop subprocess ─────────────────────────────────────────────────────
# Every method here is `async`, but many call sites ran subprocess.run
# directly — each blocking the event loop for up to its timeout, stalling
# every other request, every WebSocket and the reminder scheduler while the
# shell command ran. The handful of sites already wrapped in
# `run_in_executor(None, lambda: ...)` were correct and are left alone.
import functools as _functools


async def _run_off_loop(*args, **kwargs):
    """subprocess.run, executed in a worker thread."""
    return await asyncio.to_thread(_functools.partial(subprocess.run, *args, **kwargs))


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
        # Off macOS, replace every public async method with a stub that
        # returns a friendly "not supported on this platform" payload. This
        # avoids editing each individual method body.
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
        """Mute the system, then read-back to confirm."""
        await self._async_script("set volume with output muted")
        # Read-back: AppleScript returns 0 even when blocked, so verify.
        state = await self._async_script("output muted of (get volume settings)")
        if state.get("success") and "true" in state.get("output", "").lower():
            return {"success": True, "muted": True, "message": "Muted."}
        return {
            "success": False,
            "muted": False,
            "error": "Could not mute — Bluetooth or external audio devices may not respect software mute. Try the keyboard mute key.",
        }

    async def unmute(self) -> Dict[str, Any]:
        """Unmute the system, then read-back to confirm."""
        await self._async_script("set volume without output muted")
        state = await self._async_script("output muted of (get volume settings)")
        if state.get("success") and "false" in state.get("output", "").lower():
            return {"success": True, "muted": False, "message": "Unmuted."}
        return {
            "success": False,
            "muted": True,
            "error": "Could not unmute — try the keyboard volume key.",
        }

    async def get_mute(self) -> Dict[str, Any]:
        """Return whether the system is currently muted."""
        result = await self._async_script("output muted of (get volume settings)")
        if not result.get("success"):
            return result
        is_muted = "true" in result.get("output", "").lower()
        return {
            "success": True,
            "muted": is_muted,
            "message": "Muted." if is_muted else "Not muted.",
        }

    # ── Brightness ─────────────────────────────────────────────────────────

    async def set_brightness(self, level: float) -> Dict[str, Any]:
        """
        Set display brightness on a 0–100 scale (to match volume).

        Backward compatible: a value in the 0.0–1.0 range is also accepted
        and treated as a fraction, so older callers passing e.g. 0.5 still
        mean "50%". Anything > 1 is treated as a 0–100 percentage.

        Apple removed direct AppleScript brightness control. We try three
        paths in order:
          1. `brightness` CLI from Homebrew — exact, deterministic.
          2. Keyboard keystroke F1/F2 nudges — approximate. Estimate the
             number of nudges needed by reading the *current* brightness
             via `brightness -l` if available; otherwise default to 8
             nudges (covers the full range from min→max in either direction).
          3. Honest failure with a one-line install command.
        """
        import subprocess
        # Normalise to a 0.0–1.0 fraction for the `brightness` CLI.
        # 0–100 scale is the documented input; 0–1 is accepted for back-compat.
        level = float(level)
        if level > 1:
            level = level / 100.0
        level = max(0.0, min(1.0, level))

        # Path 1: precise via Homebrew `brightness`
        try:
            result = (await _run_off_loop(
                ["brightness", str(level)],
                capture_output=True, text=True, timeout=5,
            ))
            if result.returncode == 0:
                # Verify by reading back so we report the REAL state.
                read = (await _run_off_loop(
                    ["brightness", "-l"], capture_output=True, text=True, timeout=5,
                ))
                actual_level = level
                if read.returncode == 0:
                    import re as _re
                    m = _re.search(r"brightness\s+([\d.]+)", read.stdout)
                    if m:
                        try:
                            actual_level = float(m.group(1))
                        except ValueError:
                            pass
                return {
                    "success": True,
                    "level": actual_level,
                    "message": f"Brightness set to {int(round(actual_level * 100))}.",
                }
            # CLI present but the command failed — surface the real reason.
            return {
                "success": False,
                "error": f"Brightness command failed: {result.stderr.strip() or 'unknown error'}",
            }
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Brightness command timed out."}
        except Exception as e:
            return {"success": False, "error": f"Brightness error: {e}"}

        # No usable `brightness` CLI. We attempt a best-effort keystroke
        # nudge, but we CANNOT verify it moved the slider — the media keys
        # silently no-op without Accessibility permission and don't map to
        # brightness on every Mac. So we report FAILURE honestly rather than
        # claiming a success that may not have happened (which previously made
        # the LLM tell the user "brightness adjusted" when nothing changed).
        try:
            key_code = 145 if level < 0.5 else 144  # 145 dim, 144 brighten
            script = "\n".join([
                'tell application "System Events"',
                *[f'    key code {key_code}' for _ in range(16)],
                'end tell',
            ])
            await self._async_script(script)
        except Exception:
            pass

        return {
            "success": False,
            "error": (
                f"Couldn't reliably set brightness to {int(round(level * 100))}. "
                "The 'brightness' helper isn't installed, so I can't set the "
                "display level directly. Install it once with: brew install "
                "brightness — then brightness commands work exactly. (I sent "
                "brightness keys as a fallback, but that's unverified and may "
                "need Accessibility permission.)"
            ),
        }

    async def get_brightness(self) -> Dict[str, Any]:
        """Read current display brightness (0.0–1.0) via Homebrew `brightness -l`."""
        import subprocess, re as _re
        try:
            result = (await _run_off_loop(
                ["brightness", "-l"], capture_output=True, text=True, timeout=5,
            ))
            if result.returncode == 0:
                m = _re.search(r"brightness\s+([\d.]+)", result.stdout)
                if m:
                    level = float(m.group(1))
                    return {
                        "success": True,
                        "level": level,
                        "message": f"Brightness is at {int(level * 100)}%.",
                    }
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return {
            "success": False,
            "error": "Brightness read requires `brew install brightness`.",
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
            _safe_app = _as_quote(app_name)
            script = (
                f'tell application "{_safe_app}"\n'
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
        script = f'tell application "{_as_quote(app_name)}" to quit'
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
        """
        Read the current clipboard contents.

        Uses pbpaste rather than `the clipboard` AppleScript, because
        AppleScript's getter blows up on non-text clipboard content (images,
        files) with a confusing "Can't make «class furl» into type Unicode
        text" error. pbpaste returns text or empty cleanly.
        """
        import asyncio as _aio, subprocess as _sp
        loop = _aio.get_event_loop()
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: _sp.run(["pbpaste"], capture_output=True, text=True, timeout=5),
            )
            text = proc.stdout or ""
            return {
                "success": True,
                "text": text,
                "message": f"Clipboard: {text[:200]}" if text else "Clipboard is empty.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_clipboard(self, text: str) -> Dict[str, Any]:
        """
        Write text to the clipboard.

        OLD: piped through `osascript -e 'set the clipboard to "..."'`
        which only escaped `"` — backslashes, newlines, and embedded
        single quotes broke the AppleScript with cryptic syntax errors.

        NEW: pipe via pbcopy through stdin. Any byte sequence works,
        no escaping needed. Returns success only after a pbpaste read-back
        confirms the write took.
        """
        import asyncio as _aio, subprocess as _sp
        loop = _aio.get_event_loop()
        try:
            def _do_set():
                proc = _sp.run(
                    ["pbcopy"],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr or "pbcopy failed")
                # Read-back to confirm
                back = _sp.run(["pbpaste"], capture_output=True, text=True, timeout=5)
                return back.stdout == text
            ok = await loop.run_in_executor(None, _do_set)
            if ok:
                return {
                    "success": True,
                    "message": f"Copied {len(text)} characters to the clipboard.",
                }
            return {
                "success": False,
                "error": "Clipboard write didn't verify — system may have refused it.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Notifications ──────────────────────────────────────────────────────

    async def send_notification(
        self,
        message: str,
        title: str = "Jarvis",
        subtitle: str = "",
    ) -> Dict[str, Any]:
        """Send a macOS notification."""
        # _as_quote escapes backslashes BEFORE quotes — the old quote-only
        # escaping still let `\"` in a message break out of the literal.
        safe_msg = _as_quote(message)
        safe_title = _as_quote(title)
        safe_sub = _as_quote(subtitle)
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
            result = (await _run_off_loop(
                ["pmset", "-g", "batt"],
                capture_output=True, text=True, timeout=5
            ))
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
        script = f'tell application "System Events" to set visible of process "{_as_quote(app_name)}" to false'
        return await self._async_script(script)

    async def take_screenshot(self, save_path: str = "~/Desktop/screenshot.png") -> Dict[str, Any]:
        """Take a screenshot and save to file."""
        import subprocess, os
        path = os.path.expanduser(save_path)
        try:
            result = (await _run_off_loop(
                ["screencapture", "-x", path],
                capture_output=True, text=True, timeout=10
            ))
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
            result = (await _run_off_loop(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=5,
            ))
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
            cpu = (await _run_off_loop(
                ["bash", "-c", "top -l 1 | grep 'CPU usage' | awk '{print $3}'"],
                capture_output=True, text=True, timeout=5
            )).stdout.strip()

            # RAM
            ram = (await _run_off_loop(
                ["bash", "-c", "vm_stat | grep 'Pages active' | awk '{print $3}' | tr -d '.'"],
                capture_output=True, text=True, timeout=5
            )).stdout.strip()

            # Disk
            disk = (await _run_off_loop(
                ["df", "-h", "/"],
                capture_output=True, text=True, timeout=5
            )).stdout.split("\n")[1].split() if True else []

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
        """
        Bump volume up or down by `amount`. Returns the *actual* new
        volume (read back from the system), not the target — Bluetooth and
        external audio sometimes refuse software volume changes silently.
        """
        current = await self.get_volume()
        current_vol = current.get("volume", 50)
        if direction == "up":
            target = min(100, current_vol + amount)
        else:
            target = max(0, current_vol - amount)
        result = await self.set_volume(target)
        # set_volume already does a read-back and populates "volume" with
        # the actual measured value. Do NOT overwrite that with the target.
        if "volume" not in result:
            result["volume"] = target
        result["target"] = target
        result["previous"] = current_vol
        actual = result.get("volume", target)
        verb = "up" if direction == "up" else "down"
        if actual != current_vol:
            result["message"] = f"Volume turned {verb} from {current_vol}% to {actual}%."
        else:
            result["success"] = False
            result["message"] = (
                f"Volume didn't change (still {actual}%). External or Bluetooth "
                "output may be ignoring software volume — try the keyboard volume keys."
            )
        return result

    async def empty_trash(self) -> Dict[str, Any]:
        """Empty the trash."""
        script = 'tell application "Finder" to empty trash'
        return await self._async_script(script)

    async def get_wifi_info(self) -> Dict[str, Any]:
        """
        Get current WiFi network info.

        Dynamically discovers the actual WiFi interface name from
        `networksetup -listallhardwareports` instead of guessing en0/en1.
        On modern Macs the WiFi adapter can be en0, en1, or something else
        depending on which other hardware is plugged in.
        """
        import subprocess
        try:
            # Step 1: find the WiFi interface name from the hardware ports
            # listing. Format:
            #   Hardware Port: Wi-Fi
            #   Device: en0
            #   Ethernet Address: ...
            ports = (await _run_off_loop(
                ["networksetup", "-listallhardwareports"],
                capture_output=True, text=True, timeout=5,
            ))
            wifi_iface = None
            lines = (ports.stdout or "").splitlines()
            for i, line in enumerate(lines):
                if "wi-fi" in line.lower() or "airport" in line.lower():
                    # next "Device:" line carries the interface name
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if "device:" in lines[j].lower():
                            wifi_iface = lines[j].split(":", 1)[1].strip()
                            break
                    if wifi_iface:
                        break

            # Step 2: query the active SSID on that interface
            candidates = []
            if wifi_iface:
                candidates.append(wifi_iface)
            # Belt-and-braces: also try the legacy interface names so we
            # still work on a stripped-down macOS where listallhardwareports
            # returns unexpected output.
            for c in ("en0", "en1", "en2"):
                if c not in candidates:
                    candidates.append(c)

            for iface in candidates:
                result = (await _run_off_loop(
                    ["networksetup", "-getairportnetwork", iface],
                    capture_output=True, text=True, timeout=5,
                ))
                output = (result.stdout or "").strip()
                if ":" in output:
                    ssid = output.split(":", 1)[1].strip()
                    if ssid and "not associated" not in ssid.lower() and "you are not" not in ssid.lower():
                        # Best-effort signal strength via `ipconfig getsummary`
                        # (no admin needed, available on all modern macOS).
                        signal = None
                        try:
                            summary = (await _run_off_loop(
                                ["ipconfig", "getsummary", iface],
                                capture_output=True, text=True, timeout=3,
                            ))
                            import re as _re
                            m = _re.search(r"RSSI\s*[:=]\s*(-?\d+)", summary.stdout)
                            if m:
                                signal = int(m.group(1))
                        except Exception:
                            pass
                        msg = f"Connected to WiFi: {ssid}"
                        if signal is not None:
                            quality = (
                                "excellent" if signal > -50
                                else "good" if signal > -65
                                else "fair" if signal > -75
                                else "weak"
                            )
                            msg += f" (signal {signal} dBm, {quality})"
                        return {
                            "success": True,
                            "ssid": ssid,
                            "interface": iface,
                            "rssi": signal,
                            "message": msg,
                        }

            return {
                "success": True,
                "ssid": "Not connected",
                "message": "Not connected to any WiFi network.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}