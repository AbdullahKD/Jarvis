"""
Platform guard — detects whether Jarvis is running on macOS, so Mac-only tools
(AppleScript, screencapture, launchctl, etc.) can degrade gracefully instead of
crashing. Jarvis is a Mac-local project, but the tests run on whatever machine
is to hand, and that's enough to need the check.

Usage:
    from tools.platform_guard import is_mac, mac_only_response

    async def open_app(name):
        if not is_mac():
            return mac_only_response("open_app")
        ... real implementation ...
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def is_mac() -> bool:
    """True iff we're running on macOS (Darwin)."""
    return sys.platform == "darwin" or platform.system() == "Darwin"


def mac_only_response(feature: str) -> Dict[str, Any]:
    """
    Standard payload returned when a Mac-only feature is invoked off macOS.
    Shaped like the success-case dicts elsewhere in the codebase so the
    orchestrator's response renderers don't choke.
    """
    return {
        "success": False,
        "error": (
            f"'{feature}' is a macOS-only capability and this machine isn't "
            "running macOS."
        ),
        "unsupported_platform": True,
        "output": "",
    }
