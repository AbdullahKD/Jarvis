"""
Platform guard — detects whether Jarvis is running on the user's Mac or
inside a Linux cloud container, so Mac-only tools (AppleScript, screencapture,
launchctl, etc.) can degrade gracefully instead of crashing.

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


def is_cloud() -> bool:
    """
    True iff we look like a cloud Linux container. Heuristic — checks for
    Fly.io's FLY_APP_NAME env var first, then falls back to non-macOS.
    """
    import os
    if os.getenv("FLY_APP_NAME"):
        return True
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return True
    if os.getenv("RUNNING_IN_DOCKER"):
        return True
    return not is_mac()


def mac_only_response(feature: str) -> Dict[str, Any]:
    """
    Standard payload returned when a Mac-only feature is invoked from a
    cloud deployment. Shaped like the success-case dicts elsewhere in the
    codebase so the orchestrator's response renderers don't choke.
    """
    return {
        "success": False,
        "error": (
            f"'{feature}' is a macOS-only capability and is not available in "
            "the cloud deployment of Jarvis. Run Jarvis locally on your Mac "
            "to use this feature."
        ),
        "cloud_disabled": True,
        "output": "",
    }
