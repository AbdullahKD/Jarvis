"""
Prayer Times Tool
Uses the free Aladhan API — no key required.
Returns prayer times for any location by coordinates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import aiohttp

from config.settings import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_LOCATION_NAME

TIMEOUT = aiohttp.ClientTimeout(total=10)
API_URL = "https://api.aladhan.com/v1/timings/{date}"


class PrayerTimesTool:
    """Fetches daily prayer times via Aladhan API (free, no key)."""

    async def get_times(
        self,
        lat: float = DEFAULT_LATITUDE,
        lon: float = DEFAULT_LONGITUDE,
        location: str = DEFAULT_LOCATION_NAME,
        method: int = 2,  # 2 = Islamic Society of North America
    ) -> Dict[str, Any]:
        """
        Get prayer times for today.

        Args:
            lat, lon:  Coordinates (defaults to High Wycombe)
            location:  Display name
            method:    Calculation method (2 = ISNA, 3 = MWL, 4 = Makkah)

        Returns:
            Dict with prayer times and meta info
        """
        date_str = datetime.now().strftime("%d-%m-%Y")
        url = API_URL.format(date=date_str)

        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(url, params={
                    "latitude": lat,
                    "longitude": lon,
                    "method": method,
                }) as resp:
                    if resp.status != 200:
                        return {"success": False, "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            timings = data["data"]["timings"]
            return {
                "success": True,
                "location": location,
                "date": date_str,
                "fajr":    timings.get("Fajr", ""),
                "sunrise": timings.get("Sunrise", ""),
                "dhuhr":   timings.get("Dhuhr", ""),
                "asr":     timings.get("Asr", ""),
                "maghrib": timings.get("Maghrib", ""),
                "isha":    timings.get("Isha", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def format_times(self, data: Dict[str, Any]) -> str:
        """Format prayer times for display."""
        if not data.get("success"):
            return ""
        return (
            f"PRAYER TIMES — {data['location']}\n"
            f"  Fajr {data['fajr']}  |  Dhuhr {data['dhuhr']}  |  "
            f"Asr {data['asr']}  |  Maghrib {data['maghrib']}  |  Isha {data['isha']}"
        )

    def get_next_prayer(self, data: Dict[str, Any]) -> str:
        """Return the next upcoming prayer."""
        if not data.get("success"):
            return ""
        now = datetime.now().strftime("%H:%M")
        prayers = [
            ("Fajr",    data["fajr"]),
            ("Dhuhr",   data["dhuhr"]),
            ("Asr",     data["asr"]),
            ("Maghrib", data["maghrib"]),
            ("Isha",    data["isha"]),
        ]
        for name, time in prayers:
            if time > now:
                return f"Next prayer: {name} at {time}"
        return f"Next prayer: Fajr tomorrow at {data['fajr']}"