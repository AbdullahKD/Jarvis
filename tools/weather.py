"""
Weather Tool
Uses Open-Meteo API — completely free, no API key required.
Supports location lookup via Open-Meteo geocoding API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import aiohttp

from config.settings import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_LOCATION_NAME

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEO_URL      = "https://geocoding-api.open-meteo.com/v1/search"


class WeatherTool:
    """Fetches current weather and 7-day forecast via Open-Meteo."""

    # ── Geocoding ──────────────────────────────────────────────────────────

    async def geocode(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Convert a city name to lat/lon using Open-Meteo geocoding.
        Returns None if not found.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GEO_URL,
                    params={"name": location, "count": 1, "language": "en", "format": "json"}
                ) as resp:
                    data = await resp.json(content_type=None)
            results = data.get("results", [])
            if not results:
                return None
            r = results[0]
            return {
                "name": f"{r.get('name')}, {r.get('country', '')}".strip(", "),
                "lat": r["latitude"],
                "lon": r["longitude"],
            }
        except Exception:
            return None

    # ── Current weather ────────────────────────────────────────────────────

    async def get_current(
        self,
        lat: float = DEFAULT_LATITUDE,
        lon: float = DEFAULT_LONGITUDE,
        location_name: str = DEFAULT_LOCATION_NAME,
    ) -> Dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m",
                "apparent_temperature", "weather_code",
                "wind_speed_10m", "precipitation",
                "is_day",
            ],
            "timezone": "auto",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(FORECAST_URL, params=params) as resp:
                    data = await resp.json()

            current = data.get("current", {})
            code = current.get("weather_code", 0)

            return {
                "success": True,
                "location": location_name,
                "condition": WMO_CODES.get(code, "Unknown"),
                "temperature_c": current.get("temperature_2m"),
                "feels_like_c": current.get("apparent_temperature"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "wind_kph": current.get("wind_speed_10m"),
                "precipitation_mm": current.get("precipitation"),
                "is_day": bool(current.get("is_day", 1)),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "location": location_name}

    async def get_current_for_location(self, location: str) -> Dict[str, Any]:
        """Get weather for a named city, geocoding it first."""
        geo = await self.geocode(location)
        if not geo:
            return {"success": False, "error": f"Could not find location: {location}"}
        return await self.get_current(geo["lat"], geo["lon"], geo["name"])

    # ── Forecast ───────────────────────────────────────────────────────────

    async def get_forecast(
        self,
        lat: float = DEFAULT_LATITUDE,
        lon: float = DEFAULT_LONGITUDE,
        location_name: str = DEFAULT_LOCATION_NAME,
        days: int = 7,
    ) -> Dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "weather_code", "temperature_2m_max",
                "temperature_2m_min", "precipitation_sum",
            ],
            "timezone": "auto",
            "forecast_days": days,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(FORECAST_URL, params=params) as resp:
                    data = await resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            codes = daily.get("weather_code", [])
            highs = daily.get("temperature_2m_max", [])
            lows  = daily.get("temperature_2m_min", [])
            rain  = daily.get("precipitation_sum", [])

            forecast = [
                {
                    "date": dates[i],
                    "condition": WMO_CODES.get(codes[i], "Unknown"),
                    "high_c": highs[i],
                    "low_c": lows[i],
                    "rain_mm": rain[i],
                }
                for i in range(len(dates))
            ]
            return {"success": True, "location": location_name, "forecast": forecast}
        except Exception as exc:
            return {"success": False, "error": str(exc), "location": location_name}

    async def get_forecast_for_location(self, location: str, days: int = 7) -> Dict[str, Any]:
        """Get forecast for a named city."""
        geo = await self.geocode(location)
        if not geo:
            return {"success": False, "error": f"Could not find location: {location}"}
        return await self.get_forecast(geo["lat"], geo["lon"], geo["name"], days)

    # ── Formatters ─────────────────────────────────────────────────────────

    def format_current(self, weather: Dict[str, Any]) -> str:
        if not weather.get("success"):
            return f"Could not retrieve weather: {weather.get('error', 'unknown error')}"
        return (
            f"Weather in {weather['location']}: {weather['condition']}, "
            f"{weather['temperature_c']}°C "
            f"(feels like {weather['feels_like_c']}°C), "
            f"humidity {weather['humidity_pct']}%, "
            f"wind {weather['wind_kph']} km/h."
        )

    def format_forecast(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return f"Could not retrieve forecast: {data.get('error', 'unknown error')}"
        lines = [f"7-day forecast for {data['location']}:"]
        for day in data["forecast"]:
            lines.append(
                f"  {day['date']}: {day['condition']}, "
                f"{day['low_c']}–{day['high_c']}°C, "
                f"rain {day['rain_mm']}mm"
            )
        return "\n".join(lines)