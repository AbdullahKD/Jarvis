"""
Markets Tool
Free stock and crypto prices via Yahoo Finance public API.
No API key required.
Tracks: Bitcoin, top tech stocks (AAPL, NVDA, TSLA, MSFT, AMZN, GOOGL)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

TIMEOUT = aiohttp.ClientTimeout(total=10)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Default symbols to track
DEFAULT_SYMBOLS = {
    "BTC-USD":  "Bitcoin",
    "AAPL":     "Apple",
    "NVDA":     "NVIDIA",
    "TSLA":     "Tesla",
    "MSFT":     "Microsoft",
}


class MarketsTool:
    """Fetches real-time stock and crypto prices via Yahoo Finance."""

    async def get_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a single symbol."""
        url = YAHOO_URL.format(symbol=symbol.upper())
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url, params={"interval": "1d", "range": "2d"}) as resp:
                    if resp.status != 200:
                        return {"success": False, "symbol": symbol, "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            result = data["chart"]["result"][0]
            meta = result["meta"]
            current = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("previousClose", meta.get("chartPreviousClose", current))
            change = current - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0

            return {
                "success": True,
                "symbol": symbol,
                "name": DEFAULT_SYMBOLS.get(symbol, symbol),
                "price": round(current, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "currency": meta.get("currency", "USD"),
                "direction": "▲" if change >= 0 else "▼",
            }
        except Exception as e:
            return {"success": False, "symbol": symbol, "error": str(e)}

    async def get_all(self, symbols: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Fetch all tracked symbols concurrently."""
        targets = symbols or DEFAULT_SYMBOLS
        tasks = [self.get_price(sym) for sym in targets.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = []
        for sym, result in zip(targets.keys(), results):
            if isinstance(result, dict) and result.get("success"):
                result["name"] = targets[sym]
                prices.append(result)

        return {
            "success": len(prices) > 0,
            "prices": prices,
        }

    def format_prices(self, data: Dict[str, Any]) -> str:
        """Format market prices for display."""
        if not data.get("success"):
            return "Could not fetch market data."

        prices = data.get("prices", [])
        if not prices:
            return "No market data available."

        lines = ["MARKETS"]
        for p in prices:
            direction = p.get("direction", "")
            change_pct = p.get("change_pct", 0)
            color_hint = "+" if change_pct >= 0 else ""
            currency = "$" if p.get("currency") == "USD" else p.get("currency", "$")

            # Format price nicely
            price = p.get("price", 0)
            if price > 1000:
                price_str = f"{currency}{price:,.0f}"
            else:
                price_str = f"{currency}{price:.2f}"

            lines.append(
                f"  {p['name']:12} {price_str:>12}  "
                f"{direction} {color_hint}{change_pct:.2f}%"
            )

        return "\n".join(lines)