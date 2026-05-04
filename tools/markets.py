"""
Markets Tool
Real-time stock and crypto prices.

Primary:  yfinance library   — handles Yahoo Finance auth automatically
Fallback: Yahoo Finance v7   — direct HTTP (no API key needed)
Fallback: Yahoo Finance v8   — chart endpoint

No API key required for any method.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

import aiohttp

TIMEOUT = aiohttp.ClientTimeout(total=10)
_HEADERS_V7 = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# Shared thread pool for blocking yfinance calls
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="markets")

# Default symbols to track
DEFAULT_SYMBOLS = {
    "BTC-USD":  "Bitcoin",
    "AAPL":     "Apple",
    "NVDA":     "NVIDIA",
    "TSLA":     "Tesla",
    "MSFT":     "Microsoft",
}


class MarketsTool:
    """Fetches real-time stock and crypto prices."""

    async def get_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price for a single symbol.
        Tries yfinance first, falls back to direct Yahoo Finance HTTP.
        """
        sym = symbol.upper()

        # ── Primary: yfinance (handles auth) ───────────────────────────
        result = await self._try_yfinance(sym)
        if result.get("success"):
            return result

        # ── Fallback: v7 quote endpoint ─────────────────────────────────
        result = await self._try_v7([sym])
        if result.get("success") and result.get("prices"):
            p = result["prices"][0]
            p["name"] = DEFAULT_SYMBOLS.get(sym, sym)
            return p

        # ── Last resort: v8 chart ────────────────────────────────────────
        return await self._try_v8(sym)

    async def get_all(self, symbols: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fetch all tracked symbols concurrently.
        Tries bulk yfinance first (most reliable), falls back per-symbol.
        """
        targets = symbols or DEFAULT_SYMBOLS
        sym_list = list(targets.keys())

        # ── Bulk yfinance download ───────────────────────────────────────
        bulk = await self._try_yfinance_bulk(sym_list)
        if bulk.get("success") and bulk.get("prices"):
            for p in bulk["prices"]:
                p["name"] = targets.get(p["symbol"], p.get("name", p["symbol"]))
            return bulk

        # ── Bulk v7 HTTP ─────────────────────────────────────────────────
        bulk = await self._try_v7(sym_list)
        if bulk.get("success") and bulk.get("prices"):
            for p in bulk["prices"]:
                p["name"] = targets.get(p["symbol"], p.get("name", p["symbol"]))
            return bulk

        # ── Concurrent per-symbol yfinance ───────────────────────────────
        tasks = [self._try_yfinance(sym) for sym in sym_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = []
        for sym, result in zip(sym_list, results):
            if isinstance(result, dict) and result.get("success"):
                result["name"] = targets[sym]
                prices.append(result)

        return {"success": bool(prices), "prices": prices}

    # ── yfinance methods ───────────────────────────────────────────────────

    async def _try_yfinance(self, symbol: str) -> Dict[str, Any]:
        """Single symbol via yfinance fast_info."""
        loop = asyncio.get_event_loop()

        def _fetch():
            try:
                import yfinance as yf
                t = yf.Ticker(symbol)
                fi = t.fast_info
                current = fi.last_price
                prev    = fi.previous_close or current
                if current is None:
                    return {"success": False, "symbol": symbol, "error": "no price"}
                change     = current - prev
                change_pct = (change / prev * 100) if prev else 0
                return {
                    "success":    True,
                    "symbol":     symbol,
                    "name":       DEFAULT_SYMBOLS.get(symbol, symbol),
                    "price":      round(float(current), 2),
                    "change":     round(float(change), 2),
                    "change_pct": round(float(change_pct), 2),
                    "currency":   getattr(fi, "currency", "USD") or "USD",
                    "direction":  "▲" if change >= 0 else "▼",
                }
            except Exception as e:
                return {"success": False, "symbol": symbol, "error": str(e)}

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_thread_pool, _fetch), timeout=9
            )
        except asyncio.TimeoutError:
            return {"success": False, "symbol": symbol, "error": "yfinance timeout"}

    async def _try_yfinance_bulk(self, symbols: List[str]) -> Dict[str, Any]:
        """Bulk fetch via yfinance download (one call for all tickers)."""
        loop = asyncio.get_event_loop()

        def _fetch():
            try:
                import yfinance as yf
                import pandas as _pd
                tickers = " ".join(symbols)
                # download returns DataFrame; use period="1d" for speed
                df = yf.download(tickers, period="2d", interval="1d",
                                 auto_adjust=True, progress=False, threads=True)
                if df.empty:
                    return {"success": False, "prices": []}

                prices = []
                for sym in symbols:
                    try:
                        t = yf.Ticker(sym)
                        fi = t.fast_info
                        current = fi.last_price
                        prev    = fi.previous_close or current
                        if current is None:
                            continue
                        change     = current - prev
                        change_pct = (change / prev * 100) if prev else 0
                        prices.append({
                            "success":    True,
                            "symbol":     sym,
                            "name":       DEFAULT_SYMBOLS.get(sym, sym),
                            "price":      round(float(current), 2),
                            "change":     round(float(change), 2),
                            "change_pct": round(float(change_pct), 2),
                            "currency":   getattr(fi, "currency", "USD") or "USD",
                            "direction":  "▲" if change >= 0 else "▼",
                        })
                    except Exception:
                        continue
                return {"success": bool(prices), "prices": prices}
            except Exception as e:
                return {"success": False, "prices": [], "error": str(e)}

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_thread_pool, _fetch), timeout=15
            )
        except asyncio.TimeoutError:
            return {"success": False, "prices": []}

    # ── HTTP fallback methods ──────────────────────────────────────────────

    async def _try_v7(self, symbols: List[str]) -> Dict[str, Any]:
        """Yahoo Finance v7 /quote — single HTTP round-trip for many symbols."""
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        params = {
            "symbols": ",".join(symbols),
            "fields": "regularMarketPrice,regularMarketPreviousClose,"
                      "regularMarketChange,regularMarketChangePercent,currency",
        }
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=_HEADERS_V7
            ) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        return {"success": False, "prices": []}
                    data = await resp.json(content_type=None)

            result_list = data.get("quoteResponse", {}).get("result", [])
            if not result_list:
                return {"success": False, "prices": []}

            prices = []
            for q in result_list:
                current    = q.get("regularMarketPrice", 0)
                prev       = q.get("regularMarketPreviousClose", current)
                change     = q.get("regularMarketChange", current - prev)
                change_pct = q.get("regularMarketChangePercent", 0)
                sym        = q.get("symbol", "")
                if not current:
                    continue
                prices.append({
                    "success":    True,
                    "symbol":     sym,
                    "name":       DEFAULT_SYMBOLS.get(sym, sym),
                    "price":      round(float(current), 2),
                    "change":     round(float(change), 2),
                    "change_pct": round(float(change_pct), 2),
                    "currency":   q.get("currency", "USD"),
                    "direction":  "▲" if change >= 0 else "▼",
                })
            return {"success": bool(prices), "prices": prices}
        except Exception:
            return {"success": False, "prices": []}

    async def _try_v8(self, symbol: str) -> Dict[str, Any]:
        """Yahoo Finance v8 /chart — last-resort single symbol fetch."""
        for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
            url = f"https://{host}/v8/finance/chart/{symbol}"
            try:
                async with aiohttp.ClientSession(
                    timeout=TIMEOUT, headers=_HEADERS_V7
                ) as session:
                    async with session.get(url, params={"interval": "1d", "range": "2d"}) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json(content_type=None)
                chart_result = data.get("chart", {}).get("result", [])
                if not chart_result:
                    continue
                meta       = chart_result[0].get("meta", {})
                current    = meta.get("regularMarketPrice", 0)
                prev       = meta.get("previousClose", meta.get("chartPreviousClose", current))
                change     = current - prev
                change_pct = (change / prev * 100) if prev else 0
                return {
                    "success":    True,
                    "symbol":     symbol,
                    "name":       DEFAULT_SYMBOLS.get(symbol, symbol),
                    "price":      round(float(current), 2),
                    "change":     round(float(change), 2),
                    "change_pct": round(float(change_pct), 2),
                    "currency":   meta.get("currency", "USD"),
                    "direction":  "▲" if change >= 0 else "▼",
                }
            except Exception:
                continue
        return {"success": False, "symbol": symbol, "error": "all endpoints failed"}

    # ── Formatting ─────────────────────────────────────────────────────────

    def format_prices(self, data: Dict[str, Any]) -> str:
        """Format market prices for display."""
        if not data.get("success"):
            return "Could not fetch market data."

        prices = data.get("prices", [])
        if not prices:
            return "No market data available."

        lines = ["MARKETS"]
        for p in prices:
            direction  = p.get("direction", "")
            change_pct = p.get("change_pct", 0)
            color_hint = "+" if change_pct >= 0 else ""
            currency   = "$" if p.get("currency") == "USD" else p.get("currency", "$")

            price = p.get("price", 0)
            price_str = f"{currency}{price:,.0f}" if price > 1000 else f"{currency}{price:.2f}"

            lines.append(
                f"  {p['name']:12} {price_str:>12}  "
                f"{direction} {color_hint}{change_pct:.2f}%"
            )

        return "\n".join(lines)
