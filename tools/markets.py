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
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp

TIMEOUT = aiohttp.ClientTimeout(total=10)
_HEADERS_V7 = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# Shared thread pool for blocking yfinance calls
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="markets")

# ── Price cache with last-known-good fallback ──────────────────────────────────
# The FinEx dashboard polls every ~20s across six symbol baskets. Yahoo Finance
# aggressively rate-limits an IP that fetches this often, after which EVERY
# request returns "possibly delisted; no price data found" and the widgets blank
# out mid-session. Two defences:
#   1. Serve a fresh symbol straight from cache instead of re-hitting Yahoo
#      (TTL below) — collapses the request volume dramatically.
#   2. When a live fetch fails, fall back to the last value we ever saw so a
#      transient block never wipes a populated widget.
_CACHE_TTL = float(os.environ.get("MARKETS_CACHE_TTL_S", "15"))
# {symbol: {"price": {...}, "ts": epoch_seconds}}
_price_cache: Dict[str, Dict[str, Any]] = {}


def _cache_put(price: Dict[str, Any]) -> None:
    sym = price.get("symbol")
    if sym and price.get("success"):
        _price_cache[sym] = {"price": dict(price), "ts": time.time()}


def _cache_get(symbol: str, max_age: Optional[float] = None) -> Optional[Dict[str, Any]]:
    entry = _price_cache.get(symbol)
    if not entry:
        return None
    if max_age is not None and (time.time() - entry["ts"]) > max_age:
        return None
    return dict(entry["price"])


def _add_pct_alias(price: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every price dict carries BOTH change_pct and change_percent.

    Backend historically emitted only `change_pct`; the FinEx UI reads
    `change_percent`. Emit both so neither client breaks.
    """
    if "change_pct" in price and "change_percent" not in price:
        price["change_percent"] = price["change_pct"]
    elif "change_percent" in price and "change_pct" not in price:
        price["change_pct"] = price["change_percent"]
    return price

# Default symbols to track
DEFAULT_SYMBOLS = {
    # Major indices
    "^GSPC":    "S&P 500",
    "^IXIC":    "NASDAQ",
    "^DJI":     "Dow Jones",
    "^FTSE":    "FTSE 100",
    "^RUT":     "Russell 2000",
    # Commodities
    "GC=F":     "Gold",
    "CL=F":     "Crude Oil",
    # Crypto
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    # Big Tech
    "AAPL":     "Apple",
    "MSFT":     "Microsoft",
    "NVDA":     "NVIDIA",
    "GOOGL":    "Google",
    "AMZN":     "Amazon",
    "META":     "Meta",
    "TSLA":     "Tesla",
}


class MarketsTool:
    """Fetches real-time stock and crypto prices."""

    async def get_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price for a single symbol.
        Tries yfinance first, falls back to direct Yahoo Finance HTTP.
        """
        sym = symbol.upper()

        # ── Fresh cache hit ─────────────────────────────────────────────
        cached = _cache_get(sym, max_age=_CACHE_TTL)
        if cached:
            return cached

        # ── Primary: yfinance (handles auth) ───────────────────────────
        result = await self._try_yfinance(sym)
        if not result.get("success"):
            # ── Fallback: v7 quote endpoint ────────────────────────────
            v7 = await self._try_v7([sym])
            if v7.get("success") and v7.get("prices"):
                result = v7["prices"][0]
            else:
                # ── Last resort: v8 chart ──────────────────────────────
                result = await self._try_v8(sym)

        if result.get("success"):
            result["name"] = DEFAULT_SYMBOLS.get(sym, sym)
            _add_pct_alias(result)
            _cache_put(result)
            return result

        # Live fetch failed — serve last-known-good rather than an error.
        lkg = _cache_get(sym)
        if lkg:
            lkg["stale"] = True
            return lkg
        return result

    async def get_all(self, symbols: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fetch all tracked symbols, with a cache + last-known-good fallback so
        widgets never blank out when Yahoo rate-limits us.

        Strategy:
          1. Serve symbols that are still fresh in cache (no network call).
          2. Fetch only the stale symbols (bulk v7 HTTP → bulk yfinance →
             per-symbol yfinance).
          3. For any symbol still missing after fetching, fall back to its last
             known value from cache (even if stale).
        """
        targets = symbols or DEFAULT_SYMBOLS
        sym_list = list(targets.keys())

        by_symbol: Dict[str, Dict[str, Any]] = {}

        # ── 1. Fresh from cache ──────────────────────────────────────────
        stale: List[str] = []
        for sym in sym_list:
            cached = _cache_get(sym, max_age=_CACHE_TTL)
            if cached:
                by_symbol[sym] = cached
            else:
                stale.append(sym)

        # ── 2. Fetch the stale ones ──────────────────────────────────────
        if stale:
            fetched = await self._fetch_fresh(stale)
            for p in fetched:
                p["name"] = targets.get(p["symbol"], p.get("name", p["symbol"]))
                _add_pct_alias(p)
                _cache_put(p)
                by_symbol[p["symbol"]] = p

        # ── 3. Last-known-good backfill for anything still missing ───────
        for sym in sym_list:
            if sym not in by_symbol:
                lkg = _cache_get(sym)  # any age
                if lkg:
                    lkg["stale"] = True
                    by_symbol[sym] = lkg

        # Preserve the requested order and re-stamp names
        prices = []
        for sym in sym_list:
            if sym in by_symbol:
                p = by_symbol[sym]
                p["name"] = targets.get(sym, p.get("name", sym))
                _add_pct_alias(p)
                prices.append(p)

        return {"success": bool(prices), "prices": prices}

    async def _fetch_fresh(self, sym_list: List[str]) -> List[Dict[str, Any]]:
        """Fetch a list of symbols live. Tries the lightweight v7 HTTP batch
        first (one round-trip, no yfinance internals), then yfinance bulk, then
        per-symbol yfinance. Returns whatever succeeded."""
        # ── Bulk v7 HTTP — cheapest, single request, no 1y/5d history calls ──
        bulk = await self._try_v7(sym_list)
        if bulk.get("success") and bulk.get("prices"):
            return bulk["prices"]

        # ── Bulk yfinance download ───────────────────────────────────────
        bulk = await self._try_yfinance_bulk(sym_list)
        if bulk.get("success") and bulk.get("prices"):
            return bulk["prices"]

        # ── Concurrent per-symbol yfinance (last resort) ─────────────────
        tasks = [self._try_yfinance(sym) for sym in sym_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        prices = []
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                prices.append(result)
        return prices

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
                    "change_percent": round(float(change_pct), 2),
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
        """Bulk fetch via a single yfinance download for all tickers.

        Prices are read straight out of the returned DataFrame. We deliberately
        do NOT touch ``Ticker.fast_info`` here: accessing fast_info per symbol
        fires extra history requests (period=1y and period=5d) which both
        multiplies our request volume ~Nx and produces the
        "possibly delisted; no price data found" log spam when Yahoo
        rate-limits. One download call → one set of prices.
        """
        loop = asyncio.get_event_loop()

        def _fetch():
            try:
                import yfinance as yf
                tickers = " ".join(symbols)
                # Two days of daily candles: last close = current, the one
                # before = previous close, for the day-change calculation.
                df = yf.download(tickers, period="2d", interval="1d",
                                 auto_adjust=True, progress=False, threads=True)
                if df is None or df.empty:
                    return {"success": False, "prices": []}

                # yfinance returns a single-level frame for one ticker and a
                # column-MultiIndex (field, ticker) for many. Normalise to a
                # helper that pulls the Close series for a symbol.
                multi = isinstance(df.columns, type(df.columns)) and getattr(df.columns, "nlevels", 1) > 1

                def _close_series(sym):
                    try:
                        if multi:
                            return df["Close"][sym].dropna()
                        return df["Close"].dropna()
                    except Exception:
                        return None

                prices = []
                for sym in symbols:
                    closes = _close_series(sym)
                    if closes is None or len(closes) == 0:
                        continue
                    current = float(closes.iloc[-1])
                    prev    = float(closes.iloc[-2]) if len(closes) >= 2 else current
                    if not current:
                        continue
                    change     = current - prev
                    change_pct = (change / prev * 100) if prev else 0
                    prices.append({
                        "success":    True,
                        "symbol":     sym,
                        "name":       DEFAULT_SYMBOLS.get(sym, sym),
                        "price":      round(current, 2),
                        "change":     round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "change_percent": round(change_pct, 2),
                        "currency":   "USD",
                        "direction":  "▲" if change >= 0 else "▼",
                    })
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
                    "change_percent": round(float(change_pct), 2),
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
                    "change_percent": round(float(change_pct), 2),
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
