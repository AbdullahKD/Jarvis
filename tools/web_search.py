"""
Web Search Tool
Multi-source search pipeline:
1. DuckDuckGo Instant Answer API
2. DuckDuckGo HTML scraping (when API returns no content)
3. Wikipedia API (for factual/encyclopedic queries)
4. Direct page scraping

Query parser cleans and optimises queries before searching.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlencode

import aiohttp
from bs4 import BeautifulSoup

TIMEOUT = aiohttp.ClientTimeout(total=20)

# Browser-like headers for DuckDuckGo HTML scraping
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Wikipedia requires descriptive User-Agent
WIKI_HEADERS = {
    "User-Agent": "Jarvis/1.0 (BNU Computer Science Dissertation) Python/aiohttp"
}

DDG_API_URL  = "https://api.duckduckgo.com/"
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
WIKI_URL     = "https://en.wikipedia.org/w/api.php"

# Words to strip from queries before searching
# Only strip explicit search trigger phrases — NOT question words like "who is", "what is"
# Question words help the search engine understand context
NOISE_PHRASES = [
    "search for", "search the web for", "search the web",
    "look up information about", "look up info on", "look up",
    "google for", "google search for",
    "find information about", "find info on", "find out about",
    "can you find", "can you search for", "please search for",
    "give me information about", "give me info on",
    "i want to know about", "i need information on",
    "tell me more about", "tell me about",
]


class WebSearchTool:
    """
    Multi-source web search with intelligent query parsing.
    Automatically selects the best source based on query type.
    """

    # ── Query parser ───────────────────────────────────────────────────────

    def parse_query(self, raw: str) -> str:
        """
        Clean and optimise a natural language query for web search.
        Strips trigger phrases, stop words, and normalises the query.
        """
        q = raw.strip()

        # Strip noise phrases from the start
        q_lower = q.lower()
        for phrase in sorted(NOISE_PHRASES, key=len, reverse=True):
            if q_lower.startswith(phrase):
                q = q[len(phrase):].strip()
                q_lower = q.lower()
                break
            # Also check mid-sentence
            if " " + phrase + " " in " " + q_lower + " ":
                idx = q_lower.index(phrase)
                q = q[idx + len(phrase):].strip()
                q_lower = q.lower()
                break

        # Remove trailing punctuation and filler
        q = q.rstrip("?.!,").strip()
        q = re.sub(r'\s+', ' ', q)

        return q if q else raw.strip()

    def detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to select the best search source.
        Returns: 'factual', 'technical', 'person', 'news', 'general'
        """
        q = query.lower()

        # Technical / software topics — Wikipedia often has good coverage
        tech_signals = ["framework", "library", "database", "algorithm", "protocol",
                       "api", "architecture", "model", "system", "language", "tool"]
        if any(s in q for s in tech_signals):
            return "technical"

        # Person queries
        person_signals = ["who is", "who was", "founder", "ceo", "inventor", "born"]
        if any(s in q for s in person_signals):
            return "person"

        # News/current events — DDG is better
        news_signals = ["latest", "recent", "2024", "2025", "2026", "news", "today",
                       "announced", "released", "launched"]
        if any(s in q for s in news_signals):
            return "news"

        return "general"

    # ── Main search ────────────────────────────────────────────────────────

    async def search(self, raw_query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web with automatic source selection and fallbacks.

        Pipeline:
        1. Parse and clean the query
        2. Try DuckDuckGo Instant Answer API
        3. Try DuckDuckGo HTML scraping (with relevance check)
        4. Try Wikipedia
        5. Return best results
        """
        query = self.parse_query(raw_query)
        query_type = self.detect_query_type(query)

        # Try DuckDuckGo API first
        ddg_api = await self._ddg_api(query, max_results)
        if ddg_api.get("success") and ddg_api.get("results"):
            return ddg_api

        # Try DuckDuckGo HTML scraping with relevance check
        ddg_html = await self._ddg_html(query, max_results)
        if ddg_html.get("success") and ddg_html.get("results"):
            # Relevance check — trust DDG HTML results, they are generally accurate
            # Only filter if we have a clear mismatch (e.g. ads that slipped through)
            query_words = set(w.lower() for w in query.split() if len(w) > 3)
            if query_words:
                relevant = [
                    r for r in ddg_html["results"]
                    if any(w in r["title"].lower() or w in r["snippet"].lower()[:300]
                           for w in query_words)
                ]
                # Only apply filter if it keeps at least 1 result
                # Otherwise trust DDG's ranking
                if relevant:
                    ddg_html["results"] = relevant
            if ddg_html["results"]:
                return ddg_html

        # Try Wikipedia
        wiki = await self._wiki(query, max_results)
        if wiki.get("success") and wiki.get("results"):
            return wiki

        # Last resort: try core topic only
        words = [w for w in query.split() if len(w) > 3]
        if len(words) > 1:
            core = " ".join(words[:2])
            wiki2 = await self._wiki(core, max_results)
            if wiki2.get("success") and wiki2.get("results"):
                return wiki2

            # Also try DDG HTML with core topic
            ddg2 = await self._ddg_html(core, max_results)
            if ddg2.get("success") and ddg2.get("results"):
                return ddg2

        return {
            "success": False,
            "query": query,
            "original_query": raw_query,
            "results": [],
            "error": "No results found across all sources",
        }

    # ── DuckDuckGo Instant Answer API ─────────────────────────────────────

    async def _ddg_api(self, query: str, max_results: int) -> Dict[str, Any]:
        """DuckDuckGo Instant Answer API — best for direct factual answers."""
        params = {
            "q": query, "format": "json",
            "no_html": "1", "no_redirect": "1", "skip_disambig": "1",
        }
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=BROWSER_HEADERS
            ) as s:
                async with s.get(DDG_API_URL, params=params) as r:
                    if r.status not in (200, 202):
                        return {"success": False, "results": []}
                    text = await r.text()
                    if not text or not text.strip().startswith("{"):
                        return {"success": False, "results": []}
                    import json
                    data = json.loads(text)

            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", "DuckDuckGo"),
                })

            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(topic, dict) and topic.get("Text") and len(topic["Text"]) > 30:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    })

            return {
                "success": len(results) > 0,
                "query": query,
                "results": results[:max_results],
                "source": "duckduckgo_api",
            }
        except Exception as e:
            return {"success": False, "results": [], "error": str(e)}

    # ── DuckDuckGo HTML scraping ───────────────────────────────────────────

    async def _ddg_html(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Scrape DuckDuckGo HTML results — more results than the API.
        Uses the lite HTML endpoint which is scrapable.
        """
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=BROWSER_HEADERS
            ) as s:
                async with s.post(
                    DDG_HTML_URL,
                    data={"q": query, "b": "", "kl": "uk-en"},
                ) as r:
                    if r.status != 200:
                        return {"success": False, "results": []}
                    html = await r.text()

            soup = BeautifulSoup(html, "html.parser")
            results = []

            # DuckDuckGo HTML result structure
            # Ad indicators to filter out
            ad_signals = [
                "result--ad", "badge--ad", "is-ad",
            ]
            ad_text_signals = [
                "Ad", "Sponsored", "privacy protected by DuckDuckGo. Ad",
                "managed by Microsoft", "Viewing ads is privacy"
            ]

            for result in soup.select(".result"):
                if len(results) >= max_results:
                    break

                # Skip ads by CSS class
                result_classes = " ".join(result.get("class", []))
                if any(ad in result_classes for ad in ad_signals):
                    continue

                title_el = result.select_one(".result__title")
                snippet_el = result.select_one(".result__snippet")
                url_el = result.select_one(".result__url")

                if not title_el or not snippet_el:
                    continue

                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True)
                url = url_el.get_text(strip=True) if url_el else ""

                # Skip ads by content
                if any(ad in title or ad in snippet for ad in ad_text_signals):
                    continue

                # Skip if URL looks like an ad redirect
                if "ad_domain" in url or url.startswith("//duckduckgo.com/y.js"):
                    continue

                if title and snippet and len(snippet) > 20:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "source": "DuckDuckGo",
                    })

            return {
                "success": len(results) > 0,
                "query": query,
                "results": results,
                "source": "duckduckgo_html",
            }
        except Exception as e:
            return {"success": False, "results": [], "error": str(e)}

    # ── Wikipedia ─────────────────────────────────────────────────────────

    async def _wiki(self, query: str, max_results: int) -> Dict[str, Any]:
        """Wikipedia API — best for technical, historical, and encyclopedic queries."""
        try:
            # Search
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=WIKI_HEADERS
            ) as s:
                async with s.get(WIKI_URL, params={
                    "action": "query", "list": "search",
                    "srsearch": query, "srlimit": str(min(max_results, 5)),
                    "format": "json", "utf8": "1",
                }) as r:
                    if r.status != 200:
                        return {"success": False, "results": []}
                    search_data = await r.json(content_type=None)

            hits = search_data.get("query", {}).get("search", [])
            if not hits:
                return {"success": False, "results": []}

            # Relevance filter — title must share a meaningful word with query
            query_words = set(w.lower() for w in query.split() if len(w) > 3)
            relevant_hits = [
                h for h in hits
                if any(w in h["title"].lower() for w in query_words)
            ] or hits[:3]  # fallback: take top 3 if none match

            # Get extracts
            page_ids = "|".join(str(h["pageid"]) for h in relevant_hits[:3])
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=WIKI_HEADERS
            ) as s:
                async with s.get(WIKI_URL, params={
                    "action": "query", "pageids": page_ids,
                    "prop": "extracts|info",
                    "exintro": "1", "explaintext": "1", "exsentences": "5",
                    "inprop": "url", "format": "json", "utf8": "1",
                }) as r:
                    extract_data = await r.json(content_type=None)

            results = []
            for page in extract_data.get("query", {}).get("pages", {}).values():
                extract = page.get("extract", "").strip()
                if extract and len(extract) > 50:
                    results.append({
                        "title": page.get("title", ""),
                        "snippet": extract[:600],
                        "url": page.get("fullurl",
                            "https://en.wikipedia.org/wiki/" + quote_plus(page.get("title", ""))),
                        "source": "Wikipedia",
                    })

            # Final relevance check
            final = [
                r for r in results
                if any(w in r["title"].lower() or w in r["snippet"].lower()[:200]
                       for w in query_words)
            ] or results

            return {
                "success": len(final) > 0,
                "query": query,
                "results": final[:max_results],
                "source": "wikipedia",
            }
        except Exception as e:
            return {"success": False, "results": [], "error": str(e)}

    # ── Page scraper ───────────────────────────────────────────────────────

    async def scrape(self, url: str, max_chars: int = 3000) -> Dict[str, Any]:
        """Scrape text content from any URL."""
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=BROWSER_HEADERS
            ) as s:
                async with s.get(url) as r:
                    html = await r.text()

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else url
            text = " ".join(
                p.get_text(strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 40
            )
            return {
                "success": True, "url": url,
                "title": title, "text": text[:max_chars],
            }
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}

    def format_results(self, data: Dict[str, Any]) -> str:
        if not data.get("success") or not data.get("results"):
            return f"No results found for: {data.get('query', '')}"
        lines = [f"Results for \"{data['query']}\" (via {data.get('source', 'web')}):"]
        for i, r in enumerate(data["results"], 1):
            lines.append(f"\n{i}. {r['title']}")
            lines.append(f"   {r['snippet'][:250]}")
        return "\n".join(lines)