"""
Web Search Tool
Uses DuckDuckGo Instant Answer API — no API key, no rate limits.
Also provides a basic page scraper using aiohttp + BeautifulSoup.
"""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

DDG_URL = "https://api.duckduckgo.com/"
TIMEOUT = aiohttp.ClientTimeout(total=15)
HEADERS = {"User-Agent": "Jarvis/1.0 (AI Assistant; educational project)"}


class WebSearchTool:
    """DuckDuckGo search + lightweight web scraper."""

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search DuckDuckGo and return structured results.

        Args:
            query:       Natural language search query
            max_results: Maximum number of results to return

        Returns:
            Dict with abstract, results list, and related topics
        """
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "no_redirect": 1,
            "skip_disambig": 1,
        }

        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(DDG_URL, params=params) as resp:
                    data = await resp.json(content_type=None)

            results = []

            # Instant answer / abstract
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    })

            return {
                "success": True,
                "query": query,
                "results": results[:max_results],
                "answer_type": data.get("Type", ""),
            }

        except Exception as exc:
            return {"success": False, "query": query, "error": str(exc), "results": []}

    async def scrape(self, url: str, max_chars: int = 3000) -> Dict[str, Any]:
        """
        Scrape text content from a URL.

        Args:
            url:       Page to scrape
            max_chars: Maximum characters to return

        Returns:
            Dict with title and text content
        """
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else url
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = " ".join(p for p in paragraphs if len(p) > 40)

            return {
                "success": True,
                "url": url,
                "title": title,
                "text": text[:max_chars],
                "char_count": len(text),
            }

        except Exception as exc:
            return {"success": False, "url": url, "error": str(exc)}

    def format_results(self, data: Dict[str, Any]) -> str:
        if not data.get("success") or not data.get("results"):
            return f"No results found for: {data.get('query', '')}"
        lines = [f"Search results for \"{data['query']}\":"]
        for i, r in enumerate(data["results"], 1):
            lines.append(f"\n{i}. {r['title']}")
            lines.append(f"   {r['snippet'][:200]}")
            if r.get("url"):
                lines.append(f"   {r['url']}")
        return "\n".join(lines)
