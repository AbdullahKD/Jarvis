"""
News Tool
Aggregates headlines from free RSS feeds.
No API key required. Sources: BBC, Reuters, Hacker News, TechCrunch, Guardian.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import RSS_FEEDS

TIMEOUT = aiohttp.ClientTimeout(total=10)
HEADERS = {"User-Agent": "Jarvis/1.0 (AI Assistant; educational project)"}


class NewsTool:
    """Fetches and filters headlines from RSS feeds."""

    async def get_headlines(
        self,
        source: str = "bbc",
        topic: Optional[str] = None,
        max_items: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch headlines from an RSS feed.

        Args:
            source:    Feed name ('bbc', 'reuters', 'hackernews', etc.)
            topic:     Optional keyword filter
            max_items: Maximum headlines to return

        Returns:
            Dict with source name and list of headline dicts
        """
        url = RSS_FEEDS.get(source.lower())
        if not url:
            return {
                "success": False,
                "error": f"Unknown source '{source}'. Available: {list(RSS_FEEDS.keys())}",
            }

        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    xml_text = await resp.text()

            items = self._parse_rss(xml_text)

            if topic:
                topic_lower = topic.lower()
                items = [
                    i for i in items
                    if topic_lower in i["title"].lower()
                    or topic_lower in i["description"].lower()
                ]

            return {
                "success": True,
                "source": source,
                "topic_filter": topic,
                "items": items[:max_items],
                "fetched_at": datetime.now().isoformat(),
            }

        except Exception as exc:
            return {"success": False, "source": source, "error": str(exc)}

    async def get_all_headlines(
        self,
        topic: Optional[str] = None,
        max_per_source: int = 3,
    ) -> Dict[str, Any]:
        """Fetch headlines from all sources simultaneously."""
        import asyncio

        tasks = [
            self.get_headlines(source, topic, max_per_source)
            for source in RSS_FEEDS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items = []
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                all_items.extend(result.get("items", []))

        return {
            "success": True,
            "topic_filter": topic,
            "items": all_items,
            "total": len(all_items),
        }

    def _parse_rss(self, xml_text: str) -> List[Dict[str, str]]:
        """Parse RSS XML into a list of item dicts."""
        items = []
        try:
            root = ET.fromstring(xml_text)
            channel = root.find("channel") or root

            for item in channel.findall("item"):
                title = item.findtext("title", "").strip()
                desc  = item.findtext("description", "").strip()
                link  = item.findtext("link", "").strip()
                pub   = item.findtext("pubDate", "").strip()

                # Strip basic HTML from description
                desc = ET.fromstring(f"<x>{desc}</x>").text or desc if "<" in desc else desc

                if title:
                    items.append({
                        "title": title,
                        "description": desc[:300],
                        "url": link,
                        "published": pub,
                    })
        except ET.ParseError:
            pass
        return items

    def format_headlines(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return f"Could not fetch news: {data.get('error', 'unknown error')}"
        items = data.get("items", [])
        if not items:
            return "No headlines found."
        source = data.get("source", "all sources")
        lines = [f"Headlines from {source}:"]
        for i, item in enumerate(items, 1):
            lines.append(f"\n{i}. {item['title']}")
            if item.get("description"):
                lines.append(f"   {item['description'][:150]}")
        return "\n".join(lines)
