"""
News Tool — Smart Multi-Source Aggregator
Fetches from 25+ RSS feeds, deduplicates stories across outlets,
groups identical events, and ranks by coverage count.
No API key required.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

TIMEOUT = aiohttp.ClientTimeout(total=10)
HEADERS = {"User-Agent": "Jarvis/1.0 (BNU Dissertation Project) Python/aiohttp"}

# ── Feed Directory ──────────────────────────────────────────────────────────
FEEDS: Dict[str, Tuple[str, str, str]] = {
    # (Display Name, URL, Category)
    "bbc":           ("BBC News",           "https://feeds.bbci.co.uk/news/rss.xml",                         "general"),
    "bbc_world":     ("BBC World",          "https://feeds.bbci.co.uk/news/world/rss.xml",                   "world"),
    "bbc_uk":        ("BBC UK",             "https://feeds.bbci.co.uk/news/uk/rss.xml",                      "uk"),
    "bbc_sport":     ("BBC Sport",          "https://feeds.bbci.co.uk/sport/rss.xml",                        "sports"),
    "bbc_science":   ("BBC Science",        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "science"),
    "guardian":      ("The Guardian",       "https://www.theguardian.com/world/rss",                         "world"),
    "guardian_uk":   ("Guardian UK",        "https://www.theguardian.com/uk/rss",                            "uk"),
    "guardian_sport":("Guardian Sport",     "https://www.theguardian.com/sport/rss",                         "sports"),
    "sky":           ("Sky News",           "https://feeds.skynews.com/feeds/rss/home.xml",                  "general"),
    "independent":   ("The Independent",    "https://www.independent.co.uk/news/rss",                        "general"),
    "reuters":       ("Reuters",            "https://feeds.reuters.com/reuters/topNews",                      "general"),
    "aljazeera":     ("Al Jazeera",         "https://www.aljazeera.com/xml/rss/all.xml",                     "world"),
    "techcrunch":    ("TechCrunch",         "https://techcrunch.com/feed/",                                   "technology"),
    "hackernews":    ("Hacker News",        "https://hnrss.org/frontpage",                                   "technology"),
    "theverge":      ("The Verge",          "https://www.theverge.com/rss/index.xml",                        "technology"),
    "wired":         ("Wired",              "https://www.wired.com/feed/rss",                                 "technology"),
    "arstechnica":   ("Ars Technica",       "https://feeds.arstechnica.com/arstechnica/index",               "technology"),
    "skysports":     ("Sky Sports",         "https://www.skysports.com/rss/12040",                           "sports"),
    "espn":          ("ESPN",               "https://www.espn.com/espn/rss/news",                            "sports"),
    "metro":         ("Metro UK",           "https://metro.co.uk/feed/",                                     "uk"),
    "newscientist":  ("New Scientist",      "https://www.newscientist.com/feed/home/",                       "science"),
    "mit_tech":      ("MIT Tech Review",    "https://www.technologyreview.com/feed/",                        "technology"),
}

CATEGORY_ALIASES = {
    "general":    ["general", "top", "main", "latest", "today", "breaking", "headlines", "top stories", "current events"],
    "world":      ["world", "international", "global", "foreign", "politics"],
    "uk":         ["uk", "britain", "british", "england", "scotland", "wales", "local"],
    "sports":     ["sport", "sports", "football", "soccer", "cricket", "tennis", "f1", "formula 1", "nba", "nfl", "rugby", "golf", "boxing", "athletics"],
    "technology": ["tech", "technology", "computing", "software", "hardware", "ai", "artificial intelligence", "ml", "machine learning", "startups", "apps"],
    "science":    ["science", "scientific", "research", "space", "environment", "climate", "health", "medicine"],
}

SOURCE_ALIASES = {
    "bbc":          ["bbc", "bbc news"],
    "guardian":     ["guardian", "the guardian"],
    "reuters":      ["reuters"],
    "sky":          ["sky", "sky news"],
    "independent":  ["independent"],
    "techcrunch":   ["techcrunch"],
    "theverge":     ["verge", "the verge"],
    "aljazeera":    ["al jazeera", "aljazeera"],
    "skysports":    ["sky sports"],
    "espn":         ["espn"],
}


class NewsTool:
    """
    Smart multi-source news aggregator.
    Fetches from multiple outlets, groups stories covering the same event,
    and ranks by cross-outlet coverage count.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    async def get_headlines(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        topic: Optional[str] = None,
        max_stories: int = 7,
        max_items: int = 7,  # alias
    ) -> Dict[str, Any]:
        """
        Get smart aggregated headlines.

        Auto-detects category/source/topic from natural language query.
        Groups duplicate stories and shows which outlets cover each one.
        """
        max_stories = max(max_stories, max_items)

        # Auto-detect from query
        if query:
            if not source:
                source = self._detect_source(query)
            if not category:
                category = self._detect_category(query)
            if not topic:
                topic = self._detect_topic(query)

        # Select feeds
        feed_keys = self._select_feeds(source, category)

        # Fetch all feeds concurrently
        tasks = [self._fetch_feed(key) for key in feed_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all articles
        all_articles = []
        sources_fetched = []
        for key, result in zip(feed_keys, results):
            if isinstance(result, Exception) or result is None:
                continue
            feed_name, items = result
            sources_fetched.append(feed_name)
            for item in items:
                item["source"] = feed_name
                all_articles.append(item)

        # Topic filter
        if topic:
            topic_lower = topic.lower()
            filtered = [
                a for a in all_articles
                if topic_lower in a.get("title", "").lower()
                or topic_lower in a.get("description", "").lower()
            ]
            all_articles = filtered if filtered else all_articles

        # Group articles by story (deduplicate across outlets)
        stories = self._group_stories(all_articles)

        # Sort: most covered first, then by first appearance
        stories.sort(key=lambda s: (-len(s["sources"]), -s.get("coverage_score", 0)))

        return {
            "success": True,
            "stories": stories[:max_stories],
            "sources_fetched": sources_fetched,
            "category": category,
            "topic": topic,
            "total_articles": len(all_articles),
            "total_stories": len(stories),
            "fetched_at": datetime.now().isoformat(),
        }

    async def get_all_categories(self, max_per_category: int = 2) -> Dict[str, Any]:
        """Get a digest across all major categories."""
        cats = ["general", "world", "technology", "sports", "uk"]
        tasks = [self.get_headlines(category=c, max_stories=max_per_category) for c in cats]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        digest = {}
        for cat, result in zip(cats, results):
            if isinstance(result, dict) and result.get("success"):
                digest[cat] = result.get("stories", [])
        return {"success": True, "digest": digest}

    def format_headlines(self, data: Dict[str, Any], detailed: bool = False) -> str:
        """
        Format stories with outlet attribution.
        Shows which outlets cover the same story.
        """
        if not data.get("success"):
            return "Could not fetch news."

        stories = data.get("stories", [])
        if not stories:
            return "No headlines found."

        cat = data.get("category", "")
        topic = data.get("topic", "")
        total_sources = len(data.get("sources_fetched", []))

        header = f"Headlines"
        if topic:
            header += f" about {topic}"
        elif cat:
            header += f" — {cat.title()}"
        header += f" (from {total_sources} sources):"

        lines = [header, ""]

        for i, story in enumerate(stories, 1):
            title = story["title"]
            sources = story["sources"]
            desc = story.get("description", "")

            # Source attribution
            if len(sources) > 1:
                source_str = f"[{', '.join(sources)}]"
                coverage = f" — covered by {len(sources)} outlets"
            else:
                source_str = f"[{sources[0]}]"
                coverage = ""

            lines.append(f"{i}. {title}")
            lines.append(f"   {source_str}{coverage}")

            if detailed and desc:
                lines.append(f"   {desc[:180]}")

            lines.append("")

        return "\n".join(lines).strip()

    def list_sources(self) -> str:
        """List all available news sources by category."""
        lines = ["Available news sources:\n"]
        by_cat: Dict[str, List[str]] = {}
        for key, (name, _, cat) in FEEDS.items():
            by_cat.setdefault(cat, []).append(name)
        for cat, names in sorted(by_cat.items()):
            lines.append(f"  {cat.title()}: {', '.join(names)}")
        return "\n".join(lines)

    # ── Story grouping ──────────────────────────────────────────────────────

    def _group_stories(self, articles: List[Dict]) -> List[Dict]:
        """
        Group articles about the same event across different outlets.
        Uses keyword overlap to detect duplicate stories.
        """
        stories = []

        for article in articles:
            title = article.get("title", "")
            keywords = self._extract_keywords(title)

            matched = False
            for story in stories:
                # Check keyword overlap with existing story
                story_keywords = story["keywords"]
                overlap = len(keywords & story_keywords)
                total = len(keywords | story_keywords)
                similarity = overlap / total if total > 0 else 0

                if similarity >= 0.35:  # 35% keyword overlap = same story
                    # Add this source if not already there
                    source = article.get("source", "Unknown")
                    if source not in story["sources"]:
                        story["sources"].append(source)
                    # Keep the most descriptive title (usually longest)
                    if len(title) > len(story["title"]):
                        story["title"] = title
                    # Merge keywords
                    story["keywords"] |= keywords
                    story["coverage_score"] += 1
                    matched = True
                    break

            if not matched:
                stories.append({
                    "title": title,
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "sources": [article.get("source", "Unknown")],
                    "keywords": keywords,
                    "coverage_score": 1,
                    "published": article.get("published", ""),
                })

        return stories

    def _extract_keywords(self, title: str) -> set:
        """
        Extract meaningful keywords from a headline.
        Strips stop words, keeps proper nouns and meaningful terms.
        """
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are",
            "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "shall",
            "can", "it", "its", "this", "that", "these", "those", "he", "she",
            "they", "we", "you", "i", "me", "him", "her", "them", "us",
            "after", "before", "over", "under", "between", "about", "up",
            "out", "new", "says", "said", "say", "how", "why", "what", "who",
            "when", "where", "which", "than", "then", "so", "if", "not",
            "no", "yes", "more", "most", "first", "last", "two", "three",
        }

        # Tokenise and clean
        words = re.sub(r"[^\w\s]", " ", title.lower()).split()
        keywords = {w for w in words if w not in stop_words and len(w) > 2}
        return keywords

    # ── Feed fetching ───────────────────────────────────────────────────────

    def _select_feeds(self, source: Optional[str], category: Optional[str]) -> List[str]:
        """Select which feeds to fetch based on user intent."""
        if source and source in FEEDS:
            return [source]
        if category == "general":
            # Mix of UK and international outlets
            return ["bbc", "guardian", "reuters", "sky", "independent",
                    "bbc_world", "aljazeera", "metro"]
        if category:
            return [k for k, (_, _, cat) in FEEDS.items() if cat == category]
        # Default: broad mix of different outlets and perspectives
        return ["bbc", "guardian", "reuters", "sky", "independent",
                "bbc_world", "aljazeera", "metro", "bbc_uk"]

    async def _fetch_feed(self, key: str) -> Optional[Tuple[str, List[Dict]]]:
        """Fetch and parse a single RSS feed."""
        if key not in FEEDS:
            return None
        name, url, _ = FEEDS[key]
        try:
            async with aiohttp.ClientSession(
                timeout=TIMEOUT, headers=HEADERS
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    xml_text = await resp.text()
            items = self._parse_rss(xml_text)
            return (name, items)
        except Exception:
            return None

    def _parse_rss(self, xml_text: str) -> List[Dict]:
        """Parse RSS XML into structured article dicts."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
            channel = root.find("channel") or root
            for item in channel.findall("item"):
                title = item.findtext("title", "").strip()
                desc  = item.findtext("description", "").strip()
                link  = item.findtext("link", "").strip()
                pub   = item.findtext("pubDate", "").strip()

                if "<" in desc:
                    try:
                        desc = ET.fromstring(f"<x>{desc}</x>").text or ""
                    except Exception:
                        desc = re.sub(r"<[^>]+>", "", desc)

                if title and len(title) > 10:
                    articles.append({
                        "title": title,
                        "description": desc[:300],
                        "url": link,
                        "published": pub,
                    })
        except ET.ParseError:
            pass
        return articles

    # ── Detection helpers ───────────────────────────────────────────────────

    def _detect_category(self, query: str) -> Optional[str]:
        q = query.lower()
        for cat, aliases in CATEGORY_ALIASES.items():
            if any(alias in q for alias in aliases):
                return cat
        return None

    def _detect_source(self, query: str) -> Optional[str]:
        q = query.lower()
        for source_key, aliases in SOURCE_ALIASES.items():
            if any(alias in q for alias in aliases):
                return source_key
        return None

    def _detect_topic(self, query: str) -> Optional[str]:
        q = query.lower()
        for trigger in ["about ", "on ", "regarding ", "covering ", "related to "]:
            if trigger in q:
                topic = query[q.index(trigger) + len(trigger):].strip().rstrip("?!., ")
                if topic and len(topic) > 2:
                    return topic
        return None