"""
Summariser Agent
Condenses long content into concise briefings.
Used internally by Research, Email, News, and Document agents.
"""

from __future__ import annotations

from config.llm_client import OllamaClient


class SummariserAgent:
    """
    Summarises arbitrary text into concise briefings.
    Called by other agents — not directly by the user.
    """

    def __init__(self, llm_client: OllamaClient | None = None):
        self.llm = llm_client or OllamaClient()
        print("📝 SummariserAgent ready")

    async def summarise(
        self,
        text: str,
        max_words: int = 150,
        style: str = "concise",
    ) -> str:
        """
        Summarise text into a concise briefing.

        Args:
            text:      Text to summarise
            max_words: Target word count
            style:     'concise' | 'bullet_points' | 'executive'

        Returns:
            Summary string
        """
        if len(text.split()) <= max_words:
            return text  # Already short enough

        style_instruction = {
            "concise": "Write a concise paragraph summary.",
            "bullet_points": "Write 3-5 bullet points covering the key points.",
            "executive": "Write a 2-sentence executive summary.",
        }.get(style, "Write a concise paragraph summary.")

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a summarisation assistant. {style_instruction} "
                    f"Target length: ~{max_words} words. "
                    "Be factual, remove filler, preserve key details."
                ),
            },
            {"role": "user", "content": f"Summarise this:\n\n{text[:4000]}"},
        ]

        try:
            return await self.llm.chat(messages)
        except Exception as exc:
            print(f"⚠️  Summariser error: {exc}")
            # Fallback: truncate
            words = text.split()
            return " ".join(words[:max_words]) + "..."

    async def summarise_list(
        self,
        items: list[str],
        topic: str = "",
        max_words: int = 200,
    ) -> str:
        """Summarise a list of items into a cohesive paragraph."""
        combined = "\n".join(f"- {item}" for item in items)
        prompt = f"Topic: {topic}\n\nItems:\n{combined}" if topic else combined
        return await self.summarise(prompt, max_words=max_words)
