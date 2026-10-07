"""Intent router — the tool-calling seam.

Tonight the StubBrain routes with keyword patterns. When the real LLM lands it
replaces this with native function calling; the tool functions stay unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent_core.config import Config
from agent_core.tools.search import web_search
from agent_core.tools.telegram import send_telegram
from agent_core.tools.weather import get_weather


@dataclass(frozen=True)
class Intent:
    tool: str
    arg: str


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("weather", re.compile(r"\bweather\b|\btemperature\b.*\b(?:in|at)\b|\brain\b.*\btoday\b", re.I)),
    ("search", re.compile(r"\b(?:search|look ?up|google|find|latest|news|released?)\b|\bwho is\b|\bwhat happened\b", re.I)),
    ("telegram", re.compile(r"\b(?:send|share|forward)\b.*\b(?:telegram|phone|my (?:mobile|cell))\b|\btelegram\b", re.I)),
]


def parse_intent(text: str) -> Intent | None:
    for tool, pat in _PATTERNS:
        if pat.search(text):
            return Intent(tool, text)
    return None


class ToolRouter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def is_tool_query(self, text: str) -> bool:
        return parse_intent(text) is not None

    def run(self, text: str, *, transcript_for_sending: str | None = None) -> str:
        """Execute the tool the query implies. Only this method may use the network."""
        intent = parse_intent(text)
        if intent is None:
            return "I do not have a tool for that yet."
        tool = intent.tool
        if tool == "weather":
            if not self.cfg.tools.weather.enabled:
                return "The weather tool is disabled."
            place = _place_from(text)
            return get_weather(place)
        if tool == "search":
            if not self.cfg.tools.search.enabled:
                return "Search is disabled."
            return web_search(text, max_results=self.cfg.tools.search.max_results)
        if tool == "telegram":
            if not self.cfg.tools.telegram.enabled:
                return "Telegram is disabled."
            body = transcript_for_sending or text
            return send_telegram(self.cfg.tools.telegram.token_env,
                                 self.cfg.tools.telegram.chat_id_env, body)
        return "Unknown tool."


def _place_from(text: str) -> str:
    m = re.search(r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z ]{2,30})", text, re.I)
    return (m.group(1).strip() if m else "Hyderabad").rstrip("?.!")
