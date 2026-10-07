"""StubBrain — hardcoded streaming responses.

Proves the whole loop (streaming text -> sentence TTS -> barge-in) before the
real SLM lands. Later: brain/llm_openai.py pointing at a localhost llama.cpp
server implements the same interface.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent_core.brain.base import Brain

_REPLIES = [
    "I hear you. I am a hardcoded stub right now, but the whole voice loop around me is live.",
    "Copy that. Everything you just heard ran on this device, fully offline.",
    "Understood. Swap me for a real language model and I will actually think about that.",
]


class StubBrain(Brain):
    def __init__(self, pace_ms: float = 25.0) -> None:
        self.pace_ms = pace_ms
        self._i = 0

    async def respond(self, user_text: str, history: list) -> AsyncIterator[str]:
        reply = _REPLIES[self._i % len(_REPLIES)]
        self._i += 1
        if user_text.strip():
            reply += f" You said: {user_text.strip()[:80]}."
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(self.pace_ms / 1000)
