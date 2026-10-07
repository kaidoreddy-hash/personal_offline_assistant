"""v2 agent_core — standalone full-duplex S2S with stub LLM."""

from .duplex import FullDuplex, TurnEvent
from .llm_stub import HARDCODED_RESPONSE, stream_reply

__all__ = ["FullDuplex", "TurnEvent", "HARDCODED_RESPONSE", "stream_reply"]
