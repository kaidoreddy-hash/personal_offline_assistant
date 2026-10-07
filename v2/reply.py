"""Hardcoded reply for v2 testbed. SLM/LLM replaces this later."""

FIXED = "Got it. I heard what you said."


def reply_for(text: str, lang: str) -> str:
    """Returns a fixed reply. The text arg is for future LLM routing."""
    return FIXED


def reply_stream(text: str, lang: str):
    """Seam for the later LLM: yields reply sentence-chunks. One fixed chunk
    today; the pipeline will iterate chunks -> synthesize per sentence."""
    yield FIXED
