"""LLM slot: hardcoded stub now, OpenAI-compatible drop-in for friend's offline SLM."""

from __future__ import annotations
from typing import Iterator

HARDCODED_RESPONSE = (
    "Got it. This is Tobi on the stub brain. "
    "Your voice path works end to end — VAD, turn detection, STT and TTS are live. "
    "Tell your friend to point llm.backend at openai-compatible and I will swap in the offline model."
)


def stream_reply(user_text: str, cfg: dict | None = None) -> Iterator[str]:
    """Yields word chunks instantly (simulates streaming LLM)."""
    if (cfg or {}).get("llm", {}).get("backend") == "openai-compatible":
        yield from _proxy_stream(user_text, cfg or {})
        return
    for i, w in enumerate(words := HARDCODED_RESPONSE.split(" ")):
        yield w + (" " if i < len(words) - 1 else "")


def _proxy_stream(user_text: str, cfg: dict) -> Iterator[str]:
    import httpx  # lazy so stub needs zero deps

    llm = cfg.get("llm", {})
    try:
        r = httpx.post(
            f"{llm.get('base_url')}/chat/completions",
            json={
                "model": llm.get("model", "local-slm"),
                "messages": [{"role": "user", "content": user_text}],
                "stream": False,
            },
            timeout=60.0,
        )
        r.raise_for_status()
        yield r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        yield f"{HARDCODED_RESPONSE} (proxy failed: {e})"
