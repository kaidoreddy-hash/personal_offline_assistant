"""Speaker-echo suppression. Ported from core_pipeline.py:488-512.

If the transcript IS what we just said, the mic heard our own speaker,
not the user. Entries carry timestamps (30s TTL) so old replies can't
suppress new chat. WORDS=2 because the v2 fixed reply is 2 words.
"""

import re
import time
from collections import deque

TTL_S = 30.0
WORDS = 2


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").lower(), flags=re.UNICODE).strip()


class EchoGate:
    def __init__(self):
        self._recent: deque = deque(maxlen=8)

    def note_spoken(self, text: str):
        n = _norm(text)
        if len(n.split()) >= WORDS:
            self._recent.append((n, time.perf_counter()))

    def is_echo(self, text: str) -> bool:
        t = _norm(text)
        if len(t.split()) < WORDS:
            return False
        now = time.perf_counter()
        self._recent = deque(
            ((s, ts) for s, ts in self._recent if now - ts < TTL_S), maxlen=8
        )
        return any(t in s or s in t for s, _ in self._recent)
