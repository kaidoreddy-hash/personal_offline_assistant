"""Core S2S state machine for v2.

States: DORMANT -> LISTENING -> THINKING -> SPEAKING -> LISTENING (loop)
        LISTENING --inactivity--> DORMANT
        SPEAKING --barge-in--> LISTENING

Handsfree full-duplex: mic always open; DORMANT only gates turn detection.
The pipeline owns the SessionLog; the engine only carries the current
attempt id so every state edge lands in the right attempt.
"""

import threading
import time
from typing import Callable, Optional

DORMANT = "DORMANT"
LISTENING = "LISTENING"
THINKING = "THINKING"
SPEAKING = "SPEAKING"


class Engine:
    def __init__(
        self,
        on_state: Optional[Callable[[str], None]] = None,
        inactivity_ms: int = 20000,
    ):
        self.state = DORMANT
        self.on_state = on_state
        self.inactivity_ms = inactivity_ms
        self._epoch = 0
        self._lock = threading.Lock()
        self._last_activity = time.perf_counter()
        self.session = None  # SessionLog, set by pipeline
        self.attempt: Optional[str] = None  # current attempt id

    def _set(self, s: str):
        with self._lock:
            if self.state != s:
                src, self.state = self.state, s
                if self.session:
                    self.session.state(src, s, self.attempt)
                if self.on_state:
                    self.on_state(s)
        self._last_activity = time.perf_counter()

    def epoch(self) -> int:
        return self._epoch

    def bump(self):
        with self._lock:
            self._epoch += 1

    def note_activity(self):
        """Real voice seen by VAD. Resets the idle-to-DORMANT clock."""
        self._last_activity = time.perf_counter()

    # ---- DORMANT: only wake word listens (pipeline owns the session file)
    def on_wake(self):
        if self.state == DORMANT:
            self._set(LISTENING)

    def tick_idle(self):
        if self.state in (LISTENING, DORMANT):
            if (
                time.perf_counter() - self._last_activity
            ) * 1000.0 > self.inactivity_ms:
                if self.session and self.attempt:
                    self.session.attempt_end(self.attempt, "idle_timeout")
                    self.attempt = None
                self._set(DORMANT)

    def force_state(self, s: str):
        self._set(s)
