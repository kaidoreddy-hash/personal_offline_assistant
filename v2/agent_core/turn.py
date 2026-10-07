"""End-of-speech detector. No user-facing fixed timer, no word-X heuristic.

Adaptive tail only: long utterances get a tighter finish so a trailing pause
in a short question doesn't cost the same as a mid-sentence hesitation.
"""

from __future__ import annotations
import time


class TurnDetector:
    """update(is_speech, partial_text, frame_duration_s) -> 'continue' | 'end'."""

    def __init__(self, cfg: dict):
        self._silence_start: float | None = None
        self._speech_ms = 0.0
        self._min_speech_ms = float(cfg.get("turn", {}).get("min_speech_ms", 300))

    def update(
        self, is_speech: bool, partial_text: str = "", frame_duration_s: float = 0.096
    ) -> str:
        if is_speech:
            self._silence_start = None
            self._speech_ms += frame_duration_s * 1000.0
            return "continue"
        if self._speech_ms < self._min_speech_ms:  # never end before real speech
            return "continue"
        if self._silence_start is None:
            self._silence_start = time.monotonic()
        tail = 0.85 if len(partial_text.strip()) < 30 else 0.55
        return "end" if (time.monotonic() - self._silence_start) >= tail else "continue"

    def reset(self) -> None:
        self._silence_start = None
        self._speech_ms = 0.0
