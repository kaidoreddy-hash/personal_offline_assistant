"""Full-duplex: DORMANT -> LISTENING -> RESPONDING, stays responding through
playback so barge-in works. Client reports playback_done to return to listening.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass
from pathlib import Path
from . import logger as _log
from .turn import TurnDetector
from .vad import VAD
from .wakeword import WakeWordDetector
from .stt import STT
from .tts import TTS
from .llm_stub import stream_reply


@dataclass
class TurnEvent:
    kind: str  # state|live_transcript|llm_chunk|tts|barge_in|log
    data: str = ""


class FullDuplex:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = "dormant"
        self.vad = VAD(cfg)
        self.turn = TurnDetector(cfg)
        self.ww = WakeWordDetector(cfg)
        self.stt = STT(cfg)
        self.tts = TTS(cfg)
        self.wakeword = cfg.get("wakeword", "hey tobi").lower()
        self._silent_since = time.monotonic()
        self._partial = ""
        self.log_path: Path | None = None
        self._cancel = False

    def _ensure_log(self, source="mic") -> Path:
        if self.log_path is None:
            self.log_path = _log.new_attempt(self.cfg, source)
        return self.log_path

    def on_wakeword(self) -> list[TurnEvent]:
        self._ensure_log()
        _log.log(self.log_path, "wakeword", backend=self.ww.backend)
        self.state, self._partial, self._cancel = "listening", "", False
        self.turn.reset()
        self.vad.reset()
        self._silent_since = time.monotonic()
        return [TurnEvent("state", "listening"), TurnEvent("log", str(self.log_path))]

    def on_partial(self, text: str) -> list[TurnEvent]:
        # whole-word "tobi" so "october"/"autobiography" can't wake it
        if self.state == "dormant" and (
            self.wakeword in text.lower() or re.search(r"\btobi\b", text, re.I)
        ):
            return self.on_wakeword()  # PC path: wakeword matched in STT text
        if self.state != "listening":
            return []
        self._ensure_log()
        self._partial = text
        self._silent_since = time.monotonic()
        _log.log(self.log_path, "partial", text=text)
        return [TurnEvent("live_transcript", text)]

    def on_final(self, text: str) -> list[TurnEvent]:
        if self.state == "dormant":
            return self.on_partial(text)
        self._ensure_log()
        _log.log(self.log_path, "final", text=text)
        self.state, self._cancel = "responding", False
        self._silent_since = time.monotonic()
        evs = [TurnEvent("state", "responding"), TurnEvent("live_transcript", text)]
        reply = []
        for chunk in stream_reply(text, self.cfg):
            if self._cancel:
                _log.log(self.log_path, "barge_in_cancel")
                self.state = "listening"
                self.turn.reset()
                return evs + [
                    TurnEvent("barge_in", "cancelled"),
                    TurnEvent("state", "listening"),
                ]
            reply.append(chunk)
            evs.append(TurnEvent("llm_chunk", chunk))
        full = "".join(reply)
        _log.log(self.log_path, "reply", text=full, tts=self.tts.backend)
        # stay "responding" — client sends playback_done when audio ends
        return evs + [TurnEvent("tts", full)]

    def on_playback_done(self) -> list[TurnEvent]:
        if self.state == "responding":
            self.state = "listening"
            self.turn.reset()
            self._silent_since = time.monotonic()
            return [TurnEvent("state", "listening")]
        return []

    def on_barge_in(self) -> list[TurnEvent]:
        self._ensure_log()
        _log.log(self.log_path, "barge_in")
        self._cancel = True
        self.state = "listening"
        self.turn.reset()
        self._silent_since = time.monotonic()
        return [TurnEvent("barge_in", "interrupted"), TurnEvent("state", "listening")]

    def tick(self) -> list[TurnEvent]:
        if self.state == "listening" and time.monotonic() - self._silent_since > float(
            self.cfg.get("dormant_after_silence_s", 30)
        ):
            self.state = "dormant"
            if self.log_path:
                _log.log(self.log_path, "dormant")
            self.log_path = None  # next attempt gets a fresh timestamped file
            return [TurnEvent("state", "dormant")]
        return []
