"""Full-duplex orchestrator: DORMANT -> LISTENING -> RESPONDING with barge-in."""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from . import logger as _log
from .turn import TurnDetector
from .vad import VAD
from .stt import STT
from .tts import TTS
from .llm_stub import stream_reply


@dataclass
class TurnEvent:
    kind: str  # state|live_transcript|llm_chunk|tts|log
    data: str = ""


class FullDuplex:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = "dormant"  # dormant|listening|responding
        self.vad, self.turn, self.stt, self.tts = (
            VAD(cfg),
            TurnDetector(cfg),
            STT(cfg),
            TTS(cfg),
        )
        self.wakeword = cfg.get("wakeword", "hey tobi").lower()
        self._silent_since = time.monotonic()
        self._partial = ""
        self.log_path: Path | None = None
        self._cancel = False

    def _ensure_log(self, source="mic"):
        if self.log_path is None:
            self.log_path = _log.new_attempt(self.cfg, source)
        return self.log_path

    def on_wakeword(self) -> list[TurnEvent]:
        self._ensure_log()
        _log.log(self.log_path, "wakeword")
        self.state, self._partial, self._cancel = "listening", "", False
        self.turn.reset()
        self._silent_since = time.monotonic()
        return [TurnEvent("state", "listening"), TurnEvent("log", str(self.log_path))]

    def on_partial(self, text: str) -> list[TurnEvent]:
        if self.state == "dormant" and self.wakeword in text.lower():
            return (
                self.on_wakeword()
            )  # transcript-based fallback when openwakeword absent
        if self.state != "listening":
            return []
        self._ensure_log()
        self._partial = text
        _log.log(self.log_path, "partial", text=text)
        return [TurnEvent("live_transcript", text)]

    def on_final(self, text: str) -> list[TurnEvent]:
        if self.state == "dormant":
            return self.on_partial(text)  # may contain wakeword
        self._ensure_log()
        _log.log(self.log_path, "final", text=text)
        self.state = "responding"
        evs = [TurnEvent("state", "responding"), TurnEvent("live_transcript", text)]
        reply = []
        for chunk in stream_reply(text, self.cfg):
            if self._cancel:
                _log.log(self.log_path, "barge_in_cancel")
                evs.append(TurnEvent("state", "listening"))
                self.state = "listening"
                return evs
            reply.append(chunk)
            evs.append(TurnEvent("llm_chunk", chunk))
        full = "".join(reply)
        _log.log(self.log_path, "reply", text=full, tts=self.tts.backend)
        evs.append(TurnEvent("tts", full))
        self.state = "listening"
        self.turn.reset()
        self._silent_since = time.monotonic()
        evs.append(TurnEvent("state", "listening"))
        return evs

    def on_barge_in(self) -> list[TurnEvent]:
        self._ensure_log()
        _log.log(self.log_path, "barge_in")
        if self.state == "responding":
            self._cancel = True
        self.state = "listening"
        self.turn.reset()
        return [TurnEvent("state", "listening")]

    def tick(self) -> list[TurnEvent]:  # dormancy watchdog
        if self.state == "listening" and time.monotonic() - self._silent_since > float(
            self.cfg.get("dormant_after_silence_s", 30)
        ):
            self.state = "dormant"
            if self.log_path:
                _log.log(self.log_path, "dormant")
            self.log_path = None  # next attempt gets a fresh timestamped file
            return [TurnEvent("state", "dormant")]
        return []
