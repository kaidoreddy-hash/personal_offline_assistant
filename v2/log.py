"""Session log. ONE file per Start->Stop: logs/session_<ts>.jsonl.

Every event carries wall-clock ts, ms since session start, and the attempt
id when turn-scoped. Read top-to-bottom: session_start -> config ->
attempt_start -> turn_start -> candidate* -> stt -> smart_turn -> reply ->
tts -> playback_done -> attempt_end -> ... -> session_end.
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)


class SessionLog:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(_LOG_DIR, f"session_{self.session_id}.jsonl")
        self._fh = open(self.path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.closed = False
        self._write("session_start", session_id=self.session_id)

    def _write(self, event: str, attempt: Optional[str] = None, **fields):
        if self.closed:
            return
        rec = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "t_ms": round((time.perf_counter() - self.t0) * 1000.0, 1),
            "event": event,
        }
        if attempt is not None:
            rec["attempt"] = attempt
        rec.update(fields)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ---- session scope
    def config(self, **fields):
        self._write("config", **fields)

    def state(self, src: str, dst: str, attempt: Optional[str] = None):
        self._write("state", attempt, src=src, dst=dst)

    def session_end(self, reason: str = "stop"):
        self._write("session_end", reason=reason)
        self._fh.close()
        self.closed = True

    # ---- attempt scope
    def attempt_start(self, aid: str):
        self._write("attempt_start", aid)

    def attempt_end(self, aid: str, reason: str):
        self._write("attempt_end", aid, reason=reason)

    def turn_start(self, aid: str, db: float, prob: float):
        self._write(
            "turn_start", aid, db=round(float(db), 1), vad_prob=round(float(prob), 3)
        )

    def candidate(
        self, aid: str, p: float, speech_ms: float, silence_ms: float, threshold: float
    ):
        self._write(
            "candidate",
            aid,
            p_complete=round(float(p), 3),
            speech_ms=round(float(speech_ms), 1),
            silence_ms=round(float(silence_ms), 1),
            threshold=threshold,
        )

    def stt(self, aid: str, text: str, lang: str, prob: float, stt_ms: float):
        self._write(
            "stt",
            aid,
            text=text,
            lang=lang,
            lang_prob=round(float(prob), 3),
            stt_ms=round(float(stt_ms), 1),
        )

    def smart_turn(self, aid: str, p_complete: Optional[float], threshold: float):
        self._write(
            "smart_turn",
            aid,
            p_complete=None if p_complete is None else round(float(p_complete), 3),
            threshold=threshold,
        )

    def reply(self, aid: str, text: str, ttfa_ms: float):
        self._write("reply", aid, text=text, ttfa_ms=round(float(ttfa_ms), 1))

    def tts(self, aid: str, text: str, lang: str, synth_ms: float, audio_ms: float):
        self._write(
            "tts",
            aid,
            text=text,
            lang=lang,
            synth_ms=round(float(synth_ms), 1),
            audio_ms=round(float(audio_ms), 1),
        )

    def playback_done(self, aid: str, audio_ms: float):
        self._write("playback_done", aid, audio_ms=round(float(audio_ms), 1))

    def barge_in(self, aid: Optional[str], voice_ms: float):
        self._write("barge_in", aid, voice_ms=round(float(voice_ms), 1))

    def wake_word(self, phrase: str, threshold: float):
        self._write("wake_word", phrase=phrase, threshold=threshold)

    def crash(self, where: str, error: str):
        self._write("loop_crash", where=where, error=error)

    def partial(self, aid: str, text: str):
        self._write("partial", aid, text=text)
