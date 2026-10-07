"""VAD frame -> speech segments, plus the dB helpers every module needs.

A segment is final once silence persists past `min_silence_ms` AND SmartTurn
P(complete) >= threshold. No user-facing silence knob; these are tuned."""

import math

import numpy as np

from v2.smart_turn import SmartTurnJudge
from v2.vad_eou import SileroVAD

FLOOR_DBFS = -45.0  # frames quieter than this never start a turn (room tone)
MAX_BUF_S = 10.0  # bound SmartTurn/STT cost: keep the tail only
PREROLL_S = 0.1


def frame_dbfs(pcm: bytes) -> float:
    a = np.frombuffer(pcm, dtype=np.int16)
    if len(a) == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    return float(20 * math.log10(rms / 32768.0)) if rms > 1e-9 else -100.0


class TurnSegmenter:
    """VAD frames -> speech segments. A segment is final once silence persists
    past `min_silence_ms` AND SmartTurn P(complete) >= threshold commits it.
    No user-facing 'silence ms' knob; those are tuned, not exposed."""

    def __init__(
        self,
        vad: SileroVAD,
        judge: SmartTurnJudge,
        min_silence_ms: int = 250,
        max_silence_ms: int = 2400,
        start_thresh: float = 0.5,
        continue_thresh: float = 0.35,
        min_speech_ms: int = 250,
        p_complete: float = 0.75,
    ):
        self.vad = vad
        self.judge = judge
        self.min_silence_ms = min_silence_ms
        self.max_silence_ms = max_silence_ms
        self.start_thresh = start_thresh
        self.continue_thresh = continue_thresh
        self.min_speech_ms = min_speech_ms
        self.p_complete = p_complete
        self._speaking = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._buf = bytearray()
        self._preroll = bytearray()
        self.last_p = None
        self.last_prob = 0.0  # latest VAD prob (loop updates idle clock from it)
        self._voice_db = -100.0  # loudest frame this turn (noise reject)
        self._noise_floor = None  # room-noise EMA, learned from quiet frames
        self._quiet_n = 0
        self.candidates = []  # SmartTurn checks this turn: (p, speech_ms, silence_ms)
        self.last_candidates = []  # committed turn's checks (read post-commit)
        self.turn_started = False  # set once when a turn begins (loop consumes it)
        self.start_db = -100.0
        self.start_prob = 0.0
        self._last_check = -1e9  # silence_ms at last SmartTurn check (throttle)

    def reset(self):
        self.vad.reset()
        self._speaking = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._buf = bytearray()
        self._preroll = bytearray()
        self.last_p = None
        self.last_prob = 0.0
        self._voice_db = -100.0
        self.candidates = []
        self.turn_started = False
        self.start_db = -100.0
        self.start_prob = 0.0
        self._last_check = -1e9
        # NOTE: noise floor + last_p/last_candidates survive resets — the room
        # doesn't change per turn, and the loop reads the verdict post-commit.

    def effective_floor(self) -> float:
        """Room-adaptive gate: noise floor + 12dB, clamped. Fixed -45 until
        ~0.6s of quiet history exists."""
        if self._noise_floor is None or self._quiet_n < 20:
            return FLOOR_DBFS
        return max(-55.0, min(-30.0, self._noise_floor + 12.0))

    def _commit(self) -> tuple:
        """Freeze the current turn and hand it back, tagged by voice energy."""
        seg = bytes(self._buf)
        noisy = self._voice_db < self.effective_floor() + 6.0
        self.last_candidates = list(self.candidates)
        self.reset()
        return seg, ("noise" if noisy else "speech")

    def tail(self, max_s: float = 5.0) -> bytes:
        """Last N seconds of the in-progress turn, for live partial decodes."""
        return bytes(self._buf[-int(16000 * 2 * max_s) :])

    def feed(self, pcm: bytes, chunk_ms: float = 32.0):
        """Returns (bytes, kind) when a segment is final, kind 'speech'|'noise'.
        None while in-progress."""
        db = frame_dbfs(pcm)
        floor = self.effective_floor()
        if db < floor:  # quiet frame: learn the room
            self._quiet_n += 1
            self._noise_floor = (
                db
                if self._noise_floor is None
                else (0.95 * self._noise_floor + 0.05 * db)
            )
        p = self.vad.predict(pcm)
        self.last_prob = p

        if not self._speaking:
            if db < floor:
                return None  # room tone: never starts a turn
            self._preroll.extend(pcm)
            if len(self._preroll) > 3200:  # 100ms @16k16bit
                self._preroll = self._preroll[-3200:]
            if p >= self.start_thresh:
                self._speaking = True
                self._buf = bytearray(self._preroll)
                self._buf.extend(pcm)
                self._preroll = bytearray()
                self._speech_ms = chunk_ms
                self._silence_ms = 0.0
                self._voice_db = db
                self.candidates = []
                self.last_p = None
                self._last_check = -1e9
                self.turn_started = True
                self.start_db = db
                self.start_prob = p
            return None

        self._buf.extend(pcm)
        if len(self._buf) > int(16000 * 2 * MAX_BUF_S):
            self._buf = self._buf[-int(16000 * 2 * MAX_BUF_S) :]  # bound cost
        self._voice_db = max(self._voice_db, db)
        if p >= self.continue_thresh and db >= floor - 3.0:
            self._speech_ms += chunk_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += chunk_ms

        if (
            self._speech_ms < self.min_speech_ms
            and self._silence_ms >= self.max_silence_ms
        ):
            self.reset()
            return None, "noise"  # too short, ran out of energy
        if self._silence_ms >= self.max_silence_ms:
            return self._commit()  # hard silence settles even an open turn
        if self._silence_ms >= self.min_silence_ms:
            # ponytail: one neural check per 500ms of silence, not per chunk.
            # predict() costs ~150ms; per-chunk checks stall the consumer and
            # overflow the mic queue (the CFFI dialog).
            if self._silence_ms - self._last_check < 500.0:
                return None
            self._last_check = self._silence_ms
            pc = self.judge.predict(bytes(self._buf))
            self.last_p = pc
            if pc is not None:
                self.candidates.append((pc, self._speech_ms, self._silence_ms))
                if pc >= self.p_complete:
                    return self._commit()
            # incomplete: keep listening until the next check or max silence
        return None
