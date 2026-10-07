"""v2 S2S loop driver. One pipeline object owned by app.py.
Thread: the mic capture thread produces chunks; this loop consumes them at a
fixed poll rate, feeds turn detection, runs STT/SmartTurn/EOU, owns state,
and runs TTS into the speaker.
"""

import threading
import time
from typing import Callable, Optional

from v2.audio_io import AudioIO
from v2.echo import EchoGate
from v2.engine import DORMANT, LISTENING, SPEAKING, THINKING, Engine
from v2.log import SessionLog
from v2.reply import reply_for
from v2.smart_turn import SmartTurnJudge
from v2.stt import transcribe
from v2.tts import synthesize
from v2.turn import TurnSegmenter, frame_dbfs
from v2.vad_eou import SileroVAD
from v2.wakeword import WakeWord

BARGE_IN_DBFS = -35.0
BARGE_IN_VOICE_MS = 120.0
BARGE_IN_GRACE_MS = 300.0  # reply onset can't cut itself (own-voice echo)
PART_WINDOW_S = 5.0  # live partials re-decode at most this much (flat cost)
PART_CADENCE_MS = 350.0  # one live pass per turn per this interval
PART_MIN_S = 0.75  # don't decode before the turn holds this much audio


class Pipeline:
    def __init__(self, lang: str = "en", p_complete: float = 0.75):
        self.lang = lang
        self.vad = SileroVAD()
        self.judge = SmartTurnJudge()
        self.seg = TurnSegmenter(self.vad, self.judge, p_complete=p_complete)
        self.audio = AudioIO()
        self.engine = Engine(on_state=self._on_state)
        self.echo = EchoGate()
        self.wakeword = WakeWord()  # lazy: loads ONNX on first DORMANT chunk
        self._state_cb: Optional[Callable[[str], None]] = None
        self._partial_cb: Optional[Callable[[str, str, float], None]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._speak_until = 0.0  # wall time when current reply finishes
        self._speak_epoch = 0
        self._speak_ms = 0.0
        self._speak_grace_until = 0.0  # onset grace: own voice can't cut itself
        self.session: Optional[SessionLog] = None
        self._n = 0  # attempt counter within the session
        self._live_words: list = []  # anchored live transcript (never rewrites)
        self._live_at = 0.0

    # ---- UI callbacks (set by app.py)
    def on_state(self, cb: Callable[[str], None]):
        self._state_cb = cb

    def on_partial(self, cb: Callable[[str, str, float, bool], None]):
        self._partial_cb = cb

    def _on_state(self, s: str):
        if self._state_cb:
            self._state_cb(s)

    # _partial_cb(text, lang, ms, partial) — final commits pass partial=False
    # positionally, live passes pass partial=True.

    # ---- session lifecycle (UI Start/Stop). One file per session.
    def start_session(self) -> str:
        """Open the session log and go LISTENING. Returns the log path."""
        if self.session is None:
            self.session = SessionLog()
            self.engine.session = self.session
            self.session.config(**self._config())
        self.engine.on_wake()
        return self.session.path

    def stop_session(self):
        if self.session is None:
            return
        if self.engine.attempt:
            self.session.attempt_end(self.engine.attempt, "stopped")
            self.engine.attempt = None
        self.engine.force_state(DORMANT)
        self.session.session_end("stop")
        self.session = None
        self.engine.session = None

    def _config(self) -> dict:
        try:
            import sounddevice as sd

            devs = sd.query_devices()
            mic = self.audio.mic_device
            spk = self.audio.spk_device
            mic_n = (
                devs[mic]["name"]
                if mic is not None
                else devs[sd.default.device[0]]["name"]
            )
            spk_n = (
                devs[spk]["name"]
                if spk is not None
                else devs[sd.default.device[1]]["name"]
            )
        except Exception:
            mic_n, spk_n = "unknown", "unknown"
        return {
            "lang": self.lang,
            "p_complete": self.seg.p_complete,
            "min_silence_ms": self.seg.min_silence_ms,
            "max_silence_ms": self.seg.max_silence_ms,
            "floor": "adaptive(-45 init)",
            "mic": mic_n,
            "spk": spk_n,
            "vad": self.vad.model_path,
            "smartturn": self.judge.model_path,
            "reply": "fixed",
        }

    def _ensure_turn(self):
        """New attempt id inside the open session. No session -> no turns."""
        if self.session is None or self.engine.attempt is not None:
            return
        self._n += 1
        aid = f"{self._n:03d}"
        self.engine.attempt = aid
        self.session.attempt_start(aid)

    def _on_turn_started(self):
        """VAD edge -> open attempt, log levels. Single caller: the loop."""
        self.seg.turn_started = False
        self._ensure_turn()
        self._live_words = []
        self._live_at = 0.0
        if self.session and self.engine.attempt:
            self.session.turn_start(
                self.engine.attempt, self.seg.start_db, self.seg.start_prob
            )

    def _live_extend(self, hypo: str):
        """Anchored commit: committed words are append-only. A hypothesis that
        revises already-committed words is ignored (waits for the next pass);
        one that agrees extends. Text on screen never walks backwards."""
        hw = hypo.split()
        n = 0
        for a, b in zip(self._live_words, hw):
            if a != b:
                break
            n += 1
        if n >= len(self._live_words):
            self._live_words = hw

    def _live_pass(self, now: float):
        """One bounded live decode of the growing turn. Flat cost: the window
        is capped, so pass time never grows with turn length."""
        if now - self._live_at < PART_CADENCE_MS / 1000.0:
            return
        self._live_at = now
        tail = self.seg.tail(PART_WINDOW_S)
        if len(tail) < int(16000 * 2 * PART_MIN_S):
            return
        text, lang, _lp = transcribe(tail, language=self.lang)
        if not text.strip():
            return
        before = len(self._live_words)
        self._live_extend(text)
        if len(self._live_words) <= before:
            return  # revision, not progress: stay quiet
        shown = " ".join(self._live_words)
        aid = self.engine.attempt
        if self.session and aid:
            self.session.partial(aid, shown)
        if self._partial_cb:
            self._partial_cb(shown, lang, 0.0, True)

    def _end_turn(self, reason: str):
        if self.session and self.engine.attempt:
            self.session.attempt_end(self.engine.attempt, reason)
        self.engine.attempt = None

    def interrupt(self):
        if self.session:
            self.session.barge_in(self.engine.attempt, 0.0)
        self._end_turn("barged")
        self.engine.bump()
        self.audio.abort_playback()
        self._speak_until = 0.0
        self.engine.force_state(LISTENING)

    def _maybe_done(self, now: float):
        """Reply finished playing -> back to LISTENING. Epoch guard: a
        barge-in that already bumped must not be resurrected."""
        if (
            self.engine.state == SPEAKING
            and self._speak_until
            and now >= self._speak_until
            and self.audio.player_level < 0.08
            and self.engine.epoch() == self._speak_epoch
        ):
            if self.session and self.engine.attempt:
                self.session.playback_done(self.engine.attempt, self._speak_ms)
            self._end_turn("turn_done")
            self.engine.force_state(LISTENING)

    # ---- run
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.audio.start_mic()
        self.audio.start_speaker()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._guarded, daemon=True, name="pipeline"
        )
        self._thread.start()

    def _guarded(self):
        """A dead loop thread looks exactly like this dialog: mic keeps
        producing, nobody consumes. Log the crash instead of dying silent."""
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001 — boundary guard, must not raise
            try:
                if self.session:
                    self.session.crash("loop", repr(e))
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self.audio.stop()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        last_tick = time.perf_counter()
        barge_ms = 0.0
        while not self._stop.is_set():
            chunk = self.audio.next_chunk(timeout=0.05)
            now = time.perf_counter()
            if now - last_tick > 0.1:
                self.engine.tick_idle()
                last_tick = now
            if chunk is None:
                self._maybe_done(now)
                continue

            if self.engine.state == SPEAKING:
                # VAD-gated barge-in: sustained real voice over playback.
                # Two guards against self-trigger: onset grace (reply onset
                # can't cut itself) and playback level (quiet room first).
                p = self.seg.vad.predict(chunk)
                db = frame_dbfs(chunk)
                if (
                    now >= self._speak_grace_until
                    and p >= self.seg.start_thresh
                    and db >= BARGE_IN_DBFS
                    and self.audio.player_level > 0.05
                ):
                    barge_ms += 32.0
                    if barge_ms >= BARGE_IN_VOICE_MS:
                        self.interrupt()
                        barge_ms = 0.0
                else:
                    barge_ms = 0.0
                self._maybe_done(now)
                continue

            if self.engine.state == DORMANT:
                # Wake-word only. VAD/STT/SmartTurn stay off; ~0.6ms/chunk.
                # A miss must never kill the loop, so errors are swallowed
                # here and surface as silence (no wake).
                try:
                    if self.wakeword.available and self.wakeword.feed(chunk):
                        self.start_session()
                        if self.session:
                            self.session.wake_word(
                                self.wakeword.phrase, self.wakeword.threshold
                            )
                except Exception:
                    pass
                continue

            if self.engine.state != LISTENING:
                # SPEAKING handled above; THINKING is transient: skip.
                continue

            out = self.seg.feed(chunk, chunk_ms=32.0)
            if self.seg.last_prob >= self.seg.start_thresh:
                self.engine.note_activity()
            if self.seg.turn_started:
                self._on_turn_started()
            if out is None:
                if self.seg._speaking:
                    self._live_pass(now)
                continue
            if isinstance(out, tuple) and out[1] == "noise":
                self._live_words = []
                self._end_turn("noise_only")
                continue
            seg_bytes, kind = out
            if kind != "speech":
                continue
            # EOU committed -> final STT
            t0 = time.perf_counter()
            text, lang, lp = transcribe(seg_bytes, language=self.lang)
            stt_ms = (time.perf_counter() - t0) * 1000.0
            aid = self.engine.attempt
            if self.session and aid:
                for cp, csp, csi in self.seg.last_candidates:
                    self.session.candidate(aid, cp, csp, csi, self.seg.p_complete)
                self.session.stt(aid, text, lang, lp, stt_ms)
                self.session.smart_turn(aid, self.seg.last_p, self.seg.p_complete)
            if not text.strip():
                # nothing heard: no reply, no TTS, stay LISTENING
                self._live_words = []
                self._end_turn("empty_stt")
                continue
            if self.session and aid and self.echo.is_echo(text):
                # mic heard our own speaker, not the user (v1 is_echo pattern)
                self._live_words = []
                self._end_turn("echo_self")
                continue
            if self._partial_cb:
                self._partial_cb(text, lang, stt_ms, False)
            self._live_words = []
            self.engine.force_state(THINKING)
            reply = reply_for(text, lang)
            t1 = time.perf_counter()
            pcm = synthesize(reply, lang=lang)
            synth_ms = (time.perf_counter() - t1) * 1000.0
            audio_ms = len(pcm) / 32000.0
            if self.session and aid:
                self.session.reply(aid, reply, (time.perf_counter() - t0) * 1000.0)
                self.session.tts(aid, reply, lang, synth_ms, audio_ms)
            # TTS + playback: SPEAKING persists until audio drains (attempt
            # stays open so playback_done/barge-in land in the same file)
            self.engine.force_state(SPEAKING)
            self._speak_epoch = self.engine.epoch()
            self._speak_ms = audio_ms
            self._speak_until = time.perf_counter() + audio_ms / 1000.0 + 0.3
            self._speak_grace_until = time.perf_counter() + BARGE_IN_GRACE_MS / 1000.0
            self.audio.speak_chunk(pcm)
            self.echo.note_spoken(reply)
