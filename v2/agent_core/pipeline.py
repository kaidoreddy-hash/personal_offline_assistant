"""Full-duplex voice pipeline — the deep module behind the WebSocket transport.

States: DORMANT (wake word only) -> LISTENING (open mic, partials) -> THINKING
-> SPEAKING (TTS out, barge-in armed) -> LISTENING. Long silence returns to
DORMANT. Meeting mode skips wake word and brain: everything is transcribed.

feed() only runs VAD + wake word (a few ms) so the 32 ms frame budget holds;
every heavy step (smart-turn, STT, TTS, brain) runs via the executor.

Echo/barge-in honesty: the browser sends AEC-clean mic audio, so speaking over
the agent works on the web path. A raw server-side mic will hear the agent's
own voice — the Pi hardware path needs the echo guard (see docs/pi5_deploy.md).
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np

from agent_core.brain.base import Brain, Turn
from agent_core.config import Config
from agent_core.debuglog import SessionLog
from agent_core.stt import WhisperSTT
from agent_core.tts import PiperTTS
from agent_core.turn import SmartTurn
from agent_core.vad import SileroVAD, UtteranceTracker
from agent_core.wakeword import WakeWord

DORMANT = "dormant"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"

_SENTENCE_END = re.compile(r"[.!?]\s*$")

Emitter = Callable[[dict], None]      # JSON events
AudioOut = Callable[[bytes], None]    # int16 PCM chunks


class VoicePipeline:
    def __init__(
        self,
        cfg: Config,
        log: SessionLog,
        emit: Emitter,
        emit_audio: AudioOut,
        brain: Brain,
        *,
        stt: WhisperSTT,
        turn: SmartTurn,
        tts: PiperTTS,
        meeting_mode: bool = False,
        store=None,
    ) -> None:
        self.cfg = cfg
        self.log = log
        self.emit = emit
        self.emit_audio = emit_audio
        self.brain = brain
        self.stt = stt
        self.turn = turn
        self.tts = tts
        self.meeting_mode = meeting_mode
        self.store = store
        self._session_id = store.start_session("meeting" if meeting_mode else "assistant") if store else None
        self._meeting_audio: list[tuple[int, np.ndarray]] = []  # (t_ms, audio) — RAM only, discarded after diarization

        self.vad = SileroVAD(str(Path(cfg.root) / "models" / "silero_vad_v5.onnx"))
        self.tracker = UtteranceTracker(
            self.vad,
            threshold=cfg.vad.threshold,
            min_silence_ms=cfg.vad.min_silence_ms,
            speech_pad_ms=cfg.vad.speech_pad_ms,
            min_speech_ms=cfg.vad.min_speech_ms,
        )
        self.wake: WakeWord | None = None if meeting_mode else WakeWord(
            cfg.wake.dir, cfg.wake.names, cfg.wake.threshold
        )

        self.state = LISTENING if meeting_mode else DORMANT
        self.muted = False
        self.history: deque[Turn] = deque(maxlen=20)

        self._loop = asyncio.get_running_loop()
        self._t0 = time.monotonic()
        self._frames: list[np.ndarray] = []   # current utterance
        self._pending: list[np.ndarray] = []  # audio from held (incomplete) turns
        self._partial_at = 0                  # utterance sample count for the next partial
        self._partial_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._force_task: asyncio.Task | None = None
        self._barge = asyncio.Event()
        self._last_voice = time.monotonic()
        self._wake_ok_at = 0.0          # wake refractory deadline
        self._wake_grace_until = 0.0    # drop wake-phrase tails right after a wake
        self.emit({"type": "state", "state": self.state, "meeting": meeting_mode})

    # ------------------------------------------------------------------ feed

    def feed(self, frame: np.ndarray) -> None:
        """One 512-sample float32 frame from the transport (sync, ~ms budget)."""
        if self.muted:
            return
        if self.state == DORMANT:
            assert self.wake is not None
            scores = self.wake.process_frame(frame)
            if scores and self.wake.triggered(scores) and time.monotonic() >= self._wake_ok_at:
                self._wake_ok_at = time.monotonic() + self.cfg.wake.refractory_ms / 1000
                self._wake_grace_until = time.monotonic() + 1.5
                self.log.log("wake", scores=scores)
                self.emit({"type": "wake", "scores": scores})
                self._set_state(LISTENING)
                self._last_voice = time.monotonic()
                self._speak_text(self.cfg.greeting)
            return

        utterance = self.tracker.process(frame)
        if self.tracker.in_utterance:
            self._last_voice = time.monotonic()

        if self.state == SPEAKING:
            # Barge-in is disarmed during the wake grace window: the user's own
            # wake-phrase tail is near-end speech (AEC won't remove it) and must
            # not kill the greeting.
            if self.tracker.in_utterance and time.monotonic() >= self._wake_grace_until:
                self._barge_in()
            return

        if self.tracker.in_utterance:
            self._frames.append(frame)
            self._maybe_partial()
        if utterance is not None:
            if time.monotonic() < self._wake_grace_until and len(utterance) < 0.7 * 16000:
                # Tail of the wake phrase itself — not a user request.
                self.log.log("wake_tail_dropped", dur_s=round(len(utterance) / 16000, 2))
            else:
                self._on_utterance_end(utterance)

    # ---------------------------------------------------------- turn handling

    def _on_utterance_end(self, utterance: np.ndarray) -> None:
        if self.meeting_mode:
            self._turn_task = self._loop.create_task(self._meeting_segment(utterance))
            return
        if self._turn_task and not self._turn_task.done():
            # Previous turn is still deciding/playing; queue as a continuation.
            self._pending.append(utterance)
            self._schedule_force()
            return
        self._frames = []
        self._set_state(THINKING)
        self._turn_task = self._loop.create_task(self._process_turn(utterance))

    async def _process_turn(self, utterance: np.ndarray) -> None:
        try:
            result = await self._loop.run_in_executor(None, self.turn.predict, utterance)
            self.log.log("turn_check", probability=round(result.probability, 3), complete=result.complete)
            if not result.complete:
                # Hold the turn. If the user resumes, audio merges; if not, force after the delay.
                self._pending.append(utterance)
                self.log.log("turn_hold", pending_segments=len(self._pending))
                self._set_state(LISTENING)
                self._schedule_force()
                return
            if self._pending:
                utterance = np.concatenate([*self._pending, utterance])
                self._pending = []
            await self._respond_to(utterance)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a dead turn must never kill the session
            self.log.log("error", where="process_turn", error=repr(exc))
            self.emit({"type": "error", "message": f"turn failed: {exc}"})
            self._set_state(LISTENING)

    def _schedule_force(self) -> None:
        if self._force_task and not self._force_task.done():
            return
        delay = self.cfg.turn.force_after_ms / 1000

        async def force() -> None:
            await asyncio.sleep(delay)
            if self._pending and self.state in (LISTENING, THINKING) and not self.tracker.in_utterance:
                audio = np.concatenate(self._pending)
                self._pending = []
                self.log.log("turn_force")
                self._set_state(THINKING)
                self._turn_task = self._loop.create_task(self._respond_to(audio))

        self._force_task = self._loop.create_task(force())

    async def _respond_to(self, utterance: np.ndarray) -> None:
        t_eot = time.monotonic()
        transcript = await self._loop.run_in_executor(None, self.stt.transcribe, utterance)
        self.log.log("final", text=transcript.text, lang=transcript.language,
                     stt_ms=round(transcript.inference_ms), dur_s=round(len(utterance) / 16000, 2))
        if self.log.save_utterance_audio:
            self.log.save_wav("utterance", utterance)
        self.emit({"type": "final", "text": transcript.text, "lang": transcript.language})
        if not transcript.text:
            self._set_state(LISTENING)
            return

        self.history.append(Turn("user", transcript.text))
        first_audio_ms, reply_text = await self._play_sentences(
            self._sentences(self.brain.respond(transcript.text, list(self.history))), t_eot
        )
        if self.store is not None and transcript.text:
            self.store.add_utterance(self._session_id, round((time.monotonic() - self._t0) * 1000),
                                     "user", transcript.text, lang=transcript.language)
        if reply_text:
            self.history.append(Turn("assistant", reply_text))
            if self.store is not None:
                self.store.add_utterance(self._session_id, round((time.monotonic() - self._t0) * 1000),
                                         "assistant", reply_text)
        if first_audio_ms is not None:
            self.emit({"type": "latency", "eot_to_first_audio_ms": round(first_audio_ms)})
        self._set_state(LISTENING)

    async def _meeting_segment(self, utterance: np.ndarray) -> None:
        t_ms = round((time.monotonic() - self._t0) * 1000)
        transcript = await self._loop.run_in_executor(None, self.stt.transcribe, utterance)
        self.log.log("meeting_segment", text=transcript.text, t_ms=t_ms)
        if self.log.save_utterance_audio:
            self.log.save_wav("meeting_segment", utterance)
        if transcript.text:
            self._meeting_audio.append((t_ms, utterance))
            if self.store is not None:
                self.store.add_utterance(self._session_id, t_ms, "speaker", transcript.text,
                                         speaker="SPEAKER_?", lang=transcript.language)
            self.emit({"type": "meeting_segment", "text": transcript.text, "t_ms": t_ms})

    def take_meeting_audio(self) -> list[tuple[int, np.ndarray]]:
        """Hand the meeting buffer to the diarizer and empty it."""
        buf, self._meeting_audio = self._meeting_audio, []
        return buf

    # ------------------------------------------------------------- speaking

    async def _play_sentences(self, sentences: AsyncIterator[str], t_eot: float) -> tuple[float | None, str]:
        """Synthesize + emit sentence by sentence. Returns (first_audio_ms, reply_text).

        first_audio_ms is None when barged in (reply aborted).
        """
        self._barge.clear()
        first_audio_ms: float | None = None
        reply_text = ""
        self._set_state(SPEAKING)
        try:
            async for sentence in sentences:
                if self._barge.is_set():
                    break
                if not sentence.strip():
                    continue
                reply_text += sentence
                audio = await self._loop.run_in_executor(None, self.tts.synth_all, sentence)
                if self._barge.is_set():
                    break
                if first_audio_ms is None:
                    first_audio_ms = (time.monotonic() - t_eot) * 1000
                self.emit_audio((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
                await asyncio.sleep(0)  # let the transport flush between sentences
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.log("error", where="play_sentences", error=repr(exc))
            self.emit({"type": "error", "message": f"tts failed: {exc}"})
        if self._barge.is_set():
            return None, reply_text
        self._set_state(LISTENING)  # greeting/turn finished speaking: open the mic again
        return first_audio_ms, reply_text

    def _speak_text(self, text: str) -> None:
        """Short agent line (greeting). Barge-in can cut it."""

        async def go() -> None:
            await self._play_sentences(_one(text), time.monotonic())

        self._turn_task = self._loop.create_task(go())

    def _sentences(self, word_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Merge a streamed word async-iterator into sentence-sized chunks."""

        async def gen() -> AsyncIterator[str]:
            buf = ""
            async for word in word_stream:
                buf += word
                if _SENTENCE_END.search(buf) or len(buf) >= 140:
                    yield buf
                    buf = ""
            if buf.strip():
                yield buf

        return gen()

    # -------------------------------------------------------------- barge-in

    def _barge_in(self) -> None:
        self.log.log("barge_in")
        self.emit({"type": "bargein"})
        self._barge.set()
        if self._force_task and not self._force_task.done():
            self._force_task.cancel()
        self._set_state(LISTENING)

    # -------------------------------------------------------------- partials

    def _maybe_partial(self) -> None:
        samples = sum(len(f) for f in self._frames)
        if samples < self._partial_at:
            return
        self._partial_at = samples + self.cfg.stt.partial_every_ms * 16
        if self._partial_task and not self._partial_task.done():
            return
        audio = np.concatenate(self._frames)

        async def run_partial() -> None:
            transcript = await self._loop.run_in_executor(None, self.stt.transcribe, audio)
            if transcript.text:
                self.emit({"type": "partial", "text": transcript.text})
                self.log.log("partial", text=transcript.text)

        self._partial_task = self._loop.create_task(run_partial())

    # ------------------------------------------------------------- lifecycle

    def check_dormancy(self) -> None:
        if self.state == LISTENING and not self.meeting_mode and not self.tracker.in_utterance:
            if time.monotonic() - self._last_voice > self.cfg.dormancy.timeout_s:
                self.log.log("dormant")
                self._set_state(DORMANT)
                self._pending = []

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self.log.log("mute", muted=muted)
        self.emit({"type": "muted", "muted": muted})

    def close(self) -> None:
        for t in (self._turn_task, self._partial_task, self._force_task):
            if t and not t.done():
                t.cancel()
        if self.store is not None and self._session_id is not None:
            try:
                self.store.end_session(self._session_id)
            except Exception:  # noqa: BLE001 — session close must never crash shutdown
                pass
            self._session_id = None

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.log.log("state", state=state)
            self.emit({"type": "state", "state": state})


async def _one(text: str) -> AsyncIterator[str]:
    yield text
