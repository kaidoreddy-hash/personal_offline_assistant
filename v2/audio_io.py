"""Full-duplex audio I/O for Pi 5. One mic callback thread, one speaker
ring buffer. Both run at 16kHz mono int16. No push-to-talk: mic never mutes."""

import queue
import threading
import time
from typing import Optional

import numpy as np

_SR = 16000
_CHUNK = 512  # 32ms @16k — matches Silero v5 frame


class AudioIO:
    def __init__(
        self, mic_device: Optional[int] = None, spk_device: Optional[int] = None
    ):
        self.mic_device = mic_device
        self.spk_device = spk_device
        self._mic_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._spk_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self.player_level = 0.0  # smoothed, scaled echo-gate input [0..1]
        self._spk_thread: Optional[threading.Thread] = None
        self._mic_thread: Optional[threading.Thread] = None

    # ---- mic: capture thread pushes 32ms chunks; consumer reads via next_chunk()
    def _push_mic(self, data: bytes):
        """Callback-safe: never raises. Queue full -> drop oldest, keep live."""
        try:
            self._mic_q.put_nowait(data)
        except queue.Full:
            try:
                self._mic_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mic_q.put_nowait(data)
            except queue.Full:
                pass

    def start_mic(self):
        import sounddevice as sd

        def _cb(indata, frames, tinfo, status):
            self._push_mic(bytes(indata))

        def _run():
            with sd.InputStream(
                samplerate=_SR,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK,
                device=self.mic_device,
                callback=_cb,
            ):
                while not self._stop.is_set():
                    time.sleep(0.05)

        self._mic_thread = threading.Thread(target=_run, daemon=True, name="mic")
        self._mic_thread.start()

    def next_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        try:
            return self._mic_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---- speaker: blocking write_output, abortable
    def start_speaker(self):
        import sounddevice as sd

        def _run():
            with sd.OutputStream(
                samplerate=_SR,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK,
                device=self.spk_device,
            ) as st:
                while not self._stop.is_set():
                    try:
                        pcm = self._spk_q.get(timeout=0.05)
                    except queue.Empty:
                        # decay player_level while idle
                        self.player_level *= 0.9
                        continue
                    st.write(np.frombuffer(pcm, dtype=np.int16))
                    arr = np.frombuffer(pcm, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)) / 32768.0)
                    self.player_level = 0.7 * self.player_level + 0.3 * min(
                        1.0, rms * 4.0
                    )

        self._spk_thread = threading.Thread(target=_run, daemon=True, name="spk")
        self._spk_thread.start()

    def speak_chunk(self, pcm16: bytes):
        try:
            self._spk_q.put_nowait(pcm16)
        except queue.Full:
            pass  # drop rather than block mic

    def abort_playback(self):
        while not self._spk_q.empty():
            try:
                self._spk_q.get_nowait()
            except queue.Empty:
                break
        self.player_level = 0.0

    def stop(self):
        self._stop.set()
        if self._mic_thread:
            self._mic_thread.join(timeout=1.0)
        if self._spk_thread:
            self._spk_thread.join(timeout=1.0)
