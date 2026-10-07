"""TTS: Piper (ONNX) — sentence-level streaming synthesis, torch-free."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PiperTTS:
    def __init__(self, voice_path: str, gain: float = 1.0) -> None:
        from piper import PiperVoice

        self.voice = PiperVoice.load(voice_path)
        self.sample_rate = self.voice.config.sample_rate
        self.gain = gain

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 mono audio chunks for the given text (blocking; call from a worker)."""
        for chunk in self.voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if self.gain != 1.0:
                audio = np.clip(audio * self.gain, -1.0, 1.0)
            yield audio

    def synth_all(self, text: str) -> np.ndarray:
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
