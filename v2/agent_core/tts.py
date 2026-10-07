"""TTS: Piper binary if voice exists, else mock (browser speaks)."""

from __future__ import annotations
from pathlib import Path
import subprocess, tempfile


class TTS:
    def __init__(self, cfg: dict):
        t = cfg.get("tts", {})
        self.voice = t.get("voice", "")
        self.backend = "mock"
        if (
            t.get("backend", "auto") in ("auto", "piper")
            and self.voice
            and Path(self.voice).exists()
        ):
            self.backend = "piper"

    def synth(self, text: str) -> bytes | None:
        if self.backend != "piper":
            return None  # client-side speechSynthesis handles PC test
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            subprocess.run(
                ["piper", "--model", self.voice, "--output_file", f.name],
                input=text.encode(),
                check=True,
            )
            return Path(f.name).read_bytes()
