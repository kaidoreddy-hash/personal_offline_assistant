"""TTS: Piper offline synthesis. Never raises into the WS handler."""

from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE = Path(__file__).resolve().parent.parent


class TTS:
    def __init__(self, cfg: dict):
        t = cfg.get("tts", {})
        vs = t.get("voice", "models/en_US-lessac-low.onnx")
        self.voice = Path(vs) if Path(vs).is_absolute() else BASE / vs
        self.backend = "mock"
        if t.get("backend", "auto") in ("auto", "piper") and self.voice.exists():
            if shutil.which("piper"):
                self.backend = "piper"
            else:
                try:
                    import piper  # type: ignore  # noqa: F401

                    self.backend = "piper-python"
                except ImportError:
                    self.backend = "mock"

    def synth(self, text: str) -> bytes | None:
        if not text.strip() or self.backend == "mock":
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                if self.backend == "piper":
                    subprocess.run(
                        ["piper", "--model", str(self.voice), "--output_file", f.name],
                        input=text.encode("utf-8"),
                        check=True,
                        capture_output=True,
                    )
                else:
                    import wave
                    from piper import PiperVoice  # type: ignore

                    voice = PiperVoice.load(str(self.voice))
                    with wave.open(f.name, "wb") as w:
                        voice.synthesize(text, w)
                return Path(f.name).read_bytes()
        except Exception:
            return None  # browser speechSynthesis takes over
