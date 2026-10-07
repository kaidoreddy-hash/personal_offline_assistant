"""STT: Whistle fast-path (EN, 16.9MB) -> faster-whisper multilingual fallback. LOCAL-ONLY.

Sovereignty rule: both models load from LOCAL files cached by
scripts/download_models.py. Names/URLs at runtime are refused — that would
hit the network. Whistle covers EN/DE/FR/ES/IT/NL/PL at ~4x speed;
faster-whisper tiny covers Hindi/Telugu/Tamil/code-switch. Nothing mocked.
"""

from __future__ import annotations
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _local(name: str) -> Path | None:
    p = BASE / name
    return p if p.exists() and p.stat().st_size > 0 else None


class STT:
    def __init__(self, cfg: dict):
        s = cfg.get("stt", {})
        want = s.get("backend", "auto")
        self.backend = "none"
        self._whistle = None
        self._fw = None
        if want in ("auto", "whistle") and _local(
            s.get("whistle_path", "models/whistle.cact")
        ):
            try:
                from needle import Whistle  # type: ignore

                self._whistle = Whistle(
                    weights=str(BASE / s.get("whistle_path", "models/whistle.cact"))
                )
                self.backend = "whistle"
            except Exception:
                self._whistle = None
        if want in ("auto", "faster-whisper"):
            mp = s.get("model_path", "models/stt-tiny")
            p = Path(mp) if Path(mp).is_absolute() else BASE / mp
            if (p.is_dir() and any(p.iterdir())) or (
                p.is_file() and p.stat().st_size > 0
            ):
                try:
                    from faster_whisper import WhisperModel  # type: ignore

                    self._fw = WhisperModel(
                        str(p), device="cpu", compute_type=s.get("compute_type", "int8")
                    )
                    self.backend = "whistle+fw" if self._whistle else "faster-whisper"
                except Exception:
                    self._fw = None

    def transcribe(self, pcm16: bytes, language: str | None = None) -> str:
        # Whistle fast-path for European langs (incl. auto-detect EN); fw for the rest.
        if self._whistle is not None and (
            language is None or language in ("en", "de", "fr", "es", "it", "nl", "pl")
        ):
            try:
                import numpy as np, tempfile, wave  # type: ignore

                a = np.frombuffer(pcm16, dtype=np.int16)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    with wave.open(f.name, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes(a.tobytes())
                    r = (
                        self._whistle.transcribe(f.name)
                        if hasattr(self._whistle, "transcribe")
                        else None
                    )
                    if isinstance(r, dict) and r.get("text"):
                        return str(r["text"]).strip()
            except Exception:
                pass  # fall through to faster-whisper
        if self._fw is not None:
            import numpy as np  # type: ignore

            a = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
            segs, _ = self._fw.transcribe(a, language=language)
            return " ".join(s.text for s in segs).strip()
        return ""
