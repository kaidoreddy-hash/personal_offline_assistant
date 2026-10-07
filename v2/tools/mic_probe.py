"""Probe mic + VAD + wakeword from CLI. Writes timestamped JSONL log."""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_core.config import load
from agent_core.vad import VAD
from agent_core.wakeword import WakeWordDetector
from agent_core import logger as _log

cfg = load()
logp = _log.new_attempt(cfg, "mic-probe")
vad, ww = VAD(cfg), WakeWordDetector(cfg)
print(
    f"VAD: {'silero-v5-onnx' if vad._sess is not None else 'energy-fallback'} | wake: {ww.backend}"
)
print(f"log: {logp}")

try:
    import sounddevice as sd  # type: ignore

    def cb(indata, frames, t, status):
        raw = bytes(indata)
        speech = vad.is_speech(raw)
        wake = ww.predict(raw) if ww.backend != "transcript" else False
        _log.log(logp, "frame", speech=speech, wake=wake)
        print("[WAKE!]" if wake else ("S" if speech else "."), end="", flush=True)

    print("\nrecording 10s — speak now")
    with sd.RawInputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=1536, callback=cb
    ):
        time.sleep(10)
except ImportError:
    _log.log(logp, "no-sounddevice")
    print("\nsounddevice missing — pip install sounddevice for live probe")
print(f"\ndone: {logp}")
