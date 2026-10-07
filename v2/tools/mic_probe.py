"""Probe mic + VAD without starting server. Writes timestamped log."""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_core.config import load
from agent_core.vad import VAD
from agent_core import logger as _log

cfg = load()
logp = _log.new_attempt(cfg, "mic-probe")
vad = VAD(cfg)
print(
    f"backend: {'silero' if vad._silero is not None else 'energy-fallback'} -> {logp}"
)
try:
    import sounddevice as sd  # optional

    def cb(indata, frames, t, status):
        speech = vad.is_speech(bytes(indata))
        _log.log(logp, "frame", speech=speech)
        print("SPEECH" if speech else ".", end="", flush=True)

    with sd.RawInputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=1600, callback=cb
    ):
        time.sleep(10)
except ImportError:
    _log.log(
        logp, "no-sounddevice", hint="pip install sounddevice; energy check skipped"
    )
    print("sounddevice missing — log written, install it on Pi for live probe.")
print(f"\nlog: {logp}")
