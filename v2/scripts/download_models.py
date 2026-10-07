"""Download quantized/tiny models once while online. Idempotent, skips existing."""

from __future__ import annotations
import shutil
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# quantized / tiny only — Pi 5 4GB budget.
FILES = {
    # VAD Silero (~2.3MB) via onnxruntime, no torch.
    "silero_vad_v5.onnx": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    # TTS Piper lessac LOW (~63MB) + config. -low not -medium saves RAM/disk.
    "en_US-lessac-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx",
    "en_US-lessac-low.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
}
# STT tiny.en (~75MB int8) fetched by faster-whisper on first use, then cached for HF_HUB_OFFLINE=1.
# Friend's SLM later: Qwen2.5-0.5B-Instruct Q4_K_M (~400MB) via llama.cpp on :8080 — NOT here.


def _get(name: str, url: str) -> None:
    dest = MODELS / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {name} ({dest.stat().st_size // 1024}KB)")
        return
    print(f"get {name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"ok {name} ({dest.stat().st_size // 1024}KB)")


def _wakeword() -> None:
    """openWakeWord via official API (hey_jarvis stand-in until hey_tobi trained)."""
    dest = MODELS / "hey_jarvis_v0.1.onnx"
    if dest.exists():
        print(f"skip hey_jarvis_v0.1.onnx ({dest.stat().st_size // 1024}KB)")
        return
    try:
        from openwakeword.utils import download_models  # type: ignore
        import openwakeword  # type: ignore

        download_models(["hey_jarvis"])
        src = (
            Path(openwakeword.__file__).parent
            / "resources"
            / "models"
            / "hey_jarvis_v0.1.onnx"
        )
        if src.exists():
            shutil.copy(src, dest)
            print(f"ok hey_jarvis_v0.1.onnx ({dest.stat().st_size // 1024}KB)")
        else:
            print(
                f"wakeword downloaded but not found at {src} — transcript fallback still works"
            )
    except ImportError:
        print("skip wakeword: pip install -r requirements-pi.txt first (openwakeword)")
    except Exception as e:
        print(f"FAIL wakeword: {e} — transcript fallback still works")


def main() -> None:
    for name, url in FILES.items():
        try:
            _get(name, url)
        except Exception as e:
            print(f"FAIL {name}: {e}")
    _wakeword()
    print(
        "done. First STT use downloads tiny.en (online), then HF_HUB_OFFLINE=1 works offline."
    )


if __name__ == "__main__":
    main()
