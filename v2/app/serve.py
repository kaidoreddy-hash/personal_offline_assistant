"""v2 serve: FastAPI + WS full-duplex. Fully offline — no runtime downloads.

Offline enforcement: HF/transformers offline flags are set at import unless
V2_ALLOW_NET=1 (pre-cache run). STT/VAD/TTS load LOCAL files only and fall
back to mock/energy — they never hit the network.
Audio path is local: browser streams 16k PCM frames over WS, server runs
faster-whisper locally, replies with Piper WAV (local) — no Web Speech API.
"""

from __future__ import annotations
import asyncio, base64, json, os, sys
from pathlib import Path

if os.environ.get("V2_ALLOW_NET") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent_core.config import load
from agent_core.duplex import FullDuplex

cfg = load()
app = FastAPI(title="tobi v2 s2s offline")
WEB = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    dx = FullDuplex(cfg)  # cheap: reports which backends resolved locally
    return {
        "ok": True,
        "llm": cfg.get("llm", {}).get("backend"),
        "vad": "silero-onnx" if dx.vad._sess is not None else "energy",
        "stt": dx.stt.backend,
        "tts": dx.tts.backend,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
    }


def _emit(e):
    if e.kind == "state":
        return {"event": e.kind, "state": e.data}
    if e.kind in ("live_transcript", "llm_chunk", "tts_text"):
        return {"event": e.kind, "text": e.data}
    if e.kind == "tts_audio":
        return {"event": e.kind, "wav_b64": e.data}
    return {"event": e.kind, "file": e.data}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    dx = FullDuplex(cfg)
    await sock.send_json({"event": "state", "state": "dormant"})
    pcm_buf = bytearray()

    async def watchdog():
        while True:
            await asyncio.sleep(2)
            for e in dx.tick():
                await sock.send_json({"event": e.kind, "state": e.data})

    wd = asyncio.create_task(watchdog())
    try:
        while True:
            raw = await sock.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            ev = msg.get("event")
            if ev == "wakeword":
                out = dx.on_wakeword()
            elif ev == "partial":
                out = dx.on_partial(msg.get("text", ""))
            elif ev == "final":
                out = dx.on_final(msg.get("text", ""))
            elif ev == "barge_in":
                out = dx.on_barge_in()
            elif ev == "audio":  # local PCM 16k mono base64 from browser
                try:
                    pcm_buf.extend(base64.b64decode(msg.get("pcm_b64", "")))
                except Exception:
                    pass
                out = []
                # 1.5s window -> local STT -> live transcript (offline)
                while len(pcm_buf) >= 16000 * 2 * 3 // 2:
                    chunk = bytes(pcm_buf[: 16000 * 2 * 2])
                    del pcm_buf[: 16000 * 2 * 2]
                    is_speech = dx.vad.is_speech(chunk)
                    if dx.state == "responding" and is_speech:
                        out += dx.on_barge_in()
                    text = dx.stt.transcribe(chunk) if dx.stt.backend != "mock" else ""
                    if text:
                        if dx.state == "dormant":
                            out += dx.on_partial(text)
                        else:
                            out += dx.on_partial(text)
            else:
                out = []
            for e in out:
                if e.kind == "tts":  # synth WAV locally, send audio not just text
                    wav = dx.tts.synth(e.data)
                    await sock.send_json({"event": "tts_text", "text": e.data})
                    if wav:
                        await sock.send_json(
                            {
                                "event": "tts_audio",
                                "wav_b64": base64.b64encode(wav).decode(),
                            }
                        )
                else:
                    await sock.send_json(_emit(e))
    except Exception:
        pass
    finally:
        wd.cancel()


def demo():  # ponytail: one runnable check
    from agent_core.llm_stub import HARDCODED_RESPONSE

    dx = FullDuplex(cfg)
    assert dx.on_wakeword()[0].data == "listening"
    evs = dx.on_final("hello tobi")
    assert (
        "".join(e.data for e in evs if e.kind == "llm_chunk") == HARDCODED_RESPONSE
    ), "stub reply broke"
    print("demo ok:", HARDCODED_RESPONSE[:60])


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8765)
