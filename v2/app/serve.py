"""v2 serve: FastAPI + WS full-duplex. Fully offline — no runtime downloads.

Frame-aligned 96ms audio pump: wakeword (dormant) -> VAD -> accumulate ->
live STT -> turn detect -> final -> respond, with barge-in during playback.
"""

from __future__ import annotations
import asyncio, base64, json, os, sys, time
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
FRAME_BYTES = 3072  # 96ms @ 16kHz 16-bit mono


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    dx = FullDuplex(cfg)
    return {
        "ok": True,
        "llm": cfg.get("llm", {}).get("backend"),
        "vad": "silero-onnx" if dx.vad._sess is not None else "energy",
        "wakeword": dx.ww.backend,
        "stt": dx.stt.backend,
        "tts": dx.tts.backend,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
    }


def _emit(e):
    if e.kind == "state":
        return {"event": e.kind, "state": e.data}
    if e.kind in ("live_transcript", "llm_chunk", "tts_text", "barge_in"):
        return {"event": e.kind, "text": e.data}
    if e.kind == "tts_audio":
        return {"event": e.kind, "wav_b64": e.data}
    return {"event": e.kind, "file": e.data}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    dx = FullDuplex(cfg)
    await sock.send_json({"event": "state", "state": dx.state})
    pcm_buf, turn_buf = bytearray(), bytearray()
    last_stt = 0.0

    async def watchdog():
        while True:
            await asyncio.sleep(2)
            for e in dx.tick():
                await sock.send_json({"event": e.kind, "state": e.data})

    wd = asyncio.create_task(watchdog())
    try:
        while True:
            try:
                msg = json.loads(await sock.receive_text())
            except Exception:
                continue
            ev, out = msg.get("event"), []

            if ev == "wakeword":
                out += dx.on_wakeword()
                turn_buf.clear()
            elif ev == "playback_done":
                out += dx.on_playback_done()
                turn_buf.clear()
            elif ev == "barge_in":
                out += dx.on_barge_in()
                turn_buf.clear()
            elif ev == "final":
                if msg.get("text", "").strip():
                    out += dx.on_final(msg["text"])
                    turn_buf.clear()
            elif ev == "audio":
                try:
                    pcm_buf.extend(base64.b64decode(msg.get("pcm_b64", "")))
                except Exception:
                    pass
                while len(pcm_buf) >= FRAME_BYTES:
                    chunk = bytes(pcm_buf[:FRAME_BYTES])
                    del pcm_buf[:FRAME_BYTES]
                    if dx.state == "dormant":
                        if dx.ww.predict(chunk):
                            out += dx.on_wakeword()
                            turn_buf.clear()
                            break
                    is_speech = dx.vad.is_speech(chunk)
                    if dx.state == "responding" and is_speech:
                        out += dx.on_barge_in()
                        turn_buf.clear()
                    if dx.state == "listening":
                        turn_buf.extend(chunk)
                        now = time.monotonic()
                        # live partial transcript ~2x/sec for responsive UI
                        if now - last_stt > 0.5 and len(turn_buf) >= 16000:
                            pt = dx.stt.transcribe(bytes(turn_buf))
                            if pt:
                                out += dx.on_partial(pt)
                            last_stt = now
                        if dx.turn.update(is_speech, dx._partial, 0.096) == "end":
                            final = dx.stt.transcribe(bytes(turn_buf))
                            turn_buf.clear()
                            dx.turn.reset()
                            if final.strip():
                                out += dx.on_final(final)
                            else:
                                dx._partial = ""
            for e in out:
                if e.kind == "tts":
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


def demo():
    from agent_core.llm_stub import HARDCODED_RESPONSE

    dx = FullDuplex(cfg)
    assert dx.on_wakeword()[0].data == "listening"
    evs = dx.on_final("hello tobi")
    chunks = "".join(e.data for e in evs if e.kind == "llm_chunk")
    assert chunks == HARDCODED_RESPONSE, f"stub reply failed: {chunks}"
    assert dx.state == "responding", "must stay responding until playback_done"
    assert dx.on_playback_done()[0].data == "listening"
    print("demo ok:", HARDCODED_RESPONSE[:60])


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8765)
