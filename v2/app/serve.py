"""v2 serve: FastAPI + WS full-duplex stub. Offline-first, no network model loads."""

from __future__ import annotations
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent_core.config import load
from agent_core.duplex import FullDuplex

cfg = load()
app = FastAPI(title="tobi v2 s2s stub")
WEB = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "state": "stub", "llm": cfg.get("llm", {}).get("backend")}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    dx = FullDuplex(cfg)
    await sock.send_json({"event": "state", "state": "dormant"})

    async def watchdog():
        while True:
            await asyncio.sleep(2)
            for e in dx.tick():
                await sock.send_json({"event": e.kind, "state": e.data})

    wd = asyncio.create_task(watchdog())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            ev, text = msg.get("event"), msg.get("text", "")
            if ev == "wakeword":
                out = dx.on_wakeword()
            elif ev == "partial":
                out = dx.on_partial(text)
            elif ev == "final":
                out = dx.on_final(text)
            elif ev == "barge_in":
                out = dx.on_barge_in()
            else:
                out = []
            for e in out:
                await sock.send_json(
                    {
                        "event": e.kind,
                        **(
                            {"state": e.data}
                            if e.kind == "state"
                            else {"text": e.data}
                            if e.kind in ("live_transcript", "llm_chunk", "tts")
                            else {"file": e.data}
                        ),
                    }
                )
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
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8765)
