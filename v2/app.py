"""v2 web testbed. Server captures the mic, browser shows state + transcript.
Minimal HTML, no build step. Start/Stop own the session file; Interrupt
bumps the epoch and flushes playback (barge-in)."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from v2.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("v2")

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>v2 s2s</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center}
 .card{background:#1e293b;border-radius:16px;padding:28px 34px;width:520px;max-width:92vw;box-shadow:0 10px 40px #0008}
 h1{margin:0 0 4px;font-size:20px}
 .sub{color:#94a3b8;font-size:13px;margin-bottom:18px}
 .row{display:flex;align-items:center;gap:10px;margin:10px 0}
 .dot{width:12px;height:12px;border-radius:50%;background:#475569}
 .dot.listening{background:#22d3ee;box-shadow:0 0 10px #22d3ee}
 .dot.thinking{background:#f59e0b;box-shadow:0 0 10px #f59e0b}
 .dot.speaking{background:#34d399;box-shadow:0 0 10px #34d399}
 .dot.dormant{background:#f43f5e}
 .pill{border:1px solid #334155;border-radius:999px;padding:2px 10px;font-size:12px;color:#cbd5e1}
 .box{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px;min-height:70px;font-size:15px;line-height:1.5}
 .meta{color:#94a3b8;font-size:12px;margin-top:12px}
 button{background:#38bdf8;border:0;color:#0f172a;font-weight:600;border-radius:10px;padding:10px 16px;font-size:14px;cursor:pointer}
 button.secondary{background:#334155;color:#e2e8f0}
</style></head><body><div class="card">
<h1>v2 · sovereign companion</h1>
<div class="sub">open-mic handsfree · say “hey jarvis” or press Start · Silero VAD → SmartTurn → Whistle STT → Piper TTS</div>
<div class="row"><div id="dot" class="dot dormant"></div><span id="state" class="pill">DORMANT</span><span id="lang" class="pill"></span></div>
<div class="box" id="text">press Start, then speak…</div>
<div class="meta" id="meta"></div>
<div class="row" style="margin-top:16px">
 <button id="start">Start</button>
 <button id="stop" class="secondary">Stop</button>
 <button id="interrupt" class="secondary">Interrupt</button>
</div>
</div>
<script>
const dot=document.getElementById('dot'), st=document.getElementById('state'), lang=document.getElementById('lang'),
      box=document.getElementById('text'), meta=document.getElementById('meta');
let ws;
function conn(){ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');
 ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.state){st.textContent=m.state; dot.className='dot '+m.state.toLowerCase();}
  if(m.transcript!==undefined) box.textContent=m.transcript||'(listening…)';
  if(m.lang) lang.textContent=m.lang;
  if(m.meta) meta.textContent=m.meta;};
 ws.onclose=()=>setTimeout(conn,1000);}
conn();
document.getElementById('start').onclick=()=>ws.send(JSON.stringify({cmd:'start'}));
document.getElementById('stop').onclick=()=>ws.send(JSON.stringify({cmd:'stop'}));
document.getElementById('interrupt').onclick=()=>ws.send(JSON.stringify({cmd:'interrupt'}));
</script></body></html>"""

_pipeline: Pipeline | None = None
_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


async def _broadcast(payload: dict):
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for d in dead:
        _clients.discard(d)


def _push(payload: dict):
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)


def _state_cb(s: str):
    _push({"state": s})


def _partial_cb(text: str, lang: str, ms: float):
    _push({"transcript": text, "lang": lang, "meta": f"stt {ms:.0f} ms"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _loop
    _loop = asyncio.get_running_loop()
    _pipeline = Pipeline(lang="en", p_complete=0.75)
    _pipeline.on_state(_state_cb)
    _pipeline.on_partial(_partial_cb)
    _pipeline.start()
    log.info("v2 pipeline started")
    yield
    _pipeline.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                cmd = json.loads(msg).get("cmd")
            except Exception:
                continue
            if cmd == "start" and _pipeline:
                path = _pipeline.start_session()
                await ws.send_text(json.dumps({"meta": f"session: {path}"}))
            if cmd == "stop" and _pipeline:
                _pipeline.stop_session()
            if cmd == "interrupt" and _pipeline:
                _pipeline.interrupt()
    except WebSocketDisconnect:
        _clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("v2.app:app", host="127.0.0.1", port=8011, log_level="info")
