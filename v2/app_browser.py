"""PARKED browser-mic variant (other session's work, kept verbatim).

Browser AudioWorklet captures mic -> 16kHz PCM over WebSocket; server does
VAD + fixed-900ms turn end + STT + Piper, audio back to the browser.
NOT the Pi path: needs a browser for the mic, browser SpeechRecognition for
captions/wake word (cloud), and faster-whisper (not installed). The served
app is app.py (server owns the mic, offline, neural end-of-turn).
Delete this file if the browser testbed is abandoned.
"""

"""v2 full-duplex voice companion — browser-mic WebSocket app.

Run:  cd personal_assistant && uvicorn v2.app:app --host 127.0.0.1 --port 8011
Open: http://127.0.0.1:8011

Architecture:
  Browser AudioWorklet captures mic -> 16kHz PCM frames over WebSocket
  Server: Silero VAD -> 900ms silence = turn end -> faster-whisper STT
          -> hardcoded reply -> Piper TTS -> audio back to browser
  Browser SpeechRecognition: live captions + wake word ("hey tobi")
"""

import asyncio
import functools
import logging
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from v2.vad import SileroVAD, frame_dbfs
from v2.stt import transcribe
from v2.tts import synthesize, pcm_to_wav_b64
from v2.reply import reply_for
from v2.log import SessionLog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2")

app = FastAPI(title="v2 Sovereign Companion")

# ---- States
DORMANT = "DORMANT"
LISTENING = "LISTENING"
THINKING = "THINKING"
SPEAKING = "SPEAKING"

# ---- Turn detection config
SILENCE_COMMIT_MS = 900.0  # silence after speech -> commit turn
MIN_SPEECH_MS = 200.0  # ignore blips shorter than this
START_THRESH = 0.50  # VAD prob to start a turn
CONTINUE_THRESH = 0.35  # VAD prob to continue speech
FLOOR_DBFS = -45.0  # energy floor (room tone gate)
MAX_BUF_S = 10.0  # cap audio buffer sent to STT

# ---- Barge-in config
BARGE_SUSTAIN_FRAMES = 4  # consecutive speech frames to trigger barge-in
BARGE_DBFS = -35.0  # minimum dB to qualify for barge-in
BARGE_GRACE_MS = 300.0  # don't barge-in within this window of reply start

# ---- Inactivity
INACTIVITY_S = 20.0  # seconds of silence -> DORMANT


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=PAGE_HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # ---- Per-connection state
    vad = SileroVAD()
    state = DORMANT
    session: Optional[SessionLog] = None
    audio_buf = bytearray()  # incoming PCM frames (re-framing)
    speech_buf = bytearray()  # accumulated speech for this turn
    is_speaking = False  # user is currently speaking
    silence_ms = 0.0  # ms of silence since last speech
    speech_ms = 0.0  # ms of speech in current turn
    last_activity = time.perf_counter()
    barge_run = 0  # consecutive barge-in qualifying frames
    reply_start_t = 0.0  # when current reply started playing
    last_vad_push = 0.0  # throttle VAD meter updates
    noise_floor: Optional[float] = None
    quiet_n = 0

    async def send(obj: dict):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    def set_state(new: str):
        nonlocal state
        if state != new:
            old = state
            state = new
            if session:
                session.state_change(old, new)
            asyncio.ensure_future(send({"type": "state", "value": new}))

    def effective_floor() -> float:
        if noise_floor is None or quiet_n < 20:
            return FLOOR_DBFS
        return max(-55.0, min(-30.0, noise_floor + 12.0))

    def reset_turn():
        nonlocal is_speaking, silence_ms, speech_ms, speech_buf
        is_speaking = False
        silence_ms = 0.0
        speech_ms = 0.0
        speech_buf = bytearray()

    async def do_barge_in(auto: bool):
        nonlocal barge_run, reply_start_t
        barge_run = 0
        reply_start_t = 0.0
        reset_turn()
        if session:
            session.barge_in(auto)
        set_state(LISTENING)
        await send({"type": "barged_in", "auto": auto})
        log.info("barge-in (%s)", "auto" if auto else "manual")

    async def commit_turn():
        """Turn ended by silence. Run STT -> reply -> TTS -> send audio."""
        nonlocal reply_start_t
        if speech_ms < MIN_SPEECH_MS:
            if session:
                session.turn_end("too_short", speech_ms, silence_ms)
            reset_turn()
            return

        # Cap buffer to last MAX_BUF_S seconds
        max_bytes = int(16000 * 2 * MAX_BUF_S)
        pcm = (
            bytes(speech_buf[-max_bytes:])
            if len(speech_buf) > max_bytes
            else bytes(speech_buf)
        )
        turn_speech_ms = speech_ms
        turn_silence_ms = silence_ms
        reset_turn()

        if session:
            session.turn_end("silence", turn_speech_ms, turn_silence_ms)

        set_state(THINKING)
        await send({"type": "turn", "phase": "committed", "note": "transcribing..."})

        # STT (runs in executor to not block the event loop)
        t0 = time.perf_counter()
        text, lang, prob = await loop.run_in_executor(
            None, functools.partial(transcribe, pcm, "en")
        )
        stt_ms = (time.perf_counter() - t0) * 1000.0

        if session:
            session.stt(text, lang, prob, stt_ms)

        if not text.strip():
            log.info("STT returned empty (%.0fms)", stt_ms)
            await send(
                {"type": "transcript", "text": "(silence)", "stt_ms": round(stt_ms)}
            )
            set_state(LISTENING)
            return

        log.info("STT: %r lang=%s (%.0fms)", text[:80], lang, stt_ms)
        await send(
            {"type": "transcript", "text": text, "lang": lang, "stt_ms": round(stt_ms)}
        )

        # Reply
        reply_text = reply_for(text, lang)
        t1 = time.perf_counter()
        if session:
            session.reply_event(reply_text, (t1 - t0) * 1000.0)

        await send({"type": "reply", "text": reply_text})

        # TTS
        pcm_out = await loop.run_in_executor(
            None, functools.partial(synthesize, reply_text, lang)
        )
        synth_ms = (time.perf_counter() - t1) * 1000.0
        audio_ms = len(pcm_out) / (16000 * 2) * 1000.0

        if session:
            session.tts_event(reply_text, synth_ms, audio_ms)

        log.info("TTS: %.0fms synth, %.0fms audio", synth_ms, audio_ms)

        # Send audio to browser
        set_state(SPEAKING)
        reply_start_t = time.perf_counter()
        wav_b64 = await loop.run_in_executor(None, pcm_to_wav_b64, pcm_out)
        await send({"type": "audio", "wav_base64": wav_b64})
        await send(
            {"type": "done", "stt_ms": round(stt_ms), "synth_ms": round(synth_ms)}
        )

    async def handle_audio(chunk: bytes):
        """Process a 32ms PCM16 frame through VAD and turn detection."""
        nonlocal is_speaking, silence_ms, speech_ms, last_activity
        nonlocal barge_run, last_vad_push, noise_floor, quiet_n

        db = frame_dbfs(chunk)
        floor = effective_floor()

        # Learn room noise from quiet frames
        if db < floor:
            quiet_n += 1
            noise_floor = (
                db if noise_floor is None else (0.95 * noise_floor + 0.05 * db)
            )

        prob = 0.0 if db < floor else vad.predict(chunk)

        # Throttled VAD meter to browser (~8Hz)
        now = time.perf_counter()
        if now - last_vad_push > 0.12:
            last_vad_push = now
            await send(
                {
                    "type": "vad",
                    "db": round(db, 1),
                    "prob": round(prob, 2),
                    "speaking": is_speaking,
                    "state": state,
                }
            )

        # ---- SPEAKING state: check for barge-in
        if state == SPEAKING:
            if (
                prob >= START_THRESH
                and db >= BARGE_DBFS
                and (time.perf_counter() - reply_start_t) > (BARGE_GRACE_MS / 1000.0)
            ):
                barge_run += 1
                if barge_run >= BARGE_SUSTAIN_FRAMES:
                    await do_barge_in(auto=True)
            else:
                barge_run = 0
            return

        # ---- DORMANT state: don't process turn detection
        if state == DORMANT:
            return

        # ---- LISTENING state: VAD turn detection
        thresh = CONTINUE_THRESH if is_speaking else START_THRESH
        speech = prob >= thresh

        if not is_speaking:
            if speech and db >= floor:
                # Turn started
                is_speaking = True
                speech_buf = bytearray(chunk)
                silence_ms = 0.0
                speech_ms = 32.0
                last_activity = time.perf_counter()
                if session:
                    session.turn_start(db, prob)
                await send({"type": "turn", "phase": "speech", "note": "speaking..."})
                log.info("turn start db=%.1f prob=%.2f", db, prob)
        else:
            speech_buf.extend(chunk)

            # Cap buffer size
            max_bytes = int(16000 * 2 * MAX_BUF_S)
            if len(speech_buf) > max_bytes:
                speech_buf = speech_buf[-max_bytes:]

            if speech and db >= floor - 3.0:
                silence_ms = 0.0
                speech_ms += 32.0
                last_activity = time.perf_counter()
            else:
                silence_ms += 32.0

            # Silence too long for a blip -> discard
            if speech_ms < MIN_SPEECH_MS and silence_ms >= SILENCE_COMMIT_MS * 2:
                reset_turn()
                return

            # 900ms silence after speech -> commit turn
            if silence_ms >= SILENCE_COMMIT_MS and speech_ms >= MIN_SPEECH_MS:
                await send(
                    {
                        "type": "turn",
                        "phase": "end",
                        "note": "turn ended (900ms silence)",
                    }
                )
                log.info(
                    "turn end: %.0fms speech, %.0fms silence", speech_ms, silence_ms
                )
                await commit_turn()
                return

            # Show silence progress
            if silence_ms > 0 and silence_ms % 128 < 33:  # ~every 128ms
                await send(
                    {
                        "type": "turn",
                        "phase": "silence",
                        "silence_ms": round(silence_ms),
                        "note": f"silence {silence_ms:.0f}ms / {SILENCE_COMMIT_MS:.0f}ms",
                    }
                )

    # ---- Main WebSocket loop
    try:
        while True:
            msg = await ws.receive()

            # Binary: PCM audio frames
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                audio_buf.extend(raw)
                # Re-frame to 32ms chunks (1024 bytes = 512 samples @16kHz 16-bit)
                while len(audio_buf) >= 1024:
                    frame = bytes(audio_buf[:1024])
                    del audio_buf[:1024]
                    await handle_audio(frame)

                # Inactivity check
                if state == LISTENING and not is_speaking:
                    if (time.perf_counter() - last_activity) > INACTIVITY_S:
                        set_state(DORMANT)
                        await send(
                            {
                                "type": "turn",
                                "phase": "dormant",
                                "note": f"no speech for {INACTIVITY_S:.0f}s",
                            }
                        )
                        log.info("inactivity -> DORMANT")
                continue

            # Text: JSON control messages
            if "text" in msg and msg["text"]:
                import json

                try:
                    ctl = json.loads(msg["text"])
                except Exception:
                    continue

                cmd = ctl.get("type", "")

                if cmd == "start":
                    if session is None:
                        session = SessionLog()
                        session.config(
                            silence_ms=SILENCE_COMMIT_MS,
                            min_speech_ms=MIN_SPEECH_MS,
                            vad_thresh=START_THRESH,
                        )
                    reset_turn()
                    last_activity = time.perf_counter()
                    set_state(LISTENING)
                    await send(
                        {"type": "meta", "session": session.path if session else ""}
                    )
                    log.info("session started: %s", session.path if session else "?")

                elif cmd == "stop":
                    reset_turn()
                    set_state(DORMANT)
                    if session:
                        session.session_end("user_stop")
                        session = None
                    log.info("session stopped")

                elif cmd == "wake":
                    phrase = ctl.get("phrase", "hey tobi")
                    if state == DORMANT:
                        if session is None:
                            session = SessionLog()
                            session.config(
                                silence_ms=SILENCE_COMMIT_MS,
                                min_speech_ms=MIN_SPEECH_MS,
                                vad_thresh=START_THRESH,
                            )
                        if session:
                            session.wake_word(phrase)
                        reset_turn()
                        last_activity = time.perf_counter()
                        set_state(LISTENING)
                        log.info("wake word: %r", phrase)

                elif cmd == "interrupt":
                    if state == SPEAKING:
                        await do_barge_in(auto=False)

                elif cmd == "audio_done":
                    # Browser finished playing TTS audio
                    if state == SPEAKING:
                        set_state(LISTENING)
                        last_activity = time.perf_counter()

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.exception("WS error: %s", e)
    finally:
        if session and not session.closed:
            session.session_end("disconnect")


# ---- HTML + JS frontend (inline)
PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v2 - Sovereign Companion</title>
<style>
:root{--bg:#0a0e1a;--surface:#141825;--accent:#22d3ee;--text:#e2e8f0;
      --dim:#64748b;--border:#1e293b;--red:#ef4444;--green:#34d399;--amber:#f59e0b}
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',system-ui,sans-serif}
body{background:var(--bg);color:var(--text);display:flex;min-height:100vh;
     align-items:center;justify-content:center;padding:20px}
.wrap{max-width:640px;width:100%;display:flex;flex-direction:column;gap:14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px}
h1{font-size:20px;margin-bottom:2px}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.row{display:flex;align-items:center;gap:10px;margin:8px 0;flex-wrap:wrap}
.dot{width:14px;height:14px;border-radius:50%;background:#475569;transition:all .3s}
.dot.LISTENING{background:var(--accent);box-shadow:0 0 14px var(--accent)}
.dot.THINKING{background:var(--amber);box-shadow:0 0 14px var(--amber)}
.dot.SPEAKING{background:var(--green);box-shadow:0 0 14px var(--green)}
.dot.DORMANT{background:var(--red);box-shadow:0 0 8px var(--red)}
.pill{font-size:12px;padding:3px 12px;border-radius:999px;border:1px solid var(--border);color:var(--dim)}
.bar{height:10px;background:#0f172a;border-radius:6px;border:1px solid var(--border);overflow:hidden;margin:6px 0}
.barfill{height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--green));transition:width .1s}
.vadlbl{font-size:11px;color:var(--dim)}
.box{background:#0a0e1a;border:1px solid var(--border);border-radius:10px;padding:12px;
     min-height:60px;font-size:15px;line-height:1.5;margin:6px 0}
.box .interim{color:#7dd3fc}.box .final{color:#fff}
.reply{background:#0d1520;border:1px solid var(--border);border-radius:10px;
       padding:12px;min-height:48px;font-size:14px;color:var(--green)}
button{border-radius:10px;border:1px solid var(--border);padding:10px 18px;
       font-size:14px;font-weight:700;cursor:pointer;background:var(--accent);color:#000}
button.ghost{background:var(--surface);color:var(--text)}
button.danger{background:var(--red);color:#fff}
button:disabled{opacity:.4;cursor:default}
button.active{background:var(--red);color:#fff;animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.6}}
#log{font-size:11px;color:var(--dim);max-height:140px;overflow-y:auto;margin-top:8px}
#log div{padding:1px 0}
.foot{font-size:11px;color:var(--dim);text-align:center;margin-top:6px}
.foot b{color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>v2 Sovereign Companion</h1>
    <div class="sub">open-mic full-duplex | VAD + 900ms silence | faster-whisper STT | Piper TTS | say "hey tobi" to wake</div>
    <div class="row">
      <div id="dot" class="dot DORMANT"></div>
      <span id="statePill" class="pill">DORMANT</span>
      <span id="vadPill" class="pill">VAD: --</span>
    </div>
    <div class="bar"><div id="barfill" class="barfill"></div></div>
    <div class="vadlbl" id="vadlbl">waiting for mic...</div>
  </div>

  <div class="card">
    <div class="row">
      <button id="micBtn">Start Mic</button>
      <button id="stopBtn" class="ghost">Stop</button>
      <button id="intBtn" class="ghost">Interrupt</button>
    </div>
  </div>

  <div class="card">
    <div style="font-size:12px;color:var(--dim);margin-bottom:4px">LIVE TRANSCRIPT</div>
    <div class="box" id="live"><span class="interim">press Start, then speak...</span></div>
  </div>

  <div class="card">
    <div style="font-size:12px;color:var(--dim);margin-bottom:4px">ASSISTANT REPLY</div>
    <div class="reply" id="replyBox">...</div>
  </div>

  <div class="card">
    <div style="font-size:12px;color:var(--dim)">EVENT LOG</div>
    <div id="log"></div>
  </div>

  <div class="foot">
    VAD: <b>Silero</b> | Turn: <b>900ms silence</b> | STT: <b>faster-whisper</b> | TTS: <b>Piper</b>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
let ws, actx, micStream, workNode, micOn=false, rec=null, recFatal=null;
let audioQ=[], playing=false, curAudio=null, currentState='DORMANT';

function addLog(msg){
  const d=document.createElement('div');
  d.textContent=new Date().toLocaleTimeString()+' '+msg;
  $('log').prepend(d);
  if($('log').children.length>100)$('log').lastChild.remove();
}

function setState(s){
  currentState=s;
  $('dot').className='dot '+s;
  $('statePill').textContent=s;
}

// ---- WebSocket
function connect(){
  ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');
  ws.binaryType='arraybuffer';
  ws.onopen=()=>addLog('connected');
  ws.onclose=()=>{addLog('disconnected');setTimeout(connect,2000);};
  ws.onmessage=e=>{
    const m=JSON.parse(e.data);
    if(m.type==='state'){
      setState(m.value);
      addLog('state: '+m.value);
      // Auto-start speech recognition when listening
      if(m.value==='LISTENING'&&micOn)startRec();
      if(m.value==='DORMANT')startWakeWordListener();
    }
    else if(m.type==='vad'){
      $('barfill').style.width=Math.min(100,Math.max(0,(m.db+60)*1.8))+'%';
      $('vadlbl').textContent='VAD: '+m.db+'dB p='+m.prob+(m.speaking?' SPEECH':' quiet');
      $('vadPill').textContent='p='+m.prob;
    }
    else if(m.type==='turn'){
      addLog('turn: '+m.phase+' - '+m.note);
    }
    else if(m.type==='transcript'){
      $('live').innerHTML='<span class="final">'+esc(m.text)+'</span>';
      addLog('STT ('+m.stt_ms+'ms): '+m.text);
    }
    else if(m.type==='reply'){
      $('replyBox').textContent=m.text;
      addLog('reply: '+m.text);
    }
    else if(m.type==='audio'){
      queueAudio(m.wav_base64);
    }
    else if(m.type==='done'){
      addLog('done (stt='+m.stt_ms+'ms synth='+m.synth_ms+'ms)');
    }
    else if(m.type==='barged_in'){
      stopAllAudio();
      addLog('barge-in! '+(m.auto?'(auto)':'(manual)'));
      if(micOn)startRec();
    }
    else if(m.type==='meta'){
      addLog('session: '+m.session);
    }
  };
}

// ---- Mic capture: AudioWorklet -> 16kHz Int16 -> WebSocket binary
async function startMic(){
  if(micOn)return;
  if(!ws||ws.readyState!==1)connect();
  try{
    micStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
    actx=actx||new(window.AudioContext||window.webkitAudioContext)();
    if(actx.state==='suspended')await actx.resume();
    const src=actx.createMediaStreamSource(micStream);
    const blob=new Blob([`registerProcessor('v2cap',class extends AudioWorkletProcessor{
      constructor(){super();this.buf=[];}
      process(inp){
        const ch=inp[0][0];if(!ch)return true;
        const r=sampleRate/16000;
        for(let i=0;i<ch.length;i+=r){const j=Math.floor(i);if(j<ch.length)this.buf.push(Math.max(-1,Math.min(1,ch[j])));}
        while(this.buf.length>=512){
          const o=new Int16Array(512);
          for(let k=0;k<512;k++)o[k]=this.buf[k]*32767;
          this.buf=this.buf.slice(512);
          this.port.postMessage(o.buffer,[o.buffer]);
        }
        return true;
      }
    });`],{type:'application/javascript'});
    await actx.audioWorklet.addModule(URL.createObjectURL(blob));
    workNode=new AudioWorkletNode(actx,'v2cap');
    workNode.port.onmessage=ev=>{if(ws&&ws.readyState===1)ws.send(ev.data);};
    src.connect(workNode);
    micOn=true;
    $('micBtn').textContent='Mic ON';
    $('micBtn').classList.add('active');
    addLog('mic started');

    // Send start command
    ws.send(JSON.stringify({type:'start'}));
  }catch(e){
    addLog('mic error: '+(e.message||e));
  }
}

function stopMic(){
  micOn=false;
  try{workNode&&workNode.disconnect();}catch(e){}
  try{micStream&&micStream.getTracks().forEach(t=>t.stop());}catch(e){}
  stopRec();
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'stop'}));
  $('micBtn').textContent='Start Mic';
  $('micBtn').classList.remove('active');
  addLog('mic stopped');
}

// ---- Browser SpeechRecognition for live captions + wake word
function startRec(){
  if(!micOn)return;
  stopRec();
  if(recFatal)return;
  setTimeout(()=>{
    if(!micOn)return;
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){$('live').innerHTML='<span class="interim">live captions need Chrome/Edge</span>';return;}
    rec=new SR();rec.lang='en-IN';rec.interimResults=true;rec.continuous=true;
    rec.onresult=e=>{
      let inter='',fin='';
      for(let i=e.resultIndex;i<e.results.length;i++){
        if(e.results[i].isFinal)fin+=e.results[i][0].transcript;
        else inter+=e.results[i][0].transcript;
      }
      if(inter)$('live').innerHTML='<span class="interim">'+esc(inter)+'</span>';
      if(fin)$('live').innerHTML='<span class="final">'+esc(fin)+'</span>';
    };
    rec.onerror=e=>{
      if(e.error==='not-allowed'||e.error==='service-not-allowed'){
        recFatal=e.error;addLog('captions blocked: '+e.error);return;
      }
      if(e.error!=='no-speech')addLog('caption: '+e.error);
    };
    rec.onend=()=>{if(micOn&&!recFatal)setTimeout(startRec,200);};
    try{rec.start();}catch(e){}
  },150);
}
function stopRec(){if(rec){try{rec.onend=null;rec.stop();}catch(e){}rec=null;}}

// ---- Wake word listener (runs in DORMANT state)
function startWakeWordListener(){
  if(!micOn)return;
  stopRec();
  if(recFatal)return;
  setTimeout(()=>{
    if(!micOn||currentState!=='DORMANT')return;
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR)return;
    rec=new SR();rec.lang='en-IN';rec.interimResults=true;rec.continuous=true;
    rec.onresult=e=>{
      for(let i=e.resultIndex;i<e.results.length;i++){
        const t=e.results[i][0].transcript.toLowerCase();
        if(t.includes('hey tobi')||t.includes('hey toby')||t.includes('a tobi')||t.includes('hey tobey')){
          addLog('wake word detected: '+t);
          $('live').innerHTML='<span class="final">Wake word: '+esc(t)+'</span>';
          if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'wake',phrase:t}));
          stopRec();
          return;
        }
      }
      // Show what we're hearing during dormant
      const last=e.results[e.results.length-1][0].transcript;
      $('live').innerHTML='<span class="interim">(dormant) '+esc(last)+'</span>';
    };
    rec.onerror=e=>{
      if(e.error==='not-allowed'||e.error==='service-not-allowed'){recFatal=e.error;return;}
    };
    rec.onend=()=>{if(micOn&&currentState==='DORMANT'&&!recFatal)setTimeout(startWakeWordListener,200);};
    try{rec.start();}catch(e){}
  },150);
}

// ---- Audio playback queue
function queueAudio(b64){audioQ.push(b64);if(!playing)playNext();}
function playNext(){
  if(!audioQ.length){
    playing=false;
    if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'audio_done'}));
    return;
  }
  playing=true;
  const b64=audioQ.shift();
  curAudio=new Audio('data:audio/wav;base64,'+b64);
  curAudio.onended=playNext;
  curAudio.onerror=playNext;
  curAudio.play().catch(()=>{
    addLog('audio blocked - tap page to enable');
    document.body.addEventListener('click',function unblock(){
      addLog('audio enabled');playNext();
      document.body.removeEventListener('click',unblock);
    });
  });
}
function stopAllAudio(){audioQ=[];playing=false;try{curAudio&&curAudio.pause();}catch(e){}curAudio=null;}

// ---- Button handlers
$('micBtn').onclick=()=>{if(micOn)stopMic();else startMic();};
$('stopBtn').onclick=stopMic;
$('intBtn').onclick=()=>{
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'interrupt'}));
  stopAllAudio();
};

connect();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" v2 Sovereign Companion")
    print(" http://127.0.0.1:8011")
    print("=" * 60 + "\n")

    # Warmup VAD
    print("[warmup] Silero VAD...", flush=True)
    SileroVAD().predict(b"\x00" * 1024)
    print("[warmup] READY.", flush=True)

    import webbrowser, threading

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8011")).start()
    uvicorn.run(app, host="127.0.0.1", port=8011)
