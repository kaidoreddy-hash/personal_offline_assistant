# tobi v2 — offline full-duplex speech-to-speech (Pi 5 4GB)

Wake word → open mic → live transcription → neural end-of-turn → streaming reply → voice barge-in
→ dormancy → meeting mode with speaker labels → memory that answers from past transcripts.
100% offline after a one-time model download; torch-free (onnxruntime + ctranslate2 + piper).
Only weather/search/telegram touch the network — that boundary is the product.

## Quickstart (desktop)

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # once
python tools/download_models.py        # once, then runtime is offline forever
python tools/check_offline.py          # proves it: all models load with sockets blocked
python -m pytest tests/ -q             # 26 tests green
python -m server.ws_server --config configs/desktop.yaml
# open http://127.0.0.1:8000 → "enable microphone" → say "hey jarvis"
```

## What lives where

```
agent_core/  pipeline.py (state machine) · vad.py · turn.py · wakeword.py · stt.py · tts.py
             brain/ (base + stub — the LLM seam) · tools/ (router, weather, search, telegram)
             meeting.py (diarization) · store.py (SQLite + FTS5 + embeddings) · config.py · debuglog.py
server/      ws_server.py (FastAPI transport) · ui/ (single-page client)
tools/       download_models · check_offline · e2e_offline · gen_fixtures
tests/       fixtures + 26 tests (VAD/turn/E2E/WS/store/tools)
docs/        MASTER_PLAN.md (read this) · TESTING.md · pi5_deploy.md · wiring.md
vendors/     reference clones — never delete (huggingface-speech-to-speech, on-device-s2s, xtalk)
models/      local weights (gitignored) · logs/ auto debug sessions · data/ SQLite
```

Note: models/, vendors/, and the venv live on `D:\tobi_v2_assets` with directory junctions from
this folder (C: was full during the build). Everything resolves transparently.

## Next steps

See docs/MASTER_PLAN.md phases P10-P14 (Pi deploy, LLM brain swap, multilingual, hey_tobi wake
model, hardware GPIO). Testing checklist: docs/TESTING.md.
