"""
Web dashboard server: FastAPI + WebSocket for real-time training visualization.

Endpoints:
  GET  /                    - Dashboard HTML
  GET  /api/status          - Current training status
  GET  /api/checkpoints     - List checkpoints
  POST /api/config          - Update config
  POST /api/start           - Start training
  POST /api/stop            - Stop training
  WS   /ws/live             - Real-time step data stream

Run: .venv/bin/python -m zebrav2.web.server
"""
import os
import sys
import json
import asyncio
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from zebrav2.engine.trainer import TrainingEngine
from zebrav2.engine.config import TrainingConfig, REPERTOIRES

app = FastAPI(title="Zebrafish Brain v2 Dashboard")
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

# Global engine
engine = TrainingEngine()
ws_clients = set()
_main_loop = None  # store uvicorn's event loop
_latest_step = {}  # latest step data for polling


@app.on_event("startup")
async def on_startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()


# --- WebSocket broadcast ---
async def broadcast(data):
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


def on_step_callback(step_data):
    """Called by engine on each step from training thread."""
    global _latest_step
    _latest_step = step_data
    if _main_loop and not _main_loop.is_closed():
        _main_loop.call_soon_threadsafe(
            _main_loop.create_task,
            broadcast({'type': 'step', 'data': step_data}))


def on_round_end_callback(metrics):
    if _main_loop and not _main_loop.is_closed():
        _main_loop.call_soon_threadsafe(
            _main_loop.create_task,
            broadcast({'type': 'round_end', 'data': metrics}))


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Dashboard not found. Place index.html in zebrav2/web/static/</h1>")


@app.get("/api/status")
async def get_status():
    return engine.get_status()


@app.get("/api/checkpoints")
async def get_checkpoints():
    return engine.checkpoint_mgr.list_checkpoints()


class ConfigUpdate(BaseModel):
    config: dict


@app.post("/api/config")
async def update_config(body: ConfigUpdate):
    engine.config = TrainingConfig(body.config)
    return {"status": "ok", "config": engine.config.data}


class StartRequest(BaseModel):
    n_rounds: Optional[int] = 10
    config: Optional[dict] = None


@app.post("/api/start")
async def start_training(req: StartRequest):
    if engine.running:
        return {"status": "error", "message": "Training already running"}
    if req.config:
        engine.config = TrainingConfig(req.config)
    engine.on_step = on_step_callback
    engine.on_round_end = on_round_end_callback
    engine.train_async(n_rounds=req.n_rounds)
    return {"status": "started", "n_rounds": req.n_rounds}


@app.post("/api/stop")
async def stop_training():
    engine.stop()
    return {"status": "stopped"}


@app.get("/api/repertoires")
async def get_repertoires():
    return {k: {'name': v['name'], 'description': v['description']} for k, v in REPERTOIRES.items()}


@app.post("/api/repertoire/{name}")
async def start_repertoire(name: str):
    if name not in REPERTOIRES:
        return {"status": "error", "message": f"Unknown repertoire: {name}"}
    if engine.running:
        return {"status": "error", "message": "Training already running"}
    rep = REPERTOIRES[name]
    config = TrainingConfig(rep)
    engine.config = config
    n_rounds = rep.get('training', {}).get('n_rounds', 5)
    engine.on_step = on_step_callback
    engine.on_round_end = on_round_end_callback
    engine.train_async(n_rounds=n_rounds)
    return {"status": "started", "repertoire": name, "n_rounds": n_rounds}


@app.post("/api/load_checkpoint")
async def load_checkpoint(body: dict):
    path = body.get('path')
    if path and os.path.exists(path):
        if engine.brain is None:
            engine.brain = engine._create_brain()
        rnd, metrics = engine.checkpoint_mgr.load(engine.brain, path)
        engine.total_rounds_done = rnd
        return {"status": "loaded", "round": rnd, "metrics": metrics}
    return {"status": "error", "message": "Checkpoint not found"}


# --- WebSocket ---
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            # Always send latest state for polling fallback
            if _latest_step:
                await ws.send_json({
                    'type': 'step',
                    'data': _latest_step
                })
            await ws.send_json({
                'type': 'status',
                'data': {
                    'running': engine.running,
                    'round': engine.current_round,
                }
            })
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        ws_clients.discard(ws)


def main():
    print(f"\n{'='*60}")
    print(f"  Zebrafish Brain v2 Dashboard")
    print(f"  Open: http://localhost:8765")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")


if __name__ == '__main__':
    main()
