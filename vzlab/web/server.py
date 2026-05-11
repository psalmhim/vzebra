"""
vzlab Virtual Laboratory Dashboard — FastAPI + WebSocket server.

Endpoints:
  GET  /              Dashboard HTML
  GET  /api/species   List species
  GET  /api/protocols List protocols + episode_length
  POST /api/start     Start experiment {species, protocol, n_episodes, seed}
  POST /api/stop      Stop experiment
  POST /api/ablate    Ablate/restore region {region, enabled}
  POST /api/param     Set param {path, value}
  GET  /api/status    Run status
  WS   /ws/live       10 Hz frame stream
"""
from __future__ import annotations
import asyncio, os, sys, threading, time
from dataclasses import asdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = FastAPI(title="vzlab Virtual Laboratory Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global state
_runner = None
_organism = None
_protocol_obj = None
_running = False
_latest_frame: dict = {}
_ws_clients: set[WebSocket] = set()
_main_loop: asyncio.AbstractEventLoop | None = None
_status = {"species": None, "protocol": None, "episode": 0, "step": 0}
_stop_event = threading.Event()


def _load_vzlab():
    from vzlab.core.registry import list_species, build
    from vzlab.protocols import PROTOCOLS
    from vzlab.engine.organism import VirtualOrganism
    return list_species, PROTOCOLS, build, VirtualOrganism


@app.on_event("startup")
async def on_startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()


async def _broadcast(data: dict) -> None:
    dead: set = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def _broadcast_threadsafe(data: dict) -> None:
    if _main_loop and not _main_loop.is_closed():
        try:
            _main_loop.call_soon_threadsafe(_main_loop.create_task, _broadcast(data))
        except RuntimeError:
            pass


def _sf(v) -> float:
    try: return float(v)
    except: return 0.0


def _sanitize(obj):
    if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_sanitize(x) for x in obj]
    if hasattr(obj, "item"): return obj.item()
    if isinstance(obj, float) and obj != obj: return 0.0
    return obj


def _build_frame(ws, hs, species, protocol, episode):
    agents = []
    for a in ws.agents:
        p, v = a.position, a.velocity
        agents.append({"id": a.agent_id,
                        "x": _sf(p[0]) if len(p) > 0 else 0.0,
                        "y": _sf(p[1]) if len(p) > 1 else 0.0,
                        "heading": _sf(a.orientation),
                        "alive": bool(a.alive),
                        "tags": a.tags})
    food = [[_sf(fp[0]), _sf(fp[1])] for fp in ws.food_positions] \
        if ws.food_positions is not None and len(ws.food_positions) else []
    preds = [[_sf(pp[0]), _sf(pp[1])] for pp in ws.predator_positions] \
        if ws.predator_positions is not None and len(ws.predator_positions) else []
    extras = {}
    for k, v in ws.extras.items():
        try: extras[k] = float(v) if hasattr(v, "__float__") else v
        except: extras[k] = str(v)
    try: hier = {k: v for k, v in asdict(hs).items() if k not in ("t", "agent_id")}
    except: hier = {}
    return {"type": "step", "species": species, "protocol": protocol,
            "t": int(ws.t), "episode": episode,
            "agents": agents, "food": food, "predators": preds,
            "extras": extras, "hier": _sanitize(hier)}


def _run_experiment(species, protocol_name, n_episodes, seed):
    global _running, _latest_frame, _status, _organism, _protocol_obj
    try:
        list_species, PROTOCOLS, build, VirtualOrganism = _load_vzlab()
        _connectome, brain, env = build(species)
        organism = VirtualOrganism(brain, env, agent_id=0)
        _organism = organism
        protocol = PROTOCOLS[protocol_name]()
        _protocol_obj = protocol
    except Exception as e:
        _running = False
        _latest_frame = {"type": "error", "message": str(e)}
        _broadcast_threadsafe(_latest_frame)
        return

    for ep in range(n_episodes):
        if _stop_event.is_set(): break
        _status["episode"] = ep
        try:
            env.clear_stimuli()
            protocol.setup(env, brain)
            for stim in protocol.get_stimuli(): env.inject(stim)
        except Exception: pass
        try:
            world_state = organism.reset(seed=seed + ep)
        except Exception as e:
            _broadcast_threadsafe({"type": "error", "message": f"Reset: {e}"})
            break
        for t in range(protocol.episode_length):
            if _stop_event.is_set(): break
            _status["step"] = t
            try:
                motor, hier_state = organism.step()
                world_state = env.current_state
            except Exception as e:
                _broadcast_threadsafe({"type": "error", "message": f"Step: {e}"})
                break
            frame = _build_frame(world_state, hier_state, species, protocol_name, ep)
            _latest_frame = frame
            _broadcast_threadsafe(frame)
            try:
                if protocol.is_done(world_state, t): break
            except Exception: pass
            time.sleep(0.01)

    _running = False
    _status["species"] = None
    done = {"type": "done", "species": species, "protocol": protocol_name,
            "n_episodes": n_episodes}
    _latest_frame = done
    _broadcast_threadsafe(done)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    p = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else HTMLResponse(
        "<h1>index.html not found in vzlab/web/static/</h1>")


@app.get("/api/species")
async def get_species():
    try:
        list_species, _, _, _ = _load_vzlab()
        return {"species": list_species()}
    except Exception as e:
        return {"species": [], "error": str(e)}


@app.get("/api/protocols")
async def get_protocols():
    try:
        _, PROTOCOLS, _, _ = _load_vzlab()
        return {name: {"episode_length": cls().episode_length}
                for name, cls in PROTOCOLS.items()}
    except Exception as e:
        return {"error": str(e)}


class StartRequest(BaseModel):
    species: str = "c_elegans"
    protocol: str = "chemotaxis"
    n_episodes: int = 5
    seed: int = 0


@app.post("/api/start")
async def start_experiment(req: StartRequest):
    global _running, _stop_event, _status
    if _running:
        return {"status": "error", "message": "Already running"}
    try:
        _, PROTOCOLS, _, _ = _load_vzlab()
        if req.protocol not in PROTOCOLS:
            return {"status": "error", "message": f"Unknown protocol '{req.protocol}'"}
    except Exception as e:
        return {"status": "error", "message": f"vzlab import failed: {e}"}
    _stop_event.clear()
    _running = True
    _status.update({"species": req.species, "protocol": req.protocol,
                     "episode": 0, "step": 0})
    threading.Thread(target=_run_experiment,
                     args=(req.species, req.protocol, req.n_episodes, req.seed),
                     daemon=True).start()
    return {"status": "started", "species": req.species,
            "protocol": req.protocol, "n_episodes": req.n_episodes}


@app.post("/api/stop")
async def stop_experiment():
    global _running
    _stop_event.set()
    _running = False
    return {"status": "stopped"}


class AblationRequest(BaseModel):
    region: str
    enabled: bool = False   # False = ablate, True = restore


@app.post("/api/ablate")
async def ablate_region(req: AblationRequest):
    if _organism is None:
        return {"status": "error", "message": "No organism loaded"}
    try:
        (_organism.restore if req.enabled else _organism.ablate)(req.region)
        return {"status": "ok", "region": req.region, "enabled": req.enabled}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class ParamRequest(BaseModel):
    path: str
    value: float


@app.post("/api/param")
async def set_param(req: ParamRequest):
    if _organism is None:
        return {"status": "error", "message": "No organism loaded"}
    try:
        _organism.set_param(req.path, req.value)
        return {"status": "ok", "path": req.path, "value": req.value}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/status")
async def get_status():
    return {"running": _running, "species": _status.get("species"),
            "protocol": _status.get("protocol"),
            "episode": _status.get("episode", 0),
            "step": _status.get("step", 0)}


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            if _latest_frame:
                await ws.send_json(_latest_frame)
            await ws.send_json({"type": "status", "running": _running,
                                 "species": _status.get("species"),
                                 "protocol": _status.get("protocol"),
                                 "episode": _status.get("episode", 0),
                                 "step": _status.get("step", 0)})
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, Exception):
        _ws_clients.discard(ws)


def main():
    print("vzlab dashboard: http://localhost:8766")
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="warning")


if __name__ == "__main__":
    main()
