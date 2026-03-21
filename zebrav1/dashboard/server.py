"""
Dashboard Server — FastAPI + WebSocket for zebrafish brain monitoring.

Runs the simulation in a background thread, streams diagnostics to
connected WebSocket clients, and exposes REST API for control.

Run: python -m zebrav1.dashboard.server
Open: http://localhost:8000
"""
import os
import sys
import json
import math
import time
import asyncio
import threading
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="vzebra Dashboard")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Global simulation state
sim_state = {
    "running": False,
    "paused": False,
    "step": 0,
    "max_steps": 1000,
    "speed": 1.0,  # steps per update
    "diagnostics": {},
    "env_config": {
        "n_food": 15,
        "pred_speed": 2.7,
        "fish_speed_base": 3.0,
        "max_steps": 1000,
    },
    "training": {
        "hebbian_active": True,
        "stdp_active": True,
        "learning_rate": 0.005,
    },
    "modules": {},
    "agent": None,
    "env": None,
}

# Connected WebSocket clients
clients = set()


def get_module_info():
    """Extract module hierarchy and status from the brain agent."""
    agent = sim_state["agent"]
    if agent is None:
        return []

    modules = []

    # SNN layers
    snn_layers = [
        ("Retina L/R", "sensory", 1600, True),
        ("OT_L/OT_R", "sensory", 1200, True),
        ("OT_F (fused)", "sensory", 800, True),
        ("PT_L (pretectum)", "processing", 400, True),
        ("PC_per", "processing", 120, True),
        ("PC_int", "processing", 30, True),
        ("Motor (200)", "motor", 200, True),
        ("Eye (100)", "motor", 100, True),
        ("DA (50)", "neuromod", 50, True),
    ]
    for name, category, neurons, active in snn_layers:
        modules.append({
            "name": name, "category": category,
            "neurons": neurons, "active": active, "type": "snn"
        })

    # Other modules
    other = [
        ("Classifier", "perception", 133, True),
        ("Attention", "attention", 8, True),
        ("Spinal CPG", "motor", 32, True),
        ("Goal Policy", "decision", 4, True),
        ("Working Memory", "memory", 30, True),
        ("Dopamine", "neuromod", 10, agent.dopa_sys is not None),
        ("Basal Ganglia", "decision", 30 if sim_state.get("use_spiking") else 1, True),
        ("Amygdala", "emotion", 15 if sim_state.get("use_spiking") else 1,
         agent.amygdala is not None),
        ("Allostasis", "interoception", 3, agent.allostasis is not None),
        ("Insula", "emotion", 5, agent.insula is not None),
        ("Habenula", "emotion", 5, agent.habenula is not None),
        ("Predator Model", "world_model", 5, agent.predator_model is not None),
        ("Geographic Model", "world_model", 1200,
         agent.geographic_model is not None),
        ("Internal State", "world_model", 10,
         agent.interoceptive is not None),
        ("Lateral Line", "sensory", 16, agent.lateral_line is not None),
        ("Cerebellum", "motor", 44, agent.cerebellum is not None),
        ("Olfaction", "sensory", 10, agent.olfaction is not None),
        ("Vestibular", "sensory", 5, agent.vestibular is not None),
        ("Color Vision", "sensory", 20, agent.color_vision is not None),
        ("Circadian", "neuromod", 1, agent.circadian is not None),
        ("Proprioception", "sensory", 5, agent.proprioception is not None),
    ]
    for name, category, neurons, active in other:
        modules.append({
            "name": name, "category": category,
            "neurons": neurons, "active": active, "type": "module"
        })

    return modules


def run_simulation():
    """Background thread: run simulation loop."""
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav1.gym_env.brain_agent import BrainAgent

    cfg = sim_state["env_config"]
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=cfg["n_food"],
        max_steps=cfg["max_steps"], side_panels=False)
    agent = BrainAgent(use_allostasis=True)

    # Auto-load checkpoint
    ckpt = "zebrav1/weights/brain_checkpoint.pt"
    if os.path.exists(ckpt):
        try:
            agent.load_checkpoint(ckpt)
        except Exception:
            pass

    sim_state["agent"] = agent
    sim_state["env"] = env
    sim_state["modules"] = get_module_info()

    obs, info = env.reset(seed=42)
    agent.reset()
    sim_state["running"] = True
    sim_state["step"] = 0

    while sim_state["running"] and sim_state["step"] < cfg["max_steps"]:
        if sim_state["paused"]:
            time.sleep(0.05)
            continue

        action = agent.act(obs, env)
        obs, rew, term, trunc, info = env.step(action)
        agent.update_post_step(info, reward=rew, done=term, env=env)
        sim_state["step"] += 1

        # Collect diagnostics
        d = agent.last_diagnostics
        diag = {
            "step": sim_state["step"],
            "goal": d.get("goal", 2),
            "goal_name": ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"][d.get("goal", 2)],
            "energy": env.fish_energy,
            "fish_x": env.fish_x,
            "fish_y": env.fish_y,
            "pred_x": env.pred_x,
            "pred_y": env.pred_y,
            "pred_state": env.pred_state,
            "food_count": len(env.foods),
            "total_eaten": env.total_eaten,
            "cls_probs": [float(x) for x in d.get("cls_probs", [0]*5)],
            "dopa": d.get("dopa", 0.5),
            "rpe": d.get("rpe", 0.0),
            "heart_rate": d.get("heart_rate", 0.3),
            "alive": env.alive,
            "speed": d.get("speed", 0.0),
            "turn_rate": d.get("turn_rate", 0.0),
        }

        # Insula
        ins = d.get("insula", {})
        diag["arousal"] = ins.get("arousal", 0)
        diag["fear"] = ins.get("fear", 0)
        diag["valence"] = ins.get("valence", 0)

        # Amygdala
        amyg = d.get("amygdala", {})
        diag["threat_arousal"] = amyg.get("threat_arousal", 0)
        diag["fear_baseline"] = amyg.get("fear_baseline", 0)

        # SNN layer activity
        out = getattr(agent, '_last_snn_out', {})
        layer_rms = {}
        for key in ['oF', 'pt', 'per', 'intent', 'motor', 'eye', 'DA']:
            if key in out:
                v = out[key].detach().cpu().numpy()
                layer_rms[key] = float(np.sqrt(np.mean(v**2)))
        diag["layer_rms"] = layer_rms

        sim_state["diagnostics"] = diag

        if term:
            # Reset on death
            obs, info = env.reset()
            agent.reset()
            sim_state["step"] = 0

        # Rate control
        time.sleep(max(0.01, 0.05 / max(0.1, sim_state["speed"])))

    env.close()
    sim_state["running"] = False


# --- REST API ---

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>vzebra Dashboard</h1><p>Place index.html in zebrav1/dashboard/static/</p>")


@app.get("/api/status")
async def get_status():
    return {
        "running": sim_state["running"],
        "paused": sim_state["paused"],
        "step": sim_state["step"],
        "speed": sim_state["speed"],
    }


@app.get("/api/modules")
async def get_modules():
    return sim_state.get("modules", [])


@app.get("/api/diagnostics")
async def get_diagnostics():
    return sim_state.get("diagnostics", {})


@app.post("/api/start")
async def start_sim():
    if not sim_state["running"]:
        t = threading.Thread(target=run_simulation, daemon=True)
        t.start()
    return {"status": "started"}


@app.post("/api/stop")
async def stop_sim():
    sim_state["running"] = False
    return {"status": "stopped"}


@app.post("/api/pause")
async def toggle_pause():
    sim_state["paused"] = not sim_state["paused"]
    return {"paused": sim_state["paused"]}


@app.post("/api/speed/{value}")
async def set_speed(value: float):
    sim_state["speed"] = max(0.1, min(10.0, value))
    return {"speed": sim_state["speed"]}


@app.post("/api/config")
async def update_config(config: dict):
    sim_state["env_config"].update(config)
    return sim_state["env_config"]


@app.post("/api/save_checkpoint")
async def save_checkpoint():
    agent = sim_state.get("agent")
    if agent:
        agent.save_checkpoint("zebrav1/weights/brain_checkpoint.pt")
        return {"status": "saved"}
    return {"status": "no agent"}


# --- WebSocket for real-time streaming ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # Stream diagnostics at ~20 fps
            diag = sim_state.get("diagnostics", {})
            if diag:
                await websocket.send_json(diag)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    print("Starting vzebra Dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
