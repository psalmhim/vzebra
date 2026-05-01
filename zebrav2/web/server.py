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
import math
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Lazy imports — avoid loading torch at module level for test compatibility
_TrainingEngine = None
_TrainingConfig = None
_REPERTOIRES = None

def _lazy_engine_imports():
    global _TrainingEngine, _TrainingConfig, _REPERTOIRES
    if _TrainingEngine is None:
        from zebrav2.engine.trainer import TrainingEngine as _TE
        from zebrav2.engine.config import TrainingConfig as _TC, REPERTOIRES as _R
        _TrainingEngine, _TrainingConfig, _REPERTOIRES = _TE, _TC, _R

from contextlib import asynccontextmanager

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

# Global engine (lazy init)
engine = None
ws_clients = set()
_main_loop = None  # store uvicorn's event loop
_latest_step = {}  # latest step data for polling
_demo_t = 0        # idle demo timer
_demo_food_eaten = 0  # cumulative food eaten count

# ── Tunable circuit / environment parameters (changed via /api/demo_params) ──
_demo_params = {
    # Circuit gains
    'amygdala_gain':    2.0,   # threat→amygdala multiplier     (0.5–4.0)
    'da_gain':          3.0,   # RPE→DA sigmoid steepness       (0.5–6.0)
    'sht_gain':         1.0,   # 5-HT accumulation rate mult    (0.2–3.0)
    'bg_bias':          0.3,   # BG gate offset (higher→easier to move) (-0.5–1.0)
    'flee_weight':      1.0,   # flee goal weight multiplier    (0.2–3.0)
    # Locomotion
    'fish_speed_mult':  1.0,   # global fish speed scale        (0.3–2.0)
    # Environment
    'energy_drain':     0.012, # energy cost per sub-step       (0.002–0.04)
    'food_radius':      25.0,  # eat distance in px             (10–60)
    'food_value':       5.0,   # energy gained per food item    (1–20)
    'pred_speed':       2.8,   # predator chase speed px/step   (1.0–6.0)
    'pred_range':       220.0, # predator detection range px    (80–400)
}


@asynccontextmanager
async def lifespan(app):
    global _main_loop, engine
    _main_loop = asyncio.get_event_loop()
    _lazy_engine_imports()
    engine = _TrainingEngine()
    yield


app = FastAPI(title="Zebrafish Brain v2 Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
        try:
            _main_loop.call_soon_threadsafe(
                _main_loop.create_task,
                broadcast({'type': 'step', 'data': step_data}))
        except RuntimeError:
            pass  # loop closed


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
    status = engine.get_status()
    # Merge latest step data from callback
    if _latest_step:
        status['current_step'] = _latest_step
    return status


@app.get("/api/checkpoints")
async def get_checkpoints():
    return engine.checkpoint_mgr.list_checkpoints()


@app.get("/api/replays")
async def list_replays():
    return [{'round': r['round'], 'n_steps': len(r['steps']),
             'metrics': r.get('metrics', {})} for r in engine.saved_replays]


@app.get("/api/replay/{round_num}")
async def get_replay(round_num: int):
    for r in engine.saved_replays:
        if r['round'] == round_num:
            return r
    return {"status": "error", "message": "Replay not found"}


class AblationRequest(BaseModel):
    region: str
    enabled: bool


@app.post("/api/ablate")
async def ablate_region(req: AblationRequest):
    if engine.brain is None:
        return {"status": "error", "message": "No brain loaded — start training first"}
    engine.brain.set_region_enabled(req.region, req.enabled)
    return {"status": "ok", "region": req.region, "enabled": req.enabled,
            "ablated": list(engine.brain._ablated)}


class DisorderRequest(BaseModel):
    disorder: str
    intensity: float = 1.0


@app.post("/api/disorder")
async def apply_disorder_api(req: DisorderRequest):
    if engine.brain is None:
        return {"status": "error", "message": "No brain loaded — start training first"}
    try:
        from zebrav2.brain.disorder import apply_disorder, DISORDER_DESCRIPTIONS
        changes = apply_disorder(engine.brain, req.disorder, intensity=req.intensity)
        return {
            "status": "ok",
            "disorder": req.disorder,
            "intensity": req.intensity,
            "changes": {k: [float(v[0]), float(v[1])] for k, v in changes.items()},
            "description": DISORDER_DESCRIPTIONS.get(req.disorder, ''),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/disorders")
async def list_disorders_api():
    try:
        from zebrav2.brain.disorder import DISORDER_DESCRIPTIONS
        return {"disorders": DISORDER_DESCRIPTIONS}
    except ImportError:
        return {"disorders": {}}


class ConfigUpdate(BaseModel):
    config: dict


@app.post("/api/config")
async def update_config(body: ConfigUpdate):
    _lazy_engine_imports()
    engine.config = _TrainingConfig(body.config)
    return {"status": "ok", "config": engine.config.data}


class StartRequest(BaseModel):
    n_rounds: Optional[int] = 10
    config: Optional[dict] = None


@app.post("/api/start")
async def start_training(req: StartRequest):
    if engine.running:
        return {"status": "error", "message": "Training already running"}
    if req.config:
        _lazy_engine_imports()
        engine.config = _TrainingConfig(req.config)
    engine.on_step = on_step_callback
    engine.on_round_end = on_round_end_callback
    engine.train_async(n_rounds=req.n_rounds)
    return {"status": "started", "n_rounds": req.n_rounds}


@app.post("/api/start-multi")
async def start_multi(req: StartRequest):
    if engine.running:
        return {"status": "error", "message": "Training already running"}
    if req.config:
        _lazy_engine_imports()
        engine.config = _TrainingConfig(req.config)
    engine.on_step = on_step_callback
    engine.on_round_end = on_round_end_callback
    n_rounds = req.n_rounds or 10
    engine.train_async_multi(n_rounds=n_rounds, n_fish=5)
    return {"status": "started", "mode": "multi-agent", "n_rounds": n_rounds, "n_fish": 5}


@app.post("/api/stop")
async def stop_training():
    engine.stop()
    return {"status": "stopped"}


@app.get("/api/repertoires")
async def get_repertoires():
    _lazy_engine_imports()
    return {k: {'name': v['name'], 'description': v['description']} for k, v in _REPERTOIRES.items()}


@app.post("/api/repertoire/{name}")
async def start_repertoire(name: str):
    _lazy_engine_imports()
    if name not in _REPERTOIRES:
        return {"status": "error", "message": f"Unknown repertoire: {name}"}
    if engine.running:
        return {"status": "error", "message": "Training already running"}
    rep = _REPERTOIRES[name]
    config = _TrainingConfig(rep)
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
def _make_rock_poly(cx, cy, r, n_verts=8, seed=0):
    """Generate irregular polygon vertices for a rock."""
    import random as _rnd
    _rnd.seed(seed)
    verts = []
    for i in range(n_verts):
        a = 2 * math.pi * i / n_verts + _rnd.uniform(-0.3, 0.3)
        rr = r * _rnd.uniform(0.7, 1.3)
        verts.append([round(cx + rr * math.cos(a), 1), round(cy + rr * math.sin(a), 1)])
    return verts


# Rock clusters with polygon shapes — each rock shelters a food cluster nearby
_demo_rock_defs = [
    # [cx, cy, radius, seed] — food spawns near each rock
    [150, 140, 40, 10], [160, 200, 25, 11],   # cluster A (top-left)
    [600, 420, 45, 20], [550, 460, 30, 21],   # cluster B (bottom-right)
    [380, 100, 35, 30],                         # cluster C (top-center)
    [680, 160, 38, 40], [720, 200, 22, 41],   # cluster D (right)
    [250, 480, 42, 50], [200, 520, 28, 51],   # cluster E (bottom-left)
    [500, 300, 30, 60],                         # cluster F (center)
]
_demo_rocks = []
for _rd in _demo_rock_defs:
    _demo_rocks.append([_rd[0], _rd[1], _rd[2],
                        _make_rock_poly(_rd[0], _rd[1], _rd[2], seed=_rd[3])])

# Food clusters: sparse groups near rocks (3-5 food per cluster, within 50px of rock)
import random as _init_rnd
_init_rnd.seed(99)


def _is_inside_rock(x, y, margin=5):
    """Check if point is inside any rock (with margin)."""
    for rd in _demo_rock_defs:
        dx, dy = x - rd[0], y - rd[1]
        if math.sqrt(dx * dx + dy * dy) < rd[2] + margin:
            return True
    return False


_demo_foods_init = []
_food_clusters = [
    (150, 170, 4), (600, 440, 5), (380, 130, 3),
    (680, 180, 4), (250, 500, 4), (500, 320, 3),
]
for _fcx, _fcy, _fn in _food_clusters:
    for _ in range(_fn):
        # Retry until food lands outside all rocks
        for _try in range(50):
            _fx = round(_fcx + _init_rnd.uniform(-45, 45))
            _fy = round(_fcy + _init_rnd.uniform(-45, 45))
            if not _is_inside_rock(_fx, _fy):
                break
        _demo_foods_init.append([_fx, _fy])
_demo_foods = list(_demo_foods_init)

# Demo state: fish + predators with position/heading/energy
_demo_fish = {
    'x': 400.0, 'y': 300.0, 'h': 0.0, 'energy': 80.0,
    'target_food': None, 'goal': 'FORAGE', 'speed': 1.8,
}
_demo_conspecifics = [
    {'x': 350.0, 'y': 280.0, 'h': 0.5, 'energy': 65.0, 'goal': 'FORAGE'},
    {'x': 420.0, 'y': 350.0, 'h': -0.3, 'energy': 60.0, 'goal': 'SOCIAL'},
    {'x': 300.0, 'y': 320.0, 'h': 1.0, 'energy': 70.0, 'goal': 'FORAGE'},
    {'x': 450.0, 'y': 260.0, 'h': -1.2, 'energy': 55.0, 'goal': 'EXPLORE'},
]
_demo_predators = [
    {'x': 700.0, 'y': 100.0, 'h': math.pi, 'energy': 80.0,
     'state': 'patrol', 'patrol_cx': 650, 'patrol_cy': 150, 'patrol_r': 100},
    {'x': 100.0, 'y': 500.0, 'h': 0.0, 'energy': 75.0,
     'state': 'patrol', 'patrol_cx': 150, 'patrol_cy': 450, 'patrol_r': 120},
]


def _collide_rocks(x, y, margin=8):
    """Push entity out of any rock. Returns corrected (x, y)."""
    for rd in _demo_rock_defs:
        rcx, rcy, rr = rd[0], rd[1], rd[2]
        dx, dy = x - rcx, y - rcy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < rr + margin:
            if dist < 1:
                dx, dy = 1, 0
                dist = 1
            push = (rr + margin) / dist
            x = rcx + dx * push
            y = rcy + dy * push
    return x, y


_round_counter = 0

def _reset_round():
    """Reset the entire round: respawn fish, conspecifics, predators, food.
    Each round uses a different random seed for varied layouts."""
    global _demo_t, _demo_foods, _demo_fish, _demo_conspecifics, _demo_predators, _neural, _round_counter, _demo_food_eaten
    _demo_food_eaten = 0
    _round_counter += 1
    rnd = _init_rnd
    rnd.seed(_round_counter * 1000 + int(_demo_t) % 10000)

    _demo_t = 2000  # start in daytime

    # Randomize food positions near clusters
    _demo_foods = []
    for _fcx, _fcy, _fn in _food_clusters:
        for _ in range(_fn):
            for _try in range(50):
                fx = round(_fcx + rnd.uniform(-45, 45))
                fy = round(_fcy + rnd.uniform(-45, 45))
                if not _is_inside_rock(fx, fy):
                    break
            _demo_foods.append([fx, fy])

    # Randomize fish spawn (center area, avoid rocks)
    def _safe_spawn(cx, cy, spread=80):
        for _ in range(50):
            x = cx + rnd.uniform(-spread, spread)
            y = cy + rnd.uniform(-spread, spread)
            x = max(40, min(760, x))
            y = max(40, min(560, y))
            if not _is_inside_rock(x, y, margin=15):
                return x, y
        return cx, cy

    fx, fy = _safe_spawn(400, 300, 150)
    _demo_fish.update({
        'x': fx, 'y': fy, 'h': rnd.uniform(-math.pi, math.pi),
        'energy': 80.0, 'target_food': None, 'goal': 'FORAGE', 'speed': 1.8,
    })

    _demo_conspecifics[:] = []
    for _ in range(4):
        cx, cy = _safe_spawn(400, 300, 200)
        _demo_conspecifics.append({
            'x': cx, 'y': cy, 'h': rnd.uniform(-math.pi, math.pi),
            'energy': rnd.uniform(55, 75), 'goal': 'FORAGE',
        })

    # Randomize predator patrol centers (avoid center where fish spawns)
    pred_zones = [
        (150, 150), (650, 150), (150, 450), (650, 450), (400, 500),
        (250, 100), (550, 500),
    ]
    rnd.shuffle(pred_zones)
    _pred_seq = [1, 2, 1, 2, 2, 1, 2]
    n_predators = _pred_seq[(_round_counter - 1) % len(_pred_seq)]
    _demo_predators[:] = []
    for i in range(n_predators):
        pcx, pcy = pred_zones[i]
        px, py = _safe_spawn(pcx, pcy, 60)
        _demo_predators.append({
            'x': px, 'y': py, 'h': rnd.uniform(-math.pi, math.pi),
            'energy': rnd.uniform(70, 90),
            'state': 'patrol',
            'patrol_cx': pcx, 'patrol_cy': pcy,
            'patrol_r': rnd.uniform(70, 130),
        })
    # Reset neural state
    for k, v in {
        'DA': 0.5, 'NA': 0.3, '5HT': 0.6, 'ACh': 0.7,
        'amygdala_trace': 0.0, 'V_prev': 0.0, 'cb_pred': [0.0, 0.0],
        'place_fam': 0.0, 'goal_lock': 0, 'locked_goal': None,
        'cstart_timer': 0, 'cstart_dir': 0.0, 'sht_acc': 0.0,
        'energy_prev': 80.0, 'energy_rate': 0.0,
        'steps_since_food': 0, 'starvation_anxiety': 0.0,
        'food_memory_xy': None, 'food_memory_age': 999,
        'dead': False, 'death_timer': 0, 'death_x': 0, 'death_y': 0,
        'prev_retina_L': 0.0, 'prev_retina_R': 0.0,
        'orient_dir': 0.0, 'orient_habituation': 0.0,
    }.items():
        _neural[k] = v
    _neural['frustration'] = [0.0, 0.0, 0.0, 0.0]


# ── Real-brain demo (background thread) ──────────────────────────────────────
_brain_demo_active = False
_brain_demo_thread = None
_brain_latest_step: dict = {}
_brain_demo_epochs = 0       # episodes completed during this demo session
_brain_demo_saves = 0        # how many epoch-checkpoints saved
_GOAL_NAMES_V2 = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']
EPOCH_SIZE = 10              # save checkpoint every N episodes


def _run_brain_demo():
    """Background thread: ZebrafishBrainV2 + gym env, updates _brain_latest_step."""
    global _brain_latest_step, _brain_demo_active
    import numpy as np, torch
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav2.brain.sensory_bridge import inject_sensory

    ckpt_dir = os.path.join(PROJECT_ROOT, 'zebrav2/checkpoints')
    CKPT = os.path.join(ckpt_dir, 'ckpt_latest.pt')
    if not os.path.exists(CKPT):
        pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]) if os.path.isdir(ckpt_dir) else []
        CKPT = os.path.join(ckpt_dir, pts[-1]) if pts else ''

    brain = ZebrafishBrainV2(device='cpu')
    if CKPT and os.path.exists(CKPT):
        state = torch.load(CKPT, map_location='cpu', weights_only=False)
        brain.load_state_dict(state.get('brain', state), strict=False)
        print(f'[brain-demo] loaded {CKPT}')
    else:
        print('[brain-demo] no checkpoint found — using random weights')

    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=3000)
    obs, info = env.reset()
    brain.reset()
    step = total_eaten = 0
    global _brain_demo_epochs, _brain_demo_saves

    def _save_epoch_ckpt():
        global _brain_demo_saves
        try:
            latest = os.path.join(ckpt_dir, 'ckpt_latest.pt')
            state = {
                'brain': brain.state_dict() if hasattr(brain, 'state_dict') else {},
                'critic_state': brain.critic.state_dict(),
                'classifier_state': brain.classifier.state_dict(),
                'pallium_state': brain.pallium.state_dict(),
                'habit_state': brain.habit.state_dict(),
                'amygdala_W_la_cea': brain.amygdala.W_la_cea.cpu().numpy().tolist(),
                'amygdala_fear_baseline': brain.amygdala.fear_baseline,
                'cerebellum_W_pf': brain.cerebellum.W_pf.data.cpu().numpy().tolist(),
                'web_epochs': _brain_demo_epochs,
                'web_saves': _brain_demo_saves + 1,
            }
            import torch as _torch
            _torch.save(state, latest)
            _brain_demo_saves += 1
            print(f'[brain-demo] epoch checkpoint saved (epoch {_brain_demo_epochs}, save #{_brain_demo_saves})')
        except Exception as e:
            print(f'[brain-demo] save failed: {e}')

    while _brain_demo_active:
        try:
            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            ate = getattr(env, 'food_eaten_this_step', info.get('food_eaten_this_step', 0))
            total_eaten += ate
            step += 1

            spikes = {}
            try:
                t = brain.tectum
                spikes = {
                    'sfgs_b':     float((t.sfgs_b_L.spike_E.mean() + t.sfgs_b_R.spike_E.mean()) * 0.5),
                    'sgc':        float((t.sgc_L.spike_E.mean()    + t.sgc_R.spike_E.mean())    * 0.5),
                    'amygdala':   float(brain.amygdala_alpha),
                    'd1':         float(brain.bg.d1.spike_E.mean()),
                    'd2':         float(brain.bg.d2.spike_E.mean()),
                    'critic':     float(out.get('critic_value', 0)),
                    'habenula':   float(out.get('habenula_disappoint', 0)),
                    'cerebellum': float(out.get('cerebellum_pe', 0)),
                }
            except Exception:
                pass

            rocks = [[float(r[0]), float(r[1]), float(r[2])]
                     for r in getattr(env, 'rock_formations', [])]

            _brain_latest_step = {
                'fish_x': float(env.fish_x),
                'fish_y': float(env.fish_y),
                'fish_heading': float(env.fish_heading),
                'goal': _GOAL_NAMES_V2[brain.current_goal],
                'energy': float(brain.energy),
                'step': step,
                'turn': float(out.get('turn', 0)),
                'speed': float(out.get('speed', 0)),
                'DA':  float(out.get('DA',  brain.neuromod.DA.item())),
                'NA':  float(out.get('NA',  brain.neuromod.NA.item())),
                '5HT': float(out.get('5HT', brain.neuromod.HT5.item())),
                'ACh': float(out.get('ACh', brain.neuromod.ACh.item())),
                'amygdala': float(brain.amygdala_alpha),
                'free_energy': float(out.get('free_energy', 0)),
                'critic_value': float(out.get('critic_value', 0)),
                'heart_rate': float(out.get('heart_rate', 2.0)),
                'novelty': float(out.get('novelty', 0)),
                'valence': float(out.get('valence', 0)),
                'surprise': float(out.get('surprise', 0)),
                'hunger': float(getattr(brain, 'allostasis', None) and brain.allostasis.hunger or 0),
                'stress': float(getattr(brain, 'allostasis', None) and brain.allostasis.stress or 0),
                'foods': [[float(f[0]), float(f[1])] for f in env.foods],
                'rocks': rocks,
                'pred_x': float(env.pred_x),
                'pred_y': float(env.pred_y),
                'pred_heading': float(env.pred_heading),
                'pred_state': env.pred_state.lower(),
                'pred_energy': float(1.0 - getattr(env, 'pred_hunger', 0)),
                'arena_w': float(env.arena_w),
                'arena_h': float(env.arena_h),
                'food_total': total_eaten,
                'ate_food': bool(ate),
                'light_level': 1.0,
                'spikes': spikes,
                'is_brain_demo': True,
            }

            if terminated or truncated:
                _brain_demo_epochs += 1
                if _brain_demo_epochs % EPOCH_SIZE == 0:
                    _save_epoch_ckpt()
                obs, info = env.reset()
                brain.reset()
                total_eaten = step = 0

        except Exception:
            import traceback; traceback.print_exc()
            time.sleep(0.5)


def _steer_away_from_rocks(x, y, h, lookahead=40):
    """Steer heading away from nearby rocks before moving. Returns adjusted heading."""
    for rd in _demo_rock_defs:
        rcx, rcy, rr = rd[0], rd[1], rd[2]
        # Check if moving toward this rock
        fx = x + math.cos(h) * lookahead
        fy = y + math.sin(h) * lookahead
        dx, dy = fx - rcx, fy - rcy
        if dx * dx + dy * dy < (rr + 15) ** 2:
            # Steer perpendicular to rock (away from center)
            away = math.atan2(y - rcy, x - rcx)
            diff = math.atan2(math.sin(away - h), math.cos(away - h))
            h += diff * 0.3
    return h


# --- Circadian clock ---
_CIRCADIAN_PERIOD = 6000  # steps per full day/night cycle
_DAWN = 0.2    # fraction: night→dawn
_DUSK = 0.75   # fraction: day→dusk


def _circadian_phase(step):
    """Returns (phase 0-1, is_day bool, light_level 0-1, label)."""
    phase = (step % _CIRCADIAN_PERIOD) / _CIRCADIAN_PERIOD
    if phase < _DAWN:
        # Night → dawn transition
        light = 0.1 + 0.9 * (phase / _DAWN)
        return phase, False, light, 'NIGHT'
    elif phase < 0.35:
        # Dawn
        return phase, True, 0.8 + 0.2 * ((phase - _DAWN) / 0.15), 'DAWN'
    elif phase < _DUSK:
        # Day
        return phase, True, 1.0, 'DAY'
    elif phase < 0.9:
        # Dusk
        light = 1.0 - 0.8 * ((phase - _DUSK) / 0.15)
        return phase, False, light, 'DUSK'
    else:
        # Night
        return phase, False, 0.1, 'NIGHT'


def _steer_toward(cur_h, target_h, rate=0.08):
    diff = math.atan2(math.sin(target_h - cur_h), math.cos(target_h - cur_h))
    return cur_h + diff * rate


def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ──────────────────────────────────────────────────────────────
# Neurobiological processing pipeline — rate-coded simplified
# model that mirrors BrainV2 signal flow (brain_v2.py).
#
# Signal flow (matches real brain):
#   1. Sensory input:  Retina L/R, lateral line, olfaction
#   2. Optic chiasm:   L eye → R tectum, R eye → L tectum
#   3. Tectum:         SFGS-b (objects), SFGS-d (motion),
#                      SGC (looming/threat), SO (direction)
#   4. Thalamus:       TC relay (gated by TRN + NA arousal)
#   5. Pallium:        Pal-S (sensory repr) → Pal-D (decision)
#   6. Amygdala:       Fear assessment (LA→CeA, episodic)
#   7. Neuromod:       DA (RPE), NA (arousal), 5-HT (patience),
#                      ACh (circadian attention)
#   8. EFE goal:       G_forage, G_flee, G_explore, G_social,
#                      G_sleep → winner-take-all
#   9. Basal ganglia:  D1 (go) / D2 (no-go) gate
#  10. Motor output:   Reticulospinal → CPG → heading/speed
#  11. Homeostasis:    Insula (energy), allostasis, circadian
# ──────────────────────────────────────────────────────────────

# Persistent neural state across steps (simulates membrane/synaptic state)
_neural = {
    # Neuromodulators (slow dynamics, τ ~ 20-100 steps)
    'DA': 0.5, 'NA': 0.3, '5HT': 0.6, 'ACh': 0.7,
    # Amygdala fear trace (episodic conditioning)
    'amygdala_trace': 0.0,
    # Habenula frustration accumulator (per-goal)
    'frustration': [0.0, 0.0, 0.0, 0.0],  # forage, flee, explore, social
    # Critic expected value (for RPE computation)
    'V_prev': 0.0,
    # Cerebellum forward model prediction error
    'cb_pred': [0.0, 0.0],  # predicted [turn, speed]
    # Place cell familiarity (decays)
    'place_fam': 0.0,
    # Goal lock timer (basal ganglia attractor persistence)
    'goal_lock': 0, 'locked_goal': None,
    # Mauthner C-start reflex state
    'cstart_timer': 0,
    'cstart_dir': 0.0,
    # 5-HT patience accumulator
    'sht_acc': 0.0,
    # Energy trajectory for starvation prediction
    'energy_prev': 80.0,         # previous step energy
    'energy_rate': 0.0,          # EMA of energy change per step
    'steps_since_food': 0,       # steps since last food eaten
    'starvation_anxiety': 0.0,   # predicted future starvation (0-1)
    # Place cell food memory — remembers last food location for anxious foraging
    'food_memory_xy': None,      # (x, y) of last eaten food
    'food_memory_age': 999,      # steps since food was eaten there
    # Death state
    'dead': False,               # fish killed by predator
    'death_timer': 0,            # countdown to new round (steps)
    'death_x': 0, 'death_y': 0, # position where fish died
    # Orienting response — tectal novelty-driven attention saccade
    'prev_retina_L': 0.0,       # previous-step retinal activation (left)
    'prev_retina_R': 0.0,       # previous-step retinal activation (right)
    'orient_dir': 0.0,          # current orienting turn signal (rad/step)
    'orient_habituation': 0.0,  # habituation (dampens repeated stimuli)
    # --- Visceral organ modules ---
    # Vagus nerve: parasympathetic tone (inversely related to stress)
    'vagal_tone': 1.0,
    # Pituitary–adrenal axis (HPA)
    'crh': 0.0,                 # hypothalamic CRH (stress proxy)
    'acth': 0.0,                # pituitary ACTH output
    'cortisol_drive': 0.0,      # adrenal cortisol drive
    # Area postrema: circumventricular chemosensing
    'glucose_status': 0.0,      # blood glucose proxy (−1 low … +1 high)
    'nausea': 0.0,              # toxin detection (always 0 in demo)
    # NTS: nucleus tractus solitarius — vagal afferent relay
    'satiety_signal': 0.0,      # accumulates with eating, decays over time
    'taste_relay': 0.0,         # gustatory salience from food proximity
    # Lateral line efferent: self-motion suppression (corollary discharge)
    'll_efferent_gain': 1.0,    # 1.0 = no suppression, 0 = full suppression
}


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))


def _ema(old, new, tau):
    """Exponential moving average with time constant tau."""
    a = 1.0 / max(1, tau)
    return old * (1 - a) + new * a


def _idle_demo_step():
    """Neurobiologically-grounded idle demo.

    Each brain region computes its activation from upstream inputs.
    Behavior (goal, heading, speed) EMERGES from the neural cascade,
    not from game-programming if/else logic.
    """
    global _demo_t, _demo_foods, _demo_fish, _demo_food_eaten
    _demo_t += 1
    t = _demo_t * 0.01
    f = _demo_fish
    n = _neural

    # ================================================================
    # DEATH STATE: fish was killed — freeze and show death screen
    # ================================================================
    # Auto-clear stale state if energy was externally restored (e.g., test/round reset)
    if n['dead'] and f['energy'] > 0:
        n['dead'] = False
        n['death_timer'] = 0
    if f['energy'] - n.get('energy_prev', f['energy']) > 10:
        # Large energy jump indicates external reset — clear accumulated state
        n['steps_since_food'] = 0
        n['starvation_anxiety'] = 0.0
        n['energy_rate'] = 0.0
        n['food_memory_xy'] = None
        n['food_memory_age'] = 999

    if n['dead']:
        n['death_timer'] -= 1
        if n['death_timer'] <= 0:
            # Timer expired — start new round
            _reset_round()
            n = _neural  # re-bind after reset
            f = _demo_fish
        else:
            # Still dead — return frozen state with kill flag
            remaining_sec = round(n['death_timer'] * 0.15, 1)
            p0 = _demo_predators[0] if _demo_predators else {'x': -100, 'y': -100, 'h': 0, 'energy': 0, 'state': 'patrol'}
            return {
                'fish_x': n['death_x'], 'fish_y': n['death_y'],
                'fish_heading': f['h'],
                'goal': 'DEAD', 'energy': 0,
                'step': _demo_t, 'heart_rate': 0,
                'turn': 0, 'speed': 0,
                'food_total': 0, 'ate_food': False,
                'arena_w': 800, 'arena_h': 600,
                'foods': list(_demo_foods),
                'rocks': list(_demo_rocks),
                'pred_x': p0['x'], 'pred_y': p0['y'],
                'pred_heading': p0['h'],
                'pred_energy': p0['energy'], 'pred_state': p0['state'],
                'extra_predators': [
                    {'x': p['x'], 'y': p['y'], 'heading': p['h'],
                     'energy': p['energy'], 'state': p['state']}
                    for p in _demo_predators[1:]
                ],
                'other_fish': [
                    {'x': c['x'], 'y': c['y'], 'heading': c['h'],
                     'goal': c['goal'], 'energy': c['energy'],
                     'alive': c.get('alive', True)}
                    for c in _demo_conspecifics
                ],
                'spikes': {},
                'DA': 0, 'NA': 0, '5HT': 0, 'ACh': 0,
                'light_level': 1.0, 'circ_label': '',
                'is_sleeping': False,
                'starvation_anxiety': 0,
                'left_eye_food': 0, 'right_eye_food': 0,
                'pred_in_left': False, 'pred_in_right': False,
                'killed': True,
                'death_countdown': remaining_sec,
            }

    # ================================================================
    # STAGE 0: Circadian & homeostatic context
    # ================================================================
    circ_phase, is_day, light, circ_label = _circadian_phase(_demo_t)
    # ACh modulated by circadian (low at night → reduced attention)
    n['ACh'] = _ema(n['ACh'], light * 0.8 + 0.2, tau=30)
    # Energy-derived hunger/stress signal (allostasis)
    energy_ratio = f['energy'] / 100.0
    starvation = max(0, (0.75 - energy_ratio) / 0.75)
    fatigue = max(0, (0.5 - energy_ratio) / 0.5)

    # --- Predictive starvation anxiety (EFE: expected future energy) ---
    # Track energy depletion rate (EMA of energy change per step)
    energy_delta = f['energy'] - n['energy_prev']
    n['energy_prev'] = f['energy']
    n['energy_rate'] = _ema(n['energy_rate'], energy_delta, tau=20)

    # Count steps since last food — longer gap → more anxiety
    n['steps_since_food'] += 1

    # Predicted energy in ~100 steps (look-ahead horizon)
    predicted_energy = f['energy'] + n['energy_rate'] * 100
    predicted_ratio = max(0, min(1, predicted_energy / 100))

    # Starvation anxiety: high when predicted future energy is low
    # Even if current energy is 70%, if depletion rate is fast and no food
    # in sight, anxiety rises — drives preemptive foraging
    food_scarcity = min(1.0, n['steps_since_food'] / 80)  # no food for 80 steps → max
    predicted_starvation = max(0, (0.6 - predicted_ratio) / 0.6)
    n['starvation_anxiety'] = _ema(
        n['starvation_anxiety'],
        max(predicted_starvation, food_scarcity * 0.5) * (1.0 - energy_ratio * 0.3),
        tau=10,
    )

    # ================================================================
    # STAGE 1: Sensory input — bilateral retina + lateral line + olfaction
    # ================================================================
    fov_half = math.pi * 100 / 180  # 100° per eye (200° total binocular)

    # --- Retina: scan environment objects per eye ---
    # Left eye sees left visual field (negative rel_angle)
    # Right eye sees right visual field (positive rel_angle)
    retina_L_food = 0.0   # food salience in left eye
    retina_R_food = 0.0
    retina_L_pred = 0.0   # predator salience in left eye
    retina_R_pred = 0.0
    retina_L_conspec = 0.0
    retina_R_conspec = 0.0
    left_eye_count = 0
    right_eye_count = 0

    visible_food = []
    for i, fd in enumerate(_demo_foods):
        dx, dy = fd[0] - f['x'], fd[1] - f['y']
        d = math.sqrt(dx * dx + dy * dy)
        if d > 250:
            continue
        angle_to = math.atan2(dy, dx)
        rel_angle = math.atan2(math.sin(angle_to - f['h']),
                               math.cos(angle_to - f['h']))
        if abs(rel_angle) < fov_half:
            salience = max(0, 1.0 - d / 250)  # distance-weighted
            visible_food.append((i, d, rel_angle, salience))
            if rel_angle < 0:
                retina_L_food += salience
                left_eye_count += 1
            else:
                retina_R_food += salience
                right_eye_count += 1

    # Predator detection (visual + lateral line below)
    visible_preds = []
    for pred in _demo_predators:
        dx, dy = pred['x'] - f['x'], pred['y'] - f['y']
        d = math.sqrt(dx * dx + dy * dy)
        if d > 300:
            continue
        angle_to = math.atan2(dy, dx)
        rel_angle = math.atan2(math.sin(angle_to - f['h']),
                               math.cos(angle_to - f['h']))
        # Visual detection within FoV
        in_fov = abs(rel_angle) < fov_half and d < 200
        salience = max(0, 1.0 - d / 200) if in_fov else 0
        # Looming signal: angular expansion rate (closer = faster expansion)
        looming = max(0, (150 - d) / 150) ** 2 if d < 150 else 0
        visible_preds.append({
            'ref': pred, 'd': d, 'angle': rel_angle,
            'salience': salience, 'looming': looming, 'in_fov': in_fov,
        })
        if in_fov:
            if rel_angle < 0:
                retina_L_pred += salience
            else:
                retina_R_pred += salience

    # Conspecific detection (visual)
    for cf in _demo_conspecifics:
        if not cf.get('alive', True):
            continue
        dx, dy = cf['x'] - f['x'], cf['y'] - f['y']
        d = math.sqrt(dx * dx + dy * dy)
        if d > 200:
            continue
        angle_to = math.atan2(dy, dx)
        rel_angle = math.atan2(math.sin(angle_to - f['h']),
                               math.cos(angle_to - f['h']))
        if abs(rel_angle) < fov_half:
            sal = max(0, 1.0 - d / 200) * 0.3
            if rel_angle < 0:
                retina_L_conspec += sal
            else:
                retina_R_conspec += sal

    # Aggregate retinal ganglion cell output (ON-sustained pathway)
    retina_L = min(5.0, 0.1 + retina_L_food + retina_L_pred * 2.0
                   + retina_L_conspec)
    retina_R = min(5.0, 0.1 + retina_R_food + retina_R_pred * 2.0
                   + retina_R_conspec)

    # --- Binocular depth estimation (frontal convergence zone) ---
    # Zebrafish ~20° frontal overlap per eye → objects at |angle| < 20°
    # appear in both retinae, enabling stereo disparity distance
    bino_half = math.pi * 20 / 180  # ±20° binocular zone
    bino_food_dist = 999.0
    bino_food_conf = 0.0
    bino_pred_dist = 999.0
    bino_pred_conf = 0.0
    for item in visible_food:
        _, d, rel_angle, sal = item
        if abs(rel_angle) < bino_half:
            bino_food_dist = min(bino_food_dist, d)
            bino_food_conf = max(bino_food_conf, sal)
    for vp in visible_preds:
        if vp['in_fov'] and abs(vp['angle']) < bino_half:
            bino_pred_dist = min(bino_pred_dist, vp['d'])
            bino_pred_conf = max(bino_pred_conf, vp['salience'])
    # Approach gain: slow down when food is close in binocular zone (prey capture)
    bino_approach_gain = 1.0
    if bino_food_dist < 50 and bino_food_conf > 0.3:
        bino_approach_gain = 0.6  # precise strike approach
    elif bino_food_dist < 100 and bino_food_conf > 0.2:
        bino_approach_gain = 0.8

    # --- Lateral line: mechanoreceptive, detects water disturbance ---
    # Omnidirectional, range ~150px, noisy
    ll_signal = 0.0
    ll_pred_dir = 0.0  # directional component
    for vp in visible_preds:
        if vp['d'] < 150:
            ll_signal += max(0, (150 - vp['d']) / 150) * 1.5
            ll_pred_dir += math.sin(vp['angle']) * max(0, (150 - vp['d']) / 150)

    # --- Olfaction: chemical gradient (food odor, alarm substance) ---
    nearest_food_idx = None
    nearest_food_dist = 9999
    if visible_food:
        visible_food.sort(key=lambda x: x[1])
        nearest_food_idx = visible_food[0][0]
        nearest_food_dist = visible_food[0][1]
    # Olfactory food gradient (bilateral nostrils)
    olf_food = max(0, 1.0 - nearest_food_dist / 300) * 0.8 if nearest_food_idx is not None else 0
    # Alarm substance: released by injured conspecifics near predators
    olf_alarm = 0.0
    for cf in _demo_conspecifics:
        if cf.get('energy', 50) < 30:
            ad = _dist(f['x'], f['y'], cf['x'], cf['y'])
            if ad < 120:
                olf_alarm += max(0, (120 - ad) / 120) * 0.5

    # ================================================================
    # STAGE 2: Optic tectum — contralateral processing
    # Zebrafish: full decussation — L eye → R tectum, R eye → L tectum
    # ================================================================
    # SFGS-b (superficial): object salience processing
    # Input: ON-sustained RGCs (contralateral retina)
    sfgs_b_L = min(4.0, retina_R * 0.8 + 0.1)  # L tectum ← R eye
    sfgs_b_R = min(4.0, retina_L * 0.8 + 0.1)  # R tectum ← L eye

    # SFGS-d (deep): motion/change detection (OFF-transient pathway)
    # Temporal derivative approximation — responds to changes
    sfgs_d_L = min(3.0, retina_R_pred * 1.2 + retina_R_food * 0.4)
    sfgs_d_R = min(3.0, retina_L_pred * 1.2 + retina_L_food * 0.4)

    # SGC: looming detection → escape trigger
    max_looming = max((vp['looming'] for vp in visible_preds), default=0)
    sgc_L = min(3.0, max_looming * 2.5 + ll_signal * 0.3)
    sgc_R = sgc_L  # bilateral convergence for looming

    # SO: direction-selective — encodes movement direction of objects
    so_L = min(2.0, retina_R_food * 1.0 + retina_R_pred * 0.5)
    so_R = min(2.0, retina_L_food * 1.0 + retina_L_pred * 0.5)

    # --- Inter-tectal rivalry (binocular rivalry / hemispheric competition) ---
    # The intertectal commissure mediates mutual suppression: when both
    # hemispheres have strong, conflicting signals, the dominant side
    # suppresses the weaker side. This implements winner-take-all at the
    # tectal level, analogous to binocular rivalry in mammals.
    tect_L_total = sfgs_b_L + sfgs_d_L + so_L
    tect_R_total = sfgs_b_R + sfgs_d_R + so_R
    rivalry_imbalance = abs(tect_L_total - tect_R_total)
    # Suppression only when both sides are active (genuine conflict)
    rivalry_active = min(tect_L_total, tect_R_total) > 0.5
    if rivalry_active and rivalry_imbalance > 0.3:
        # Dominant side suppresses weaker side by 20-40%
        suppress = min(0.4, rivalry_imbalance * 0.15)
        if tect_L_total > tect_R_total:
            sfgs_b_R *= (1.0 - suppress)
            so_R *= (1.0 - suppress)
        else:
            sfgs_b_L *= (1.0 - suppress)
            so_L *= (1.0 - suppress)

    # Aggregate tectum output
    tectum_threat = (sgc_L + sgc_R) / 2  # bilateral threat signal
    tectum_food = (so_L + so_R) / 2      # food direction signal

    # --- Tectal orienting response (novelty-driven attention saccade) ---
    # The optic tectum (superior colliculus homologue) drives reflexive
    # orienting toward novel stimuli. Change detection in SFGS-d compares
    # current vs previous retinal activation; a sudden increase triggers
    # a turn toward the novel hemifield.
    delta_L = max(0, retina_L - n['prev_retina_L'])  # new input in L eye
    delta_R = max(0, retina_R - n['prev_retina_R'])  # new input in R eye
    n['prev_retina_L'] = retina_L
    n['prev_retina_R'] = retina_R

    # Novelty signal: asymmetric change → orienting direction
    # Positive = turn right (toward R eye novelty), negative = turn left
    novelty_contrast = delta_R - delta_L
    novelty_magnitude = abs(novelty_contrast)

    # Habituation: repeated similar stimuli → diminishing orienting
    if novelty_magnitude > 0.1:
        n['orient_habituation'] = min(1.0, n['orient_habituation'] + 0.15)
    else:
        n['orient_habituation'] *= 0.92  # decay toward zero

    # Orienting signal: modulated by NA (arousal amplifies), habituated
    orient_gain = (1.0 + n['NA'] * 0.5) * max(0, 1.0 - n['orient_habituation'])
    orient_signal = novelty_contrast * orient_gain * 0.08  # rad/step

    # Clamp and store
    n['orient_dir'] = max(-0.15, min(0.15, orient_signal))

    # ================================================================
    # STAGE 3: Thalamus — relay gating (modulated by NA arousal)
    # ================================================================
    # TRN (inhibitory gating): high NA → open gate (aroused, pass everything)
    # Low NA → filter, only salient signals pass
    trn_gate = _sigmoid((n['NA'] - 0.4) * 4)  # 0-1 gate
    trn_L = 0.2 + trn_gate * 0.6
    trn_R = trn_L

    # TC (relay): passes tectum SFGS-b to pallium, gated by TRN
    tc_L = sfgs_b_L * trn_gate * n['ACh']  # ACh modulates attention
    tc_R = sfgs_b_R * trn_gate * n['ACh']

    # ================================================================
    # STAGE 4: Pallium (cortical homologue)
    # ================================================================
    # Pal-S (superficial): sensory representation — what is present?
    pal_s = min(4.0, (tc_L + tc_R) * 0.6 + olf_food * 0.5
                + ll_signal * 0.3 + olf_alarm * 0.8)

    # Pal-D (deep): decision/intent integration
    # Receives Pal-S + amygdala (fear) + place cells + allostasis
    pal_d_L = min(3.0, tc_L * 0.5 + n['amygdala_trace'] * 0.3)
    pal_d_R = min(3.0, tc_R * 0.5 + n['amygdala_trace'] * 0.3)
    pal_d = (pal_d_L + pal_d_R) / 2 + pal_s * 0.3

    # ================================================================
    # STAGE 5: Amygdala — fear circuit (episodic conditioning)
    # ================================================================
    # LA (lateral): threat input
    nearest_pred_dist = min((vp['d'] for vp in visible_preds), default=9999)
    pred_proximity = max(0, 1.0 - nearest_pred_dist / 200) if visible_preds else 0
    amyg_input = (pred_proximity * _demo_params['amygdala_gain'] + tectum_threat * 0.5
                  + olf_alarm * 1.5 + ll_signal * 0.3)

    # CeA (central): fear output with trace (persists after threat gone)
    # Episodic conditioning: near-death → LTP (trace grows)
    if pred_proximity > 0.8:
        n['amygdala_trace'] = min(3.0, n['amygdala_trace'] + 0.2)
    # Asymmetric dynamics: fast rise (tau=8), slow decay (tau=60)
    # Biological basis: fear conditioning is rapid, extinction is slow
    if amyg_input > n['amygdala_trace']:
        n['amygdala_trace'] = _ema(n['amygdala_trace'], amyg_input, tau=8)
    else:
        n['amygdala_trace'] = _ema(n['amygdala_trace'], amyg_input, tau=60)

    amygdala_out = min(3.0, n['amygdala_trace'])

    # ================================================================
    # STAGE 6: Neuromodulation — 4-axis update
    # ================================================================
    # --- Insula: interoception (hunger, stress, heart rate) ---
    insula = (0.2 + starvation * 1.5 + pred_proximity * 0.5 + fatigue * 0.8
              + n['starvation_anxiety'] * 1.2)  # anxiety about future starvation
    heart_rate = 2.0 + amygdala_out * 0.8 + fatigue * 0.3 + math.sin(t * 0.8) * 0.3

    # --- Critic: temporal difference value ---
    reward = 0.0
    ate_food = False
    if nearest_food_idx is not None and nearest_food_dist < _demo_params['food_radius']:
        reward = 1.0
        ate_food = True
        _demo_food_eaten += 1
        f['energy'] = min(100, f['energy'] + _demo_params['food_value'])
        n['steps_since_food'] = 0
        n['food_memory_xy'] = (f['x'], f['y'])
        n['food_memory_age'] = 0
        # Remove eaten food and respawn at a random safe location
        _demo_foods.pop(nearest_food_idx)
        for _try in range(40):
            nx = _init_rnd.uniform(40, 760)
            ny = _init_rnd.uniform(40, 560)
            if not _is_inside_rock(nx, ny):
                break
        _demo_foods.append([round(nx, 1), round(ny, 1)])
    if pred_proximity > 0.8:
        reward -= 0.5  # punishment for near-death

    V_current = pal_s * 0.3 + olf_food * 0.5 - amygdala_out * 0.4
    RPE = reward + 0.95 * V_current - n['V_prev']  # TD error
    n['V_prev'] = V_current
    critic = 0.2 + abs(RPE) * 2.0  # critic activity ∝ |RPE|

    # DA: reward prediction error → sigmoid(da_gain × RPE)
    n['DA'] = _ema(n['DA'], _sigmoid(_demo_params['da_gain'] * RPE), tau=8)

    # NA: arousal = 0.3 + 0.5×amygdala + 0.2×CMS
    CMS = 0.3 * amygdala_out + 0.1 * starvation
    n['NA'] = _ema(n['NA'], 0.3 + 0.5 * amygdala_out + 0.2 * CMS, tau=12)

    # 5-HT: rises when NOT fleeing (patience/habituation)
    if amygdala_out < 0.5:
        n['sht_acc'] = min(1.0, n['sht_acc'] + 0.005 * _demo_params['sht_gain'])
    else:
        n['sht_acc'] = max(0, n['sht_acc'] - 0.02)
    n['5HT'] = _ema(n['5HT'], 0.3 + 0.4 * n['sht_acc'], tau=20)

    # ================================================================
    # STAGE 7: Place cells — spatial memory (8Hz theta modulated)
    # ================================================================
    # 8Hz theta with ~50ms/step equivalent → ~1 cycle per 25 demo steps
    theta = 0.5 + 0.5 * math.sin(_demo_t * 0.25)
    n['place_fam'] = _ema(n['place_fam'], 0.3 + olf_food * 0.3, tau=50)
    place_cells = 0.3 + theta * 0.4 + n['place_fam'] * 0.3

    # --- Food memory: hippocampal replay under anxiety ---
    # When anxious and no food visible, recall where food was last found
    # This drives the fish to return to food-rich areas (place cell → forage)
    n['food_memory_age'] = n.get('food_memory_age', 999) + 1
    food_memory_signal = 0.0
    if n.get('food_memory_xy') and n['food_memory_age'] < 300:
        fm_x, fm_y = n['food_memory_xy']
        fm_dist = math.sqrt((f['x'] - fm_x)**2 + (f['y'] - fm_y)**2)
        # Memory strength decays with age (exponential) and distance
        fm_recency = math.exp(-n['food_memory_age'] / 100.0)
        fm_proximity = max(0, 1.0 - fm_dist / 400.0)
        # Only replay under anxiety — hippocampal replay is stress-gated
        food_memory_signal = fm_recency * 0.6 * n['starvation_anxiety']

    # ================================================================
    # STAGE 8: Expected Free Energy — goal selection
    # Matches brain_v2.py EFE computation (lines 479-596)
    # ================================================================
    U = max(0, 1.0 - 0.5 * (CMS + 0.3))  # uncertainty/novelty

    # Food evidence: retinal + olfactory + place cell food memory
    p_food = min(1.0, (retina_L_food + retina_R_food) * 0.4
                 + olf_food * 0.6
                 + food_memory_signal)  # remembered food location

    # Enemy evidence: retinal + amygdala + tectum + lateral line
    p_enemy = min(1.0, pred_proximity * 0.6 + amygdala_out * 0.25
                  + tectum_threat * 0.2 + ll_signal * 0.15)

    # Circadian sleep drive
    sleep_drive = max(0, (0.3 - light) / 0.3) if not is_day else 0

    # EFE per goal (lower = more attractive, matches brain_v2.py convention)
    # Starvation anxiety biases FORAGE strongly — fish fears future energy crisis
    anx = n['starvation_anxiety']
    # Melatonin-mediated circadian suppression of arousal/foraging at night
    night_factor = max(0, (0.3 - light) / 0.3) if not is_day else 0.0

    G_forage = (0.2 * U - 0.8 * p_food + 0.15
                - starvation * 0.5      # current hunger bias
                - anx * 1.2 * (1.0 - night_factor * 0.7)  # melatonin blunts foraging drive at night
                - olf_food * 0.3        # smell guides foraging
                - food_memory_signal * 0.5  # remembered food pulls toward foraging
                + night_factor * 0.4    # circadian penalty: foraging suppressed at night
                + n['frustration'][0] * 0.3)

    G_flee = (0.1 * CMS - 1.2 * p_enemy + 0.20
              - amygdala_out * 0.4          # fear drives escape
              - ll_signal * 0.2             # lateral line: behind-detection triggers flee
              + n['5HT'] * 0.1             # patience mildly suppresses flee
              + anx * 0.10                 # starving fish takes slightly more risk
              + n['frustration'][1] * 0.15)

    G_explore = (0.3 * U - 0.3 + 0.20
                 + anx * 0.4             # anxiety PENALIZES aimless explore → drives FORAGE
                 + n['frustration'][2] * 0.2)

    G_social = (0.18 + starvation * 0.2
                + anx * 0.3              # anxiety reduces social drive
                + n['frustration'][3] * 0.2
                - 0.4 * min(1.0, retina_L_conspec + retina_R_conspec))  # attractive when conspecifics visible

    G_sleep = (0.5 - sleep_drive * 1.5  # circadian pull (strong night drive)
               - fatigue * 0.5           # tired → sleep
               - night_factor * fatigue * 0.8  # low energy at night → strong sleep pull
               + amygdala_out * 0.5      # fear prevents sleep
               + anx * 0.4 * (1.0 - night_factor * 0.8))  # melatonin blunts anxiety at night

    goals = {'FORAGE': G_forage,
             'FLEE':   G_flee   - (_demo_params['flee_weight'] - 1.0) * 0.4,
             'EXPLORE': G_explore, 'SOCIAL': G_social, 'SLEEP': G_sleep}

    # --- Reflexive overrides (hard-wired, like Mauthner escape) ---
    # C-start reflex: looming > threshold → immediate escape
    if max_looming > 0.5 and n['cstart_timer'] <= 0:
        n['cstart_timer'] = 4  # 4-step motor sequence
        nearest_vp = min(visible_preds, key=lambda vp: vp['d'])
        n['cstart_dir'] = math.atan2(f['y'] - nearest_vp['ref']['y'],
                                     f['x'] - nearest_vp['ref']['x'])

    # Starvation override
    if starvation > 0.7 and p_food > 0.2 and pred_proximity < 0.3:
        goals['FORAGE'] -= 0.5

    # --- Winner-take-all with BG attractor persistence ---
    if n['goal_lock'] > 0:
        n['goal_lock'] -= 1
        winner = n['locked_goal']
    else:
        winner = min(goals, key=goals.get)
        # BG attractor: lock into goal for 8-15 steps
        if winner != f.get('goal', 'EXPLORE'):
            n['goal_lock'] = 10
            n['locked_goal'] = winner
            # Habenula: strategy switch frustration on old goal
            goal_idx = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL'].index(
                f.get('goal', 'EXPLORE')) if f.get('goal', 'EXPLORE') in \
                ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL'] else 2
            n['frustration'][goal_idx] = min(1.0,
                                             n['frustration'][goal_idx] + 0.1)

    f['goal'] = winner

    # Frustration decay
    for i in range(4):
        n['frustration'][i] *= 0.995

    # ================================================================
    # STAGE 9: Basal ganglia — D1 (go) / D2 (no-go) gate
    # ================================================================
    # D1: direct pathway, enhanced by DA → disinhibit motor
    d1_input = pal_d * 0.5 + (0.8 if winner in ('FORAGE', 'EXPLORE') else 0.3)
    d1 = d1_input * (1.0 + 0.5 * n['DA'])

    # D2: indirect pathway, suppressed by DA → inhibit motor
    d2_input = pal_d * 0.3 + (0.8 if winner == 'FLEE' else 0.2) + amygdala_out * 0.3
    d2 = d2_input * (1.0 - 0.3 * n['DA'])

    # BG gate: D1 wins → gate opens → motor permitted
    bg_gate = _sigmoid(4.0 * (d1 * 0.4 - d2 * 0.3) - _demo_params['bg_bias'])

    # ================================================================
    # STAGE 10: Motor output — reticulospinal + CPG
    # ================================================================
    cerebellum = 0.3  # default; updated by voluntary motor branch
    # --- Mauthner C-start reflex (overrides voluntary motor) ---
    if n['cstart_timer'] > 0:
        cst = n['cstart_timer']
        n['cstart_timer'] -= 1
        if cst == 4:
            # Stage 1: sharp ipsilateral bend
            f['h'] = n['cstart_dir'] + 0.3 * (1 if _init_rnd.random() > 0.5 else -1)
            f['speed'] = 0.3
        elif cst == 3:
            f['speed'] = 0.8
        elif cst == 2:
            f['speed'] = 3.5  # propulsive burst
        else:
            f['speed'] = 3.0
        f['goal'] = 'FLEE'  # override goal display during C-start
    else:
        # --- Voluntary motor: Pal-D L/R contrast → turn direction ---
        # Left pallium stronger → turn right, right → turn left
        voluntary_turn = (pal_d_R - pal_d_L) * 0.3

        # Tectal orienting response: reflexive turn toward novel stimulus
        # Bottom-up, pre-attentive — fires regardless of current goal
        voluntary_turn += n['orient_dir']

        # Goal-specific motor programs (from reticulospinal named neurons)
        if winner == 'FORAGE' and nearest_food_idx is not None:
            target = _demo_foods[nearest_food_idx]
            target_h = math.atan2(target[1] - f['y'], target[0] - f['x'])
            turn_toward = math.atan2(math.sin(target_h - f['h']),
                                     math.cos(target_h - f['h']))
            voluntary_turn += turn_toward * 0.08
            goal_speed = (1.2 + 0.5 * min(1.0, nearest_food_dist / 200)) * bino_approach_gain
        elif winner == 'FORAGE' and n.get('food_memory_xy') and n['food_memory_age'] < 300:
            # No visible food but anxious → navigate to remembered food area
            fm_x, fm_y = n['food_memory_xy']
            target_h = math.atan2(fm_y - f['y'], fm_x - f['x'])
            turn_toward = math.atan2(math.sin(target_h - f['h']),
                                     math.cos(target_h - f['h']))
            voluntary_turn += turn_toward * 0.06  # gentler: memory is uncertain
            goal_speed = 1.0 + anx * 0.5  # anxiety makes search more urgent
        elif winner == 'FLEE':
            if visible_preds:
                nearest_vp = min(visible_preds, key=lambda vp: vp['d'])
                away_h = math.atan2(f['y'] - nearest_vp['ref']['y'],
                                    f['x'] - nearest_vp['ref']['x'])
                turn_away = math.atan2(math.sin(away_h - f['h']),
                                       math.cos(away_h - f['h']))
                voluntary_turn += turn_away * 0.18
            goal_speed = 3.0
        elif winner == 'SOCIAL':
            # Steer toward nearest conspecific
            nearest_cf = min(_demo_conspecifics,
                             key=lambda c: _dist(f['x'], f['y'], c['x'], c['y']),
                             default=None)
            if nearest_cf:
                soc_h = math.atan2(nearest_cf['y'] - f['y'],
                                   nearest_cf['x'] - f['x'])
                turn_to = math.atan2(math.sin(soc_h - f['h']),
                                     math.cos(soc_h - f['h']))
                voluntary_turn += turn_to * 0.05
            goal_speed = 0.9
        elif winner == 'SLEEP':
            # Seek nearest rock shelter
            nearest_rock = min(_demo_rock_defs,
                               key=lambda r: _dist(f['x'], f['y'], r[0], r[1]))
            shelter_d = _dist(f['x'], f['y'], nearest_rock[0], nearest_rock[1])
            if shelter_d > nearest_rock[2] + 20:
                sh = math.atan2(nearest_rock[1] - f['y'],
                                nearest_rock[0] - f['x'])
                voluntary_turn += math.atan2(math.sin(sh - f['h']),
                                             math.cos(sh - f['h'])) * 0.03
                goal_speed = 0.3
            else:
                goal_speed = 0.05
                voluntary_turn = math.sin(t * 0.1) * 0.005
            f['energy'] = min(100, f['energy'] + 0.03)
        else:  # EXPLORE
            # Sinusoidal scanning (saccade-like) + random walk
            voluntary_turn += 0.02 + math.sin(t * 0.3) * 0.03
            goal_speed = 1.0

        # Cerebellum: forward model error (prediction vs actual)
        actual_turn = voluntary_turn
        actual_speed = goal_speed
        cb_pe_turn = abs(actual_turn - n['cb_pred'][0])
        cb_pe_speed = abs(actual_speed - n['cb_pred'][1])
        n['cb_pred'][0] = _ema(n['cb_pred'][0], actual_turn, tau=5)
        n['cb_pred'][1] = _ema(n['cb_pred'][1], actual_speed, tau=5)
        cerebellum = 0.3 + (cb_pe_turn + cb_pe_speed) * 0.8

        # Apply BG gate to motor amplitude
        # U-shaped motility: peak at ~50% energy, low at both extremes
        # Inverted-U: motility = 4 * e * (1-e) where e = energy_ratio
        # e=0 → 0 (exhausted), e=0.5 → 1.0 (peak), e=1.0 → 0 (satiated)
        e = energy_ratio
        motility = max(0.15, 4.0 * e * (1.0 - e))  # floor at 0.15 so fish never freezes
        f['h'] += voluntary_turn * bg_gate
        f['speed'] = goal_speed * bg_gate * motility * _demo_params['fish_speed_mult']

    # CPG: rhythmic pattern modulates speed
    cpg_rhythm = 0.8 + 0.2 * math.sin(t * 6)  # ~6Hz swimming rhythm
    f['speed'] *= cpg_rhythm

    # Reticulospinal: descending command amplitude
    reticulospinal = 0.1 + f['speed'] * 0.6 + (1.5 if n['cstart_timer'] > 0 else 0)
    cpg_act = 0.2 + f['speed'] * 0.8

    # ================================================================
    # STAGE 11: Physical movement + collision
    # ================================================================
    f['h'] = _steer_away_from_rocks(f['x'], f['y'], f['h'])
    f['x'] = _clamp(f['x'] + math.cos(f['h']) * f['speed'], 20, 780)
    f['y'] = _clamp(f['y'] + math.sin(f['h']) * f['speed'], 20, 580)
    f['x'], f['y'] = _collide_rocks(f['x'], f['y'])
    is_sleeping = winner == 'SLEEP'
    f['energy'] = max(0, f['energy'] - (0.003 if is_sleeping else _demo_params['energy_drain']))

    # Starvation death: energy depleted
    if f['energy'] <= 0 and not n['dead']:
        n['dead'] = True
        n['death_timer'] = 33
        n['death_x'] = f['x']
        n['death_y'] = f['y']
        f['goal'] = 'DEAD'


    # ================================================================
    # Conspecifics (simplified neural: flee/social/forage)
    # ================================================================
    for ci, cf in enumerate(_demo_conspecifics):
        if not cf.get('alive', True):
            continue
        cf_flee = None
        for pred in _demo_predators:
            pd = _dist(cf['x'], cf['y'], pred['x'], pred['y'])
            if pd < 100:
                cf_flee = pred
                break
        if cf_flee:
            away_h = math.atan2(cf['y'] - cf_flee['y'], cf['x'] - cf_flee['x'])
            cf['h'] = _steer_toward(cf['h'], away_h, 0.12)
            cf['goal'] = 'FLEE'
            spd = 3.0
        else:
            d_focal = _dist(cf['x'], cf['y'], f['x'], f['y'])
            if d_focal > 100:
                target_h = math.atan2(f['y'] - cf['y'], f['x'] - cf['x'])
                cf['h'] = _steer_toward(cf['h'], target_h, 0.05)
                cf['goal'] = 'SOCIAL'
            elif d_focal < 30:
                cf['h'] += 0.1
            else:
                cf['h'] += math.sin(t * 0.3 + ci * 2.1) * 0.04
                cf['goal'] = 'FORAGE'
            spd = 1.3 + 0.3 * math.sin(t + ci)
        cf['h'] = _steer_away_from_rocks(cf['x'], cf['y'], cf['h'])
        cf['x'] = _clamp(cf['x'] + math.cos(cf['h']) * spd, 20, 780)
        cf['y'] = _clamp(cf['y'] + math.sin(cf['h']) * spd, 20, 580)
        cf['x'], cf['y'] = _collide_rocks(cf['x'], cf['y'])
        cf['energy'] = max(20, cf['energy'] - 0.01 + 0.02 * math.sin(t + ci))

    # ================================================================
    # Predators: patrol/hunt (simplified — predator brain not modeled)
    # ================================================================
    for pred in _demo_predators:
        all_prey = [{'ref': f, 'x': f['x'], 'y': f['y']}]
        for c in _demo_conspecifics:
            if c.get('alive', True):
                all_prey.append({'ref': c, 'x': c['x'], 'y': c['y']})
        nearest_prey = min(all_prey,
                           key=lambda p: _dist(pred['x'], pred['y'], p['x'], p['y']))
        prey_dist = _dist(pred['x'], pred['y'], nearest_prey['x'], nearest_prey['y'])

        if prey_dist < _demo_params['pred_range'] and pred['energy'] > 20:
            if pred['state'] != 'hunt':
                pred['hunt_steps'] = 0  # reset on hunt transition
            pred['state'] = 'hunt'
            pred['hunt_steps'] = pred.get('hunt_steps', 0) + 1
            chase_h = math.atan2(nearest_prey['y'] - pred['y'],
                                 nearest_prey['x'] - pred['x'])
            pred['h'] = _steer_toward(pred['h'], chase_h, 0.10)
            pred_spd = _demo_params['pred_speed']
            pred['energy'] = max(0, pred['energy'] - 0.06)
            # Kill: close enough AND sustained hunt
            if prey_dist < 20 and pred.get('hunt_steps', 0) >= 20:
                pred['energy'] = min(100, pred['energy'] + 15)
                pred['hunt_steps'] = 0
                prey_ref = nearest_prey['ref']
                if prey_ref is not f:
                    # Conspecific: instant kill
                    prey_ref['alive'] = False
                    prey_ref['energy'] = 0
                else:
                    # Focal fish: KILLED — trigger death state
                    n['dead'] = True
                    n['death_timer'] = 33  # ~5 seconds at 150ms interval
                    n['death_x'] = f['x']
                    n['death_y'] = f['y']
                    f['energy'] = 0
                    f['goal'] = 'DEAD'
        else:
            pred['state'] = 'patrol'
            pred['hunt_steps'] = 0
            pa = t * 0.3 + _demo_predators.index(pred) * 2.1
            target_x = pred['patrol_cx'] + pred['patrol_r'] * math.cos(pa)
            target_y = pred['patrol_cy'] + pred['patrol_r'] * math.sin(pa)
            patrol_h = math.atan2(target_y - pred['y'], target_x - pred['x'])
            pred['h'] = _steer_toward(pred['h'], patrol_h, 0.05)
            pred_spd = 1.2
            pred['energy'] = min(100, pred['energy'] + 0.02)

        pred['h'] = _steer_away_from_rocks(pred['x'], pred['y'], pred['h'])
        pred['x'] = _clamp(pred['x'] + math.cos(pred['h']) * pred_spd, 10, 790)
        pred['y'] = _clamp(pred['y'] + math.sin(pred['h']) * pred_spd, 10, 590)
        pred['x'], pred['y'] = _collide_rocks(pred['x'], pred['y'])

    # ================================================================
    # VISCERAL ORGANS: lightweight heuristic approximations
    # ================================================================

    # --- Vagus nerve: parasympathetic tone (high when calm, low under stress) ---
    # Stress proxy from amygdala + starvation; vagal tone is inverse
    stress_level = min(1.0, amygdala_out * 0.4 + starvation * 0.3
                       + pred_proximity * 0.3)
    n['vagal_tone'] = _ema(n['vagal_tone'], 1.0 - stress_level, tau=15)

    # --- Pituitary–adrenal axis (HPA): CRH → ACTH → cortisol ---
    # CRH: hypothalamic stress signal driven by amygdala + starvation
    n['crh'] = _ema(n['crh'], amygdala_out * 0.5 + starvation * 0.4
                    + n['starvation_anxiety'] * 0.3, tau=20)
    n['acth'] = stress_level * 0.6 + n['crh'] * 0.4
    n['cortisol_drive'] = n['acth'] * 0.5

    # --- Area postrema: circumventricular organ (glucose + toxin sensing) ---
    # Glucose status: maps energy ratio [0,1] → [−1, +1]
    n['glucose_status'] = (energy_ratio - 0.5) * 2.0
    n['nausea'] = 0.0  # no toxins modeled in demo

    # --- NTS: nucleus tractus solitarius (vagal afferent relay) ---
    # Satiety: jumps on eating, decays slowly (gut stretch → NTS → satiety)
    if ate_food:
        n['satiety_signal'] = min(1.0, n['satiety_signal'] + 0.3)
    else:
        n['satiety_signal'] *= 0.995  # slow decay (~200 step half-life)
    # Taste relay: gustatory salience from food proximity (via facial nerve → NTS)
    n['taste_relay'] = max(0, 1.0 - nearest_food_dist / 60) if nearest_food_idx is not None else 0.0

    # --- Lateral line efferent: corollary discharge (self-motion suppression) ---
    # During fast swimming, efferent copy suppresses lateral-line sensitivity
    # to prevent self-generated wake from triggering false alarms.
    # gain → 0 at high speed (full suppression), gain → 1 at rest
    current_speed = f['speed']
    n['ll_efferent_gain'] = _ema(
        n['ll_efferent_gain'],
        max(0.1, 1.0 - min(1.0, current_speed / 3.0) * 0.8),
        tau=5,
    )

    # ================================================================
    # Collect all neural activations — these ARE the computation,
    # not decorative retroactive spike data
    # ================================================================
    p0 = _demo_predators[0] if _demo_predators else {'x': -100, 'y': -100, 'h': 0, 'energy': 0, 'state': 'patrol'}
    sleep_factor = 0.15 if is_sleeping else 1.0
    habenula = (sum(n['frustration']) * 0.5
                + max(0, (50 - f['energy']) / 50) * 0.5
                + pred_proximity * 0.3)
    predictive = pal_s * 0.4 + abs(RPE) * 0.8  # prediction error magnitude

    spikes = {
        # Stage 1: Sensory input
        'retina_L': retina_L * sleep_factor,
        'retina_R': retina_R * sleep_factor,
        'lateral_line': ll_signal * sleep_factor,
        'olfaction': (olf_food + olf_alarm) * sleep_factor,
        # Stage 2: Tectum (contralateral)
        'sfgs_b': ((sfgs_b_L + sfgs_b_R) / 2) * sleep_factor,
        'sfgs_b_L': sfgs_b_L * sleep_factor,
        'sfgs_b_R': sfgs_b_R * sleep_factor,
        'sfgs_d': ((sfgs_d_L + sfgs_d_R) / 2) * sleep_factor,
        'sgc': ((sgc_L + sgc_R) / 2) * sleep_factor,
        'so': ((so_L + so_R) / 2) * sleep_factor,
        # Stage 3: Thalamus
        'tc': ((tc_L + tc_R) / 2) * sleep_factor,
        'trn': ((trn_L + trn_R) / 2) * sleep_factor,
        # Stage 4: Pallium
        'pal_s': pal_s * sleep_factor,
        'pal_d': pal_d * sleep_factor,
        # Stage 5: Amygdala
        'amygdala': amygdala_out * sleep_factor,
        # Stage 6: Limbic
        'habenula': habenula * sleep_factor,
        'insula': insula * sleep_factor,
        'critic': critic * sleep_factor,
        # Stage 7: Spatial
        'place_cells': place_cells * sleep_factor,
        'predictive': predictive * sleep_factor,
        # Stage 8: Basal ganglia
        'd1': d1 * sleep_factor,
        'd2': d2 * sleep_factor,
        # Stage 9: Motor
        'cerebellum': cerebellum * sleep_factor if n['cstart_timer'] <= 0 else 2.5,
        'cpg': cpg_act * sleep_factor,
        'reticulospinal': reticulospinal * sleep_factor,
        # Visceral organs
        'vagus_nerve': n['vagal_tone'] * sleep_factor,
        'pituitary': n['acth'] * sleep_factor,
        'area_postrema': max(0, -n['glucose_status']) * sleep_factor,  # active when glucose low
        'nts': (n['satiety_signal'] + n['taste_relay']) * 0.5 * sleep_factor,
        'll_efferent': (1.0 - n['ll_efferent_gain']) * sleep_factor,  # suppression strength
    }

    # Per-eye detection data for arena FoV visualization
    pred_in_left = retina_L_pred > 0.1
    pred_in_right = retina_R_pred > 0.1

    return {
        'fish_x': f['x'], 'fish_y': f['y'], 'fish_heading': f['h'],
        'goal': f['goal'], 'energy': f['energy'],
        'step': _demo_t, 'heart_rate': heart_rate,
        'turn': (pal_d_R - pal_d_L) * 0.3, 'speed': f['speed'] / 3.0,
        'food_total': _demo_food_eaten,
        'ate_food': ate_food,
        'arena_w': 800, 'arena_h': 600,
        'foods': list(_demo_foods),
        'rocks': list(_demo_rocks),
        'pred_x': p0['x'], 'pred_y': p0['y'],
        'pred_heading': p0['h'],
        'pred_energy': p0['energy'], 'pred_state': p0['state'],
        'extra_predators': [
            {'x': p['x'], 'y': p['y'], 'heading': p['h'],
             'energy': p['energy'], 'state': p['state']}
            for p in _demo_predators[1:]
        ],
        'other_fish': [
            {'x': c['x'], 'y': c['y'], 'heading': c['h'],
             'goal': c['goal'], 'energy': c['energy'],
             'alive': c.get('alive', True)}
            for c in _demo_conspecifics
        ],
        'spikes': spikes,
        'DA': n['DA'], 'NA': n['NA'], '5HT': n['5HT'], 'ACh': n['ACh'],
        # Circadian
        'light_level': light,
        'circ_label': circ_label,
        'is_sleeping': is_sleeping,
        'starvation_anxiety': round(n['starvation_anxiety'], 3),
        # Per-eye FoV data
        'left_eye_food': left_eye_count,
        'right_eye_food': right_eye_count,
        'pred_in_left': pred_in_left,
        'pred_in_right': pred_in_right,
        'cstart_timer': n['cstart_timer'],
        # Binocular depth & rivalry
        'bino_food_dist': round(bino_food_dist, 1) if bino_food_dist < 900 else None,
        'bino_food_conf': round(bino_food_conf, 3),
        'bino_pred_dist': round(bino_pred_dist, 1) if bino_pred_dist < 900 else None,
        'bino_pred_conf': round(bino_pred_conf, 3),
        'bino_approach_gain': round(bino_approach_gain, 2),
        'rivalry_suppression': round(suppress, 3) if rivalry_active and rivalry_imbalance > 0.3 else 0.0,
        'rivalry_dominant': 'L' if rivalry_active and rivalry_imbalance > 0.3 and tect_L_total > tect_R_total
                           else ('R' if rivalry_active and rivalry_imbalance > 0.3 else None),
        # Orienting response
        'orient_dir': round(n['orient_dir'], 4),
        'orient_habituation': round(n['orient_habituation'], 3),
        'novelty_L': round(delta_L, 3),
        'novelty_R': round(delta_R, 3),
        # Visceral organs
        'vagal_tone': round(n['vagal_tone'], 3),
        'crh': round(n['crh'], 3),
        'acth': round(n['acth'], 3),
        'cortisol_drive': round(n['cortisol_drive'], 3),
        'glucose_status': round(n['glucose_status'], 3),
        'nausea': round(n['nausea'], 3),
        'satiety_signal': round(n['satiety_signal'], 3),
        'taste_relay': round(n['taste_relay'], 3),
        'll_efferent_gain': round(n['ll_efferent_gain'], 3),
        # Free energy (simulated from prediction errors)
        'free_energy': round(abs(retina_L - retina_R) * 0.05 + n.get('starvation_anxiety', 0) * 0.15, 4),
        'total_free_energy': round(abs(retina_L - retina_R) * 0.05 + n.get('starvation_anxiety', 0) * 0.15 + max_looming * 0.1, 4),
        'fe_gradient': 0.0,
        'spontaneity': 0.05,
        'ai_blend': 0.3,
        'ai_convergence': 0.0,
        'module_fe': {
            'retina': round(abs(retina_L - retina_R) * 0.03, 4),
            'olfaction': round(olf_food * 0.02, 4),
            'lateral_line': round(ll_signal * 0.03, 4),
            'proprioception': round(abs(f['speed'] - 1.0) * 0.02, 4),
            'vestibular': round(abs(n.get('orient_dir', 0)) * 0.02, 4),
            'color_vision': round(abs(light - 0.5) * 0.02, 4),
            'thalamus_L': round(tc_L * 0.02, 4),
            'thalamus_R': round(tc_R * 0.02, 4),
            'pallium': round(abs(pal_d_L - pal_d_R) * 0.04, 4),
            'active_motor': round(f['speed'] * 0.01, 4),
        },
    }


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            if _latest_step:
                await ws.send_json({'type': 'step', 'data': _latest_step})
            elif _brain_demo_active and _brain_latest_step:
                await ws.send_json({'type': 'step', 'data': _brain_latest_step})
            else:
                # No step data yet — run 8 physics sub-steps per frame so
                # the fish moves at a realistic speed (~8px / 0.15 s interval)
                for _ in range(7):
                    _idle_demo_step()
                await ws.send_json({'type': 'step', 'data': _idle_demo_step()})
            await ws.send_json({
                'type': 'status',
                'data': {'running': engine.running, 'round': engine.current_round,
                         'brain_demo': _brain_demo_active}
            })
            await asyncio.sleep(0.3 if engine.running else 0.15)
    except WebSocketDisconnect:
        ws_clients.discard(ws)


@app.get("/api/demo_params")
async def get_demo_params():
    return {**_demo_params, 'food_eaten': _demo_food_eaten,
            'fish_energy': _demo_fish.get('energy', 0),
            'fish_goal': _demo_fish.get('goal', '?')}


@app.post("/api/demo_params")
async def set_demo_params(request: Request):
    body = await request.json()
    _allowed = set(_demo_params.keys())
    for k, v in body.items():
        if k in _allowed:
            _demo_params[k] = float(v)
    return {'ok': True, 'params': _demo_params}


@app.post("/api/brain_demo/start")
async def start_brain_demo():
    global _brain_demo_active, _brain_demo_thread
    if not _brain_demo_active:
        _brain_demo_active = True
        _brain_demo_thread = threading.Thread(target=_run_brain_demo, daemon=True)
        _brain_demo_thread.start()
        print('[brain-demo] started background thread')
    return {'ok': True, 'active': True}


@app.post("/api/brain_demo/stop")
async def stop_brain_demo():
    global _brain_demo_active
    _brain_demo_active = False
    print('[brain-demo] stopped')
    return {'ok': True, 'active': False}


@app.get("/api/brain_demo/status")
async def brain_demo_status():
    return {
        'active': _brain_demo_active,
        'step': _brain_latest_step.get('step', 0),
        'goal': _brain_latest_step.get('goal', ''),
        'energy': _brain_latest_step.get('energy', 0),
        'epochs': _brain_demo_epochs,
        'saves': _brain_demo_saves,
        'next_save_in': EPOCH_SIZE - (_brain_demo_epochs % EPOCH_SIZE),
    }


@app.get("/api/checkpoint/status")
async def checkpoint_status():
    ckpt_dir = os.path.join(PROJECT_ROOT, 'zebrav2/checkpoints')
    latest = os.path.join(ckpt_dir, 'ckpt_latest.pt')
    size_mb = round(os.path.getsize(latest) / 1e6, 1) if os.path.exists(latest) else 0
    mtime = os.path.getmtime(latest) if os.path.exists(latest) else 0
    return {
        'exists': os.path.exists(latest),
        'size_mb': size_mb,
        'saved_at': mtime,
        'web_epochs': _brain_demo_epochs,
        'web_saves': _brain_demo_saves,
        'epoch_size': EPOCH_SIZE,
        'next_save_in': EPOCH_SIZE - (_brain_demo_epochs % EPOCH_SIZE) if _brain_demo_active else None,
    }


@app.get("/api/checkpoint/download")
async def download_checkpoint():
    from fastapi.responses import FileResponse
    latest = os.path.join(PROJECT_ROOT, 'zebrav2/checkpoints/ckpt_latest.pt')
    if not os.path.exists(latest):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No checkpoint found")
    return FileResponse(latest, media_type='application/octet-stream',
                        filename='ckpt_latest.pt')


def main():
    print(f"\n{'='*60}")
    print(f"  Zebrafish Brain v2 Dashboard")
    print(f"  Open: http://localhost:5001")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")


if __name__ == '__main__':
    main()
