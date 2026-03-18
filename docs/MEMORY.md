# vzebra Project Memory

## Environment
- Python: `.venv/bin/python` (Python 3.12, has numpy/torch/etc.)
- Do NOT use system python3 or conda — use `.venv` in project root
- Device: MPS (Apple Silicon) via `get_device()` in `brain/device_util.py`
- If Dropbox sync invalidates .so code signatures, recreate venv: `rm -rf .venv && python3.12 -m venv .venv && .venv/bin/pip install torch numpy matplotlib gymnasium`

## Architecture
- `zebrav1/` — main package (v1 zebrafish brain simulation)
- `zebrav1/brain/` — 25+ neural modules (SNN, optic tectum, dopamine, BG, etc.)
- `zebrav1/gym_env/` — Gymnasium environment + BrainAgent bridge
- `zebrav1/world/` — WorldEnv (ray-casting world) + renderer
- `zebrav1/tests/` — step-by-step integration tests (step1 through step30)
- `zebrav1/viz/` — neural monitor visualization
- `zebrav1/weights/` — pretrained weights (genomic, hebbian, classifier)
- `zebrav1/paper.tex` — 58-page technical report (Steps 1-29)

## SNN Architecture (Steps 25-26)
- **PredictiveTwoComp**: two-compartment neuron (soma V_s + apical V_a), PE = V_a - V_s
- **AttentionModulator**: goal → 8 attention neurons → per-layer additive somatic bias
- **Homeostatic gain control**: TARGET_RMS=5.0, divisive normalization
- Weight naming: W_FF (feedforward), W_FB (feedback)
- Free energy: F = accuracy (PE²) + 0.01*complexity (KL) + Bayesian surprise (|ΔF|)

## Classifier (Step 11) — 100% all classes
- Input: 804 dims = 800 type pixels + 4 aggregate pixel counts
- Architecture: 804→128 ReLU→5 softmax, class-weighted loss (3x environment)

## Training Pipeline (always pass goal_probs)
1. Step 8: genomic (70% FORAGE/20% FLEE/10% EXPLORE) → genomic.pt
2. Step 10: Hebbian → genomic_hebbian.pt
3. Step 11: classifier → classifier.pt
4. Step 26: W_FB (frozen W_FF, _skip_ff_update=True) → classifier_wfb.pt

## Motor Primitives (Step 28)
- Bout types: IDLE, ROUTINE_FWD, BURST, ESCAPE, FLEE_BURST, CAPTURE
- Brain controls DIRECTION, motor system controls TIMING (burst-glide-idle)
- Mauthner C-start: l/v < 10 → 1.5 rad turn + 1.6x propulsive stroke
- Prey capture: J-TURN → APPROACH → STRIKE (1.4x eat radius)
- Starvation interrupts IBI (energy < 30%)

## Ecological Features (Steps 27-30)
- Multi-size food: small (+2, radius 12), large (+5, radius 16)
- Binocular depth: intensity + stereo overlap → per-entity depth
- Optimal foraging: density-based marginal value theorem
- Adaptive predator: chase boost 1.25x, speed 1.4 base
- Social learning: observe conspecific flee/forage behavior
- Rock cover-seeking during flee
- Interactive: P=predator attack, R=retreat, F=spawn food

## Decision Rationality (Step 29) — 83/100 RATIONAL
- Scenarios: A=90[A], B=80[A], C=80[A], D=100[A], E=65[B]
- Curriculum (Step 30): 3/4 COMPETENT

## Key Parameters
- Obstacle detection margin: 25 units (world_env.py)
- Pixel-evidence cap: enemy_px / 15
- Emergency flee threshold: p_enemy > 0.25
- Looming trigger: l/v < 10, enemy_px > 3
- Flee speed: 1.5 + 0.5*p_enemy, burst cap 1.6x
- Predator speed: 1.4 base, 1.25x chase boost

## Branches
- `master` — steps 1-30 (PC, AIF, Bayesian brain, motor primitives, social, curriculum)
