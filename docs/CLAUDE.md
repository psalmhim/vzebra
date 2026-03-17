# vzebra — Zebrafish Brain Simulation

## Environment
- Python: `.venv/bin/python` (Python 3.12, has numpy/torch/etc.)
- Do NOT use system python3 or conda — use `.venv` in project root
- Device: MPS (Apple Silicon) via `get_device()` in `brain/device_util.py`
- Setup: `python -m venv .venv && .venv/bin/pip install torch numpy matplotlib gymnasium`

## Architecture
- `zebra_v60/` — main package (v60 zebrafish brain simulation)
- `zebra_v60/brain/` — 25 neural modules (SNN, optic tectum, dopamine, BG, etc.)
- `zebra_v60/gym_env/` — Gymnasium environment + BrainAgent bridge
- `zebra_v60/world/` — WorldEnv (ray-casting world) + renderer
- `zebra_v60/tests/` — step-by-step integration tests (step1 through step29)
- `zebra_v60/viz/` — neural monitor visualization
- `zebra_v60/weights/` — pretrained weights (genomic, hebbian, classifier)

## Key Files
- `brain_agent.py` — main brain pipeline (~1500 lines), orchestrates all modules
- `zebrafish_snn_v60.py` — SNN model with PredictiveTwoComp neurons + AttentionModulator
- `zebrafish_env.py` — Gymnasium env (prey-predator arena with rocks, food, predator)
- `hebbian_v60.py` — RPE-gated Hebbian + PE-driven anti-Hebbian feedback learning
- `retina_sampling_v60.py` — binocular retinal sampling (800 per eye: 400 intensity + 400 type)

## SNN Architecture
- **PredictiveTwoComp**: two-compartment neuron (soma V_s + apical V_a), PE = V_a - V_s
- **AttentionModulator**: goal → 8 attention neurons → per-layer projection
- **Homeostatic gain control**: TARGET_RMS=5.0, divisive normalization
- Layers: OT_L/OT_R (frozen TwoComp) → OT_F → PT_L → PC_per → PC_int → mot/eye/DA
- Weight naming: W_FF (feedforward), W_FB (feedback) — NOT old W or weight/bias

## Type Encoding (retina_sampling_v60.py)
- NONE=0.0, FOOD=1.0, ENEMY=0.5, COLLEAGUE=0.25, BOUNDARY=0.12, OBSTACLE=0.75, PREY=0.38
- Detection tolerance: `|type - val| < 0.1` (boundary uses 0.05)

## Classifier Architecture
- Input: 804 dims (800 raw type pixels + 4 aggregate pixel counts: obs/ene/food/boundary)
- Hidden: 128 ReLU units, Output: 5 classes
- Pixel count features enable obstacle vs enemy/food disambiguation (large vs point entities)
- Class-weighted loss (3x for environment), n_integration=8 for environment class

## Training Pipeline
1. **Step 8** (genomic pretraining): 70% FORAGE + 20% FLEE + 10% EXPLORE → `weights/genomic_v60.pt`
2. **Step 10** (Hebbian fine-tuning): online RPE-gated plasticity → `weights/genomic_hebbian_v60.pt`
3. **Step 11** (classifier): goal-matched context, obstacle scenes → `weights/classifier_v60.pt`
4. **Step 26** (W_FB training): online layer-wise PE minimization, frozen W_FF → `weights/classifier_wfb_v60.pt`
- All training steps MUST pass `goal_probs` to `model.forward()` — without it, AttentionModulator trains on zeros
- Step 26 MUST freeze W_FF (`_skip_ff_update=True`) and classifier to prevent representational drift

## Weight Loading
- `brain_agent.py` uses `model.load_saveable_state(state)` (NOT `load_state_dict`)
- `load_saveable_state()` handles migration from old format (TwoComp.W → W_FF, nn.Linear → W_FF.t())

## Free Energy & Bayesian Brain
- **Free energy**: F = accuracy (Σ π_l·PE²) + 0.01·complexity (Σ γ²) — upper bounds surprise
- **Bayesian surprise**: |F(t) - F(t-1)| — regime change detection
- **Attention = precision optimization**: goal-driven additive somatic bias (Feldman & Friston 2010)
- **Interoceptive PE**: allostatic errors are prediction errors across the interoceptive Markov blanket
- **EFE precision modulation**: σ_E tightens under hunger, σ_S tightens under stress
- **DP-like memory**: AssociativeMemory with CRP allocation (concentration α)

## Motor Primitives (zebrafish_env.py)
- Bout types: IDLE, ROUTINE_FWD/L/R, J_TURN, BURST, C_START, CAPTURE, ESCAPE, FLEE_BURST
- Brain controls DIRECTION (turn_rate), motor system controls TIMING (burst-glide-idle)
- Biological noise: σ=0.03 during burst, σ=0.015 during idle (fidgeting)
- Goal-modulated IBI: FLEE=1, FORAGE=2, EXPLORE/SOCIAL=3 steps
- Urgent stimuli + starvation interrupt IBI immediately

## Binocular Depth + Predator Speed
- `compute_binocular_depth()` in retina_sampling_v60.py
- Mono intensity + stereo overlap → per-entity depth estimate
- Predator speed from binocular depth temporal changes → TTC improvement
- Looming detector: l/v ratio < 10 triggers Mauthner C-start

## Escape Reflex (Mauthner C-start)
- Looming trigger → 4-step stereotyped escape (C-bend 1.5 rad + propulsive 1.6x)
- 12-step refractory period, away from enemy lateral bias
- Prey capture: J-TURN (3 steps) → APPROACH (5) → STRIKE (2, 1.4x eat radius)

## Optimal Foraging (density-based)
- Local density = neighbors within 80px radius
- net_value = density + gain + urgency + proximity² - distance - risk - occlusion
- Nearest food in highest-value patch pursued first

## Social Learning (shoaling_v60.py)
- observe_social_cues(): infer danger/food from conspecific behavior
- social_alarm: fast-moving neighbours → boost p_enemy (flee signal)
- social_food_bearing: slow neighbours clustered → food patch direction
- Always active (not just during SOCIAL goal)

## Decision Rationality (Step 29)
- 6 metrics: foraging efficiency, patch selection, flee timing, path efficiency, energy management, threat response
- 5 structured scenarios: safe/risky, occluded/open, predator charge, starvation dilemma, detour
- Overall: 83/100 RATIONAL (A=90, B=80, C=80, D=100, E=65)

## Curriculum Training (Step 30)
- 4 stages: FORAGE_EASY → FORAGE_HARD → SURVIVE_EASY → SURVIVE_HARD
- Current: 3/4 COMPETENT (passes foraging + easy survival, fails full env)

## Interactive Commands (during --render demo)
- P: predator ATTACK (charge at fish), R: predator RETREAT, F: spawn FOOD cluster

## Obstacle Navigation (brain_agent.py section 12b-12d)
- **Bilateral repulsion**: steer away from rock-heavy eye, 1.5x amplification above 15 pixels
- **Center-escape**: post-gain escape turn when rock centered ahead (|L-R| < 5, total > 20)
- **Stuck detection**: 8+ steps barely moving near obstacle → force EXPLORE
- **Occluded food exclusion**: skip food behind rocks when urgency < 0.5
- **Physics rebound**: heading deflection gain 0.7 on collision
- **Foraging override**: preserved for moderate coverage (<70%), removed at dense walls

## Bayesian Survival Trade-off (goal_policy_v60.py)
- `starvation_risk = max(0, (0.50 - energy_ratio) / 0.50)` modulates all 4 goals
- FORAGE urgency increases, FLEE/EXPLORE/SOCIAL become costly when starving
- When both predator + starvation active: Bayesian model comparison via EFE softmax
- Starvation mechanics: 1.3x metabolic cost below 30%, 1.15x below 50% energy
- Speed cap: below 20% energy, max speed scales down linearly

## Running
```bash
# Full demo
.venv/bin/python -m zebra_v60.gym_env.demo --brain --render --monitor --record --sound --steps 500

# Evaluation
.venv/bin/python -m zebra_v60.tests.step19_full_evaluation

# Training (in order)
.venv/bin/python -m zebra_v60.tests.step8_genomic_pretraining
.venv/bin/python -m zebra_v60.tests.step10_hebbian_finetuning
.venv/bin/python -m zebra_v60.tests.step11_object_classification
.venv/bin/python -m zebra_v60.tests.step26_wfb_pe_training
```

## Branches
- `master` — working v60 with steps 1-30 (PC, AIF, Bayesian brain, motor primitives, social learning, curriculum)
