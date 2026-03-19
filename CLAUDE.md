# vzebra — Zebrafish Brain Simulation

## Environment
- Python: `.venv/bin/python` (Python 3.12)
- Setup: `python -m venv .venv && .venv/bin/pip install torch numpy matplotlib gymnasium pygame imageio imageio-ffmpeg`
- If Dropbox sync breaks .so: `rm -rf .venv` and recreate
- Device: MPS (Apple Silicon) via `get_device()` in `brain/device_util.py`

## Architecture
- `zebrav1/` — main package (35+ brain modules, renamed from zebra_v60)
- `zebrav1/brain/` — SNN, world models, sensory, motor, neuromodulatory modules
- `zebrav1/gym_env/` — Gymnasium env + BrainAgent + MultiAgentEnv
- `zebrav1/world/` — WorldEnv (ray-casting) + renderer
- `zebrav1/tests/` — step-by-step tests (step1 through step34)
- `zebrav1/viz/` — neural monitor (920px, shows all 41 steps)
- `zebrav1/paper.tex` — 65-page technical report (Steps 1-41)

## Key Files
- `brain_agent.py` — main pipeline (~2500 lines), orchestrates all modules
- `zebrafish_snn.py` — SNN with PredictiveTwoComp + reticulospinal shortcut (disabled)
- `zebrafish_env.py` — Gymnasium env with circadian background, muscle-driven tail
- `multi_agent_env.py` — 5-fish environment (1 focal + 4 conspecific brains)
- `spinal_cpg.py` — 32-neuron spiking half-centre oscillator (LIF)

## SNN Architecture & Motor Pathway
- Layers: OT_L/OT_R → OT_F → PT_L → PC_per → PC_int → mot/eye/DA
- **Signal death**: deep layers (PC_int, motor) have near-zero activity (RMS 0.002)
- **Primary turn signal**: retinal L/R balance (retR_sum - retL_sum) — NOT SNN motor output
- **Reticulospinal shortcut**: OT_L→motor_R, OT_R→motor_L (DISABLED — random weights add noise)
- **Homeostatic gain**: unidirectional (suppress only). Bidirectional boost was tried but degraded behavior.
- **SNN motor neurons**: alive via reticulospinal (RMS ~1.8 when enabled) but untrained for correct L/R
- **Next step**: train reticulospinal weights with supervised motor targets

## Spinal CPG (Step 38)
- 32 LIF neurons: 8 V2a excitatory + 4 V0d inhibitory + 4 motor per side
- Brain provides tonic drive (speed) + turn bias → CPG adds phasic L/R oscillation
- CPG glide phase bypassed during FLEE (maintain full escape speed)
- Muscle L/R drives tail rendering asymmetry

## Modules (Steps 31-41)
- Geographic model, Predator model, Internal state model (Step 31)
- Lateral line (Step 32), Cerebellum (Step 33), Multi-agent (Step 34)
- Olfaction + alarm substance (Step 35), Habenula (Step 36)
- Vestibular (Step 37), Spiking CPG (Step 38), Color vision (Step 39)
- Circadian clock (Step 40), Proprioception (Step 41)

## Training Pipeline
1. **Step 8** genomic → `weights/genomic.pt` (74% direction accuracy)
2. **Step 10** Hebbian → `weights/genomic_hebbian.pt`
3. **Step 11** classifier → `weights/classifier.pt` (100% all 5 classes)
4. **Step 26** W_FB → `weights/classifier_wfb.pt`
- Step 11 is slow (~1hr on MPS)
- All steps MUST pass `goal_probs` to `model.forward()`

## Checkpoint System
- `save_checkpoint`: SNN, habit, critic, VAE, place cells, Hebbian, geographic, internal state, cerebellum, habenula
- `load_checkpoint`: restores all learned state
- Use `--autosave` flag in demo to persist online learning

## Current Scores
- Classifier: 100% all classes
- Curriculum: 2/4 LEARNING (regression from architecture changes)
- Decision quality: ~60/100 (regression from SNN retrain)
- Known issue: Scenario C (predator charge) — fish flees but can't outrun predator

## Flee Behavior
- Hard flee threshold: p_enemy > 0.25
- Pixel-evidence cap: enemy_pixels / 15
- Flee exit: p_enemy < 0.10 for 5 steps
- Looming trigger: l/v < 10, enemy_px > 3

## Rendering
- Circadian day/night background color (Step 40)
- Muscle-driven tail oscillation with L/R bend asymmetry (Step 38)
- Vestibular tilt effect on fish size (Step 37)
- Proprioceptive collision flash (Step 41)
- Neural monitor: CPG L/R, color UV/B/G/R, circadian dial, LL flow, olfaction, habenula, cerebellum PE

## Running
```bash
# Demo (single fish)
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --monitor --steps 500

# Multi-agent (5 fish)
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --monitor --multi-agent --steps 500

# Record video
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --record --monitor --steps 1500

# Training (in order, step11 is ~1hr)
.venv/bin/python -m zebrav1.tests.step8_genomic_pretraining
.venv/bin/python -m zebrav1.tests.step10_hebbian_finetuning
.venv/bin/python -m zebrav1.tests.step11_object_classification
.venv/bin/python -m zebrav1.tests.step26_wfb_pe_training

# Evaluation
.venv/bin/python -m zebrav1.tests.step29b_decision_scenarios
.venv/bin/python -m zebrav1.tests.step30_curriculum_motor
```

## Branches
- `main` — Steps 1-41, 35+ modules, 65-page paper
