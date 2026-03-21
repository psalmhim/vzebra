# AI Context — vzebra Project

This file contains all context needed for an AI assistant to continue
working on this project. Load this as the first instruction.

## Project Summary

Virtual zebrafish brain simulation with 42 developmental steps, 5,823
spiking neurons, 36+ neural modules. The fish lives in a 2D arena,
foraging for food while avoiding a predator. All behavior emerges from
the neural architecture — no hardcoded behavior scripts.

## Current Scores
- Decision quality: 84/100 RATIONAL
- Classifier: 90.6% (retrained on gameplay data)
- Curriculum: 2-3/4 (stochastic)
- Online RL: learning confirmed (food doubled over 100 episodes)

## Architecture (42 Steps)

### SNN (5,635 neurons)
- PredictiveTwoComp: two-compartment (soma V_s + apical V_a), PE = V_a - V_s
- Layers: Retina(1600) → OT_L/R(1200) → OT_F(800) → PT(400) → PC_per(120) → PC_int(30) → motor(200)/eye(100)/DA(50)
- Deep layers have signal death (PC_int RMS=0.002, motor=0.0001)
- Reticulospinal shortcut: OT_L→motor_R (crossed), learns via STDP from zero
- Homeostatic gain: unidirectional (suppress only). Bidirectional was tried but degraded behavior.
- Attention modulator: 8 neurons, goal→precision weighting

### Motor (32 LIF neurons)
- Spinal CPG: half-centre oscillator, 8 V2a + 4 V0d + 4 motor per side
- PRIMARY turn signal: retinal L/R balance (NOT SNN motor output)
- During FLEE: 3x turn rate, 1.5x speed
- Population vector decoding from 200 motor neurons (Phase 1)

### Perception
- Retina: 800 pixels/eye (400 intensity + 400 type), 7 entity types
- Classifier: 804→128→5 (nn.Linear, retrained on gameplay data)
- Distance-proportional threat: proximity² curve, looming from growth rate
- Lateral line: 16 neuromasts, dipole flow, reafference cancellation
- Olfaction: bilateral nostrils, food gradient + alarm substance
- Color vision: UV/blue/green/red cone channels
- Vestibular: otolith + semicircular canal

### Decision Making
- Goal policy: EFE minimization, 4 goals (FORAGE/FLEE/EXPLORE/SOCIAL)
- Spiking goal selector: WTA attractor (4 neurons)
- Habenula: frustration → strategy switching
- Flee: enemy_lateral_bias → turn AWAY, forced turn when centered

### World Models
- Predator model: Kalman filter, object permanence, intent inference
  - Non-AI mode: ground truth + 10px noise (12px error)
  - AI mode: retinal-only (50-100px error when visible)
- Geographic model: 40×30 grid, obstacle/food density, epistemic exploration
- Internal state: energy trajectory simulator, learned metabolic costs

### Neuromodulation & Emotion
- Dopamine: TD error, tonic/phasic (spiking version available)
- Amygdala: fear circuit with episodic conditioning (LTP on near-death)
- Allostasis: hunger/fatigue/stress tracking
- Insula: heart rate → arousal/fear/valence, emotional EFE bias
- Circadian: 24h oscillator, day/night background

### Learning
- Step 8: genomic pretraining (supervised, 120-300 epochs)
- Step 10: Hebbian (RPE-gated, online)
- Step 11: classifier (gameplay data, 90.6%)
- Step 26: W_FB (PE minimization, frozen W_FF)
- STDP: three-factor (pre × post × dopamine) on reticulospinal
- Online RL: shaped reward, checkpoint persistence

### Multi-Agent
- 5 fish: 1 focal (full BrainAgent) + 4 conspecific (lightweight)
- Social alarm cascading: fleeing neighbours → group flee
- Predator confusion: nearest-fish targeting with hysteresis

## Speed Parameters
- Fish: base 3.0 px/step (1x), flee 4.5 (1.5x)
- Predator: patrol 2.7 (0.9x), hunt max 4.2 (1.4x)
- Predator imperfection: 15px noise, 5% distraction, 0.12 turn limit
- Eat radius: 35px
- Flee turn boost: 3x (0.45 rad/step)

## Key Files
- `zebrav1/gym_env/brain_agent.py` — main pipeline (~2500 lines)
- `zebrav1/brain/zebrafish_snn.py` — SNN model
- `zebrav1/gym_env/zebrafish_env.py` — Gymnasium environment
- `zebrav1/dashboard/server.py` — web dashboard
- `manuscript.tex` — Neural Computation paper (32 pages)
- `zebrav1/paper.tex` — technical report (68 pages)

## Known Issues
1. SNN deep layers dead (signal death at PC_per→PC_int)
2. Fish reads own position from env (cheating in default mode)
3. Classifier outputs flat probs on mixed scenes (raw pixel count used as backup)
4. Step 11 training takes ~1hr on MPS (~20min on A100)
5. Predator model diverges when predator invisible in AI mode

## What Needs Work
- SNN Phase 2: STDP on all layers, layer-wise PE minimization
- SNN Phase 3: oscillations, Izhikevich neurons, sparse coding
- More RL training (1000+ episodes on GPU)
- Web demo (PyScript/Pyodide)
- Full active inference mode (no env reads)

## Git
- Remote: github.com/psalmhim/vzebra, branch: main
- Tag: v1.0
- No conventional commit prefixes
- .gitignore: __pycache__, *.pt, atlas/*.nii.gz, .venv/

## Running
```bash
# Demo
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --monitor --steps 500

# Training
.venv/bin/python -m zebrav1.tests.step8_genomic_pretraining
.venv/bin/python -m zebrav1.tests.step43_online_rl

# Dashboard
.venv/bin/python -m zebrav1.dashboard

# Flags: --spiking --sound --multi-agent --record --autosave
```
