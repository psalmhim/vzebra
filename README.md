# vzebra — Virtual Zebrafish Brain Simulation

A biologically grounded spiking neural network simulation of the larval zebrafish brain, implementing 42 developmental steps from retinal sampling to emotional awareness.

## Overview

The virtual zebrafish lives in a 2D prey-predator arena, foraging for food while avoiding a hunting predator. Its brain consists of 36+ neural modules spanning sensory processing, motor control, decision-making, world modeling, and emotional regulation — all based on real zebrafish neuroscience.

**Key capabilities:**
- Binocular vision with 800 pixels per eye
- Spiking neural network with predictive coding (two-compartment neurons)
- Active inference goal selection (FORAGE / FLEE / EXPLORE / SOCIAL)
- Three structured world models (geographic, predator, internal state)
- Lateral line mechanosensation + olfaction
- Cerebellum motor learning + spinal CPG (32 LIF neurons)
- Heart rate + insula emotional awareness
- Multi-agent schooling (5-fish environment)
- Online reinforcement learning (improves over episodes)

## Quick Start

```bash
# Setup
python3.12 -m venv .venv
.venv/bin/pip install torch numpy matplotlib gymnasium pygame imageio imageio-ffmpeg

# Run demo (single fish + neural monitor)
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --monitor --steps 500

# Multi-agent (5 fish)
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --monitor --multi-agent --steps 500

# Record video
.venv/bin/python -m zebrav1.gym_env.demo --brain --render --record --monitor --steps 1000

# Interactive controls during --render:
#   P = predator ATTACK    R = predator RETREAT    F = spawn FOOD
```

## Training

```bash
# Full training pipeline (run in order)
.venv/bin/python -m zebrav1.tests.step8_genomic_pretraining     # ~5 min
.venv/bin/python -m zebrav1.tests.step10_hebbian_finetuning     # ~3 min
.venv/bin/python -m zebrav1.tests.step11_object_classification  # ~60 min
.venv/bin/python -m zebrav1.tests.step11b_gameplay_classifier   # ~30 min (gameplay data)
.venv/bin/python -m zebrav1.tests.step26_wfb_pe_training        # ~5 min

# Online RL (improves over episodes)
.venv/bin/python -m zebrav1.tests.step43_online_rl
```

## Evaluation

```bash
.venv/bin/python -m zebrav1.tests.step29b_decision_scenarios    # 84/100 RATIONAL
.venv/bin/python -m zebrav1.tests.step30_curriculum_motor       # 3/4 COMPETENT
.venv/bin/python -m zebrav1.tests.step31_structured_world_models # 8/8 PASS
```

## Architecture

42 developmental steps building from basic vision to emotional awareness:

| Steps | Modules |
|-------|---------|
| 1-7 | Retina, optic tectum, precision, dopamine, basal ganglia, thalamus, locomotion |
| 8-11 | Genomic pretraining, Hebbian learning, object classification |
| 12-14 | Reactive behavior, active inference policy, Gymnasium integration |
| 15-19 | RL critic, VAE world model, place cells, allostasis, full evaluation |
| 20-24 | Amygdala, food prospection, curriculum, sleep-wake, active inference |
| 25-26 | Predictive coding (two-compartment), feedback weight training |
| 27-30 | Binocular depth, motor primitives, decision rationality, curriculum |
| 31 | Structured world models (geographic + predator + internal state) |
| 32-34 | Lateral line, cerebellum, multi-agent dynamics |
| 35-36 | Olfaction + alarm substance, habenula (behavioral flexibility) |
| 37-41 | Vestibular, spinal CPG, color vision, circadian clock, proprioception |
| 42 | Insula (heart rate + emotional awareness) |

## Performance

- **Decision quality:** 84/100 RATIONAL
- **Foraging:** 11 food / 500 steps
- **Survival:** 500+ steps (stochastic)
- **Classifier:** 90.6% on gameplay data
- **Online RL:** doubles food intake over 20 episodes

## Rendering

- 10-segment articulated fish body with gaze-tracking eyes
- Predator with jaw, danger aura, state label (PATROL/STALK/HUNT)
- Pulsing food glow, eating sparkle animation
- Energy bar, emotion icons, speed trail
- Heartbeat glow synchronized with heart rate
- Circadian day/night cycle background
- Water current particles, sandy shore border
- Neural monitor: retinal heatmaps, spike raster, classification bars, CPG oscillation, HR pulse, mini-map, goal timeline, energy sparkline

## Paper

65-page technical report: `zebrav1/paper.tex` / `zebrav1/paper.pdf`

## License

Research use. Contact author for commercial licensing.
