# ZebrafishBrainV2 System Architecture

## Overview

| Property | Value |
|----------|-------|
| Total SNN neurons | 7,158 (Izhikevich) + 160 (LIF) |
| Total modules | 35 |
| SNN modules | 27 |
| Rate/analytic modules | 8 |
| Neuron model | Izhikevich (midpoint integration, 1ms timestep) |
| Cell types | RS, IB, FS, CH, LTS, TC, MSN (7 types) |
| Substeps per behavioral step | 50 (= 50ms) |
| Synapse types | AMPA, NMDA, GABA_A (conductance-based) |
| E/I ratio | 75% excitatory / 25% inhibitory per layer |
| Neuromodulation | 4-axis (DA, NA, 5-HT, ACh) |
| Goal policies | 4 (FORAGE, FLEE, EXPLORE, SOCIAL) |
| Device | MPS (Apple Silicon) / CUDA / CPU |

---

## Signal Flow: Sensory → Motor

### Stage 1: Sensory Input

| Module | Type | Neurons | Input | Output | Key Mechanism |
|--------|------|---------|-------|--------|---------------|
| Sensory Bridge | Analytic | — | Environment geometry | 2×800 retinal arrays (L,R) | Geometric ray-casting, 200° FoV |
| Retina (RetinaV2) | Rate model | 1000/eye | 800-dim retinal arrays | ON, OFF, Looming, DS rates | DoG filter, Reichardt detector, l/v ratio |
| Classifier (ClassifierV2) | **SNN (LIF)** | **128** | 804-dim (type pixels + aggregates) | 5 class probs | LIF + lateral inhibition, hybrid spike+analog |
| Lateral Line | Analytic | — | Predator position | Distance estimate (<150px) | Mechanoreceptive, noisy |

#### Retina RGC Types

| RGC Type | Count/eye | Neuron Type | Function |
|----------|-----------|-------------|----------|
| ON-sustained | 400 | Rate | Luminance, center-surround |
| OFF-transient | 400 | Rate | Brightness decrease detection |
| Looming | 100 | Rate | Expanding object (l/v ratio) |
| Direction-selective | 100 | Rate | Motion detection (Reichardt) |

#### Classifier Classes

| Class ID | Name | Type Value | Detection |
|----------|------|------------|-----------|
| 0 | nothing | — | No significant pixels |
| 1 | food | > 0.7 | Green food items |
| 2 | enemy | ≈ 0.5 | Predator |
| 3 | colleague | — | Conspecific fish |
| 4 | environment | ≈ 0.75 | Rocks, obstacles |

---

### Stage 2: Tectum (Optic Tectum)

| Layer | Neurons | E/I | Cell Type | Input | Function |
|-------|---------|-----|-----------|-------|----------|
| SFGS-b | 1200 | 900E + 300I | CH (chattering) | ON-sustained RGC | Visual object processing |
| SFGS-d | 1200 | 900E + 300I | CH | OFF-transient RGC | Motion/change detection |
| SGC | 400 | 300E + 100I | IB (bursting) | Looming RGC | Looming detection → escape |
| SO | 400 | 300E + 100I | RS (regular) | Direction-selective RGC | Direction of motion |
| **Total** | **3200** | | | | **4 E/I layers, 50ms spiking** |

---

### Stage 3: Thalamo-Pallial Loop

| Module | Neurons | Cell Type | Input | Output | Key Mechanism |
|--------|---------|-----------|-------|--------|---------------|
| Thalamus TC | 300 | LTS | Tectum SFGS-b | Pallium-S relay | NA-gated (wake/sleep), burst/tonic |
| Thalamus TRN | 80 | FS | Pallium-S feedback + TC | TC inhibition | Attentional gating |
| Pallium-S | 1600 (1200E+400I) | RS | TC relay + goal attention | Sensory representation | Two-compartment predictive coding |
| Pallium-D | 800 (600E+200I) | IB | Pallium-S feedforward | Goal/intent representation | Apical feedback → prediction error |
| **Total** | **2780** | | | | **Predictive coding: PE = apical - somatic** |

#### Pallium Predictive Coding

| Component | Description |
|-----------|-------------|
| Somatic compartment | Bottom-up sensory (thalamic relay) |
| Apical compartment | Top-down prediction (pallium-D → pallium-S via W_FB) |
| Prediction error | PE = apical - somatic (drives learning) |
| W_FB learning | Anti-Hebbian: ΔW = -η × h_upper^T × ε |
| ACh modulation | Attention gate on goal → pallium projection |

---

### Stage 4: Goal Selection (Active Inference + SNN)

#### Expected Free Energy (EFE) Computation

| EFE Input | Source Module | Source Type | Contribution |
|-----------|-------------|-------------|--------------|
| p_food | Classifier | SNN (LIF) | Food probability → lower G_forage |
| p_enemy | Retina + Amygdala + Tectum | SNN + Rate | Threat estimate → lower G_flee |
| Starvation | Allostasis | Analytic | Energy deficit → lower G_forage |
| CMS (novelty) | Amygdala | SNN | Contextual modulation |
| 5-HT patrol bias | Neuromod | Rate | Patience → suppress G_flee |
| Place cell bonus | Place Cells | Rate | Food/risk memory → G_forage/G_flee |
| Allostatic bias | Allostasis | Analytic | Hunger/fatigue/stress → per-goal |
| World model EFE | Internal World Model | Analytic | Predicted outcomes per policy |
| Interoceptive bias | Insular Cortex | SNN (Izhikevich) | Spiking hunger/stress → per-goal |
| Cerebellum PE | Cerebellum | SNN (Izhikevich) | High PE → increase G_explore |
| Habenula frustration | Habenula | SNN (Izhikevich) | Frustrated goals penalized |

#### EFE Formula

| Goal | G(π) Formula |
|------|-------------|
| FORAGE | 0.2U - 0.8·p_food + 0.15 - 1.5·starvation + allostatic + interoceptive + place + world_model + habenula |
| FLEE | 0.1·CMS - 0.8·p_enemy + 0.20 + 0.8·starvation + 5-HT_bias + allostatic + interoceptive + habenula |
| EXPLORE | 0.3U - 0.1 + cerebellum_PE + allostatic + interoceptive + habenula |
| SOCIAL | 0.25 + allostatic + habenula |

#### Spiking WTA Goal Selector

| Property | Value |
|----------|-------|
| Neurons | 4 Izhikevich RS (one per goal) |
| Self-excitation | w = 4.0 (attractor dynamics) |
| Mutual inhibition | w = -2.0 (winner-take-all) |
| Input | Pallium-D rates + EFE bias |
| Decision rule | If WTA confidence > 0.4 → SNN winner, else analytic EFE argmax |

#### Reflexive Overrides (Hard-Wired)

| Override | Condition | Action |
|----------|-----------|--------|
| Food reflex | food_px > 1 and not fleeing | Force FORAGE (20 step lock) |
| Hard flee | p_enemy > 0.25 and not starving | Force FLEE |
| Starvation forage | starvation > 0.35 and not hunted | Force FORAGE |
| Extreme starvation | starvation > 0.7 and pred far | Force FORAGE |
| Proximity flee | pred_dist < 60 | Force FLEE |
| Stuck detection | no food for 30 steps | Force EXPLORE (15 steps) |

---

### Stage 5: Basal Ganglia Action Gating

| Component | Neurons | Function | DA Modulation |
|-----------|---------|----------|---------------|
| D1 MSNs | 400 | Direct pathway (Go) | Excited by DA |
| D2 MSNs | 300 | Indirect pathway (NoGo) | Inhibited by DA |
| GPi | 60 | Inhibitory output | — |
| Gate output | scalar [0,1] | 1 = Go, 0 = NoGo | High DA → D1 dominates → Gate opens |

---

### Stage 6: Motor Output

#### Goal-Specific Motor Computation

| Goal | Turn Source | Speed | Smoothing α |
|------|-----------|-------|-------------|
| FLEE | GT angular correction (away from predator) | 1.5× | 0.6 (fast) |
| FORAGE | Retinal food bearing (L/R pixel weighted) | 1.0-1.2× | 0.5 (responsive) |
| EXPLORE | Sinusoidal + retinal + scanning saccade | 0.8× | 0.25 (gentle) |
| SOCIAL | Retinal approach | 0.7× | 0.30 |

#### Motor Pathway

| Module | Type | Neurons | Function |
|--------|------|---------|----------|
| Active Inference Motor | **SNN (Izh 2-comp)** | **48** | Action-perception cycle: 8 proprioceptive channels × 6 neurons, iterative inference (3 passes/step) |
| Reticulospinal | Rate model | 21/side | Named neurons (Mauthner, MiD2/3, RoM2, MeM, CaD), gap junction coupling |
| Mauthner cell | Rate | 1/side | C-start escape (looming trigger, 4-step sequence), gap junction facilitation (threshold 0.05→0.04) |
| Spinal CPG | **SNN (LIF)** | **32** | Half-centre oscillator (8 V2a + 4 V0d + 4 MN per side) |
| Wall avoidance | Analytic | — | Angle-to-center proportional correction |

#### Action-Perception Cycle (Friston 2011)

Motor commands are proprioceptive **predictions** (μ) — the body moves to fulfil them via the spinal reflex arc. Within each step, 3 iterative inference passes refine predictions:

```
μ^(k) ← μ^(k-1) − η_act × ε^(k)    (η_act = 0.15)
```

**Adaptive blend**: `α_AI = clamp(0.3 + 0.4*π̄ − 0.3*ξ, 0.1, 0.8)` where π̄ = mean motor precision, ξ = spontaneity.

**FE-gradient goal modulation**: rising F penalizes current goal (dF > 0.05 → +0.3·dF to EFE).

**Spontaneity** (anti-FEP): habenula frustration, boredom, DA exploration, play — temporarily breaks FE minimization.

#### Final Motor Command

```
turn_rate = -brain_turn × brain_weight + wall_turn
speed = goal_speed × allostatic_fatigue_cap
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| brain_weight | max(0.2, 1 - wall_urgency) | Reduces brain turn near walls |
| wall_turn | urgency × 0.8 × sign(angle_to_center) | Pushes toward arena center |
| Flee turn_max | 3× normal (0.45 rad/step) | Faster turning during escape |

---

### Stage 7: Subcortical / Limbic Modules

| Module | Type | Neurons | Cell Types | Function |
|--------|------|---------|------------|----------|
| Amygdala | **SNN** | **50** | LA(20 RS) + CeA(20 IB) + ITC(10 FS) | Fear circuit, episodic LTP, threat arousal |
| Habenula | **SNN** | **50** | LHb(30 RS) + MHb(20 RS) | Disappointment, per-goal frustration, strategy switch |
| Cerebellum | **SNN** | **270** | GC(200 RS) + PC(50 IB) + DCN(20 RS) | Forward model, parallel fiber LTD, prediction error |
| Insular Cortex | **SNN** | **34** | 3×10 channels + 4 valence (RS) | Hunger/fatigue/stress encoding, heart rate, valence |
| Pretectum | **SNN** | **60** | 30 RS per hemisphere | OKR, direction-selective, retinal slip → image stabilization |
| IPN | **SNN** | **24** | vIPN(12 RS) + dIPN(12 RS) | Habenula relay, behavioral inhibition, DA/5-HT feedback |
| Raphe | **SNN** | **40** | DR(30 RS) + MR(10 RS) | Population-coded 5-HT, overrides scalar neuromod.HT5 |
| Locus Coeruleus | **SNN** | **20** | 20 RS (tonic/phasic) | Population-coded NA, overrides scalar neuromod.NA |
| Pectoral Fin | **SNN** | **8** | 4 RS per side | Slow-turn kinematics, goal-gated (suppressed in FLEE) |
| Tectal Habituation | Analytic | — | Per-synapse depletion | Short-term synaptic depression on SFGS-b input |

#### Amygdala Detail

| Nucleus | Neurons | Type | Connectivity | Function |
|---------|---------|------|-------------|----------|
| LA (Lateral) | 20 | RS excitatory | Sensory → LA | Threat input processing |
| CeA (Central) | 20 | IB bursting | LA → CeA (Hebbian LTP) | Fear output |
| ITC (Intercalated) | 10 | FS inhibitory | LA → ITC → CeA | Extinction gate |

#### Habenula Detail

| Feature | Description |
|---------|-------------|
| LHb spiking | Fires on negative RPE (reward < expected) |
| MHb spiking | Fires on aversion (amygdala threat) |
| Per-goal frustration | 4-dim vector, accumulates from negative RPE |
| Strategy switch | frustration > 0.4 → force goal change (15-step cooldown) |
| DA suppression | LHb rate × 5.0 (capped at 0.5) |
| 5-HT suppression | LHb rate × 3.0 (capped at 0.3) |

#### Cerebellum Detail

| Component | Neurons | Type | Function |
|-----------|---------|------|----------|
| Granule cells | 200 | RS (high threshold) | Sparse combinatorial encoding (~16% active) |
| Purkinje cells | 50 | IB | Inhibitory output, supervised by climbing fiber |
| Deep cerebellar nuclei | 20 | RS | Output, disinhibited when PC pauses |
| Climbing fiber | — | Signal from pallium PE | Error signal → parallel fiber LTD |
| Parallel fiber LTD | — | GC→PC weights | CF + GC active → weaken synapse (supervised learning) |

---

### Stage 8: Learning Systems

#### Reinforcement Learning

| Component | Module | Type | Mechanism |
|-----------|--------|------|-----------|
| Value estimation | RL Critic | **SNN (64 RS + 4 RS)** | V(s) per goal from state features |
| TD error | RL Critic | **SNN** | δ = r + γV(s') - V(s), γ=0.95 |
| Eligibility traces | RL Critic + Plasticity | **SNN** | τ_elig = 500ms, Hebbian co-activation |
| Consolidation | RL Critic | **SNN** | ΔW = lr × δ × DA × eligibility |
| RPE → DA | Neuromod | Rate | DA = sigmoid(3 × RPE) |
| Habit formation | Habit Network | **SNN (32+8)** | Hebbian LTP from repeated stimulus→action |

#### Active Inference

| Component | Module | Type | Mechanism |
|-----------|--------|------|-----------|
| Belief state μ | VAE Encoder | Rate + Gradient | pooled_tectum(64) + state_ctx(13) → z(16) |
| Generative model | VAE Decoder | Rate + Gradient | z(16) → reconstructed_tectum(64) |
| ELBO training | VAE | Gradient descent | recon_loss + β × KL_divergence, online ring buffer |
| Transition model | VAE | Rate + Gradient | z' = z + 0.3 × delta(z, action_ctx) |
| Planning | VAE | Rate | 3 goals × 3-step rollout → G_plan[3] |
| Associative memory | VAE | Analytic (RBF) | 64 nodes: z → (food_rate, risk) |
| Prediction error | Pallium | **SNN** | PE = apical - somatic (two-compartment) |
| W_FB learning | Feedback PE | Analytic | ΔW_FB = -η × h_upper^T × ε (anti-Hebbian) |
| Spiking prediction | Predictive Net | **SNN (80+80+32)** | Retinal → latent → predicted next frame |
| Surprise signal | Predictive Net | **SNN** | MSE of prediction error → epistemic value |

#### Adaptive EFE Parameters (MetaGoalWeights)

| Component | Parameters | Update Rule | Range |
|-----------|-----------|-------------|-------|
| Goal bias | 4 scalars (b_FORAGE…b_SOCIAL) | REINFORCE, mean log-prob, normalized advantage | [-0.5, 0.5] |
| Modulation weights | 8 scalars (wm, social, geo, novelty, vae, circ, cb, urgency) | Fitness-correlation heuristic, active sources only | [0.1, 3.0] |
| Fitness baseline | EMA (τ=0.95), bootstrapped from first episode | — | — |
| Advantage normalization | Running variance EMA (τ=0.95) | Prevents scale-dependent gradient | — |

Updates run once per episode at `brain.on_episode_end(fitness)`. All parameters saved in checkpoint under `meta_goal` key.

#### Social Memory (SocialMemory)

| Weight | Init | Update Rule | Converges To |
|--------|------|-------------|--------------|
| w_alarm | 1.0 | Precision-EMA over 20-step horizon window | 0.2 (false alarms) – 3.0 (reliable) |
| w_food_cue | 1.0 | Precision-EMA over 30-step horizon window | 0.2 (useless) – 3.0 (reliable) |
| w_competition | 1.0 | (future: episode fitness correlation) | [0.2, 3.0] |

Social state inference:
- `speed < 0.30` → eating (food cue trigger)
- `speed > 1.50` → fleeing (alarm trigger)
- ≥2 non-fleeing within 100px → competition penalty on G_FORAGE

#### Predictive Coding (Pallium)

| Component | Layer | Direction | Signal |
|-----------|-------|-----------|--------|
| Somatic input | Pallium-S | Bottom-up | Thalamic relay (sensory) |
| Apical input | Pallium-S | Top-down | Pallium-D via W_FB (prediction) |
| Prediction error | Pallium-S | Lateral | PE = apical - somatic |
| Learning | W_FB | Online | Anti-Hebbian: minimize PE |
| Free energy | Pallium | Scalar | mean(PE²) — drives exploration |

---

### Stage 9: Neuromodulation

| Axis | Symbol | Source | Target | Function | Update Rule |
|------|--------|--------|--------|----------|-------------|
| Dopamine | DA | VTA analogue | BG (D1↑ D2↓), Critic | Reward prediction error | DA = sigmoid(3 × RPE), suppressed by Habenula LHb |
| Noradrenaline | NA | LC analogue | Thalamus TC, global gain | Arousal, threat response | NA ← 0.3 + 0.5×amygdala + 0.2×CMS |
| Serotonin | 5-HT | Raphe analogue | EFE (flee suppression) | Patience, habituation | 5-HT rises when not fleeing, suppressed by Habenula |
| Acetylcholine | ACh | BF analogue | Pallium attention, plasticity | Attention gate | ACh ← circadian × (1-fatigue) × (1+CMS) |

---

### Stage 10: Spatial / Interoceptive Memory

| Module | Type | Size | Function |
|--------|------|------|----------|
| Place Cells | Rate model | 128 cells | Gaussian place fields, theta (8Hz), phase precession |
| Place cell memory | Rate model | food_rate, risk_rate per cell | Food/risk maps → EFE forage/flee bonus |
| Predator Model | Analytic (Bayesian) | 5-state [x,y,vx,vy,intent] | Kalman filter, object permanence, TTC estimation |
| Allostasis | Analytic | 3 channels | Hunger/fatigue/stress setpoint tracking, goal bias |
| Internal World Model | Analytic | — | Predicts energy trajectory, threat level per goal |

---

## Module Count Summary

| Category | SNN (Izhikevich) | SNN (LIF) | Rate Model | Analytic | Total |
|----------|-------------------|-----------|------------|----------|-------|
| Sensory | Tectum (3200) | Classifier (128) | Retina (2000) | Sensory Bridge, Lateral Line | 5 |
| Thalamo-Pallial | Thalamus (380), Pallium (2400) | — | — | — | 2 |
| Goal Selection | Goal Selector (4) | — | — | EFE computation | 2 |
| Action Gating | — | — | BG (760) | — | 1 |
| Motor | — | CPG (32) | RS (42) | Wall avoidance | 3 |
| Subcortical | Amygdala (50), Habenula (50), Cerebellum (270), Insula (34), Pretectum (60), IPN (24), Raphe (40), LC (20), Pectoral Fin (8) | — | — | Habituation | 10 |
| Learning | Critic (68), Predictive (192), Habit (40) | — | — | W_FB, ELBO | 5 |
| Memory | — | — | Place Cells (128) | Predator Model, Allostasis, World Model | 4 |
| Neuromod | — | — | DA/NA/5-HT/ACh | — | 1 |
| World Model | — | — | — | VAE (8600 params) | 1 |
| **Total** | **6,840 neurons** | **160 neurons** | **2,930 neurons** | **8 modules** | **34 files** |

---

## Paradigm Usage Summary

| Paradigm | Where Used | Key Modules |
|----------|-----------|-------------|
| **SNN (Izhikevich)** | Sensory→motor processing, fear, goal selection, value, prediction, habits, interoception, forward model, disappointment, OKR, behavioral inhibition, 5-HT, NA, slow turns | Tectum, Thalamus, Pallium, Amygdala, Goal Selector, Cerebellum, Habenula, RL Critic, Predictive Net, Habit Net, Insula, Pretectum, IPN, Raphe, LC, Pectoral Fin |
| **SNN (LIF)** | Scene classification, rhythmic locomotion | Classifier, Spinal CPG |
| **Active Inference** | Goal selection (EFE), belief state (VAE), predictive coding (pallium PE), visual prediction (predictive net), exploration drive (surprise) | EFE engine, VAE World Model, Pallium W_FB, Predictive Net, Cerebellum |
| **Reinforcement Learning** | Value estimation (TD), reward prediction error (DA), habit formation (Hebbian), strategy switching (frustration) | RL Critic, Neuromod DA, Habit Network, Habenula |
| **Bayesian Inference** | Predator tracking (Kalman filter), spatial memory (RBF associative memory) | Predator Model, VAE Associative Memory |
| **Predictive Coding** | Pallium two-compartment (somatic vs apical), feedback weight learning, free energy minimization | Pallium, W_FB, Cerebellum climbing fiber |
| **Hebbian/STDP** | Amygdala fear conditioning (LA→CeA LTP), cerebellum (parallel fiber LTD), habit (repeated pairing), critic (eligibility traces) | Amygdala, Cerebellum, Habit Net, RL Critic, Plasticity |

---

## Performance

### v1 vs v2 Comparison (matched seeds)

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| SNN neurons | 3,470 (rate-coded) | 7,166 (Izhikevich) | +106% |
| Cell types | 1 | 7 | +6 |
| Inhibitory neurons | 0% | 25% per layer | E/I balance |
| Neuromodulatory axes | 1 (DA) | 4 (DA/NA/5-HT/ACh) | +3 |
| Modules | 42 steps | 29 modules | Consolidated |
| Classifier accuracy | 90.6% | 96.2% | +5.6pp |
| Mean survival (3 seeds) | 319 steps | 422 steps | **+32%** |
| Mean food (3 seeds) | 3.0 | 7.7 | **+156%** |
| Decision rationality | 84/100 | 69/100 | -15 (different scenarios) |
| Curriculum | 2/4 LEARNING | 3/3 PASS | Improved |

### Multi-Seed Evaluation (9 seeds x 500 steps)

| Metric | Value |
|--------|-------|
| Survival | 474 +/- 72 steps |
| Food eaten | 6.0 +/- 2.4 |
| Caught | 1/9 (89% survival rate) |

### Decision Scenarios

| Scenario | Score | Description |
|----------|-------|-------------|
| A | 80/100 | Safe vs risky food |
| B | 30/100 | Predator charge |
| C | 96/100 | Starvation dilemma |
| D | 81/100 | Easy foraging |
| E | 60/100 | Explore unknown |
| **Average** | **69/100** | |

### Ablation Study

| Condition | Food (mean) | Impact |
|-----------|-------------|--------|
| Full model | 4.5 | baseline |
| No olfaction | 3.5 | **-22%** (largest) |
| No cerebellum | 4.0 | -11% |
| No amygdala | 4.0 | -11% |
| No VAE | 4.0 | -11% |
| No critic | 4.5 | 0% |
| No predictive | 4.5 | 0% |
| No habenula | 5.5 | +22% |
| No interoception | 5.5 | +22% |

### Training Pipeline Results

| Pipeline | Result |
|----------|--------|
| Hebbian STDP | 3 episodes, weights converged |
| Online RL | Critic value +113% (0.023 → 0.049) |
| MetaGoalWeights | Goal biases + 8 modulation weights, REINFORCE + correlation |
| SocialMemory | Alarm/food-cue precision-EMA, competition aversion |
| Classifier | **96.2%** accuracy |
| Curriculum | **3/3 stages PASS** |
