---
name: Session Progress (SOFAR)
description: Complete record of all work done — hemispheric vision, novelty gaze, spatial atlas, manuscript updates
type: project
---

# SOFAR — vzebra session log (continue from here)

## What was done, in order

---

## 1. Hemispheric Vision System (brain_v2.py, tectum.py, thalamus.py)

Tectum split from 4 unified EI layers → 8 bilateral:
`sfgs_b_L/R`, `sfgs_d_L/R`, `sgc_L/R`, `so_L/R`
Each: 450E (CH/IB) + 150I (FS) = 600 neurons per hemisphere = 1200 total per layer.

Thalamus split: `thalamus_L` + `thalamus_R`, each TC=300 (LTS), TRN=80 (FS).

Optic chiasm: left eye → right tectum, right eye → left tectum (contralateral).

Thalamic laterality: `thalamus_L` relays tect_L (= right visual hemifield), `thalamus_R` relays tect_R (= left hemifield). Thalamic output concatenated `[TC_R || TC_L]` before pallium so left/right halves of pallium encode correct hemifields.

---

## 2. Novelty Gaze (thalamic delta PE) — brain_v2.py

Pallium pred_error was tried as PE source but failed: `W_tc_pals` (1200×300 dense random xavier) distributes activity uniformly — no L/R spatial structure (measured asymmetry only 1.7%).

Fix: thalamic delta `|tc_now − tc_prev|` has clean spatial novelty because TC populations are physically separated by hemisphere.

**Code in brain_v2.py:**
```python
# In __init__:
self.register_buffer('_prev_tc', torch.zeros(N_TC, device=device))

# After tc_combined computed:
_tc_delta = (tc_combined.detach() - self._prev_tc).abs()
self._prev_tc.copy_(tc_combined.detach())
_n_tc_half = self.thalamus_L.TC.n  # 150
_pe_L = (float(rgc_out['L_ds'].mean()) + float(rgc_out['L_loom'].mean())
         + float(_tc_delta[:_n_tc_half].mean()) * 5.0)
_pe_R = (float(rgc_out['R_ds'].mean()) + float(rgc_out['R_loom'].mean())
         + float(_tc_delta[_n_tc_half:].mean()) * 5.0)
# Passed to saccade.forward(pe_L=_pe_L, pe_R=_pe_R)

# In reset():
self._prev_tc.zero_()
```

---

## 3. EXPLORE Saccade Fix — saccade.py

Old formula: `target_gaze = 0.5 * sin(gaze_ema * 3.0)`
Bug: for small x, sin(3x)≈3x so target≈1.5*gaze_ema. With α=0.1: new_ema = 0.9*ema + 0.1*(1.5*ema) = 1.05*ema → diverges to ±0.5 attractor.

Fix: open-loop time-based oscillation:
```python
# In __init__ and reset():
self._scan_step = 0

# EXPLORE branch:
target_gaze = 0.4 * math.sin(self._scan_step * 0.08)
self._scan_step += 1
```
Period ≈ 79 steps ≈ 4s at behavioral 20Hz.

---

## 4. Test Suite — zebrav2/tests/test_hemispheric_vision.py

7 tests, all PASS:
1. L/R tectum activation asymmetry
2. Thalamic relay L/R independence
3. Pallium top-down separate hemispheres
4. EXPLORE saccade stability (no attractor)
5. FORAGE saccade tracks food bearing
6. FLEE saccade tracks predator bearing
7. Novelty gaze toward novel hemifield (thalamic delta)

Figure: 4×2 layout, panels 1-7 + summary panel 8.
Signature: `make_figure(d1, d2, d3, d4, d5, d6, d7)`

---

## 5. Trainer / generate_paper_figures fixes — trainer.py, generate_paper_figures.py

`spikes` dict updated from old unified tectum/thalamus API:
```python
# OLD (broken):
'sfgs_b': float(self.brain.tectum.sfgs_b.spike_E.sum()),
'tc':     float(self.brain.thalamus.TC.rate.sum()),

# NEW:
'sfgs_b': float(self.brain.tectum.sfgs_b_L.spike_E.sum() + self.brain.tectum.sfgs_b_R.spike_E.sum()),
'sfgs_d': float(self.brain.tectum.sfgs_d_L.spike_E.sum() + self.brain.tectum.sfgs_d_R.spike_E.sum()),
'sgc':    float(self.brain.tectum.sgc_L.spike_E.sum()    + self.brain.tectum.sgc_R.spike_E.sum()),
'so':     float(self.brain.tectum.so_L.spike_E.sum()     + self.brain.tectum.so_R.spike_E.sum()),
'tc':     float(self.brain.thalamus_L.TC.rate.sum()      + self.brain.thalamus_R.TC.rate.sum()),
'trn':    float(self.brain.thalamus_L.TRN.rate.sum()     + self.brain.thalamus_R.TRN.rate.sum()),
```
Same fix in both single-agent (line ~347) and multi-agent (line ~662) paths.

---

## 6. 3D Spatial Atlas — generate_brain_atlas.py (complete rewrite)

### Atlas data used (CORRECT source)
- `atlas/subject_12_CellXYZ.csv` — 47010 cells, (x,y,z) in voxel space
- `atlas/subject_12_region_num.csv` — 47010 cells, format `[region_num, 'lName']`
- Both files have matching row counts (47010 rows each after header)
- Connectome `subject_12_sc_list_pre_post.csv` uses indices 0-47009 (matches this dataset)

NOT `cell_xyz_s12.npy` (58105 cells, 36 combined regions) — that's the OLD source.

### 72 bilateral atlas regions
- Regions 0-35: left hemisphere (prefix `l`), 36-71: right hemisphere (prefix `r`)
- 61 of 72 present in s12 data; missing: 9(lHc), 15(lGG), 18(lHr), 23(lPi), 27(lR/retina), 45(rHc), 51(rGG), 54(rHr), 55(rOG?), 59(rPi), 63(rR/retina)
- Retina (27/63) is missing → uses manual estimate: `_RETINA_L_POS = [220, 115, 165]`, `_RETINA_R_POS = [220, 505, 165]`

### Coordinate system (voxels)
- x = anterior→posterior (lOB at x=144, lNX/spinal at x=961)
- y = left→right (midline ≈ 310; left regions have y<310, right have y>310)
- z = ventral→dorsal

### REGIONS dict structure in generate_brain_atlas.py
```python
REGIONS = {
    'sfgs_b_L': {'n': 1200, 'atlas': [29],  'side': 'L', 'spread': 20, 'z_shift': 18, 'color': '#00cccc'},
    'pal_s':    {'n': 1600, 'atlas': [22,58], 'spread': 25, 'color': '#44cc00'},  # average L+R (midline)
    'retina_L': {'n': 1000, 'atlas': [],    'pos': [220,115,165], 'spread': 25, 'color': '#aaccff'},
    ...
}
```
Key: `atlas: [region_nums]` → average centroid; `atlas: []` → use `pos` fallback.
`side: 'L'` → left centroid; `side: 'R'` → right centroid; omitted → average L+R.

### Atlas region number → brain module mapping
| Brain module | Atlas region nums |
|---|---|
| sfgs_b_L / sfgs_b_R | 29 (lTeO) / 65 (rTeO) |
| sfgs_d_L / sfgs_d_R | 29 / 65 + z_shift |
| sgc_L / sgc_R | 29 / 65 + z_shift |
| so_L / so_R | 29 / 65 + z_shift |
| tc_L / tc_R | 30 (lTh) / 66 (rTh) |
| trn_L / trn_R | 30 / 66 + z_shift |
| pal_s, pal_d | [22 (lP), 58 (rP)] average |
| d1, d2, gpi, amygdala | [28 (lSP), 64 (rSP)] average |
| habenula | [16 (lHb), 52 (rHb)] average |
| cerebellum | [1 (lCb), 37 (rCb)] average |
| insula, allostasis, circadian, sleep_wake | [17 (lHi), 53 (rHi)] average |
| critic, da, ach | [11 (lT), 47 (rT)] average |
| habit, na | [14 (lpRF), 50 (rpRF)] average |
| serotonin | [10 (lRa), 46 (rRa)] average |
| predictive, saccade | [26 (lPrT), 62 (rPrT)] average |
| reticulospinal | [24 (lPT), 60 (rPT)] average |
| cpg_L / cpg_R | 35 (lNX) / 71 (rNX) |
| lateral_line, vestibular | [0 (lMON), 36 (rMON)] average |
| olfaction | [20 (lOB), 56 (rOB)] average |
| goal | [25 (lPO), 61 (rPO)] average |

### Output: zebrav2/web/static/brain_atlas.json
- 46 model regions, 13,766 neurons, 3,500 connections
- 293 KB
- Run to regenerate: `.venv/bin/python -m zebrav2.web.generate_brain_atlas`

### Key anatomical positions verified (voxels):
| Module | x | y | Notes |
|--------|---|---|-------|
| retina_L/R | 220 | 114/505 | ±195 from midline |
| sfgs_b_L/R | 554 | 202/410 | mid-brain |
| tc_L/R | 388 | 268/352 | diencephalon |
| habenula | 317 | 307 | near midline |
| pal_s/d1 | 209-228 | 310 | anterior telencephalon |
| cerebellum | 668 | 310 | posterior rhombencephalon |
| cpg_L/R | 961 | 259/353 | most posterior |

---

## 7. brain3d.js — spike visualization for bilateral regions

spikeMap restructured: one spike key → array of atlas region names:
```javascript
const spikeMap = {
    'sfgs_b': ['sfgs_b_L', 'sfgs_b_R'],
    'sfgs_d': ['sfgs_d_L', 'sfgs_d_R'],
    'sgc':    ['sgc_L',    'sgc_R'],
    'so':     ['so_L',     'so_R'],
    'tc':     ['tc_L',     'tc_R'],
    'trn':    ['trn_L',    'trn_R'],
    ...single regions...
};
// Loop changed from:
for (const [key, rn] of Object.entries(spikeMap))
// To:
for (const [key, rnList] of Object.entries(spikeMap))
    for (const rn of rnList)
```

---

## 8. spatial_registry.py — bilateral assign_to_brain()

`assign_to_brain()` updated to use bilateral keys for tectum/thalamus:
- `'tectum.sfgs_b_L'` → atlas region 'tectum', offset [0,-50,20]
- `'tectum.sfgs_b_R'` → atlas region 'tectum', offset [0,+50,20]
- `'thalamus.tc_L'` → atlas region 'thalamus', offset [0,-25,0]
- Plus new modules: lateral_line, olfaction, vestibular, proprioception, color_vision, circadian, sleep_wake, working_memory, saccade

---

## 9. manuscript2.tex — updated sections

### Abstract (line ~115)
Changed: "73~regions" → "72~bilateral regions, 61~with recorded cells in subject~12"

### New subsection (line ~607, before Sensory Processing)
`\subsection{Anatomically Constrained Neuron Placement}\label{sec:atlas}`
- MPIN atlas details, bilateral centroid placement
- Coordinate axes description
- Key bilateral separations (retina ±390, tectum ±208, thalamus ±84 vox)
- Gaussian sampling, connectome edges, Three.js rendering
- Citation: `\citep{kunst2019}` added

### Hemispheric Visual Organization subsection (line ~456)
Already complete from previous session. Contains:
- Optic chiasm / tectal lateralization
- Hemispheric thalamus (TC=300 per side, TRN=80 per side)
- Top-down predictive attention equation
- Active-inference gaze control (saccade, 3-branch policy)
- Novelty-driven gaze via thalamic prediction error (Eq. hemifield_pe)
- All 7 unit tests mentioned

### Bibliography
Added `\bibitem[Kunst et~al., 2019]{kunst2019}` — Neuron 103:21–38

---

## 10. technical_report_v2.tex — updated sections

### Phase 3 Tectal Layer Structure (line ~487)
- Equation updated: `(450E+150I)×2` for SFGS layers, `(150E+50I)×2` for SGC/SO
- New paragraph: "Each tectal layer is duplicated into left/right... Total tectum: 4 layers × 2 hemispheres = 8 EI populations, 4,800 neurons"

### Phase 6 Thalamo-Pallial Loop (line ~834)
- Thalamus equation: now bilateral with superscript s ∈ {L,R}
- New paragraph: two thalamic nuclei, TC=300, TRN=80, laterality, [TC_R || TC_L] ∈ ℝ^600 concatenation
- Pallium sizes corrected: pal_s = 1200 RS + 400 FS (n=1600), pal_d = 600 IB + 200 FS (n=800)

### Recent Extensions — new subsection (line ~1794)
`\subsection{Hemispheric Visual Architecture and Active-Inference Gaze}`
- Tectum refactor, SpikingSaccade
- Gaze-EMA instability fix with math (sin(3x)≈3x → diverges)
- Per-hemifield PE equation using thalamic delta
- Why pallium PE rejected (1.7% asymmetry)
- 7 unit tests

### Atlas section (line ~1808)
Rewritten: 13,766 neurons, 72 bilateral MPIN regions, voxel coordinate axes, module-to-region mapping, 3,500+400+100 connections

---

## Files Changed (complete list)

| File | What changed |
|------|-------------|
| `zebrav2/brain/brain_v2.py` | `_prev_tc` buffer, thalamic delta PE computation |
| `zebrav2/brain/saccade.py` | `_scan_step` oscillation fix for EXPLORE |
| `zebrav2/engine/trainer.py` | spikes dict: bilateral tectum/thalamus API |
| `zebrav2/tests/test_hemispheric_vision.py` | 7 tests + 4×2 figure |
| `zebrav2/tests/generate_paper_figures.py` | bilateral API fix + fig8 |
| `zebrav2/brain/spatial_registry.py` | bilateral assign_to_brain() |
| `zebrav2/web/generate_brain_atlas.py` | complete rewrite using 72-region atlas |
| `zebrav2/web/static/brain_atlas.json` | regenerated (293 KB, 13,766 neurons) |
| `zebrav2/web/static/brain3d.js` | spikeMap arrays for bilateral display |
| `manuscript2.tex` | abstract fix, new atlas subsection, kunst2019 bibitem |
| `zebrav2/technical_report_v2.tex` | Phase 3/6 bilateral updates, new Recent Extensions subsection, atlas section rewrite |

---

## 11. Neural Monitor bilateral fix — neural_monitor.py

`NeuralMonitorV2.update()` and `render()` used old unified tectum/thalamus API → crashed on `brain.tectum.sfgs_b`.

Fixed all references:
- `brain.tectum.sfgs_b.spike_E` → `concat(sfgs_b_L.spike_E, sfgs_b_R.spike_E)`
- `brain.tectum.sfgs_b.get_rate_e()` → separate L/R heatmaps side by side
- `brain.tectum.sfgs_d.get_rate_e()` → separate L/R heatmaps
- `brain.tectum.sgc/so.get_rate_e()` → `concat(sgc_L, sgc_R)` / `concat(so_L, so_R)`
- `brain.thalamus.tc_rate` → `brain.thalamus_L.TC.rate` + `brain.thalamus_R.TC.rate` (separate L/R columns)
- `brain.thalamus.trn_rate` → `concat(thalamus_L.TRN.rate, thalamus_R.TRN.rate)`

Monitor now shows bilateral structure: SFGS-b L|R, SFGS-d L|R, TC L|R columns.

Demo recorded: `plots/zebrafish_v2_demo.mp4` — 1220 steps, 22 food eaten, caught at step 1219.

---

## Files Changed (updated)

| File | What changed |
|------|-------------|
| `zebrav2/viz/neural_monitor.py` | bilateral tectum/thalamus API in update() and render() |

---

## 12. Session 2 — Bilateral Retraining, Full Test Suite, Motor Evaluation, Platform Vision

### 12a. Bilateral Retraining (train_bilateral.py)
- Ran 14 rounds (86→99), checkpoints saved at round 90, 95
- High variance: survival 23–500, avg ~211
- Round 95 checkpoint (20.4MB) is current reference

### 12b. Paper Figure Generation (generate_all_paper_figs.py)
- Updated default checkpoint from round 20 → round 85
- All 11 figures generated successfully including 9-seed survival evaluation

### 12c. Spatial Atlas Synchronization Tests (test_spatial_sync.py) — NEW
- 6 tests: atlas CSV loads, label_map coverage, centroid ranges, position shapes, weight masks, region number sync
- Fixed pytest fixture: added `@pytest.fixture(scope='module')` for `reg`

### 12d. Module Interaction Tests (test_module_interaction.py) — 9 tests
- Cross-module: amygdala-DA coupling, habenula-5HT, cerebellum-habit, circadian-sleep, allostasis-HPA, social-oxytocin, place-cell geographic, cortisol-anxiety, neuromod-crosstalk

### 12e. Spiking vs Rate-Coded Agreement Tests (test_spiking_rate_agreement.py) — 32 tests
- 6 sections: pipeline correspondence, neuromod dynamics, EFE formula, behavioral benchmarks, signal ranges, structural checks
- Fixed sys.exit() killing pytest: guarded with `if __name__ == '__main__':`

### 12f. Motor Function Evaluation (test_motor_evaluation.py) — NEW, 25 tests
8 categories:
1. RS-CPG integration (3): RS→CPG drive, speed scaling, CPG oscillation L/R alternation
2. Escape kinematics (4): C-start burst speed ≥3.0, 4-phase sequence, high threat = faster flee, duration 15–60 steps
3. Goal-specific motor programs (5): FORAGE=moderate, FLEE=fast, EXPLORE=variable heading, SOCIAL=slow, SLEEP=minimal
4. Energy-motor coupling (3): inverted-U motility `max(0.15, 4*e*(1-e))`, zero-energy minimal, speed ordering
5. Cerebellum PE adaptation (3): initial PE>0.5, adapts<0.5, sudden change re-spikes
6. Vestibular/orient (2): tilt → compensatory motor, tectal orient via BG gate
7. Prey capture (2): approach food → speed>0, correct heading toward food
8. Robustness (3): NaN input → finite output, dead fish stops, motor clipping 0–2

Key fixes during development:
- Energy mismatch: all goals tested at 50% energy for fair comparison
- BG gate: DA=0.8 for open gate in tectal orient test
- Death timer: set to 30 to prevent `_reset_round()` revival

### 12g. server.py Lazy Import Fix
- Dropbox sync corrupting .so files caused `import torch` to hang
- Moved `TrainingEngine`, `TrainingConfig`, `REPERTOIRES` imports into `_lazy_engine_imports()`
- Server module loads instantly; torch loaded on-demand in endpoints and on_startup()

### 12h. Manuscript Updates
- manuscript2.tex: updated test count to "348 automated tests" (182 spiking + 94 rate + 25 motor + 9 interaction + 6 atlas + 32 agreement)
- technical_report_v2.tex: added 3 subsections (module interaction, atlas sync, spiking-rate agreement)

### 12i. Final Test Results — 348/348 PASS
| Suite | Count | Status |
|-------|-------|--------|
| Spiking brain | 182 | ✅ PASS |
| Rate-coded pipeline | 94 | ✅ PASS |
| Motor evaluation | 25 | ✅ PASS |
| Module interaction | 9 | ✅ PASS |
| Spatial atlas sync | 6 | ✅ PASS |
| Spiking-rate agreement | 32 | ✅ PASS |
| **Total** | **348** | **✅ ALL PASS** |

---

## 13. Downloadable Virtual Zebrafish Platform — Architecture Vision

### Goal
Transform vzebra from research code into a **downloadable virtual zebrafish** for diverse workbenches — neuroscience, social science, pharmacology, education. The sensory-motor system is basic (sufficient for higher-order function study), so the platform emphasizes configurable brain modules, experiment APIs, and recording infrastructure.

### Current Architecture Inventory
- **47 brain modules** in zebrav2/brain/ (~240K code)
  - 34 nn.Module spiking classes + 13 pure Python modules
- **Checkpoint system**: 23 state components (20.4MB per checkpoint)
  - critic, classifier, pallium, cerebellum, amygdala, habit, place_cells, geographic, VAE (encoder/decoder/transition/memory), habenula, personality, HPA, oxytocin, neuromod, meta_goal, social_mem
- **100+ hardcoded parameters** in brain_v2.py (EFE coefficients, motor speeds, thresholds, neuromod coupling)
- **TrainingConfig**: nested dict with env/fish/training/objectives sections, 6 repertoire presets

### 5-Layer Platform Architecture

```
Layer 5: Experiment API          ← researcher-facing (Python, CLI, Jupyter)
Layer 4: Virtual Lab Framework   ← experiments, recording, analysis
Layer 3: Brain                   ← configurable, 3 fidelity levels
Layer 2: Body                    ← sensory-motor abstraction
Layer 1: World                   ← environments (arena, multi-agent, naturalistic)
```

**Layer 1 — World (environments)**
- Arena types: open field, Y-maze, T-maze, place preference, social interaction, predator avoidance
- Stimulus library: visual (looming, gratings, optogenetic), chemical (alarm, food odor), social (conspecific cues)
- Environmental parameters: temperature, light cycle, flow, boundaries

**Layer 2 — Body (sensory-motor abstraction)**
- Sensory channels: retina (bilateral, 4-cone color), lateral line (flow), olfaction (alarm/food/social), vestibular, proprioception
- Motor output: tail CPG, turn bias, speed, saccade/gaze
- Body parameters: size, metabolic rate, fatigue model
- Key: sensory-motor is intentionally basic — serves as I/O for higher-order brain study

**Layer 3 — Brain (configurable)**
- 3 fidelity levels:
  1. **Full spiking** (Izhikevich, ~13K neurons) — biophysically detailed
  2. **Rate-coded** (current server.py pipeline) — fast, real-time
  3. **Minimal/lesioned** — for ablation studies, custom module injection
- All 100+ parameters exposed via BrainConfig dataclass
- Module enable/disable flags for ablation studies
- Neuromodulator coupling matrix configurable
- Pre-trained weights downloadable (round 95 checkpoint)

**Layer 4 — Virtual Lab Framework**
- Recording: spike trains, firing rates, neuromodulator timeseries, behavioral trajectory, decision logs
- Perturbation API: lesion(region), inject(drug, dose), stimulate(region, pattern), silence(region)
- Analysis: PSTH, tuning curves, information-theoretic measures, behavioral ethograms
- Batch experiment runner with parameter sweeps

**Layer 5 — Experiment API**
- Python API: `fish = VirtualZebrafish.load("pretrained"); fish.lesion("habenula"); results = fish.run(arena="y_maze", steps=10000)`
- CLI: `vzebra run --arena open_field --lesion habenula --record spikes,behavior --steps 10000`
- Jupyter integration: inline visualization, interactive parameter tuning
- Export: NWB format, CSV, video

### Implementation Plan (14 tasks, 5 phases)

**Phase 1 — Package Skeleton + Config Extraction**
1. `pyproject.toml` with `vzebra` entry point
2. Extract `BrainConfig` dataclass from brain_v2.py hardcoded params
3. Extract `BodyConfig` (sensory channels, motor params)
4. Extract `WorldConfig` (arena types, stimulus params)

**Phase 2 — Brain Abstraction Layer**
5. `AbstractBrain` interface (step, reset, lesion, inject, get_state)
6. `RateCodedBrain` wrapping current server.py pipeline
7. `SpikingBrain` wrapping current brain_v2.py
8. Module registry with enable/disable flags

**Phase 3 — Virtual Lab Framework**
9. Recording system (spike recorder, behavior logger, neuromod tracker)
10. Perturbation API (lesion, drug injection, optogenetic stimulation)
11. Batch experiment runner with parameter sweeps
12. Analysis toolkit (PSTH, tuning curves, ethograms)

**Phase 4 — Experiment API + Distribution**
13. Python API + CLI + Jupyter widgets
14. Pre-trained weight packaging + download system

**Phase 5 — Documentation + Community**
- Tutorials, example experiments, API reference
- Contribution guide for adding new brain modules

### Use Cases by Field
| Field | Example Experiment |
|-------|-------------------|
| Systems neuroscience | Tectal lesion → prey capture deficit |
| Neuromodulation | DA antagonist → reward learning impairment |
| Social neuroscience | Oxytocin knockout → social preference shift |
| Pharmacology | Drug dose-response on anxiety (light/dark preference) |
| Computational psychiatry | Habenula hyperactivity → learned helplessness |
| Education | Interactive brain explorer with real-time ablation |
| Behavioral genetics | Personality parameter sweep → ethogram diversity |

---

## 14. Phase 1 — Package Skeleton + Config Extraction (DONE)

### 14a. pyproject.toml
- Package name: `vzebra` v2.0.0
- Entry point: `vzebra = zebrav2.cli:main`
- Dependencies: torch, numpy, matplotlib, gymnasium, pygame, imageio
- Optional: `[web]` (fastapi, uvicorn, websockets), `[dev]` (pytest)
- Entry points for arenas: open_field, novel_tank, light_dark, social_preference

### 14b. BrainConfig (zebrav2/config/brain_config.py)
7 subsystem dataclasses, 200+ parameters extracted from brain_v2.py:
- **EFEConfig**: 40+ goal selection coefficients (forage/flee/explore/social offsets, prediction weights, energy urgency, interoception, habenula, circadian, surprise coupling)
- **GoalSelectionConfig**: WTA confidence, food-visible reflex, starvation overrides, threat evidence, stuck detection, persistence state machine
- **NeuromodConfig**: reward computation, DA phasic burst, habenula suppression
- **PlasticityConfig**: 4 STDP pathways (tect→tc, tc→pal, pal_d + consolidation), classifier/enemy learning rates, top-down gain
- **ThreatConfig**: enemy probability fusion, CMS, rear threat memory, lateral line thresholds
- **NoveltyConfig**: place visit, VAE novelty, EMA parameters
- **AblationConfig**: 23 module enable/disable flags (habenula + insula ablated by default)

### 14c. BodyConfig (zebrav2/config/body_config.py)
3 subsystem dataclasses:
- **SensoryConfig**: retina (8 params), lateral_line (7), olfaction (10), vestibular (9), proprioception (11), color_vision (8 + 4 spectral signatures)
- **MotorConfig**: CPG (19 synaptic/neuron params), speeds by goal (5), turn limits, motor smoothing (4 alpha), RS integration, exploration oscillation, flee/forage/wall/obstacle avoidance
- **MetabolismConfig**: energy system (7 params including inverted-U motility)

### 14d. WorldConfig (zebrav2/config/world_config.py)
2 subsystem dataclasses:
- **ArenaConfig**: geometry (3), food (6), rocks (10)
- **PredatorConfig**: energy (4), stamina (5), speeds (7), detection (5), timers (4), intelligence (5), place cells (5)
- Vision physics: FOV, max distance, type codes (4), visual radii (4)

### 14e. VirtualZebrafish facade (zebrav2/virtual_fish.py)
Researcher-facing API:
```python
fish = VirtualZebrafish.load('pretrained')      # loads round 95 checkpoint
fish.lesion('cerebellum').set_personality('shy') # fluent API
fish.export_config('my_fish.json')              # save full config
fish2 = VirtualZebrafish.from_config('my_fish.json')  # reload
```
- `lesion(region)` / `enable(region)` — ablation study API
- `set_personality(preset)` — bold/shy/explorer/social
- `set_fidelity(level)` — spiking/rate_coded/minimal
- `load_checkpoint(path)` — pre-trained weights
- `export_config(path)` / `from_config(path)` — JSON persistence

### 14f. CLI entry point (zebrav2/cli.py)
```bash
vzebra run   --config fish.json --steps 1000 --lesion habenula
vzebra train --rounds 30 --checkpoint ckpt.pt --predator intelligent
vzebra serve --host 0.0.0.0 --port 8765
vzebra info                   # show model summary
vzebra export -o config.json  # export default config
```

### 14g. Tests — 46/46 PASS
| Category | Tests | Status |
|----------|-------|--------|
| BrainConfig defaults & subsystems | 7 | ✅ |
| BodyConfig defaults (sensory/motor/metabolism) | 6 | ✅ |
| WorldConfig defaults (arena/predator/vision) | 4 | ✅ |
| JSON round-trip serialization | 7 | ✅ |
| VirtualZebrafish facade | 11 | ✅ |
| CLI entry point | 4 | ✅ |
| Integration with TrainingConfig | 3 | ✅ |
| Parameter completeness | 4 | ✅ |
| **Total** | **46** | **✅ ALL PASS** |

### Files Created
| File | Lines | Description |
|------|-------|-------------|
| `pyproject.toml` | 60 | Package metadata + entry points |
| `zebrav2/config/__init__.py` | 14 | Config package exports |
| `zebrav2/config/brain_config.py` | 270 | BrainConfig (7 sub-dataclasses) |
| `zebrav2/config/body_config.py` | 240 | BodyConfig (3 sub-dataclasses) |
| `zebrav2/config/world_config.py` | 130 | WorldConfig (2 sub-dataclasses) |
| `zebrav2/virtual_fish.py` | 190 | VirtualZebrafish facade |
| `zebrav2/cli.py` | 145 | CLI entry point |
| `zebrav2/__init__.py` | 10 | Package root with version |
| `zebrav2/tests/test_platform_config.py` | 310 | 46 tests |

---

## 15. Phase 2 — Platform wiring + subsystems

### Config wiring into brain_v2.py (~30 surgical edits)
- `__init__` accepts `brain_config` and `body_config` parameters
- EFE: all 40+ coefficients wired via `_ec = self.cfg.efe`
- Threat: all detection params via `_tc = self.cfg.threat`
- Goal selection: all 25+ thresholds via `_gs = self.cfg.goal_selection`
- Motor: all speed/gate params via `_mc_cfg = self.body_cfg.motor`
- Neuromod: reward/phasic/habenula params via `_nm = self.cfg.neuromod`
- Plasticity: STDP, consolidation, classifier eta via `_plast = self.cfg.plasticity`
- Ablation: `self._ablated = self.cfg.get_ablated_set()` replaces hardcoded adds
- CPG: all 14 synaptic weights wired from `body_cfg.motor.cpg`
- Lateral line: SN/CN/prey ranges from `body_cfg.sensory.lateral_line`

### AbstractBrain protocol (`brain/abstract_brain.py`)
`runtime_checkable` Protocol: step(), reset(), current_goal, energy, set_region_enabled()

### Recording system (`recording.py`, ~158 lines)
StepRecord dataclass, Recorder with start/stop, record_step/record_event.
Analysis: get_trajectory(), get_goal_counts(), get_neuromod_timeseries().
Export: to_dict(), save(JSON), to_csv().

### Perturbation API (`perturbation.py`, ~126 lines)
- DrugEffect: agonist/antagonist/reuptake_inhibitor with dose-dependent multipliers
- DRUG_LIBRARY: haloperidol, fluoxetine, scopolamine, amphetamine, propranolol, buspirone
- PerturbationManager: lesion(), inject(), stimulate(), get_neuromod_multipliers()
- **Integrated into brain_v2.py step()**: drug multipliers applied after neuromod update

### Batch experiment runner (`batch.py`, ~165 lines)
ParameterSweep + BatchRunner: full factorial sweep with multi-seed experiments.

### Module registry (`brain/module_registry.py`, ~200 lines)
39 modules (8 required + 31 optional) with metadata: class_name, role, region_group, dependencies.
AblationConfig synced to 32 fields matching all optional modules.

### RateCodedBrain (`brain/rate_coded_brain.py`, ~200 lines)
Standalone class satisfying AbstractBrain. Implements 11-stage pipeline from server.py.
~100x faster than spiking brain — suitable for batch sweeps.

### Disorder integration
VirtualZebrafish.apply_disorder() delegates to disorder.py (8 presets: wildtype, hypodopamine, asd, schizophrenia, anxiety, depression, adhd, ptsd).

### Trainer config wiring (`engine/trainer.py`)
- Accepts brain_config, body_config, world_config
- Passes to ZebrafishBrainV2 constructor
- WorldConfig arena dimensions applied to env
- Recorder integrated: start/stop per round, record_step + food_eaten events

### Spatial registry sync (`brain/spatial_registry.py`)
label_map expanded from 16 to 40 entries covering all brain modules.
Fine-grained sub-regions: tectum_L/R, thalamus_L/R, cpg_L/R, amygdala, basal_ganglia, etc.
Synced with generate_brain_atlas.py REGIONS dict.

### Personality `name` key
All 5 personality dicts now include `'name'` field (bold, shy, explorer, social, default).

### Tests
| Suite | Tests | Status |
|-------|-------|--------|
| test_platform_config.py (Phase 1) | 46 | ALL PASS |
| test_platform_phase2.py | 37 | ALL PASS |
| test_platform_phase3.py | 17 | ALL PASS |
| test_motor_evaluation.py | 25 | ALL PASS (regression) |
| **Total** | **125** | **ALL PASS** |

### New/modified files
| File | Lines | Purpose |
|------|-------|---------|
| `zebrav2/brain/abstract_brain.py` | 20 | AbstractBrain Protocol |
| `zebrav2/recording.py` | 158 | Recording system |
| `zebrav2/perturbation.py` | 126 | Perturbation API + drug library |
| `zebrav2/batch.py` | 165 | Batch experiment runner |
| `zebrav2/brain/module_registry.py` | 200 | Module registry (39 modules) |
| `zebrav2/brain/rate_coded_brain.py` | 200 | RateCodedBrain wrapper |
| `zebrav2/brain/brain_v2.py` | ~1350 | +30 config wiring edits + perturbation integration |
| `zebrav2/brain/personality.py` | 128 | Added `name` keys |
| `zebrav2/brain/spatial_registry.py` | 243 | Expanded label_map (40 entries) |
| `zebrav2/config/brain_config.py` | 310 | +9 AblationConfig fields |
| `zebrav2/engine/trainer.py` | 920 | Config wiring + recorder integration |
| `zebrav2/virtual_fish.py` | 280 | RateCodedBrain + disorder + perturbation |
| `zebrav2/tests/test_platform_phase2.py` | 351 | 37 tests |
| `zebrav2/tests/test_platform_phase3.py` | 130 | 17 tests |

---

## 16. Phase 3 — Integration hardening + spatial priors

### 16a. WorldConfig wiring into sensory_bridge.py
`inject_sensory()` now accepts optional `world_config` parameter.
When provided, uses: `fov_degrees`, `max_vision_distance`, `retinal_intensity_scale`,
`visual_radius_predator/food/rock`, `type_code_predator/food/rock`.
Backward compatible: defaults to hardcoded values when `world_config` is None.

### 16b. Distance-dependent spatial priors on STDP pathways
`_apply_spatial_priors()` added to `ZebrafishBrainV2.__init__()`:
- Tectum→Thalamus (L/R): λ=80μm, strength=0.4
- Thalamus→Pallium-S: λ=120μm, strength=0.3
- Pallium-S→Pallium-D: λ=60μm, strength=0.3
- Pallium→Tectum (top-down): λ=150μm, strength=0.2

Formula: `w_ij ← w_ij * [(1-s) + s * exp(-d_ij/λ)]`

`SpatialRegistry.apply_distance_weights()` now handles dimension mismatches via bilinear interpolation.

### 16c. Integration tests (11/11 pass)
`test_platform_integration.py`:
- Spiking brain run (20 steps), rate-coded brain run (50 steps)
- Lesion + run, personality + run, fluent API chaining
- Drug injection + run, disorder + run
- Spatial priors applied, distance mask shape
- Rate-coded brain step output, energy tracking

### 16d. Manuscript update — Platform Architecture section
Added `\section{Software Platform Architecture}` to `technical_report_v2.tex`:
- Dataclass configuration hierarchy
- VirtualZebrafish facade API
- Distance-dependent spatial priors table
- Recording and batch execution

### 16e. pip install verification
`pip install -e .` succeeds → `vzebra==2.0.0`.
All imports work: `zebrav2`, `VirtualZebrafish`, config classes.

### 16f. Batch sweep validation
`BatchRunner` with 2 values × 2 seeds = 4 runs completed successfully.
Summary aggregation working (survived_mean, food_mean, energy_mean).

### 16g. Checkpoint verification
23 checkpoints (round 1–95). Latest (round 95):
- 129 steps survived, 1 food eaten, fitness 245.5
- Goal distribution: FORAGE 78, FLEE 13, EXPLORE 38
- Loads and runs via `VirtualZebrafish.load("pretrained")`

### Files touched

| File | Lines | What |
|------|-------|------|
| `zebrav2/brain/brain_v2.py` | +35 | `_apply_spatial_priors()` method |
| `zebrav2/brain/spatial_registry.py` | +8 | Dimension mismatch handling |
| `zebrav2/brain/sensory_bridge.py` | +20 | WorldConfig parameter wiring |
| `zebrav2/tests/test_platform_integration.py` | 165 | 11 integration tests (NEW) |
| `zebrav2/technical_report_v2.tex` | +85 | Platform architecture section |

---

## What is NOT done yet (possible next steps)

- ~~Record demo video with new bilateral tectum visible in brain3d~~ ✓ done (item 11)
- ~~Run `generate_paper_figures.py` end-to-end~~ ✓ done (item 12b)
- ~~Full motor evaluation~~ ✓ done (item 12f)
- ~~348/348 tests passing~~ ✓ done (item 12i)
- ~~Phase 1 of platform architecture~~ ✓ done (item 14)
- ~~Phase 2 — config wiring, subsystems, registry~~ ✓ done (item 15)
- ~~Wire BodyConfig into sensory_bridge.py~~ ✓ done (item 16a)
- ~~Wire distance-dependent weights from spatial_registry into STDP~~ ✓ done (item 16b)
- ~~Integration test: VirtualZebrafish end-to-end~~ ✓ done (item 16c)
- ~~pip install test~~ ✓ done (item 16e)
- ~~Batch sweep demo~~ ✓ done (item 16f)
- Continue training beyond round 95 (more rounds with intelligent predator)
- Multi-seed survival statistics (9+ seeds for paper figure)
- Social conspecific interaction module integration
- Web 3D atlas sync with new spatial registry label_map
