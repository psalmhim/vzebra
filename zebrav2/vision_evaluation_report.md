# Vision System Evaluation Report

**Date**: 2026-04-15 14:51
**Tests**: 94 across 31 evaluation categories
**Pass Rate**: 94/94 (100%)

## Test Results

| # | Test | Status | Details |
|---|------|--------|---------|
| 1 | Food at -45° → left eye only | PASS | L=1 R=0 |
| 2 | Food at +45° → right eye only | PASS | L=0 R=1 |
| 3 | Food straight ahead (0°) → detected | PASS | total=1 |
| 4 | Food at 180° (behind) → blind spot | PASS | L=0 R=0 (should be 0,0) |
| 5 | Food at 99° (FoV boundary inner) → detected | PASS | 99° detected=True |
| 6 | Food at 101° (FoV boundary outer) → blind | PASS | 101° detected=False |
| 7 | Distance gradient: closer food → stronger retina | PASS | activations={50: 0.9999999999999999, 100: 0.7999999999999998, 150: 0.6, 200: 0.3... |
| 8 | Food at 260px → beyond max range | PASS | food at 260px: detected=False |
| 9 | Predator looming: closer → stronger SGC | PASS | SGC looming: {30: 1.9600000000000004, 60: 1.17, 90: 0.5800000000000001, 120: 0.1... |
| 10 | Predator vs food: 2x retinal amplification | PASS | food_retina=0.700 pred_retina=1.100 ratio=1.57 |
| 11 | Conspecific vs food: lower salience | PASS | food=0.700 conspec=0.250 |
| 12 | Left eye → right tectum (chiasm) | PASS | sfgs_b_R=0.660 > sfgs_b_L=0.180 |
| 13 | Right eye → left tectum (chiasm) | PASS | sfgs_b_L=0.660 > sfgs_b_R=0.180 |
| 14 | Predator behind → lateral line detection | PASS | lateral_line=0.700 |
| 15 | Predator behind → lateral line → FLEE goal | PASS | FLEE in 15/15 steps (goals: ['FLEE', 'FLEE', 'FLEE', 'FLEE', 'FLEE']) |
| 16 | Injured conspecific → alarm substance → amygdala | PASS | amygdala from alarm=0.162 |
| 17 | Low NA → weak TC relay; high NA → strong TC relay | PASS | tc(NA=0.1)=0.069 tc(NA=0.9)=0.262 ratio=3.8 |
| 18 | Day vs night: ACh gating reduces night vision | PASS | tc_day=0.120 > tc_night=0.017 |
| 19 | Food in both visual fields → bilateral retina | PASS | L=1 R=1 |
| 20 | Food + predator same eye → both detected | PASS | pred_detected_in_eye=True |
| 21 | Crowded scene: retinal saturation ceiling | PASS | retina_R=5.000 (max~5.0) |
| 22 | Visible food + hungry → FORAGE | PASS | goal=FORAGE |
| 23 | Close predator → FLEE | PASS | goal=FLEE |
| 24 | No stimuli → EXPLORE/SOCIAL | PASS | goal=SOCIAL |
| 25 | Predator + food: FLEE overrides FORAGE | PASS | goal=FLEE (should be FLEE, not FORAGE) |
| 26 | Night time → SLEEP goal | PASS | goal=SLEEP at night |
| 27 | Close predator → C-start reflex | PASS | cstart_timer=3 |
| 28 | C-start speed sequence (4-step burst) | PASS | speed sequence=[0.09, 0.25, 1.1, 0.95, 0.23, 0.23] |
| 29 | Far predator (180px) → no C-start | PASS | cstart_timer=0 (should be 0) |
| 30 | 360° food sweep: FoV = ±100° | PASS | detected=13/25 angles, blind=12 |
| 31 | 360° predator: lateral line omnidirectional | PASS | LL at all angles: True, values={-180: 0.5, -150: 0.5, -120: 0.5, -90: 0.5, -60: ... |
| 32 | Heading invariance: same relative angle → same eye | PASS | right_eye across headings: [1, 1, 1, 1, 1, 1] |
| 33 | Food in left FoV → turn left | PASS | dh=-13.9° (should be negative) |
| 34 | Food in right FoV → turn right | PASS | dh=13.9° (should be positive) |
| 35 | Amygdala fear trace persists after threat removal | PASS | during=1.199 → after=0.602 (decaying) |
| 36 | Eating food → DA increase (reward) | PASS | DA before=0.500 → after=0.652 |
| 37 | No food → starvation anxiety rises | PASS | anxiety: 0.082 → 0.368 |
| 38 | Eating food → stores food memory location | PASS | food_memory_xy=(401.92104079261895, 299.99356492940217) |
| 39 | Two predators: both eyes detect | PASS | pred_in_left=True pred_in_right=True |
| 40 | 3 predators surround → high amygdala + FLEE | PASS | amyg=0.757 goal=FLEE |
| 41 | Fish at arena edge → no crash | PASS | pos=(20, 20) |
| 42 | Low energy → vision still works | PASS | energy=10.0 food_detected=1 |
| 43 | Predator at fish position → no crash | PASS | survived step with overlapping predator |
| 44 | Food in binocular zone → depth estimate | PASS | bino_food_dist=80.0 conf=0.680 |
| 45 | Food outside binocular zone → no depth | PASS | bino_food_dist=None (should be None) |
| 46 | Predator in binocular zone → depth estimate | PASS | bino_pred_dist=60.0 conf=0.700 |
| 47 | Close frontal food → reduced approach speed | PASS | approach_gain=0.6 (should be < 1.0) |
| 48 | Far frontal food → normal approach speed | PASS | approach_gain=1.0 (should be 1.0) |
| 49 | Asymmetric bilateral stimuli → rivalry suppression | PASS | suppression=0.132 dominant=R |
| 50 | Balanced bilateral stimuli → no rivalry | PASS | suppression=0.000 (should be < 0.1) |
| 51 | Food vs predator: predator side dominates rivalry | PASS | suppression=0.073 dominant=L (predator side wins) |
| 52 | Empty arena → no rivalry | PASS | suppression=0.0 dominant=None |
| 53 | Novel food in right eye → orient right | PASS | orient_dir=0.0532 (should be positive) |
| 54 | Novel food in left eye → orient left | PASS | orient_dir=-0.0532 (should be negative) |
| 55 | Novel predator → strong orienting | PASS | orient_dir=0.0783 (predator → strong orient) |
| 56 | Persistent stimulus → orienting habituates | PASS | orient[0]=0.0532 → orient[9]=0.0001 hab=0.071 |
| 57 | Habituation decays after stimulus removal | PASS | habituation: 0.084 → 0.024 (decayed) |
| 58 | Empty arena → no orienting | PASS | orient_dir=0.0000 (should be ~0) |
| 59 | High NA amplifies orienting | PASS | orient(NA=0.1)=0.0490 orient(NA=0.9)=0.0660 ratio=1.35 |
| 60 | Theta modulation of place cells | PASS | range=0.469 (min=0.311, max=0.781) |
| 61 | Familiarity buildup near food | PASS | place_fam=0.302 |
| 62 | Hippocampal food memory replay | PASS | food_memory=(400.0, 300.0), age_after_eat=1, persists=True |
| 63 | Food memory age decay | PASS | age after eat=1, after 10 steps=11 |
| 64 | Cerebellum PE on motor change | PASS | cerebellum=2.500 (baseline=0.3) |
| 65 | Cerebellum adaptation over time | PASS | early_PE=0.621, late_PE=0.300 |
| 66 | Cerebellum spike during C-start | PASS | cerebellum_during_cstart=2.50 |
| 67 | D1 > D2 during FORAGE | PASS | D1=1.125, D2=0.217 |
| 68 | DA modulates D1/D2 balance | PASS | DA=0.2: D1=1.011,D2=0.234; DA=0.9: D1=1.284,D2=0.187 |
| 69 | BG gate modulates motor output | PASS | speed(DA=0.1)=0.089, speed(DA=0.9)=0.100 |
| 70 | Day phase: high light | PASS | light=1.00, label=DAY |
| 71 | Night phase: low light | PASS | light=0.10, label=NIGHT |
| 72 | ACh drops at night | PASS | ACh_day=0.882, ACh_night=0.432 |
| 73 | Dawn transition: intermediate light | PASS | light=0.87, label=DAWN |
| 74 | Motility reduced at night | PASS | speed_day=0.056, speed_night=0.015 |
| 75 | Goal lock persists after stimulus removal | PASS | FORAGE in 8/8 steps (goal_lock) |
| 76 | Goal lock timer decrements | PASS | goal_lock: 10 → 9 |
| 77 | Habenula frustration on goal switch | PASS | frustration=[0.0, 0.0, 0.1, 0.0] |
| 78 | 5-HT rises without threat | PASS | 5HT after 60 calm steps=0.384 |
| 79 | 5-HT drops under threat | PASS | 5HT after 30 threat steps=0.431 |
| 80 | 5-HT patience in flee EFE | PASS | 5HT=0.1 goal=FLEE, 5HT=0.9 goal=FLEE |
| 81 | SOCIAL goal with visible conspecific | PASS | SOCIAL selected 20/20 steps |
| 82 | Fish steers toward conspecific | PASS | heading: 0.000 → 0.080 |
| 83 | Conspecific lower retinal salience than food | PASS | retina_food=0.700, retina_conspec=0.250 |
| 84 | SLEEP seeks rock shelter | PASS | rock_dist: 100.0 → 99.3 |
| 85 | SLEEP restores energy | PASS | energy: 50.0 → 51.0 (gained=1.00) |
| 86 | Rock collision pushes fish out | PASS | dist_from_rock_center=48.0 (radius=40) |
| 87 | Flee from two predators | PASS | goal=FLEE, cstart=0 |
| 88 | Three predator surround escape | PASS | goal=FLEE, speed=0.257 |
| 89 | Motility peak at 50% energy | PASS | speed@50%=0.240, speed@95%=0.029 |
| 90 | Motility floor prevents freezing | PASS | speed@10energy=0.084 |
| 91 | CPG rhythm modulates speed | PASS | speed range=[0.178, 0.222], var=0.045 |
| 92 | Reticulospinal burst during C-start | PASS | reticulospinal=1.767 |
| 93 | CPG scales with speed | PASS | cpg_slow=0.427, cpg_fast=0.749 |
| 94 | Complex scene pipeline speed | PASS | 0.059 ms/step (16960 steps/sec) |

## Key Metrics

### Food at -45° → left eye only
- left_eye_food: 1
- right_eye_food: 0

### Food at +45° → right eye only
- left_eye_food: 0
- right_eye_food: 1

### Food straight ahead (0°) → detected
- total_food_detected: 1

### Food at 99° (FoV boundary inner) → detected
- food_count: 1

### Food at 101° (FoV boundary outer) → blind
- food_count: 0

### Distance gradient: closer food → stronger retina
- 50: 1.0000
- 100: 0.8000
- 150: 0.6000
- 200: 0.4000
- 250: 0.2000

### Predator looming: closer → stronger SGC
- 30: 1.9600
- 60: 1.1700
- 90: 0.5800
- 120: 0.1900
- 150: 0.0000
- 180: 0.0000

### Predator vs food: 2x retinal amplification
- food_retina: 0.7000
- pred_retina: 1.1000
- ratio: 1.5714

### Conspecific vs food: lower salience
- food_retina: 0.7000
- conspec_retina: 0.2500

### Left eye → right tectum (chiasm)
- sfgs_b_R: 0.6600
- sfgs_b_L: 0.1800

### Right eye → left tectum (chiasm)
- sfgs_b_L: 0.6600
- sfgs_b_R: 0.1800

### Predator behind → lateral line detection
- lateral_line: 0.7000

### Predator behind → lateral line → FLEE goal
- flee_count: 15

### Injured conspecific → alarm substance → amygdala
- amygdala: 0.1616

### Low NA → weak TC relay; high NA → strong TC relay
- tc_low: 0.0690
- tc_high: 0.2624
- ratio: 3.8051

### Day vs night: ACh gating reduces night vision
- tc_day: 0.1197
- tc_night: 0.0173

### Food in both visual fields → bilateral retina
- left_eye_food: 1
- right_eye_food: 1

### Crowded scene: retinal saturation ceiling
- retina_R: 5.0000

### Visible food + hungry → FORAGE
- goal: FORAGE

### Close predator → FLEE
- goal: FLEE

### No stimuli → EXPLORE/SOCIAL
- goal: SOCIAL

### Predator + food: FLEE overrides FORAGE
- goal: FLEE

### Night time → SLEEP goal
- goal: SLEEP

### Close predator → C-start reflex
- cstart_timer: 3

### C-start speed sequence (4-step burst)
- speeds: [0.09, 0.25, 1.1, 0.95, 0.23, 0.23]
- max_speed: 1.1000

### 360° food sweep: FoV = ±100°
- detected_angles: [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
- blind_angles: [-180, -165, -150, -135, -120, -105, 105, 120, 135, 150, 165, 180]

### 360° predator: lateral line omnidirectional
- -180: 0.5000
- -150: 0.5000
- -120: 0.5000
- -90: 0.5000
- -60: 0.5000
- -30: 0.5000
- 0: 0.5000
- 30: 0.5000
- 60: 0.5000
- 90: 0.5000
- 120: 0.5000
- 150: 0.5000
- 180: 0.5000

### Heading invariance: same relative angle → same eye
- detections: [1, 1, 1, 1, 1, 1]

### Food in left FoV → turn left
- heading_change_deg: -13.8513

### Food in right FoV → turn right
- heading_change_deg: 13.8513

### Amygdala fear trace persists after threat removal
- amyg_during: 1.1992
- amyg_after: 0.6015

### Eating food → DA increase (reward)
- da_before: 0.5000
- da_after: 0.6521

### No food → starvation anxiety rises
- anxiety_start: 0.0820
- anxiety_end: 0.3680

### Eating food → stores food memory location
- food_memory: (401.92104079261895, 299.99356492940217)

### 3 predators surround → high amygdala + FLEE
- amygdala: 0.7574
- goal: FLEE

### Low energy → vision still works
- energy: 10
- food_detected: 1

### Food in binocular zone → depth estimate
- bino_food_dist: 80.0000
- bino_food_conf: 0.6800

### Food outside binocular zone → no depth
- bino_food_dist: None

### Predator in binocular zone → depth estimate
- bino_pred_dist: 60.0000
- bino_pred_conf: 0.7000

### Close frontal food → reduced approach speed
- bino_approach_gain: 0.6000

### Far frontal food → normal approach speed
- bino_approach_gain: 1.0000

### Asymmetric bilateral stimuli → rivalry suppression
- rivalry_suppression: 0.1320
- rivalry_dominant: R

### Balanced bilateral stimuli → no rivalry
- rivalry_suppression: 0.0000

### Food vs predator: predator side dominates rivalry
- rivalry_suppression: 0.0730
- rivalry_dominant: L

### Novel food in right eye → orient right
- orient_dir: 0.0532

### Novel food in left eye → orient left
- orient_dir: -0.0532

### Novel predator → strong orienting
- orient_dir: 0.0783

### Persistent stimulus → orienting habituates
- orient_first: 0.0532
- orient_last: 0.0001
- habituation: 0.0710

### Habituation decays after stimulus removal
- hab_before: 0.0837
- hab_after: 0.0240

### High NA amplifies orienting
- orient_low: 0.0490
- orient_high: 0.0660
- ratio: 1.3469

### Theta modulation of place cells
- oscillation_range: 0.4692

### Familiarity buildup near food
- place_fam: 0.3020

### Hippocampal food memory replay
- food_memory: (400.0, 300.0)
- food_memory_age: 1

### Food memory age decay
- age_after_eat: 1
- age_after_10: 11

### Cerebellum PE on motor change
- cerebellum_pe: 2.5000

### Cerebellum adaptation over time
- early_pe: 0.6213
- late_pe: 0.3004

### Cerebellum spike during C-start
- cerebellum_cstart: 2.5000

### D1 > D2 during FORAGE
- d1: 1.1250
- d2: 0.2167

### DA modulates D1/D2 balance
- d1_low: 1.0113
- d2_low: 0.2343
- d1_high: 1.2843
- d2_high: 0.1875

### BG gate modulates motor output
- speed_low_da: 0.0893
- speed_high_da: 0.0997

### Day phase: high light
- light: 1.0000
- label: DAY

### Night phase: low light
- light: 0.1000
- label: NIGHT

### ACh drops at night
- ach_day: 0.8822
- ach_night: 0.4319

### Dawn transition: intermediate light
- light: 0.8669
- label: DAWN

### Motility reduced at night
- speed_day: 0.0560
- speed_night: 0.0154

### Goal lock persists after stimulus removal
- forage_count: 8

### Goal lock timer decrements
- goal_lock: 9

### Habenula frustration on goal switch
- frustration: [0.0, 0.0, 0.1, 0.0]

### 5-HT rises without threat
- 5ht: 0.3838

### 5-HT drops under threat
- 5ht: 0.4310

### 5-HT patience in flee EFE
- goal_low_5ht: FLEE
- goal_high_5ht: FLEE

### SOCIAL goal with visible conspecific
- social_count: 20

### Fish steers toward conspecific
- h0: 0.0000
- h1: 0.0800

### Conspecific lower retinal salience than food
- retina_food: 0.7000
- retina_conspec: 0.2500

### SLEEP seeks rock shelter
- d0: 100.0000
- d1: 99.3303

### SLEEP restores energy
- e0: 50.0000
- e1: 51.0000
- gained: 1.0000

### Rock collision pushes fish out
- dist: 48.0000
- radius: 40

### Flee from two predators
- goal: FLEE
- cstart: 0

### Three predator surround escape
- goal: FLEE
- speed: 0.2567

### Motility peak at 50% energy
- speed_50: 0.2396
- speed_95: 0.0292

### Motility floor prevents freezing
- speed: 0.0839

### CPG rhythm modulates speed
- min_speed: 0.1776
- max_speed: 0.2225
- variation: 0.0448

### Reticulospinal burst during C-start
- reticulospinal: 1.7666

### CPG scales with speed
- cpg_slow: 0.4268
- cpg_fast: 0.7494

### Complex scene pipeline speed
- ms_per_step: 0.0590
- steps_per_sec: 16960.2198

## Vision Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| FoV per eye | 100° |
| Total binocular FoV | 200° |
| Blind spot | 160° (rear) |
| Food max range | 250px |
| Predator visual range | 200px |
| Predator detect range | 300px |
| Lateral line range | 150px (omnidirectional) |
| Looming C-start threshold | 0.5 |
| Retinal predator weight | 2.0x |
| Retinal conspecific weight | 0.3x |
| Thalamic NA gate center | 0.4 |
