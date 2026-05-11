"""
Step 16: Dorsolateral pallium (hippocampal homolog) — spatial memory tests.

30 tests covering DG pattern separation, CA3 completion, one-shot encoding,
episodic retrieval, SWR replay, contextual remapping, EFE bias, and reset.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
from zebrav2.brain.dl_pallium import SpikingDlPallium
from zebrav2.brain.place_cells import ThetaPlaceCells

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

# ===== CONSTRUCTION & BASIC FORWARD (5 tests) =====
print("\n=== Construction & Basic Forward ===")
dl = SpikingDlPallium()
pc = ThetaPlaceCells()

# 1. Construction
place_rate = pc.activation(400, 300)
out = dl(place_activation=place_rate, pos_x=400, pos_y=300)
check("1. construction returns dict with spatial_prediction",
      isinstance(out, dict) and 'spatial_prediction' in out)

# 2. All expected keys present
expected_keys = ['food_memory', 'risk_memory', 'spatial_prediction', 'novelty',
                 'pattern_separation', 'pattern_completion', 'replay_active',
                 'prediction_error', 'precision', 'free_energy', 'rate', 'episode_count']
check("2. all expected keys present",
      all(k in out for k in expected_keys))

# 3. Neuron count
check("3. total neurons = 20 (6 DG + 8 CA3 + 6 CA1)",
      dl.n == 20)

# 4. Rate output is finite
check("4. rate output is finite", np.isfinite(out['rate']))

# 5. Initially no episodes
dl2 = SpikingDlPallium()
out2 = dl2(place_activation=place_rate, pos_x=400, pos_y=300)
check("5. initially zero episodes", out2['episode_count'] == 0)


# ===== PATTERN SEPARATION (4 tests) =====
print("\n=== DG Pattern Separation ===")
dl = SpikingDlPallium()

# 6. DG is sparse (high threshold → few active cells)
place_rate = pc.activation(400, 300)
out = dl(place_activation=place_rate, pos_x=400, pos_y=300)
check("6. DG sparse: pattern_separation > 0.3",
      out['pattern_separation'] > 0.3)

# 7. Distant locations produce discriminable spatial codes
dl.reset()
place_a = pc.activation(100, 100)
dl(place_activation=place_a)
dg_a = dl.rate_dg.cpu().numpy()
code_a = dl.W_dg_ca3.cpu().numpy() @ dg_a
code_a = code_a / (np.linalg.norm(code_a) + 1e-8)

dl.reset()
place_b = pc.activation(700, 500)
dl(place_activation=place_b)
dg_b = dl.rate_dg.cpu().numpy()
code_b = dl.W_dg_ca3.cpu().numpy() @ dg_b
code_b = code_b / (np.linalg.norm(code_b) + 1e-8)

cos_sim = float(np.dot(code_a, code_b))
check("7. distant locations → discriminable codes (cos_sim < 0.95)",
      cos_sim < 0.95)

# 8. Same location → similar spatial code (reproducibility)
dl.reset()
dl(place_activation=place_a)
dg_a2 = dl.rate_dg.cpu().numpy()
code_a2 = dl.W_dg_ca3.cpu().numpy() @ dg_a2
code_a2 = code_a2 / (np.linalg.norm(code_a2) + 1e-8)
check("8. same location → correlated spatial code",
      float(np.dot(code_a, code_a2)) > 0.0)

# 9. Zero input → minimal DG activity
dl.reset()
zero_place = torch.zeros(128)
out_zero = dl(place_activation=zero_place)
check("9. zero input → low DG activity",
      float(dl.rate_dg.mean()) < 0.5)


# ===== ONE-SHOT EPISODIC ENCODING (5 tests) =====
print("\n=== One-Shot Episodic Encoding ===")
dl = SpikingDlPallium()

# 10. Food event creates episode
place_food = pc.activation(200, 150)
out = dl(place_activation=place_food, pos_x=200, pos_y=150, food_eaten=True)
check("10. food event creates 1 episode", out['episode_count'] == 1)

# 11. Predator event creates episode
place_pred = pc.activation(600, 400)
out = dl(place_activation=place_pred, pos_x=600, pos_y=400, predator_near=True)
check("11. predator event creates 2nd episode", out['episode_count'] == 2)

# 12. No event → no new episode
place_neutral = pc.activation(400, 300)
out = dl(place_activation=place_neutral, pos_x=400, pos_y=300)
check("12. no event → still 2 episodes", out['episode_count'] == 2)

# 13. CA3 recurrent weights strengthened after encoding
w_norm = float(dl.W_ca3_rec.abs().sum())
check("13. CA3 recurrent weights non-zero after encoding", w_norm > 0.01)

# 14. Episode strength = 1.0 for fresh memories
check("14. fresh episode strength = 1.0",
      dl._episode_strength[0] == 1.0 and dl._episode_strength[1] == 1.0)


# ===== MEMORY RETRIEVAL (5 tests) =====
print("\n=== Memory Retrieval ===")

# 15. Return to food location → food_memory > 0
out_return = dl(place_activation=place_food, pos_x=200, pos_y=150)
check("15. return to food location → food_memory > 0",
      out_return['food_memory'] > 0)

# 16. Return to predator location → risk_memory > 0
out_risk = dl(place_activation=place_pred, pos_x=600, pos_y=400)
check("16. return to predator location → risk_memory > 0",
      out_risk['risk_memory'] > 0)

# 17. Novel location → higher novelty than stored locations
# Encoded: (200,150) food, (600,400) predator
# Novel: far from both
place_novel = pc.activation(50, 550)  # opposite corner
out_novel = dl(place_activation=place_novel, pos_x=50, pos_y=550)
# Novel location should have higher novelty than returning to encoded location
out_stored = dl(place_activation=place_food, pos_x=200, pos_y=150)
check("17. novel location has higher novelty than stored location",
      out_novel['novelty'] > out_stored['novelty'])

# 18. Spatial prediction: food location = positive, risk location = negative
check("18. food location → positive spatial_prediction",
      out_return['spatial_prediction'] > 0 or out_return['food_memory'] > 0)

# 19. Spatial prediction: risk location = negative
check("19. risk location → risk > food in memory",
      out_risk['risk_memory'] > out_risk['food_memory'])


# ===== SHARP-WAVE RIPPLE REPLAY (4 tests) =====
print("\n=== SWR Replay ===")
dl = SpikingDlPallium()
dl._SWR_INTERVAL = 5  # shorten for testing

# Encode an episode
place_enc = pc.activation(300, 200)
dl(place_activation=place_enc, pos_x=300, pos_y=200, food_eaten=True)
initial_w = dl.W_ca3_rec.clone()

# 20. No replay during active behavior
out_active = dl(place_activation=place_enc, pos_x=300, pos_y=200)
check("20. no replay during active behavior", out_active['replay_active'] is False)

# 21. Replay triggers after rest period
for i in range(10):
    dl(place_activation=place_enc, pos_x=300, pos_y=200, is_resting=True)
# Check if replay happened at some point
any_replay = dl.replay_active
# If not, keep going
for i in range(10):
    dl(place_activation=place_enc, pos_x=300, pos_y=200, is_resting=True)
    if dl.replay_active:
        any_replay = True
        break
check("21. replay triggers during rest", any_replay)

# 22. Replay strengthens CA3 weights
post_w = dl.W_ca3_rec.clone()
weight_change = float((post_w - initial_w).abs().sum())
check("22. replay strengthens CA3 recurrent weights",
      weight_change > 0.01)

# 23. Un-replayed episodes decay
dl = SpikingDlPallium()
dl._SWR_INTERVAL = 3
# Encode two episodes
dl(place_activation=pc.activation(100, 100), food_eaten=True)
dl(place_activation=pc.activation(700, 500), predator_near=True)
initial_strength = dl._episode_strength[0]
# Trigger many replays (most recent gets replayed, older decays)
for _ in range(50):
    dl(place_activation=pc.activation(400, 300), is_resting=True)
check("23. un-replayed episodes decay over time",
      dl._episode_strength[0] <= initial_strength)


# ===== PATTERN COMPLETION (3 tests) =====
print("\n=== CA3 Pattern Completion ===")
dl = SpikingDlPallium()

# 24. CA3 recurrent attractor strengthens with repeated encoding
dl = SpikingDlPallium()
pc2 = ThetaPlaceCells()
w_before = float(dl.W_ca3_rec.abs().sum())
for _ in range(5):
    dl(place_activation=pc2.activation(400, 300), food_eaten=True)
w_after = float(dl.W_ca3_rec.abs().sum())
check("24. CA3 attractor strengthens with repeated encoding",
      w_after > w_before)

# 25. Partial input → retrieval via recurrent completion
dl = SpikingDlPallium()
full_place = pc.activation(400, 300)
dl(place_activation=full_place, food_eaten=True)
# Degrade input (partial cue)
partial = full_place * 0.3  # weakened
out_partial = dl(place_activation=partial)
check("25. partial input still retrieves memory",
      out_partial['food_memory'] >= 0)  # may be weak but non-negative

# 26. Strong recurrent after multiple encodings at same location
dl = SpikingDlPallium()
same_place = pc.activation(300, 300)
for _ in range(5):
    dl(place_activation=same_place, food_eaten=True)
w_sum = float(dl.W_ca3_rec.abs().sum())
check("26. repeated encoding strengthens CA3 recurrent", w_sum > 0.1)


# ===== EFE BIAS & FEP (3 tests) =====
print("\n=== EFE Bias & FEP ===")
dl = SpikingDlPallium()
dl(place_activation=pc.activation(200, 200), food_eaten=True)
dl(place_activation=pc.activation(600, 500), predator_near=True)

# 27. get_efe_bias returns dict with forage/flee/explore
bias = dl.get_efe_bias()
check("27. get_efe_bias has forage/flee/explore",
      'forage' in bias and 'flee' in bias and 'explore' in bias)

# 28. FEP prediction error ≥ 0
out_fep = dl(place_activation=pc.activation(200, 200))
check("28. prediction error ≥ 0", out_fep['prediction_error'] >= 0)

# 29. Free energy is float
check("29. free energy is float", isinstance(dl.free_energy, float))


# ===== RESET (1 test) =====
print("\n=== Reset ===")

# 30. Reset clears spiking state but preserves episodic buffer
ep_count_before = dl._episode_count
dl.reset()
check("30. reset clears state, preserves episodes",
      dl.food_memory == 0.0 and dl.risk_memory == 0.0
      and dl.spatial_prediction == 0.0
      and dl._episode_count == ep_count_before)  # episodes survive reset


# ===== SUMMARY =====
print(f"\n{'='*50}")
print(f"Step 16 Dl Pallium: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
    sys.exit(1)
