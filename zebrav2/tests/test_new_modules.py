"""
Comprehensive test for all new v2 SNN modules.

Tests each module individually, then runs closed-loop integration.
Generates one figure per module + summary with pass/fail criteria.

Modules tested:
  1. Cerebellum (granule-Purkinje-DCN forward model)
  2. Habenula (disappointment signal)
  3. Predictive network (spiking world model)
  4. RL Critic (TD value learning)
  5. Habit network (fast cached actions)
  6. Interoception (spiking insular cortex)
  7. Full closed-loop integration (foraging + survival)

Run:
  .venv/bin/python -u -m zebrav2.tests.test_new_modules
"""
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.cerebellum import SpikingCerebellum
from zebrav2.brain.habenula import SpikingHabenula
from zebrav2.brain.predictive_net import SpikingPredictiveNet
from zebrav2.brain.rl_critic import SpikingCritic
from zebrav2.brain.habit_network import SpikingHabitNet
from zebrav2.brain.interoception import SpikingInsularCortex

PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_modules')
WEIGHT_DIR = os.path.join(PROJECT_ROOT, 'zebrav2', 'weights')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)

results = {}


def test_cerebellum():
    """Test 1: Spiking cerebellum forward model."""
    print("\n=== Test 1: Spiking Cerebellum ===")
    cb = SpikingCerebellum(device=DEVICE)

    # Simulate varying error signals and track LTD
    n_steps = 100
    gc_sparsities = []
    pc_rates = []
    dcn_rates = []
    pred_errors = []
    pf_weight_means = []

    for t in range(n_steps):
        # Varying mossy fiber input (sensory context)
        mossy = torch.rand(128, device=DEVICE) * (0.5 + 0.5 * np.sin(t * 0.1))
        # Climbing fiber error: high early, decreases as cerebellum learns
        cf_error = max(0.0, 0.5 - t * 0.003) + 0.1 * np.random.random()
        out = cb(mossy, cf_error, DA=0.5)

        gc_sparsities.append(out['gc_sparsity'])
        pc_rates.append(out['pc_rate_mean'])
        dcn_rates.append(float(out['dcn_rate'].mean()))
        pred_errors.append(out['prediction_error'])
        pf_weight_means.append(float(cb.W_pf.data.mean()))

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 1: Spiking Cerebellum (GC-PC-DCN)', fontsize=14, fontweight='bold')

    axes[0, 0].plot(gc_sparsities, 'b-', linewidth=1.5)
    axes[0, 0].set_title('GC Sparsity (fraction active)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].axhline(0.1, color='r', linestyle='--', label='target 10%')
    axes[0, 0].legend()

    axes[0, 1].plot(pc_rates, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Purkinje Cell Mean Rate')
    axes[0, 1].set_xlabel('Step')

    axes[0, 2].plot(dcn_rates, 'g-', linewidth=1.5)
    axes[0, 2].set_title('DCN Output Rate')
    axes[0, 2].set_xlabel('Step')

    axes[1, 0].plot(pred_errors, 'm-', linewidth=1.5)
    axes[1, 0].set_title('Prediction Error (CF)')
    axes[1, 0].set_xlabel('Step')

    axes[1, 1].plot(pf_weight_means, 'k-', linewidth=1.5)
    axes[1, 1].set_title('Parallel Fiber Weight Mean (LTD)')
    axes[1, 1].set_xlabel('Step')

    # Spike raster (last 50 steps snapshot)
    cb.reset()
    raster_gc = []
    raster_pc = []
    for t in range(50):
        mossy = torch.rand(128, device=DEVICE) * 0.5
        out = cb(mossy, 0.2, DA=0.5)
        raster_gc.append((cb.GC.rate > 0.01).float().cpu().numpy())
        raster_pc.append((cb.PC.rate > 0.01).float().cpu().numpy())
    raster_gc = np.array(raster_gc)
    raster_pc = np.array(raster_pc)
    axes[1, 2].imshow(raster_gc.T[:40], aspect='auto', cmap='Greys', interpolation='none')
    axes[1, 2].set_title('GC Spike Raster (40 cells, 50 steps)')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Neuron')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test1_cerebellum.png'), dpi=150)
    plt.close(fig)

    # Pass/fail
    gc_sparse_ok = np.mean(gc_sparsities) < 0.5  # sparse coding
    pc_active = np.mean(pc_rates) > 0.0  # PC fires
    dcn_active = np.mean(dcn_rates) > 0.0  # DCN fires
    ltd_occurred = pf_weight_means[-1] < pf_weight_means[0]  # LTD reduced weights

    results['CB_gc_sparse'] = ('PASS' if gc_sparse_ok else 'FAIL',
                                f'{np.mean(gc_sparsities):.3f}', '< 0.5')
    results['CB_pc_active'] = ('PASS' if pc_active else 'FAIL',
                                f'{np.mean(pc_rates):.4f}', '> 0')
    results['CB_dcn_active'] = ('PASS' if dcn_active else 'FAIL',
                                 f'{np.mean(dcn_rates):.4f}', '> 0')
    results['CB_ltd_learning'] = ('PASS' if ltd_occurred else 'FAIL',
                                   f'{pf_weight_means[-1]:.4f} < {pf_weight_means[0]:.4f}',
                                   'decreasing')

    for k, v in results.items():
        if k.startswith('CB_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test1_cerebellum.png")


def test_habenula():
    """Test 2: Spiking habenula (disappointment signal)."""
    print("\n=== Test 2: Spiking Habenula ===")
    hab = SpikingHabenula(device=DEVICE)

    n_steps = 100
    disappointments = []
    da_supps = []
    ht5_supps = []
    lhb_rates = []
    mhb_rates = []
    explore_drives = []

    # Phase 1 (0-30): high reward → low disappointment
    # Phase 2 (30-60): reward drops → high disappointment
    # Phase 3 (60-100): reward recovers → disappointment fades
    for t in range(n_steps):
        if t < 30:
            reward = 0.5 + 0.2 * np.random.random()
            aversion = 0.1
        elif t < 60:
            reward = 0.05  # sudden reward drop
            aversion = 0.3
        else:
            reward = 0.3 + 0.1 * np.random.random()
            aversion = 0.1

        out = hab(reward=reward, aversion=aversion)
        disappointments.append(out['disappointment'])
        da_supps.append(out['da_suppression'])
        ht5_supps.append(out['ht5_suppression'])
        lhb_rates.append(out['lhb_rate'])
        mhb_rates.append(out['mhb_rate'])
        explore_drives.append(out['explore_drive'])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 2: Spiking Habenula (Disappointment)', fontsize=14, fontweight='bold')

    axes[0, 0].plot(disappointments, 'r-', linewidth=1.5)
    axes[0, 0].set_title('Disappointment Signal')
    axes[0, 0].axvspan(30, 60, alpha=0.1, color='red', label='reward drop')
    axes[0, 0].legend()

    axes[0, 1].plot(da_supps, 'b-', label='DA suppress', linewidth=1.5)
    axes[0, 1].plot(ht5_supps, 'g-', label='5-HT suppress', linewidth=1.5)
    axes[0, 1].set_title('Neuromod Suppression')
    axes[0, 1].legend()

    axes[0, 2].plot(explore_drives, 'm-', linewidth=1.5)
    axes[0, 2].set_title('Exploration Drive')

    axes[1, 0].plot(lhb_rates, 'r-', linewidth=1.5)
    axes[1, 0].set_title('Lateral Hb Spike Rate')

    axes[1, 1].plot(mhb_rates, 'b-', linewidth=1.5)
    axes[1, 1].set_title('Medial Hb Spike Rate')

    # Bar: mean disappointment per phase
    phases = ['Reward High\n(0-30)', 'Reward Drop\n(30-60)', 'Reward Recover\n(60-100)']
    means = [np.mean(disappointments[:30]), np.mean(disappointments[30:60]),
             np.mean(disappointments[60:])]
    colors = ['green', 'red', 'orange']
    axes[1, 2].bar(phases, means, color=colors)
    axes[1, 2].set_title('Mean Disappointment by Phase')
    axes[1, 2].set_ylabel('Disappointment')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test2_habenula.png'), dpi=150)
    plt.close(fig)

    # Pass/fail
    disappoint_phase2 = np.mean(disappointments[35:60])
    disappoint_phase1 = np.mean(disappointments[:25])
    disappoint_rises = disappoint_phase2 > disappoint_phase1
    da_supp_during_drop = np.mean(da_supps[35:60])
    da_supp_works = da_supp_during_drop > 0.0

    results['HB_disappoint_rises'] = ('PASS' if disappoint_rises else 'FAIL',
                                       f'{disappoint_phase2:.3f} > {disappoint_phase1:.3f}',
                                       'phase2 > phase1')
    results['HB_da_suppression'] = ('PASS' if da_supp_works else 'FAIL',
                                     f'{da_supp_during_drop:.3f}', '> 0')
    results['HB_lhb_fires'] = ('PASS' if np.max(lhb_rates) > 0 else 'FAIL',
                                f'{np.max(lhb_rates):.4f}', '> 0')

    for k, v in results.items():
        if k.startswith('HB_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test2_habenula.png")


def test_predictive_net():
    """Test 3: Spiking predictive network (world model)."""
    print("\n=== Test 3: Spiking Predictive Network ===")
    pred = SpikingPredictiveNet(device=DEVICE)

    n_steps = 80
    surprises = []
    enc_sparsities = []
    pred_errors_mean = []
    latents_over_time = []

    for t in range(n_steps):
        # Varying retinal input: sinusoidal + noise
        retinal = torch.sigmoid(torch.randn(400, device=DEVICE) * 0.3
                                + 0.5 * np.sin(t * 0.1))
        motor = torch.tensor([0.3 * np.sin(t * 0.05), 1.0],
                             dtype=torch.float32, device=DEVICE)
        out = pred(retinal, motor)

        surprises.append(out['surprise'])
        enc_sparsities.append(out['enc_sparsity'])
        pred_errors_mean.append(float(out['pred_error'].abs().mean()))
        latents_over_time.append(out['latent'].cpu().numpy())

    latents_arr = np.array(latents_over_time)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 3: Spiking Predictive Network (World Model)', fontsize=14, fontweight='bold')

    axes[0, 0].plot(surprises, 'r-', linewidth=1.5)
    axes[0, 0].set_title('Prediction Surprise (MSE)')
    axes[0, 0].set_xlabel('Step')

    axes[0, 1].plot(enc_sparsities, 'b-', linewidth=1.5)
    axes[0, 1].set_title('Encoder Sparsity')

    axes[0, 2].plot(pred_errors_mean, 'm-', linewidth=1.5)
    axes[0, 2].set_title('Mean |Prediction Error|')

    # Latent space evolution (first 8 dims)
    for i in range(min(8, latents_arr.shape[1])):
        axes[1, 0].plot(latents_arr[:, i], alpha=0.6, linewidth=0.8)
    axes[1, 0].set_title('Latent Dimensions (8 of 32)')
    axes[1, 0].set_xlabel('Step')

    # Prediction vs actual (last step)
    pred_last = pred.prediction.cpu().numpy()
    actual_last = pred.prev_retinal.cpu().numpy()
    axes[1, 1].plot(actual_last[:50], 'b-', label='Actual', alpha=0.7)
    axes[1, 1].plot(pred_last[:50], 'r--', label='Predicted', alpha=0.7)
    axes[1, 1].set_title('Prediction vs Actual (50 pixels)')
    axes[1, 1].legend()

    # Surprise reduction over time
    window = 10
    if len(surprises) >= 2 * window:
        early = np.mean(surprises[:window])
        late = np.mean(surprises[-window:])
        axes[1, 2].bar(['Early', 'Late'], [early, late], color=['red', 'green'])
        axes[1, 2].set_title(f'Surprise: Early={early:.4f} → Late={late:.4f}')
    else:
        axes[1, 2].text(0.5, 0.5, 'Not enough steps', ha='center', va='center')
        early, late = 0, 0

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test3_predictive_net.png'), dpi=150)
    plt.close(fig)

    # Pass/fail
    encoder_active = np.mean(enc_sparsities) > 0.05
    has_latent_var = np.std(latents_arr) > 0.01
    surprise_finite = np.all(np.isfinite(surprises))

    results['PN_encoder_active'] = ('PASS' if encoder_active else 'FAIL',
                                     f'{np.mean(enc_sparsities):.3f}', '> 0.05')
    results['PN_latent_varies'] = ('PASS' if has_latent_var else 'FAIL',
                                    f'{np.std(latents_arr):.4f}', '> 0.01')
    results['PN_surprise_finite'] = ('PASS' if surprise_finite else 'FAIL',
                                      f'all finite', 'finite')

    for k, v in results.items():
        if k.startswith('PN_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test3_predictive_net.png")


def test_rl_critic():
    """Test 4: Spiking RL critic."""
    print("\n=== Test 4: Spiking RL Critic ===")
    critic = SpikingCritic(device=DEVICE)

    n_steps = 150
    values = []
    td_errors = []
    hidden_activities = []

    # Simulate: goal 0 (FORAGE) with occasional rewards
    for t in range(n_steps):
        energy = max(10.0, 100.0 - t * 0.3)
        threat = 0.1 + 0.3 * np.sin(t * 0.05)
        food_visible = 5.0 if t % 20 < 5 else 0.0
        reward = 0.01
        if t % 20 == 4:  # eat food
            reward = 10.0
        goal = 0  # FORAGE

        out = critic(energy=energy, threat=threat, food_visible=food_visible,
                     goal=goal, DA=0.5, reward=reward)
        values.append(out['values'].cpu().numpy().copy())
        td_errors.append(out['td_error'])
        hidden_activities.append(out['hidden_active'])

    values_arr = np.array(values)
    goal_names = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 4: Spiking RL Critic (TD Learning)', fontsize=14, fontweight='bold')

    for g in range(4):
        axes[0, 0].plot(values_arr[:, g], label=goal_names[g], linewidth=1.5)
    axes[0, 0].set_title('Value Estimates per Goal')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xlabel('Step')

    axes[0, 1].plot(td_errors, 'r-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('TD Error (δ)')
    axes[0, 1].axhline(0, color='k', linestyle='--')
    axes[0, 1].set_xlabel('Step')

    axes[0, 2].plot(hidden_activities, 'b-', linewidth=1.5)
    axes[0, 2].set_title('Hidden Layer Activity')
    axes[0, 2].set_xlabel('Step')

    # Value at reward events
    reward_steps = list(range(4, n_steps, 20))
    reward_vals = [values_arr[min(t, n_steps-1), 0] for t in reward_steps]
    axes[1, 0].scatter(reward_steps, reward_vals, c='red', s=30, zorder=5)
    axes[1, 0].plot(values_arr[:, 0], 'b-', alpha=0.5)
    axes[1, 0].set_title('FORAGE Value at Reward Events')

    # TD error histogram
    axes[1, 1].hist(td_errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('TD Error Distribution')
    axes[1, 1].axvline(0, color='r', linestyle='--')

    # Value evolution: early vs late
    early_val = np.mean(values_arr[:20, 0])
    late_val = np.mean(values_arr[-20:, 0])
    axes[1, 2].bar(['Early (0-20)', 'Late (130-150)'], [early_val, late_val],
                    color=['red', 'green'])
    axes[1, 2].set_title(f'FORAGE Value: {early_val:.3f} → {late_val:.3f}')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test4_rl_critic.png'), dpi=150)
    plt.close(fig)

    # Pass/fail
    hidden_ok = np.mean(hidden_activities) > 0.05
    td_varies = np.std(td_errors) > 0.001
    values_finite = np.all(np.isfinite(values_arr))

    results['CR_hidden_active'] = ('PASS' if hidden_ok else 'FAIL',
                                    f'{np.mean(hidden_activities):.3f}', '> 0.05')
    results['CR_td_varies'] = ('PASS' if td_varies else 'FAIL',
                                f'{np.std(td_errors):.4f}', '> 0.001')
    results['CR_values_finite'] = ('PASS' if values_finite else 'FAIL',
                                    'all finite', 'finite')

    for k, v in results.items():
        if k.startswith('CR_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test4_rl_critic.png")


def test_habit_network():
    """Test 5: Spiking habit network."""
    print("\n=== Test 5: Spiking Habit Network ===")
    habit = SpikingHabitNet(device=DEVICE)

    n_steps = 100
    confidences = []
    habit_strengths = []
    turns_out = []
    speeds_out = []

    # Repeated stimulus→action pairing: food left → turn left
    for t in range(n_steps):
        cls_probs = torch.tensor([0.1, 0.7, 0.1, 0.05, 0.05], device=DEVICE)
        goal = 0  # FORAGE
        turn = -0.5  # consistently turn left for food
        speed = 1.0

        out = habit(cls_probs=cls_probs, goal=goal, turn=turn, speed=speed)
        confidences.append(out['confidence'])
        habit_strengths.append(out['habit_strength'])
        turns_out.append(out['turn'])
        speeds_out.append(out['speed'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Test 5: Spiking Habit Network', fontsize=14, fontweight='bold')

    axes[0, 0].plot(confidences, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Habit Confidence')
    axes[0, 0].set_xlabel('Repetition')

    axes[0, 1].plot(habit_strengths, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Habit Strength (max)')
    axes[0, 1].set_xlabel('Repetition')

    axes[1, 0].plot(turns_out, 'g-', linewidth=1.5)
    axes[1, 0].axhline(-0.5, color='r', linestyle='--', label='target turn')
    axes[1, 0].set_title('Habit Turn Output')
    axes[1, 0].legend()

    axes[1, 1].plot(speeds_out, 'm-', linewidth=1.5)
    axes[1, 1].axhline(1.0, color='r', linestyle='--', label='target speed')
    axes[1, 1].set_title('Habit Speed Output')
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test5_habit.png'), dpi=150)
    plt.close(fig)

    # Pass/fail
    strength_grows = habit_strengths[-1] > habit_strengths[0]
    output_finite = np.all(np.isfinite(turns_out)) and np.all(np.isfinite(speeds_out))

    results['HN_strength_grows'] = ('PASS' if strength_grows else 'FAIL',
                                     f'{habit_strengths[-1]:.3f} > {habit_strengths[0]:.3f}',
                                     'increasing')
    results['HN_output_finite'] = ('PASS' if output_finite else 'FAIL',
                                    'all finite', 'finite')

    for k, v in results.items():
        if k.startswith('HN_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test5_habit.png")


def test_interoception():
    """Test 6: Spiking insular cortex (interoception)."""
    print("\n=== Test 6: Spiking Interoception ===")
    insula = SpikingInsularCortex(device=DEVICE)

    n_steps = 120
    hunger_rates = []
    fatigue_rates = []
    stress_rates = []
    valences = []
    arousals = []
    heart_rates = []
    heartbeats = []

    for t in range(n_steps):
        # Phase 1 (0-40): safe, energy declining
        # Phase 2 (40-80): threat, high stress
        # Phase 3 (80-120): recovery
        if t < 40:
            energy = max(20, 100 - t * 1.5)
            stress = 0.1
            fatigue = t * 0.005
            reward = 0.01
            threat_acute = False
        elif t < 80:
            energy = max(10, 40 - (t - 40) * 0.5)
            stress = 0.7
            fatigue = 0.4
            reward = 0.0
            threat_acute = True
        else:
            energy = min(80, 10 + (t - 80) * 1.5)
            stress = max(0.1, 0.7 - (t - 80) * 0.015)
            fatigue = max(0.1, 0.4 - (t - 80) * 0.007)
            reward = 0.3
            threat_acute = False

        out = insula(energy=energy, stress=stress, fatigue=fatigue,
                     reward=reward, threat_acute=threat_acute)
        hunger_rates.append(out['hunger_rate'])
        fatigue_rates.append(out['fatigue_rate'])
        stress_rates.append(out['stress_rate'])
        valences.append(out['valence'])
        arousals.append(out['arousal'])
        heart_rates.append(out['heart_rate'])
        heartbeats.append(1.0 if out['heartbeat'] else 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 6: Spiking Insular Cortex (Interoception)', fontsize=14, fontweight='bold')

    axes[0, 0].plot(hunger_rates, 'r-', label='Hunger', linewidth=1.5)
    axes[0, 0].plot(fatigue_rates, 'b-', label='Fatigue', linewidth=1.5)
    axes[0, 0].plot(stress_rates, 'g-', label='Stress', linewidth=1.5)
    axes[0, 0].set_title('Interoceptive Spike Rates')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].axvspan(40, 80, alpha=0.1, color='red', label='threat')

    axes[0, 1].plot(valences, 'purple', linewidth=1.5)
    axes[0, 1].axhline(0, color='k', linestyle='--')
    axes[0, 1].set_title('Emotional Valence')
    axes[0, 1].set_ylabel('-1 (negative) to +1 (positive)')

    axes[0, 2].plot(arousals, 'orange', linewidth=1.5)
    axes[0, 2].set_title('Arousal Level')

    axes[1, 0].plot(heart_rates, 'red', linewidth=1.5)
    axes[1, 0].set_title('Heart Rate (Hz)')
    axes[1, 0].axhline(2.0, color='k', linestyle='--', label='baseline')
    axes[1, 0].legend()

    # Heartbeat markers
    hb_steps = [i for i, h in enumerate(heartbeats) if h > 0.5]
    axes[1, 1].eventplot([hb_steps], lineoffsets=[0], linelengths=[0.5], color='red')
    axes[1, 1].set_title('Heartbeat Events')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_xlim(0, n_steps)

    # Phase comparison
    phases = ['Safe\n(0-40)', 'Threat\n(40-80)', 'Recovery\n(80-120)']
    stress_by_phase = [np.mean(stress_rates[:40]), np.mean(stress_rates[40:80]),
                       np.mean(stress_rates[80:])]
    axes[1, 2].bar(phases, stress_by_phase, color=['green', 'red', 'orange'])
    axes[1, 2].set_title('Stress Rate by Phase')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test6_interoception.png'), dpi=150)
    plt.close(fig)

    # Pass/fail
    stress_rises = np.mean(stress_rates[40:80]) > np.mean(stress_rates[:40])
    hr_rises = np.mean(heart_rates[40:80]) > np.mean(heart_rates[:20])
    valence_drops = np.mean(valences[50:80]) < np.mean(valences[:20])

    results['IN_stress_rises'] = ('PASS' if stress_rises else 'FAIL',
                                   f'{np.mean(stress_rates[40:80]):.4f} > {np.mean(stress_rates[:40]):.4f}',
                                   'threat > safe')
    results['IN_hr_rises'] = ('PASS' if hr_rises else 'FAIL',
                               f'{np.mean(heart_rates[40:80]):.2f} > {np.mean(heart_rates[:20]):.2f}',
                               'threat > safe')
    results['IN_valence_drops'] = ('PASS' if valence_drops else 'FAIL',
                                    f'{np.mean(valences[50:80]):.3f} < {np.mean(valences[:20]):.3f}',
                                    'threat < safe')

    for k, v in results.items():
        if k.startswith('IN_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test6_interoception.png")


def test_closed_loop():
    """Test 7: Full closed-loop integration with all modules."""
    print("\n=== Test 7: Closed-Loop Integration ===")

    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav2.brain.sensory_bridge import inject_sensory

    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=500)
    brain = ZebrafishBrainV2(device=DEVICE)
    obs, info = env.reset(seed=42)
    brain.reset()

    T = 500
    goals_over_time = []
    energies = []
    food_eaten_cumul = []
    da_vals = []
    valences = []
    heart_rates = []
    surprises = []
    critic_values = []
    habit_confs = []
    cb_pes = []
    hab_disps = []
    turns = []
    speeds = []
    total_eaten = 0

    for t in range(T):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1, panic_intensity=0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)

        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        env._eaten_now = info.get('food_eaten_this_step', 0)
        total_eaten += env._eaten_now

        goals_over_time.append(brain.current_goal)
        energies.append(brain.energy)
        food_eaten_cumul.append(total_eaten)
        da_vals.append(out['DA'])
        valences.append(out.get('insula_valence', 0.0))
        heart_rates.append(out.get('insula_heart_rate', 2.0))
        surprises.append(out.get('predictive_surprise', 0.0))
        critic_values.append(out.get('critic_value', 0.0))
        habit_confs.append(out.get('habit_confidence', 0.0))
        cb_pes.append(out.get('cerebellum_pe', 0.0))
        hab_disps.append(out.get('habenula_disappoint', 0.0))
        turns.append(out['turn'])
        speeds.append(out['speed'])

        if terminated or truncated:
            break

    survived = t + 1

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f'Test 7: Closed-Loop Integration ({survived} steps, {total_eaten} food)',
                 fontsize=14, fontweight='bold')

    # Row 1
    goal_colors = {0: 'green', 1: 'red', 2: 'blue', 3: 'cyan'}
    goal_names = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']
    for i in range(len(goals_over_time)):
        axes[0, 0].axvspan(i, i+1, color=goal_colors[goals_over_time[i]], alpha=0.3)
    axes[0, 0].set_title('Goal Timeline')
    axes[0, 0].set_xlim(0, survived)
    # Legend
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(color=goal_colors[g], label=goal_names[g]) for g in range(4)],
                      fontsize=7, loc='upper right')

    axes[0, 1].plot(energies, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Energy')
    axes[0, 1].axhline(30, color='orange', linestyle='--', label='critical')
    axes[0, 1].legend()

    axes[0, 2].plot(food_eaten_cumul, 'g-', linewidth=2)
    axes[0, 2].set_title(f'Cumulative Food: {total_eaten}')

    axes[0, 3].plot(da_vals, 'orange', linewidth=1, alpha=0.7)
    axes[0, 3].set_title('Dopamine')

    # Row 2: new module outputs
    axes[1, 0].plot(valences, 'purple', linewidth=1.5)
    axes[1, 0].axhline(0, color='k', linestyle='--')
    axes[1, 0].set_title('Insula Valence')

    axes[1, 1].plot(heart_rates, 'red', linewidth=1)
    axes[1, 1].set_title('Heart Rate (Hz)')

    axes[1, 2].plot(surprises, 'blue', linewidth=1, alpha=0.7)
    axes[1, 2].set_title('Predictive Surprise')

    axes[1, 3].plot(critic_values, 'green', linewidth=1)
    axes[1, 3].set_title('Critic Value')

    # Row 3
    axes[2, 0].plot(habit_confs, 'cyan', linewidth=1)
    axes[2, 0].set_title('Habit Confidence')

    axes[2, 1].plot(cb_pes, 'magenta', linewidth=1, alpha=0.7)
    axes[2, 1].set_title('Cerebellum PE')

    axes[2, 2].plot(hab_disps, 'red', linewidth=1, alpha=0.7)
    axes[2, 2].set_title('Habenula Disappointment')

    axes[2, 3].plot(turns, 'blue', linewidth=0.5, alpha=0.5)
    axes[2, 3].plot(speeds, 'orange', linewidth=0.5, alpha=0.5)
    axes[2, 3].set_title('Motor: Turn (blue) + Speed (orange)')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test7_closed_loop.png'), dpi=150)
    plt.close(fig)
    env.close()

    # Pass/fail
    results['CL_survived'] = ('PASS' if survived >= 200 else 'FAIL',
                                str(survived), '>= 200')
    results['CL_food_eaten'] = ('PASS' if total_eaten >= 1 else 'FAIL',
                                 str(total_eaten), '>= 1')
    results['CL_goal_diversity'] = ('PASS' if len(set(goals_over_time)) >= 2 else 'FAIL',
                                     str(len(set(goals_over_time))), '>= 2')
    results['CL_energy_stable'] = ('PASS' if energies[-1] > 0 else 'FAIL',
                                    f'{energies[-1]:.1f}', '> 0')
    # New module activity checks
    results['CL_insula_active'] = ('PASS' if np.std(valences) > 0.001 else 'FAIL',
                                    f'{np.std(valences):.4f}', '> 0.001')
    results['CL_critic_active'] = ('PASS' if np.std(critic_values) > 0.001 else 'FAIL',
                                    f'{np.std(critic_values):.4f}', '> 0.001')
    results['CL_predictive_active'] = ('PASS' if np.mean(surprises) > 0.0 else 'FAIL',
                                        f'{np.mean(surprises):.4f}', '> 0')

    for k, v in results.items():
        if k.startswith('CL_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")
    print(f"  Saved: {PLOT_DIR}/test7_closed_loop.png")


def generate_summary():
    """Generate summary figure with pass/fail table."""
    print("\n=== SUMMARY ===")

    n_pass = sum(1 for v in results.values() if v[0] == 'PASS')
    n_total = len(results)

    fig, ax = plt.subplots(figsize=(12, max(6, n_total * 0.35 + 2)))
    fig.suptitle(f'v2 New Module Test Results: {n_pass}/{n_total} PASS ({100*n_pass/max(1,n_total):.0f}%)',
                 fontsize=14, fontweight='bold')

    # Table
    col_labels = ['Test', 'Status', 'Value', 'Threshold']
    cell_colors = []
    table_data = []

    for name, (status, value, threshold) in sorted(results.items()):
        table_data.append([name, status, str(value), str(threshold)])
        if status == 'PASS':
            cell_colors.append(['white', '#d4edda', 'white', 'white'])
        else:
            cell_colors.append(['white', '#f8d7da', 'white', 'white'])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors, loc='center',
                     colWidths=[0.25, 0.1, 0.35, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'summary_new_modules.png'), dpi=150)
    plt.close(fig)

    print(f"\n  Results: {n_pass}/{n_total} PASS ({100*n_pass/max(1,n_total):.0f}%)")
    print(f"  Figures saved to: {PLOT_DIR}/")
    for name, (status, value, threshold) in sorted(results.items()):
        mark = 'OK' if status == 'PASS' else 'XX'
        print(f"  [{mark}] {name}: {value} (threshold: {threshold})")


if __name__ == '__main__':
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print(f"Testing 6 new SNN modules + closed-loop integration")

    test_cerebellum()
    test_habenula()
    test_predictive_net()
    test_rl_critic()
    test_habit_network()
    test_interoception()
    test_closed_loop()
    generate_summary()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
