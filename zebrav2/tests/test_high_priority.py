"""
Tests for high-priority additions: CPG, VAE, enhanced habenula, decision scenarios.

Run: .venv/bin/python -u -m zebrav2.tests.test_high_priority
"""
import os, sys, time, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_high_priority')
os.makedirs(PLOT_DIR, exist_ok=True)
results = {}

GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']


def test_cpg():
    """Test 1: Spinal CPG half-centre oscillator."""
    print("\n=== Test 1: Spinal CPG ===")
    from zebrav2.brain.spinal_cpg import SpinalCPG
    cpg = SpinalCPG(device=DEVICE)

    n_steps = 200
    motor_Ls, motor_Rs, phases = [], [], []
    speeds_out, turns_out = [], []

    for t in range(n_steps):
        # Ramp drive from 0.2 to 1.0 (stronger to trigger oscillation)
        drive = min(1.0, 0.2 + t * 0.004)
        turn_bias = 0.3 * np.sin(t * 0.03)
        # Run multiple CPG substeps per behavioral step (like brain does)
        for _ in range(5):
            mL, mR, spd, trn, diag = cpg.step(drive, turn_bias)
        motor_Ls.append(mL)
        motor_Rs.append(mR)
        phases.append(diag['phase'])
        speeds_out.append(spd)
        turns_out.append(trn)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 1: Spinal CPG (Half-Centre Oscillator)', fontsize=14, fontweight='bold')

    axes[0, 0].plot(motor_Ls, 'b-', label='Left', linewidth=1.5)
    axes[0, 0].plot(motor_Rs, 'r-', label='Right', linewidth=1.5)
    axes[0, 0].set_title('Motor Neuron Output (L/R)')
    axes[0, 0].legend()

    axes[0, 1].plot(phases, 'g-', linewidth=1)
    axes[0, 1].set_title('Phase (R/(L+R))')
    axes[0, 1].axhline(0.5, color='k', linestyle='--')

    axes[0, 2].plot(speeds_out, 'orange', linewidth=1.5)
    axes[0, 2].set_title('CPG Speed Output')

    axes[1, 0].plot(turns_out, 'purple', linewidth=1)
    axes[1, 0].set_title('CPG Turn Output')

    # FFT of motor_L for oscillation frequency
    motor_arr = np.array(motor_Ls[50:])  # skip transient
    if motor_arr.std() > 0.001:
        fft_vals = np.abs(np.fft.rfft(motor_arr - motor_arr.mean()))
        freqs = np.fft.rfftfreq(len(motor_arr), d=1.0)
        axes[1, 1].plot(freqs[:30], fft_vals[:30], 'b-', linewidth=1.5)
        axes[1, 1].set_title('FFT of Motor L (oscillation)')
        axes[1, 1].set_xlabel('Frequency (cycles/step)')
        peak_freq = freqs[np.argmax(fft_vals[1:]) + 1] if len(fft_vals) > 1 else 0
    else:
        axes[1, 1].text(0.5, 0.5, 'No oscillation', ha='center', va='center')
        peak_freq = 0

    # L/R alternation check
    alt_count = 0
    for i in range(1, len(motor_Ls)):
        if (motor_Ls[i] > motor_Rs[i]) != (motor_Ls[i-1] > motor_Rs[i-1]):
            alt_count += 1
    axes[1, 2].bar(['Alternations', 'Steps'], [alt_count, n_steps], color=['blue', 'gray'])
    axes[1, 2].set_title(f'L/R Alternations: {alt_count}')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test1_cpg.png'), dpi=150)
    plt.close(fig)

    has_oscillation = alt_count > 3
    motor_active = max(motor_Ls) > 0.01
    results['CPG_oscillation'] = ('PASS' if has_oscillation else 'FAIL',
                                   f'{alt_count} alternations', '> 3')
    results['CPG_motor_active'] = ('PASS' if motor_active else 'FAIL',
                                    f'{max(motor_Ls):.4f}', '> 0.01')
    for k, v in results.items():
        if k.startswith('CPG_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")


def test_vae():
    """Test 2: VAE World Model with online training."""
    print("\n=== Test 2: VAE World Model ===")
    from zebrav2.brain.vae_world_model import VAEWorldModelV2
    vae = VAEWorldModelV2(tectum_dim=200, pool_dim=32, state_ctx_dim=13,
                          latent_dim=16, warmup_steps=20, device=DEVICE)

    n_steps = 100
    vae_losses = []
    memory_nodes = []
    latents = []
    z_prev = None
    last_act = None

    for t in range(n_steps):
        tect = torch.rand(1, 200, device=DEVICE) * (0.5 + 0.3 * np.sin(t * 0.1))
        ctx = np.random.randn(13).astype(np.float32) * 0.3
        vae.train_step(tect, ctx)
        z, mu = vae.encode(tect, ctx)
        latents.append(z.copy())
        # Transition training
        act = np.array([0.1 * np.sin(t * 0.05), 0.8, 1, 0, 0], dtype=np.float32)
        if z_prev is not None and last_act is not None:
            vae.update_transition(z_prev, last_act, z)
        z_prev = z.copy()
        last_act = act.copy()
        # Memory
        vae.update_memory(z, eaten=1 if t % 20 == 0 else 0, pred_dist=200.0)
        vae_losses.append(vae._last_vae_loss)
        memory_nodes.append(vae.memory.n_allocated)

    latents_arr = np.array(latents)
    diag = vae.get_diagnostics()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 2: VAE World Model (Online ELBO Training)', fontsize=14, fontweight='bold')

    axes[0, 0].plot(vae_losses, 'r-', linewidth=1)
    axes[0, 0].set_title('VAE ELBO Loss')
    axes[0, 0].set_xlabel('Step')

    axes[0, 1].plot(memory_nodes, 'b-', linewidth=1.5)
    axes[0, 1].set_title(f'Associative Memory Nodes: {memory_nodes[-1]}')

    for i in range(min(8, latents_arr.shape[1])):
        axes[0, 2].plot(latents_arr[:, i], alpha=0.6, linewidth=0.8)
    axes[0, 2].set_title('Latent z (8 dims)')

    # Planning output
    G_plan = vae.plan(z, np.array([0.1, 0.8]), dopa=0.5)
    axes[1, 0].bar(['FORAGE', 'FLEE', 'EXPLORE'], G_plan, color=['green', 'red', 'blue'])
    axes[1, 0].set_title('Planning G (lower=preferred)')

    # Loss reduction
    if len(vae_losses) > 20:
        early = np.mean([x for x in vae_losses[:10] if x > 0] or [0])
        late = np.mean([x for x in vae_losses[-10:] if x > 0] or [0])
        axes[1, 1].bar(['Early', 'Late'], [early, late], color=['red', 'green'])
        axes[1, 1].set_title(f'Loss: {early:.4f} → {late:.4f}')
    else:
        axes[1, 1].text(0.5, 0.5, 'Too few steps', ha='center')
        early, late = 0, 0

    # Latent variance
    axes[1, 2].bar(range(min(16, latents_arr.shape[1])),
                   np.std(latents_arr, axis=0)[:16], color='purple')
    axes[1, 2].set_title('Per-dim Latent Std')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test2_vae.png'), dpi=150)
    plt.close(fig)

    trained = any(l > 0 for l in vae_losses)
    mem_ok = memory_nodes[-1] > 0
    latent_ok = np.std(latents_arr) > 0.01
    results['VAE_trained'] = ('PASS' if trained else 'FAIL',
                               f'loss={vae_losses[-1]:.4f}', 'loss > 0')
    results['VAE_memory'] = ('PASS' if mem_ok else 'FAIL',
                              f'{memory_nodes[-1]} nodes', '> 0')
    results['VAE_latent_var'] = ('PASS' if latent_ok else 'FAIL',
                                  f'{np.std(latents_arr):.4f}', '> 0.01')
    for k, v in results.items():
        if k.startswith('VAE_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")


def test_habenula_enhanced():
    """Test 3: Enhanced habenula with per-goal frustration."""
    print("\n=== Test 3: Enhanced Habenula ===")
    from zebrav2.brain.habenula import SpikingHabenula
    hab = SpikingHabenula(device=DEVICE)

    n_steps = 150
    frustrations = []
    switches = []
    helplessness_vals = []
    goal_biases = []

    current_goal = 0
    # Pre-load expected reward high; slow EMA so it stays high during phase 1
    hab.expected_reward = 1.0
    hab.reward_ema_alpha = 0.01  # slow adaptation
    for t in range(n_steps):
        # Phase 1 (0-50): forage with low reward → frustration builds (RPE negative)
        if t < 50:
            reward = 0.01
            goal = 0
        # Phase 2 (50-100): switch forced, new goal gets reward
        elif t < 100:
            reward = 0.5 if t % 10 == 0 else 0.01
            goal = current_goal
        # Phase 3 (100-150): back to foraging, but now habenula remembers
        else:
            reward = 0.01
            goal = 0

        out = hab(reward=reward, aversion=0.1, current_goal=goal, DA=0.5)
        frustrations.append(out['frustration'].copy())
        switches.append(out['switch_signal'])
        helplessness_vals.append(out['helplessness'])
        goal_biases.append(out['goal_bias'].copy())

        if out['switch_signal']:
            # Pick least frustrated goal
            frust = out['frustration']
            candidates = [g for g in range(4) if g != current_goal]
            current_goal = min(candidates, key=lambda g: frust[g])

    frust_arr = np.array(frustrations)
    bias_arr = np.array(goal_biases)
    switch_steps = [i for i, s in enumerate(switches) if s]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 3: Enhanced Habenula (Per-Goal Frustration)', fontsize=14, fontweight='bold')

    for g in range(4):
        axes[0, 0].plot(frust_arr[:, g], label=GOAL_NAMES[g], linewidth=1.5)
    axes[0, 0].set_title('Per-Goal Frustration')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].axhline(0.7, color='k', linestyle='--', label='switch threshold')

    axes[0, 1].plot(helplessness_vals, 'r-', linewidth=1.5)
    axes[0, 1].set_title('Overall Helplessness')

    # Switch events
    if switch_steps:
        axes[0, 2].eventplot([switch_steps], lineoffsets=[0], linelengths=[0.5], color='red')
    axes[0, 2].set_title(f'Strategy Switches: {len(switch_steps)}')
    axes[0, 2].set_xlim(0, n_steps)

    for g in range(4):
        axes[1, 0].plot(bias_arr[:, g], label=GOAL_NAMES[g], linewidth=1)
    axes[1, 0].set_title('Goal Avoidance Bias')
    axes[1, 0].legend(fontsize=8)

    # Frustration by phase
    phases = ['No Reward\n(0-50)', 'With Reward\n(50-100)', 'No Reward\n(100-150)']
    forage_frust = [np.mean(frust_arr[:50, 0]), np.mean(frust_arr[50:100, 0]),
                    np.mean(frust_arr[100:, 0])]
    axes[1, 1].bar(phases, forage_frust, color=['red', 'green', 'orange'])
    axes[1, 1].set_title('FORAGE Frustration by Phase')

    axes[1, 2].text(0.1, 0.7, f'Switch events: {len(switch_steps)}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Max frustration: {frust_arr.max():.3f}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f'Threshold: 0.7', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test3_habenula.png'), dpi=150)
    plt.close(fig)

    frust_builds = frust_arr[49, 0] > frust_arr[0, 0]
    has_bias = bias_arr.max() > 0
    results['HAB_frust_builds'] = ('PASS' if frust_builds else 'FAIL',
                                    f'{frust_arr[49,0]:.3f} > {frust_arr[0,0]:.3f}', 'increasing')
    results['HAB_goal_bias'] = ('PASS' if has_bias else 'FAIL',
                                 f'{bias_arr.max():.3f}', '> 0')
    results['HAB_switch_works'] = ('PASS' if len(switch_steps) > 0 else 'FAIL',
                                    f'{len(switch_steps)} switches', '> 0')
    for k, v in results.items():
        if k.startswith('HAB_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")


def test_decision_scenarios():
    """Test 4: Decision scenarios (v1-equivalent A-E)."""
    print("\n=== Test 4: Decision Scenarios ===")
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav2.brain.sensory_bridge import inject_sensory

    scenario_scores = {}

    def run_scenario(label, setup_fn, score_fn, T=100):
        env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=0, max_steps=T)
        brain = ZebrafishBrainV2(device=DEVICE)
        obs, info = env.reset(seed=42)
        brain.reset()
        setup_fn(env)
        goals_log = []
        positions = []
        for t in range(T):
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)
            goals_log.append(brain.current_goal)
            positions.append((getattr(env, 'fish_x', 400), getattr(env, 'fish_y', 300)))
            if terminated or truncated:
                break
        env.close()
        score = score_fn(goals_log, positions, info)
        return score, goals_log, positions

    # A: Safe vs Risky food — should prefer safe patch (far from predator)
    def setup_A(env):
        env.fish_x, env.fish_y = env.arena_w * 0.5, env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x, env.pred_y = env.arena_w * 0.8, env.arena_h * 0.2
        env.pred_heading = math.pi
        env.pred_state = 'STALK'
        env.foods = []
        np.random.seed(101)
        for _ in range(8):
            env.foods.append([env.pred_x + np.random.uniform(-40, 40),
                              env.pred_y + np.random.uniform(-40, 40), 'small'])
        for _ in range(3):
            env.foods.append([env.arena_w * 0.2 + np.random.uniform(-30, 30),
                              env.arena_h * 0.8 + np.random.uniform(-30, 30), 'small'])

    def score_A(goals, positions, info):
        # Good if fish moves toward safe food (bottom-left) not risky (top-right)
        last_pos = positions[-1] if positions else (400, 300)
        safe_x, safe_y = 160, 480
        pred_x, pred_y = 640, 120
        d_safe = math.sqrt((last_pos[0]-safe_x)**2 + (last_pos[1]-safe_y)**2)
        d_pred = math.sqrt((last_pos[0]-pred_x)**2 + (last_pos[1]-pred_y)**2)
        return min(100, max(0, int(100 * (d_pred / (d_safe + d_pred + 1e-8)))))

    # B: Predator charge — should FLEE
    def setup_B(env):
        env.fish_x, env.fish_y = env.arena_w * 0.5, env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x, env.pred_y = env.arena_w * 0.5, env.arena_h * 0.3
        env.pred_heading = math.pi / 2
        env.pred_state = 'HUNT'
        env.foods = []

    def score_B(goals, positions, info):
        flee_frac = sum(1 for g in goals[:30] if g == 1) / max(1, min(30, len(goals)))
        return min(100, int(flee_frac * 100))

    # C: Starvation dilemma — low energy + predator nearby
    def setup_C(env):
        env.fish_x, env.fish_y = env.arena_w * 0.5, env.arena_h * 0.5
        env.fish_heading = 0.0
        env.fish_energy = 20.0
        env.pred_x, env.pred_y = env.arena_w * 0.7, env.arena_h * 0.5
        env.pred_state = 'STALK'
        env.foods = []
        for _ in range(5):
            env.foods.append([env.arena_w * 0.3 + np.random.uniform(-50, 50),
                              env.arena_h * 0.5 + np.random.uniform(-50, 50), 'small'])

    def score_C(goals, positions, info):
        # Should forage despite predator (energy critical)
        forage_frac = sum(1 for g in goals if g == 0) / max(1, len(goals))
        return min(100, int(forage_frac * 120))

    # D: No threat, food everywhere — should FORAGE efficiently
    def setup_D(env):
        env.fish_x, env.fish_y = env.arena_w * 0.5, env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x, env.pred_y = -100, -100  # far away
        env.pred_state = 'PATROL'
        env.foods = []
        for _ in range(10):
            env.foods.append([np.random.uniform(100, 700),
                              np.random.uniform(100, 500), 'small'])

    def score_D(goals, positions, info):
        forage_frac = sum(1 for g in goals if g == 0) / max(1, len(goals))
        eaten = info.get('total_eaten', 0)
        return min(100, int(50 * forage_frac + 50 * min(1, eaten / 2.0)))

    # E: Explore unknown area — should EXPLORE when no food/threat visible
    def setup_E(env):
        env.fish_x, env.fish_y = env.arena_w * 0.5, env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x, env.pred_y = -200, -200
        env.pred_state = 'PATROL'
        env.foods = []

    def score_E(goals, positions, info):
        explore_frac = sum(1 for g in goals if g == 2) / max(1, len(goals))
        if len(positions) > 10:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            spread = np.std(xs) + np.std(ys)
        else:
            spread = 0
        return min(100, int(40 * explore_frac + 60 * min(1, spread / 100.0)))

    scenarios = [('A', setup_A, score_A), ('B', setup_B, score_B),
                 ('C', setup_C, score_C), ('D', setup_D, score_D),
                 ('E', setup_E, score_E)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Test 4: Decision Scenarios (A-E)', fontsize=14, fontweight='bold')

    scores_all = []
    for idx, (label, setup, scorer) in enumerate(scenarios):
        print(f"  Running scenario {label}...", end=' ', flush=True)
        score, goals, positions = run_scenario(label, setup, scorer)
        scenario_scores[label] = score
        scores_all.append(score)
        print(f"score={score}/100")

        ax = axes[idx // 3, idx % 3]
        # Goal timeline
        goal_colors = {0: 'green', 1: 'red', 2: 'blue', 3: 'cyan'}
        for i in range(len(goals)):
            ax.axvspan(i, i+1, color=goal_colors[goals[i]], alpha=0.3)
        ax.set_title(f'Scenario {label}: {score}/100')
        ax.set_xlim(0, len(goals))
        ax.set_xlabel('Step')

    # Summary
    ax_sum = axes[1, 2]
    ax_sum.bar(list(scenario_scores.keys()), list(scenario_scores.values()),
               color=['green' if s >= 60 else 'red' for s in scenario_scores.values()])
    avg_score = np.mean(scores_all)
    ax_sum.axhline(60, color='k', linestyle='--', label='threshold')
    ax_sum.set_title(f'Average: {avg_score:.0f}/100')
    ax_sum.set_ylim(0, 100)
    ax_sum.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test4_decisions.png'), dpi=150)
    plt.close(fig)

    results['DEC_avg_score'] = ('PASS' if avg_score >= 50 else 'FAIL',
                                 f'{avg_score:.0f}/100', '>= 50')
    results['DEC_flee_B'] = ('PASS' if scenario_scores['B'] >= 50 else 'FAIL',
                              f'{scenario_scores["B"]}/100', '>= 50')
    for k, v in results.items():
        if k.startswith('DEC_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")


def test_full_integration():
    """Test 5: Full closed-loop with all new modules."""
    print("\n=== Test 5: Full Integration (all modules) ===")
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav2.brain.sensory_bridge import inject_sensory

    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=500)
    brain = ZebrafishBrainV2(device=DEVICE)
    obs, info = env.reset(seed=42)
    brain.reset()

    T = 500
    goals, energies, food_cumul = [], [], []
    cpg_Ls, cpg_Rs, vae_losses_t, vae_nodes_t = [], [], [], []
    hab_helpless, hab_switches_t = [], []
    total_eaten = 0

    for t in range(T):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        env._eaten_now = info.get('food_eaten_this_step', 0)
        total_eaten += env._eaten_now

        goals.append(brain.current_goal)
        energies.append(brain.energy)
        food_cumul.append(total_eaten)
        cpg_Ls.append(out.get('cpg_motor_L', 0))
        cpg_Rs.append(out.get('cpg_motor_R', 0))
        vae_losses_t.append(out.get('vae_loss', 0))
        vae_nodes_t.append(out.get('vae_memory_nodes', 0))
        hab_helpless.append(out.get('hab_helplessness', 0))
        hab_switches_t.append(1 if out.get('hab_switch', False) else 0)

        if terminated or truncated:
            break
    survived = t + 1
    env.close()

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Test 5: Full Integration ({survived} steps, {total_eaten} food)',
                 fontsize=14, fontweight='bold')

    # Row 1
    goal_colors = {0: 'green', 1: 'red', 2: 'blue', 3: 'cyan'}
    for i in range(len(goals)):
        axes[0, 0].axvspan(i, i+1, color=goal_colors[goals[i]], alpha=0.3)
    axes[0, 0].set_title('Goal Timeline')
    axes[0, 0].set_xlim(0, survived)

    axes[0, 1].plot(energies, 'r-', linewidth=1.5)
    axes[0, 1].set_title(f'Energy (final: {energies[-1]:.0f})')

    axes[0, 2].plot(food_cumul, 'g-', linewidth=2)
    axes[0, 2].set_title(f'Cumulative Food: {total_eaten}')

    # Row 2: CPG + VAE
    axes[1, 0].plot(cpg_Ls, 'b-', alpha=0.5, label='L', linewidth=0.5)
    axes[1, 0].plot(cpg_Rs, 'r-', alpha=0.5, label='R', linewidth=0.5)
    axes[1, 0].set_title('CPG Motor L/R')
    axes[1, 0].legend()

    axes[1, 1].plot(vae_losses_t, 'purple', linewidth=0.5, alpha=0.7)
    axes[1, 1].set_title('VAE Loss (online)')

    axes[1, 2].plot(vae_nodes_t, 'orange', linewidth=1.5)
    axes[1, 2].set_title(f'VAE Memory Nodes: {vae_nodes_t[-1] if vae_nodes_t else 0}')

    # Row 3: Habenula
    axes[2, 0].plot(hab_helpless, 'red', linewidth=1)
    axes[2, 0].set_title('Habenula Helplessness')

    switch_steps = [i for i, s in enumerate(hab_switches_t) if s]
    if switch_steps:
        axes[2, 1].eventplot([switch_steps], lineoffsets=[0], linelengths=[0.5], color='red')
    axes[2, 1].set_title(f'Strategy Switches: {len(switch_steps)}')
    axes[2, 1].set_xlim(0, survived)

    # Goal distribution
    from collections import Counter
    gc = Counter(goals)
    axes[2, 2].bar([GOAL_NAMES[g] for g in range(4)],
                    [gc.get(g, 0) for g in range(4)],
                    color=['green', 'red', 'blue', 'cyan'])
    axes[2, 2].set_title('Goal Distribution')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test5_full_integration.png'), dpi=150)
    plt.close(fig)

    results['FULL_survived'] = ('PASS' if survived >= 200 else 'FAIL',
                                 str(survived), '>= 200')
    results['FULL_food'] = ('PASS' if total_eaten >= 1 else 'FAIL',
                             str(total_eaten), '>= 1')
    results['FULL_cpg_active'] = ('PASS' if max(cpg_Ls) > 0.01 else 'FAIL',
                                   f'{max(cpg_Ls):.4f}', '> 0.01')
    results['FULL_vae_trained'] = ('PASS' if any(v > 0 for v in vae_losses_t) else 'FAIL',
                                    f'trained={any(v > 0 for v in vae_losses_t)}', 'True')
    results['FULL_vae_memory'] = ('PASS' if vae_nodes_t[-1] > 0 else 'FAIL',
                                   f'{vae_nodes_t[-1]} nodes', '> 0')
    for k, v in results.items():
        if k.startswith('FULL_'):
            print(f"  {k}: {v[0]} (value={v[1]}, threshold={v[2]})")


def generate_summary():
    print("\n=== SUMMARY ===")
    n_pass = sum(1 for v in results.values() if v[0] == 'PASS')
    n_total = len(results)

    fig, ax = plt.subplots(figsize=(12, max(6, n_total * 0.35 + 2)))
    fig.suptitle(f'High-Priority Tests: {n_pass}/{n_total} PASS ({100*n_pass/max(1,n_total):.0f}%)',
                 fontsize=14, fontweight='bold')
    col_labels = ['Test', 'Status', 'Value', 'Threshold']
    cell_colors, table_data = [], []
    for name, (status, value, threshold) in sorted(results.items()):
        table_data.append([name, status, str(value), str(threshold)])
        cell_colors.append(['white', '#d4edda' if status == 'PASS' else '#f8d7da', 'white', 'white'])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors, loc='center',
                     colWidths=[0.25, 0.1, 0.35, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'summary_high_priority.png'), dpi=150)
    plt.close(fig)

    print(f"\n  Results: {n_pass}/{n_total} PASS ({100*n_pass/max(1,n_total):.0f}%)")
    for name, (status, value, threshold) in sorted(results.items()):
        mark = 'OK' if status == 'PASS' else 'XX'
        print(f"  [{mark}] {name}: {value} (threshold: {threshold})")


if __name__ == '__main__':
    t0 = time.time()
    print(f"Device: {DEVICE}")
    test_cpg()
    test_vae()
    test_habenula_enhanced()
    test_decision_scenarios()
    test_full_integration()
    generate_summary()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
