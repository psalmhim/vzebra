"""
Step 26: W_FB Predictive Coding Training — Online Layer-Wise Free Energy

Trains feedback weights (W_FB) ONLINE in the actual simulation environment,
bottom-up, using predictive coding free energy minimization.

Unlike offline training, online training ensures W_FB co-adapts with the
actual behavioral dynamics (feedforward activity patterns during real
foraging, fleeing, exploring). This prevents the distributional mismatch
that causes offline-trained W_FB to disrupt behavior.

Free energy per layer:  F_i = ||PE_i||² / (2 σ²_i)
Update rule:            dW_FB_i = -(η_i / σ²_i) * fb_source.T @ PE_i

Layer-wise phases (bottom-up, each 2000 steps):
  Phase 1: Train OT_F.W_FB only (sensory prediction, η × 0.3)
  Phase 2: Freeze OT_F, train PT_L.W_FB (tectal, η × 0.5)
  Phase 3: Freeze PT_L, train PC_per.W_FB (perceptual, η × 1.0)
  Phase 4: Freeze PC_per, train PC_int.W_FB (intent, η × 1.0)

Run: python -m zebrav1.tests.step26_wfb_pe_training
Output: weights/classifier_wfb.pt
"""
import os
import sys
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent
from zebrav1.brain.device_util import get_device


# Layer training config: (name, eta_scale, max_w_fb)
# Only train the two large visual layers — OT_F (400→800) and PT_L (120→400).
# PC_per and PC_int have near-zero PE already; training them introduces
# instability without meaningful PE reduction.
LAYER_CONFIGS = [
    ("OT_F", 0.3, 0.06),    # sensory: noisy, conservative
    ("PT_L", 0.3, 0.08),    # tectal: conservative (prevents classifier drift)
]


def get_fb_pairs(model):
    """Return (layer, fb_source, name) tuples."""
    return [
        (model.OT_F, lambda: model.PT_L.v_s, "OT_F"),
        (model.PT_L, lambda: model.PC_per.v_s, "PT_L"),
        (model.PC_per, lambda: model.PC_int.v_s, "PC_per"),
        (model.PC_int, lambda: model.DA.v_s, "PC_int"),
    ]


def update_single_layer_wfb(model, target_name, eta_fb=3e-4,
                              max_dw_fb=0.003, decay=0.9999):
    """Update W_FB for a single target layer only.

    Uses precision-weighted free energy gradient:
      dW = -(η_scale * η_fb * precision) * fb_source.T @ PE
    """
    fb_pairs = get_fb_pairs(model)

    for layer, fb_getter, name in fb_pairs:
        if name != target_name or layer.W_FB is None:
            continue

        pe = layer.pred_error
        if pe is None:
            return 0.0

        # Find config for this layer
        config = next((c for c in LAYER_CONFIGS if c[0] == name), None)
        if config is None:
            return 0.0
        _, eta_scale, max_w_fb = config

        pe_var = pe.pow(2).mean().item() + 1e-8
        precision = min(5.0, 1.0 / pe_var)
        effective_eta = eta_fb * eta_scale * precision

        with torch.no_grad():
            fb_source = fb_getter()
            dW = -effective_eta * (fb_source.t() @ pe)
            dW.clamp_(-max_dw_fb, max_dw_fb)
            layer.W_FB.data += dW
            layer.W_FB.data *= decay
            layer.W_FB.data.clamp_(-max_w_fb, max_w_fb)

        return pe.pow(2).mean().item()

    return 0.0


def run_training_phase(phase_name, target_layer_name, steps=2000,
                       seed=42, eta_fb=3e-4):
    """Run one phase: simulate + train one layer's W_FB online."""
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=steps, side_panels=False)
    agent = BrainAgent(device="auto", world_model="place_cell",
                       use_allostasis=True)

    # Load current best weights (accumulates across phases)
    wfb_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier_wfb.pt")
    cls_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier.pt")
    weight_path = wfb_path if os.path.exists(wfb_path) else cls_path
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=agent.model.device,
                           weights_only=True)
        agent.model.load_saveable_state(state)

    # Disable default Hebbian W_FB updates (we do our own)
    agent._skip_fb_update = True
    # Freeze W_FF Hebbian updates to prevent representational drift
    agent.model._skip_ff_update = True
    # Freeze classifier head — W_FB training must not alter classification
    for name, param in agent.model.named_parameters():
        if "cls_" in name:
            param.requires_grad = False

    obs, info = env.reset(seed=seed)
    agent.reset()

    pe_per_step = []
    all_layer_pe = {"OT_F": [], "PT_L": [], "PC_per": [], "PC_int": []}
    goals = []
    eaten_total = 0

    for t in range(steps):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)

        # Custom W_FB update for target layer only
        pe_val = update_single_layer_wfb(
            agent.model, target_layer_name, eta_fb=eta_fb)
        pe_per_step.append(pe_val)

        # Track all layer PEs
        for layer, _, name in get_fb_pairs(agent.model):
            if hasattr(layer, 'pred_error') and layer.pred_error is not None:
                all_layer_pe[name].append(
                    layer.pred_error.pow(2).mean().item())

        diag = agent.last_diagnostics
        goals.append(diag.get("goal", 2))
        eaten_total = info.get("total_eaten", 0)

        if terminated or truncated:
            if terminated:
                print(f"    CAUGHT at step {t}!")
            break

    env.close()

    # Save updated weights
    state = agent.model.get_saveable_state()
    torch.save(state, wfb_path)

    # Summary
    goals_arr = np.array(goals)
    goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
    final_step = t + 1
    final_energy = diag.get("energy", 0)
    mean_pe = np.mean(pe_per_step) if pe_per_step else 0
    final_pe = np.mean(pe_per_step[-100:]) if len(pe_per_step) >= 100 else mean_pe

    print(f"    Steps={final_step} Eaten={eaten_total} "
          f"Energy={final_energy:.0f}")
    print(f"    Target PE ({target_layer_name}): "
          f"mean={mean_pe:.6f} final={final_pe:.6f}")
    goal_str = "  ".join(
        f"{goal_names[i]}:{100*np.sum(goals_arr==i)/len(goals_arr):.0f}%"
        for i in range(4))
    print(f"    Goals: {goal_str}")

    return {
        "pe_per_step": pe_per_step,
        "all_layer_pe": all_layer_pe,
        "final_step": final_step,
        "eaten": eaten_total,
        "energy": final_energy,
        "goals": goals_arr,
    }


def verify_behavior(label="Final"):
    """Quick 800-step evaluation to check behavior is preserved."""
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=800, side_panels=False)
    agent = BrainAgent(device="auto", world_model="place_cell",
                       use_allostasis=True)

    wfb_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier_wfb.pt")
    if os.path.exists(wfb_path):
        state = torch.load(wfb_path, map_location=agent.model.device,
                           weights_only=True)
        agent.model.load_saveable_state(state)

    obs, info = env.reset(seed=42)
    agent.reset()

    pe_vals = []
    for t in range(800):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)

        model = agent.model
        pe = sum(l.pred_error.pow(2).mean().item()
                 for l in [model.OT_F, model.PT_L, model.PC_per, model.PC_int]
                 if hasattr(l, 'pred_error') and l.pred_error is not None)
        pe_vals.append(pe)

        if terminated or truncated:
            if terminated:
                print(f"  [{label}] CAUGHT at step {t}!")
            break

    env.close()
    diag = agent.last_diagnostics
    print(f"  [{label}] Steps={t+1} Eaten={info['total_eaten']} "
          f"Energy={diag.get('energy',0):.0f} "
          f"PE_mean={np.mean(pe_vals):.4f}")

    # W_FB magnitudes
    for name, layer in [("OT_F", model.OT_F), ("PT_L", model.PT_L),
                        ("PC_per", model.PC_per), ("PC_int", model.PC_int)]:
        if layer.W_FB is not None:
            w_max = layer.W_FB.data.abs().max().item()
            print(f"    {name}.W_FB max={w_max:.4f}")

    return {
        "steps": t + 1, "eaten": info["total_eaten"],
        "energy": diag.get("energy", 0), "pe_mean": np.mean(pe_vals),
    }


def main():
    print("=" * 60)
    print("Step 26: W_FB Predictive Coding Training")
    print("        Online Layer-Wise Free Energy Minimization")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Start from original classifier weights
    cls_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier.pt")
    wfb_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier_wfb.pt")

    # Copy original to wfb as starting point
    if os.path.exists(cls_path):
        import shutil
        shutil.copy2(cls_path, wfb_path)
        print(f"Starting from {cls_path}")

    # Baseline behavior
    print("\n--- Baseline (original weights) ---")
    baseline = verify_behavior("Baseline")

    # Phase-wise online training (2 phases: OT_F then PT_L)
    n_phases = len(LAYER_CONFIGS)
    phase_results = {}
    for phase, (name, eta_scale, max_w_fb) in enumerate(LAYER_CONFIGS):
        print(f"\n{'='*55}")
        print(f"Phase {phase+1}/{n_phases}: Online training {name}.W_FB")
        print(f"  eta_scale={eta_scale}, max_w_fb={max_w_fb}")
        print(f"{'='*55}")

        result = run_training_phase(
            f"Phase {phase+1}", name,
            steps=3000,
            seed=42 + phase * 100,
            eta_fb=3e-4,
        )
        phase_results[name] = result

        # Quick behavior check after each phase
        print(f"\n  Post-phase behavior check:")
        verify_behavior(f"After {name}")

    # Final evaluation
    print(f"\n{'='*55}")
    print("Final evaluation")
    print(f"{'='*55}")
    final = verify_behavior("Final")

    # Generate plot
    n_phases = len(phase_results)
    fig, axes = plt.subplots(n_phases, 2, figsize=(14, 4 * n_phases))
    if n_phases == 1:
        axes = axes.reshape(1, -1)

    colors = {"OT_F": "#e74c3c", "PT_L": "#2ecc71",
              "PC_per": "#3498db", "PC_int": "#f39c12"}

    for i, (name, result) in enumerate(phase_results.items()):
        # Left: target layer PE over steps
        ax = axes[i, 0]
        pe = result["pe_per_step"]
        if len(pe) > 50:
            # Smooth with rolling mean
            k = 50
            kernel = np.ones(k) / k
            pe_smooth = np.convolve(pe, kernel, mode="valid")
            ax.plot(pe_smooth, color=colors[name], linewidth=1.5)
        else:
            ax.plot(pe, color=colors[name], linewidth=1.5)
        ax.set_ylabel(f"PE ({name})")
        ax.set_title(f"Phase {i+1}: {name} — Online PE During Training")
        if max(pe) > 0:
            ax.set_yscale("log")

        # Right: all layer PEs during this phase
        ax = axes[i, 1]
        for lname, lpe in result["all_layer_pe"].items():
            if len(lpe) > 50:
                pe_s = np.convolve(lpe, kernel, mode="valid")
                ax.plot(pe_s, color=colors[lname], linewidth=1,
                        alpha=0.7, label=lname)
            elif lpe:
                ax.plot(lpe, color=colors[lname], linewidth=1,
                        alpha=0.7, label=lname)
        ax.set_ylabel("PE (all layers)")
        ax.set_title(f"Phase {i+1}: All Layer PEs")
        ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Step")

    fig.suptitle("Online Layer-Wise W_FB Training",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(PROJECT_ROOT, "plots",
                              "v1_step26_wfb_training.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Baseline: steps={baseline['steps']} eaten={baseline['eaten']} "
          f"energy={baseline['energy']:.0f} PE={baseline['pe_mean']:.4f}")
    print(f"  Final:    steps={final['steps']} eaten={final['eaten']} "
          f"energy={final['energy']:.0f} PE={final['pe_mean']:.4f}")

    pe_improved = final["pe_mean"] < baseline["pe_mean"]
    food_ok = final["eaten"] >= baseline["eaten"] - 2
    energy_ok = final["energy"] > 20
    print(f"\n  PE improved:    {'YES' if pe_improved else 'NO'}")
    print(f"  Food preserved: {'YES' if food_ok else 'NO'}")
    print(f"  Energy safe:    {'YES' if energy_ok else 'NO'}")
    passed = pe_improved and food_ok and energy_ok
    print(f"  {'PASS' if passed else 'NEEDS TUNING'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
