"""
Step 21: Paper Figure Generation

Runs a 2000-step simulation with full brain (allostasis, amygdala, Hebbian
learning, fear conditioning, escape tracking) and generates a publication-
ready 6-panel figure.

Panels:
  1. Trajectory map — fish path colored by goal, food/predator/colleague scatter
  2. Goal selection timeline — horizontal color bar + percentage breakdown
  3. Hebbian learning curve — dW norm (smoothed) + threat-boost shaded regions
  4. Fear conditioning map — place cell risk as scatter on arena
  5. Energy + food timeline — energy line + green markers at eating events
  6. Escape success/failure — cumulative lines

Run: python -m zebra_v60.tests.step21_paper_figures
Output: plots/v60_step21_paper_figures.png, .pdf
"""
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebra_v60.gym_env.brain_agent import BrainAgent


def run_episode(T=2000, seed=42):
    """Run a full brain episode, collecting per-step data."""
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T, side_panels=False)
    agent = BrainAgent(device="auto", world_model="vae",
                       use_allostasis=True)
    obs, info = env.reset(seed=seed)
    agent.reset()

    hist = {
        "pos_x": [], "pos_y": [],
        "pred_x": [], "pred_y": [],
        "goal": [],
        "energy": [],
        "hebb_dw_norm": [],
        "threat_arousal": [],
        "threat_boost": [],
        "escape_successes": [],
        "escape_failures": [],
        "fear_memory_count": [],
        "eaten_cumulative": [],
    }

    cumulative_reward = 0.0
    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)
        cumulative_reward += reward

        diag = agent.last_diagnostics
        hist["pos_x"].append(info["fish_pos"][0])
        hist["pos_y"].append(info["fish_pos"][1])
        hist["pred_x"].append(info["pred_pos"][0])
        hist["pred_y"].append(info["pred_pos"][1])
        hist["goal"].append(diag.get("goal", 2))
        hist["energy"].append(diag.get("energy", 100.0))
        hist["hebb_dw_norm"].append(diag.get("hebb_dw_norm", 0.0))
        hist["threat_arousal"].append(
            agent.amygdala.threat_arousal if agent.amygdala else 0.0)
        hist["threat_boost"].append(diag.get("threat_boost", 1.0))
        hist["escape_successes"].append(diag.get("escape_successes", 0))
        hist["escape_failures"].append(diag.get("escape_failures", 0))
        pc_diag = diag.get("place_cells", {})
        hist["fear_memory_count"].append(pc_diag.get("fear_memory_count", 0))
        hist["eaten_cumulative"].append(info["total_eaten"])

        if t % 500 == 0:
            goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
            g = diag.get("goal", 2)
            print(f"  t={t:4d}  goal={goal_names[g]:7s}  "
                  f"eaten={info['total_eaten']}  "
                  f"energy={diag.get('energy', 0):.0f}  "
                  f"fear={hist['fear_memory_count'][-1]}")

        if terminated or truncated:
            if terminated:
                print(f"  CAUGHT at step {t}!")
            break

    # Collect place cell data for fear map
    place_data = None
    if hasattr(agent, 'place_cells'):
        pc = agent.place_cells
        n = pc.n_allocated
        if n > 0:
            place_data = {
                "centroids": pc.centroids[:n].copy(),
                "risk": pc.risk[:n].copy(),
                "visit_count": pc.visit_count[:n].copy(),
                "food_rate": pc.food_rate[:n].copy(),
            }

    # Collect rock formations for map overlay
    rocks = []
    for rock in getattr(env, 'rock_formations', []):
        rocks.append({
            "cx": rock["cx"], "cy": rock["cy"],
            "base_r": rock["base_r"],
        })

    env.close()

    print(f"\nSummary: survived {t+1} steps, "
          f"ate {info['total_eaten']} food, "
          f"reward = {cumulative_reward:.1f}")

    return hist, place_data, rocks, env.arena_w, env.arena_h


def smooth(data, k=50):
    """Rolling mean smoothing."""
    if len(data) < k:
        return np.array(data)
    kernel = np.ones(k) / k
    return np.convolve(data, kernel, mode="valid")


def generate_figures(hist, place_data, rocks, arena_w, arena_h):
    """Generate 6-panel publication figure."""
    goal_colors_map = {0: "#2ecc71", 1: "#e74c3c", 2: "#3498db", 3: "#1abc9c"}
    goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    T = len(hist["pos_x"])
    ts = np.arange(T)

    # ── Panel 1: Trajectory Map ──
    ax = axes[0, 0]
    goals = np.array(hist["goal"])
    for gi in range(4):
        mask = goals == gi
        if mask.any():
            ax.scatter(
                np.array(hist["pos_x"])[mask],
                np.array(hist["pos_y"])[mask],
                c=goal_colors_map[gi], s=1, alpha=0.5,
                label=goal_names[gi], zorder=2)

    # Predator path (subsampled)
    pred_sub = slice(0, T, 10)
    ax.plot(np.array(hist["pred_x"])[pred_sub],
            np.array(hist["pred_y"])[pred_sub],
            "r-", alpha=0.15, linewidth=0.5, zorder=1)

    # Rocks
    for rock in rocks:
        circle = plt.Circle(
            (rock["cx"], rock["cy"]), rock["base_r"],
            color="gray", alpha=0.3, zorder=0)
        ax.add_patch(circle)

    # Start/end markers
    ax.plot(hist["pos_x"][0], hist["pos_y"][0], "ko", ms=8, label="Start")
    ax.plot(hist["pos_x"][-1], hist["pos_y"][-1], "k*", ms=10, label="End")

    ax.set_xlim(0, arena_w)
    ax.set_ylim(0, arena_h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Fish Trajectory (colored by goal)")
    ax.legend(fontsize=7, loc="upper right", markerscale=3)

    # ── Panel 2: Goal Selection Timeline ──
    ax = axes[0, 1]
    goal_arr = np.array(hist["goal"]).reshape(1, -1)
    cmap = ListedColormap([goal_colors_map[i] for i in range(4)])
    ax.imshow(goal_arr, aspect="auto", cmap=cmap, vmin=0, vmax=3,
              extent=[0, T, 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Step")
    ax.set_title("Goal Selection Over Time")

    # Percentage breakdown
    for gi in range(4):
        pct = 100 * np.sum(goals == gi) / max(1, T)
        ax.text(T * (0.02 + 0.25 * gi), -0.15,
                f"{goal_names[gi]}: {pct:.1f}%",
                color=goal_colors_map[gi], fontsize=9, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # ── Panel 3: Hebbian Learning Curve ──
    ax = axes[1, 0]
    dw = np.array(hist["hebb_dw_norm"])
    dw_smooth = smooth(dw, k=50)
    ax.plot(np.arange(len(dw_smooth)), dw_smooth,
            color="#9b59b6", linewidth=1.5, label="dW norm (smoothed)")
    ax.set_ylabel("dW norm", color="#9b59b6")
    ax.set_xlabel("Step")
    ax.set_title("Hebbian Learning Activity")

    # Shade threat-boost periods
    threat_b = np.array(hist["threat_boost"])
    boosted = threat_b > 1.1
    if boosted.any():
        ax.fill_between(ts, 0, dw.max() * 1.1,
                         where=boosted, alpha=0.15, color="red",
                         label="Threat boost active")

    ax2 = ax.twinx()
    ta = np.array(hist["threat_arousal"])
    ax2.plot(ts, ta, color="#e74c3c", alpha=0.3, linewidth=0.5,
             label="Threat arousal")
    ax2.set_ylabel("Threat arousal", color="#e74c3c")
    ax2.set_ylim(0, 1.2)
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel 4: Fear Conditioning Map ──
    ax = axes[1, 1]
    if place_data is not None and len(place_data["centroids"]) > 0:
        centroids = place_data["centroids"]
        risk = place_data["risk"]
        visits = place_data["visit_count"]
        sizes = np.clip(visits / visits.max() * 100, 10, 200)
        sc = ax.scatter(centroids[:, 0], centroids[:, 1],
                        c=risk, cmap="Reds", s=sizes,
                        vmin=0, vmax=max(0.5, risk.max()),
                        edgecolors="gray", linewidths=0.3, zorder=2)
        plt.colorbar(sc, ax=ax, label="Fear Risk")
    for rock in rocks:
        circle = plt.Circle(
            (rock["cx"], rock["cy"]), rock["base_r"],
            color="gray", alpha=0.3, zorder=0)
        ax.add_patch(circle)
    ax.set_xlim(0, arena_w)
    ax.set_ylim(0, arena_h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fear_count = hist["fear_memory_count"][-1] if hist["fear_memory_count"] else 0
    ax.set_title(f"Fear Conditioning Map ({fear_count} memories)")

    # ── Panel 5: Energy + Food Timeline ──
    ax = axes[2, 0]
    energy = np.array(hist["energy"])
    ax.plot(ts, energy, color="#2980b9", linewidth=1.5, label="Energy")
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="Critical")
    ax.fill_between(ts, 0, energy, where=energy < 20,
                     alpha=0.15, color="red")
    ax.set_ylabel("Energy")
    ax.set_xlabel("Step")
    ax.set_title("Energy & Foraging")

    ax2 = ax.twinx()
    eaten = np.array(hist["eaten_cumulative"])
    ax2.plot(ts, eaten, color="#27ae60", linewidth=1.5,
             label="Cumulative food")
    ax2.set_ylabel("Food eaten", color="#27ae60")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    # ── Panel 6: Escape Success Rate ──
    ax = axes[2, 1]
    esc_s = np.array(hist["escape_successes"])
    esc_f = np.array(hist["escape_failures"])
    ax.plot(ts, esc_s, color="#27ae60", linewidth=2, label="Escapes (success)")
    ax.plot(ts, esc_f, color="#e74c3c", linewidth=2, label="Escapes (failure)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Predator Escape Learning")
    ax.legend(fontsize=8)

    if esc_s[-1] + esc_f[-1] > 0:
        rate = 100 * esc_s[-1] / (esc_s[-1] + esc_f[-1])
        ax.text(0.95, 0.95, f"Success rate: {rate:.0f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold",
                color="#27ae60" if rate > 50 else "#e74c3c")
    else:
        ax.text(0.5, 0.5, "No predator encounters",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    fig.suptitle(
        "Zebrafish Brain Simulation — Active Inference + Fear Conditioning",
        fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return fig


def main():
    print("=" * 60)
    print("Step 21: Paper Figure Generation")
    print("=" * 60)

    hist, place_data, rocks, arena_w, arena_h = run_episode(T=2000, seed=42)

    fig = generate_figures(hist, place_data, rocks, arena_w, arena_h)

    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    png_path = os.path.join(PROJECT_ROOT, "plots",
                            "v60_step21_paper_figures.png")
    pdf_path = os.path.join(PROJECT_ROOT, "plots",
                            "v60_step21_paper_figures.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
