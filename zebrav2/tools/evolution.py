"""
Evolutionary competition: evolve personality traits over generations.

Each generation:
  1. N fish with different personality traits compete in arena
  2. Fitness = survival_steps + food_eaten * 50
  3. Top K survive, produce offspring with mutated traits
  4. Repeat for G generations

Tracks: which traits evolve to dominate, convergence of population.

Run: .venv/bin/python -u -m zebrav2.tools.evolution
"""
import os, sys, time, math, copy
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.brain.personality import get_personality, random_personality
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.predator_brain import PredatorBrain
from zebrav2.spec import DEVICE
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv


def evaluate_fitness(personality, seed=42, max_steps=200):
    """Run one fish with given personality, return fitness."""
    from zebrav2.brain.sensory_bridge import inject_sensory
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=max_steps)
    brain = ZebrafishBrainV2(device=DEVICE, personality=personality)
    obs, info = env.reset(seed=seed)
    brain.reset()
    eaten = 0
    for t in range(max_steps):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)
        obs, _, term, trunc, info = env.step(np.array([out['turn'], out['speed']], dtype=np.float32))
        env._eaten_now = info.get('food_eaten_this_step', 0)
        eaten += env._eaten_now
        if term or trunc:
            break
    env.close()
    del brain
    survived = t + 1
    fitness = survived + eaten * 50
    return fitness, survived, eaten


def mutate(personality, mutation_rate=0.1):
    """Create mutated offspring from parent personality."""
    child = copy.deepcopy(personality)
    for key in ['DA_baseline', 'HT5_baseline', 'NA_baseline', 'ACh_baseline']:
        child[key] = float(np.clip(child[key] + np.random.normal(0, mutation_rate), 0.1, 0.9))
    child['amy_gain'] = float(np.clip(child['amy_gain'] + np.random.normal(0, mutation_rate * 2), 0.3, 2.0))
    child['flee_threshold'] = float(np.clip(child['flee_threshold'] + np.random.normal(0, mutation_rate * 0.3), 0.10, 0.40))
    child['explore_bias'] = float(np.clip(child['explore_bias'] + np.random.normal(0, mutation_rate), -0.4, 0.4))
    child['social_bias'] = float(np.clip(child['social_bias'] + np.random.normal(0, mutation_rate), -0.4, 0.4))
    child['habenula_threshold'] = float(np.clip(child['habenula_threshold'] + np.random.normal(0, mutation_rate), 0.2, 0.7))
    return child


def crossover(p1, p2):
    """Create offspring by mixing traits from two parents."""
    child = copy.deepcopy(p1)
    for key in p1:
        if isinstance(p1[key], (int, float)) and key != 'description':
            if np.random.random() > 0.5:
                child[key] = p2[key]
    child['description'] = 'offspring'
    return child


def run_evolution(pop_size=6, n_generations=5, top_k=3, max_steps=150,
                  seeds_per_eval=2, mutation_rate=0.1):
    """Run evolutionary optimization of personality traits."""
    print(f"\n{'='*60}")
    print(f"  Evolution: {pop_size} fish × {n_generations} generations")
    print(f"  Selection: top {top_k}, mutation rate={mutation_rate}")
    print(f"{'='*60}")

    # Initialize population
    rng = np.random.RandomState(42)
    population = [random_personality(rng) for _ in range(pop_size)]
    # Seed with named personalities
    from zebrav2.brain.personality import PERSONALITIES
    for i, name in enumerate(['bold', 'shy', 'explorer']):
        if i < pop_size:
            population[i] = get_personality(name)

    gen_stats = []

    for gen in range(n_generations):
        print(f"\n  Generation {gen+1}/{n_generations}")

        # Evaluate each individual across multiple seeds
        fitness_scores = []
        for i, p in enumerate(population):
            total_fitness = 0
            total_food = 0
            total_surv = 0
            for s in range(seeds_per_eval):
                f, surv, eaten = evaluate_fitness(p, seed=gen * 100 + s * 7 + 1,
                                                   max_steps=max_steps)
                total_fitness += f
                total_food += eaten
                total_surv += surv
            avg_fitness = total_fitness / seeds_per_eval
            avg_food = total_food / seeds_per_eval
            avg_surv = total_surv / seeds_per_eval
            fitness_scores.append((avg_fitness, i, p, avg_food, avg_surv))
            trait_str = (f"DA={p['DA_baseline']:.2f} 5HT={p['HT5_baseline']:.2f} "
                         f"amy={p['amy_gain']:.1f} flee={p['flee_threshold']:.2f}")
            print(f"    Fish {i}: fitness={avg_fitness:.0f} "
                  f"(surv={avg_surv:.0f}, food={avg_food:.1f}) [{trait_str}]")

        # Sort by fitness
        fitness_scores.sort(key=lambda x: -x[0])

        # Stats
        fitnesses = [f[0] for f in fitness_scores]
        gen_stats.append({
            'gen': gen + 1,
            'best_fitness': fitnesses[0],
            'mean_fitness': np.mean(fitnesses),
            'best_traits': {k: v for k, v in fitness_scores[0][2].items()
                           if isinstance(v, (int, float))},
        })
        print(f"    Best: {fitnesses[0]:.0f}, Mean: {np.mean(fitnesses):.0f}")

        # Selection: top K survive
        survivors = [fitness_scores[i][2] for i in range(min(top_k, len(fitness_scores)))]

        # Reproduction: fill population with offspring
        new_pop = list(survivors)  # elites survive
        while len(new_pop) < pop_size:
            # Pick two random parents from survivors
            p1 = survivors[np.random.randint(len(survivors))]
            p2 = survivors[np.random.randint(len(survivors))]
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Evolution Complete")
    print(f"{'='*60}")
    print(f"\n  Best fitness trajectory:")
    for gs in gen_stats:
        print(f"    Gen {gs['gen']}: best={gs['best_fitness']:.0f}, mean={gs['mean_fitness']:.0f}")

    # Best evolved traits
    best = gen_stats[-1]['best_traits']
    print(f"\n  Evolved optimal traits:")
    for k, v in sorted(best.items()):
        print(f"    {k}: {v:.3f}")

    # Save figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle('Personality Evolution', fontweight='bold')
        axes[0].plot([g['best_fitness'] for g in gen_stats], 'b-o', label='Best')
        axes[0].plot([g['mean_fitness'] for g in gen_stats], 'r--o', label='Mean')
        axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness Over Generations'); axes[0].legend()

        traits = ['DA_baseline', 'HT5_baseline', 'amy_gain', 'flee_threshold']
        for trait in traits:
            vals = [g['best_traits'].get(trait, 0) for g in gen_stats]
            axes[1].plot(vals, '-o', label=trait, markersize=4)
        axes[1].set_xlabel('Generation'); axes[1].set_title('Trait Evolution')
        axes[1].legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(PROJECT_ROOT, 'plots', 'v2_paper', 'fig_evolution.png'), dpi=200)
        plt.close(fig)
        print(f"\n  Figure saved: plots/v2_paper/fig_evolution.png")
    except Exception as e:
        print(f"  Figure failed: {e}")

    return gen_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop', type=int, default=6)
    parser.add_argument('--gens', type=int, default=5)
    parser.add_argument('--steps', type=int, default=150)
    args = parser.parse_args()
    run_evolution(pop_size=args.pop, n_generations=args.gens, max_steps=args.steps)
