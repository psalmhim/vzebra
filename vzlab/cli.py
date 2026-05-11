"""
vzlab CLI — virtual lab command line interface.

Usage:
  vzlab run   --species danio_rerio --protocol predator_avoidance [options]
  vzlab sweep --species danio_rerio --param efe.beta_epistemic --values 0.0,1.0,3.0
  vzlab list  (show available species and protocols)
  vzlab info  --species danio_rerio
"""
from __future__ import annotations

import argparse
import json
import sys


def cmd_list(_args) -> None:
    from . import core
    from .core.registry import list_species, get
    from .protocols import PROTOCOLS

    print("\n── Available species ──────────────────────────────────────")
    for s in list_species():
        e = get(s)
        print(f"  {s:20s}  {e.common_name}  (~{e.n_neurons_approx:,} neurons)")

    print("\n── Available protocols ────────────────────────────────────")
    for name, cls in PROTOCOLS.items():
        p = cls()
        print(f"  {name:25s}  {p.episode_length} steps/episode")
    print()


def cmd_info(args) -> None:
    from .core.registry import get
    entry = get(args.species)
    print(f"\nSpecies: {entry.common_name} ({entry.name})")
    print(f"  Neurons:     ~{entry.n_neurons_approx:,}")
    print(f"  Connectome:  {entry.connectome_cls}")
    print(f"  Brain:       {entry.brain_cls}")
    print(f"  Environment: {entry.environment_cls}")
    print(f"  Notes:       {entry.notes}\n")


def cmd_run(args) -> None:
    from .core.registry import build
    from .engine.organism import VirtualOrganism
    from .engine.experiment import ExperimentRunner
    from .protocols import PROTOCOLS

    print(f"\n── vzlab run: {args.species} / {args.protocol} ──")
    connectome, brain, env = build(args.species)
    organism = VirtualOrganism(brain, env)

    if args.species not in PROTOCOLS and args.protocol not in PROTOCOLS:
        print(f"Unknown protocol: {args.protocol}")
        sys.exit(1)
    protocol_cls = PROTOCOLS[args.protocol]
    protocol = protocol_cls()

    runner = ExperimentRunner(
        organism, protocol,
        log_levels=[int(l) for l in args.log_levels.split(",")],
        seed=args.seed,
    )

    # apply ablations
    for region in (args.ablate or "").split(","):
        if region.strip():
            runner.ablate(region.strip())
            print(f"  Ablating: {region.strip()}")

    # apply params
    for pstr in (args.params or "").split(";"):
        if "=" in pstr:
            k, v = pstr.strip().split("=", 1)
            runner.set_param(k.strip(), float(v.strip()))
            print(f"  Param: {k.strip()} = {v.strip()}")

    result = runner.run(
        n_episodes=args.episodes,
        label=f"{args.species}/{args.protocol}",
        verbose=True,
    )

    print(f"\n── Summary ────────────────────────────────────────────────")
    summary = result.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.output:
        runner.save(result, args.output)


def cmd_sweep(args) -> None:
    from .core.registry import build
    from .engine.organism import VirtualOrganism
    from .engine.experiment import ExperimentRunner
    from .protocols import PROTOCOLS

    print(f"\n── vzlab sweep: {args.species} / {args.protocol} / {args.param} ──")
    values = [float(v) for v in args.values.split(",")]
    connectome, brain, env = build(args.species)
    organism = VirtualOrganism(brain, env)
    protocol = PROTOCOLS[args.protocol]()
    runner = ExperimentRunner(organism, protocol, seed=args.seed)

    results = runner.sweep(args.param, values, n_episodes=args.episodes)

    print(f"\n── Sweep results ({args.param}) ─────────────────────────────")
    print(f"  {'value':>8}  {'survival':>10}  {'food':>8}  {'predation':>10}")
    for val, res in zip(values, results):
        print(f"  {val:8.2f}  {res.mean_survival:10.1f}  {res.mean_food:8.2f}  {res.mean_predation:10.2f}")

    if args.output:
        data = [{"param_value": v, **r.summary()} for v, r in zip(values, results)]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved: {args.output}")


def cmd_validate(args) -> None:
    from .validation import run_full_validation

    print(f"\n── vzlab validate: {args.species} ──")
    report = run_full_validation(args.species, verbose=True)
    report.print_summary()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_json(), f, indent=2)
        print(f"Results saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vzlab",
        description="Virtual Laboratory — multi-species hierarchical brain platform",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List available species and protocols")

    # info
    p_info = sub.add_parser("info", help="Show species details")
    p_info.add_argument("--species", required=True)

    # run
    p_run = sub.add_parser("run", help="Run experiment")
    p_run.add_argument("--species",  required=True)
    p_run.add_argument("--protocol", required=True)
    p_run.add_argument("--episodes", type=int, default=10)
    p_run.add_argument("--seed",     type=int, default=0)
    p_run.add_argument("--ablate",   default="", help="comma-separated regions to ablate")
    p_run.add_argument("--params",   default="", help="semicolon-separated key=value params")
    p_run.add_argument("--log-levels", dest="log_levels", default="4,5,6")
    p_run.add_argument("--output",   default="", help="save JSON results to path")

    # sweep
    p_sweep = sub.add_parser("sweep", help="Parameter sweep")
    p_sweep.add_argument("--species",  required=True)
    p_sweep.add_argument("--protocol", required=True)
    p_sweep.add_argument("--param",    required=True)
    p_sweep.add_argument("--values",   required=True, help="comma-separated floats")
    p_sweep.add_argument("--episodes", type=int, default=5)
    p_sweep.add_argument("--seed",     type=int, default=0)
    p_sweep.add_argument("--output",   default="")

    # validate
    p_val = sub.add_parser("validate", help="Run three-tier validation battery")
    p_val.add_argument("--species", required=True)
    p_val.add_argument("--output",  default="", help="save JSON results to path")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
