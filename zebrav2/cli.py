"""
vzebra CLI — entry point for the Virtual Zebrafish platform.

Usage:
    vzebra run          Run a simulation episode
    vzebra train        Train from checkpoint
    vzebra serve        Launch web dashboard
    vzebra info         Show model and config info
    vzebra export       Export config to JSON
"""
from __future__ import annotations

import argparse
import json
import sys


def cmd_run(args: argparse.Namespace) -> None:
    from zebrav2.config import WorldConfig, BodyConfig, BrainConfig
    from zebrav2.virtual_fish import VirtualZebrafish

    world = WorldConfig()
    body = BodyConfig()
    brain = BrainConfig()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        if 'world' in cfg:
            world = WorldConfig.from_dict(cfg['world'])
        if 'body' in cfg:
            body = BodyConfig.from_dict(cfg['body'])
        if 'brain' in cfg:
            brain = BrainConfig.from_dict(cfg['brain'])

    if args.lesion:
        for region in args.lesion:
            setattr(brain.ablation, region, False)

    if args.arena:
        world.arena.width = {'open_field': 600, 'novel_tank': 400,
                             'light_dark': 800, 'social': 900}.get(args.arena, 800)
        world.arena.height = {'open_field': 600, 'novel_tank': 600,
                              'light_dark': 300, 'social': 300}.get(args.arena, 600)

    world.max_steps = args.steps

    fish = VirtualZebrafish(world=world, body=body, brain=brain)
    if args.checkpoint:
        fish.load_checkpoint(args.checkpoint)

    results = fish.run()
    print(json.dumps(results, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from zebrav2.engine.trainer import TrainingEngine
    from zebrav2.engine.config import TrainingConfig

    config = TrainingConfig({
        'training': {
            'n_rounds': args.rounds,
            'load_checkpoint': args.checkpoint,
        },
        'env': {
            'predator_ai': args.predator,
        },
    })
    engine = TrainingEngine(config)
    engine.run()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run('zebrav2.web.server:app', host=args.host, port=args.port,
                reload=args.reload)


def cmd_info(args: argparse.Namespace) -> None:
    from zebrav2.config import BrainConfig, BodyConfig, WorldConfig

    brain = BrainConfig()
    body = BodyConfig()
    world = WorldConfig()

    n_ablated = len(brain.get_ablated_set())
    n_enabled = len([v for v in brain.ablation.__dict__.values() if v])

    print("Virtual Zebrafish v2.0.0")
    print(f"  Brain modules:  {n_enabled} enabled, {n_ablated} ablated")
    print(f"  Fidelity:       {brain.fidelity}")
    print(f"  Personality:    {brain.personality}")
    print(f"  Arena:          {world.arena.width}x{world.arena.height}")
    print(f"  Max steps:      {world.max_steps}")
    print(f"  Sensory:        retina({body.sensory.retina.n_per_type}/eye), "
          f"LL({body.sensory.lateral_line.sn_range}px), "
          f"olf({body.sensory.olfaction.lambda_food}px)")
    print(f"  Motor:          CPG({body.motor.cpg.n_v2a}+{body.motor.cpg.n_v0d}"
          f"+{body.motor.cpg.n_mn}/side), RS({body.motor.n_reticulospinal})")
    print(f"  EFE goals:      FORAGE({brain.efe.forage_offset}) "
          f"FLEE({brain.efe.flee_offset}) "
          f"EXPLORE({brain.efe.explore_offset}) "
          f"SOCIAL({brain.efe.social_offset})")


def cmd_export(args: argparse.Namespace) -> None:
    from zebrav2.config import WorldConfig, BodyConfig, BrainConfig

    cfg = {
        'world': WorldConfig().to_dict(),
        'body': BodyConfig().to_dict(),
        'brain': BrainConfig().to_dict(),
    }
    output = json.dumps(cfg, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Config exported to {args.output}")
    else:
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='vzebra',
        description='Virtual Zebrafish: whole-brain simulation platform')
    sub = parser.add_subparsers(dest='command')

    # run
    p_run = sub.add_parser('run', help='Run a simulation episode')
    p_run.add_argument('--config', type=str, help='JSON config file')
    p_run.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    p_run.add_argument('--steps', type=int, default=500)
    p_run.add_argument('--arena', type=str,
                       choices=['open_field', 'novel_tank', 'light_dark', 'social'])
    p_run.add_argument('--lesion', nargs='+', help='Brain regions to lesion')
    p_run.set_defaults(func=cmd_run)

    # train
    p_train = sub.add_parser('train', help='Train from checkpoint')
    p_train.add_argument('--rounds', type=int, default=10)
    p_train.add_argument('--checkpoint', type=str)
    p_train.add_argument('--predator', type=str, default='intelligent',
                         choices=['none', 'simple', 'intelligent'])
    p_train.set_defaults(func=cmd_train)

    # serve
    p_serve = sub.add_parser('serve', help='Launch web dashboard')
    p_serve.add_argument('--host', default='0.0.0.0')
    p_serve.add_argument('--port', type=int, default=8765)
    p_serve.add_argument('--reload', action='store_true')
    p_serve.set_defaults(func=cmd_serve)

    # info
    p_info = sub.add_parser('info', help='Show model info')
    p_info.set_defaults(func=cmd_info)

    # export
    p_export = sub.add_parser('export', help='Export default config to JSON')
    p_export.add_argument('--output', '-o', type=str, help='Output file path')
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()
