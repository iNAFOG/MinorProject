"""Bounded local game benchmark. No network requests unless a real config is selected."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_harness.config import Config
from agent_harness.games.benchmark import GameExperiment
from agent_harness.games.models import load_key_from_environment


def main(argv=None):
    parser=argparse.ArgumentParser(description='Hidden-fleet Battleship fixed-model adaptation benchmark (legacy games optional)')
    parser.add_argument('--config',default=str(ROOT/'configs/games-smoke.yaml'))
    parser.add_argument('--output',default='runs/games-smoke')
    parser.add_argument('--games',default='battleship')
    parser.add_argument('--adaptation',type=int,default=4)
    parser.add_argument('--verification',type=int,default=4)
    parser.add_argument('--held-out',type=int,default=12)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-requests',type=int,default=120)
    parser.add_argument('--max-total-tokens',type=int,default=200000)
    parser.add_argument('--request-interval',type=float,default=None,help='Minimum seconds between request starts (default: 6 remote, 0 offline)')
    parser.add_argument('--model',help='Explicit provider model ID, held fixed throughout the run')
    parser.add_argument('--ablate-notes',action='store_true',help='Also evaluate a frozen copy with learned notes removed')
    parser.add_argument('--check-setup',action='store_true',help='Check the configured key variable without displaying it or making an API call')
    args=parser.parse_args(argv)
    cfg=Config.load(args.config)
    if args.model:
        cfg=replace(cfg,model=replace(cfg.model,model=args.model))
    if args.check_setup:
        available=cfg.model.backend in {'game_smoke','transformers'} or load_key_from_environment(cfg.model.api_key_env)
        print(json.dumps({'model':cfg.model.model,'backend':cfg.model.backend,'key_variable':cfg.model.api_key_env,
                          'key_available':available,'network_requests':0,'free_tier_eligibility':'confirm in your AI Studio project'},indent=2))
        return 0 if available else 2
    interval=args.request_interval if args.request_interval is not None else (0 if cfg.model.backend in {'game_smoke','transformers'} else 6)
    try:
        experiment=GameExperiment(cfg,args.output,args.max_requests,args.max_total_tokens,interval)
        summary=experiment.run(tuple(args.games.split(',')),args.adaptation,args.verification,args.held_out,args.seed,args.ablate_notes)
    except (ValueError,FileExistsError) as exc:
        parser.error(str(exc))
    compact={k:summary[k] for k in ('status','stop_reason','is_real_model','interpretation','request_attempts',
                                  'total_tokens','accepted_changes','rejected_changes')}
    compact['games']={game:{'initial_score':groups['initial']['mean_game_score'],
                           'learned_score':groups['learned']['mean_game_score'],
                           'paired_score_change':groups['paired']['mean_score_delta'],
                           'held_out_episodes':groups['initial']['episodes']} for game,groups in summary['results'].items()}
    print(json.dumps(compact,indent=2))
    print(f'Report: {Path(args.output).resolve()/"report.md"}')
    return 0 if summary['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
