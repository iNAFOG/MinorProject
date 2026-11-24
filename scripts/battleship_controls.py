"""Paired non-LLM reference controls. No credentials or network calls."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_harness.games.battleship import Battleship, reference_salvo
from agent_harness.games.benchmark import paired_comparison


def run_controls(count=2000, start_seed=50000):
    if not 2 <= count <= 100000:
        raise ValueError('use 2..100000 scenarios')
    results, records = {}, {}
    for mode in ('random','hunt_target'):
        rows=[]
        for seed in range(start_seed,start_seed+count):
            game=Battleship(seed)
            while not game.terminal:
                obs=game.observe()
                # Controller receives only public observations, never the game object.
                game.step(reference_salvo(obs,mode,seed*10+game.steps))
            rows.append(game.metrics() | {'task_id':f'battleship-{seed}', 'model_calls':0,'tokens':0})
        records[mode]=rows
        results[mode]={'episodes':count,'win_rate':statistics.mean(r['outcome']=='win' for r in rows),
                       'mean_hits':statistics.mean(r['hits'] for r in rows),
                       'mean_score':statistics.mean(r['game_score'] for r in rows),
                       'mean_shots':statistics.mean(r['shots_fired'] for r in rows)}
    return {'kind':'non-LLM benchmark sensitivity controls; NOT harness learning results',
            'seed_range':[start_seed,start_seed+count-1],'results':results,
            'paired':paired_comparison(records['random'],records['hunt_target']),
            'network_requests':0,
            'limitations':['Two handwritten fixed policies, neither learns.',
                           'Random seeds may generate the same fleet; this is sampling with replacement.',
                           'These results demonstrate strategy sensitivity, not an adaptive-harness improvement.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=2000)
    parser.add_argument('--output',default='runs/battleship-controls-20261017')
    args=parser.parse_args()
    result=run_controls(args.episodes)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'controls.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Battleship benchmark sensitivity controls','',result['kind'],'',
           '| Fixed controller | Episodes | Mean targets hit / 5 | Win rate | Mean shots |',
           '|---|---:|---:|---:|---:|']
    for name, r in result['results'].items():
        lines.append(f'| {name} | {r["episodes"]} | {r["mean_hits"]:.3f} | {r["win_rate"]:.1%} | {r["mean_shots"]:.3f} |')
    lines+=['','Both controllers see only hit/miss observations. Same seeded fleets in both conditions.',
            'This comparison shows the benchmark can distinguish strategies. It does not show that the model or harness learned.',
            '', 'Paired mean-score delta: '+str(result['paired']['mean_score_delta']),
            'Exploratory paired bootstrap interval: '+str(result['paired']['exploratory_paired_episode_bootstrap_95_percent'])]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
