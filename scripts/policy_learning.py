"""Learned Battleship policy-skill experiment (fixed local model vs random-proposal control).

Protocol (declared before held-out evaluation; see docs/policy-learning-20261122.md):
  * Initial harness: the model's zero-shot policy (no practice feedback), installed as version 1.
  * Per run: ROUNDS revisions. Each uses practice-fleet feedback only, then the gate compares
    candidate vs current policy on the fixed verification fleets; accept iff mean paired
    score gain >= MIN_GAIN. Held-out fleets are evaluated only after all runs finish learning.
  * Control arm: identical, but proposals are uniform random weights (ignore feedback).
  * Fleet LAYOUTS are disjoint across practice / verification / held-out.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_harness.config import Config
from agent_harness.evaluation.benchmark import metadata
from agent_harness.games.battleship import Battleship, reference_salvo
from agent_harness.games.policy import (NAIVE, ModelProposer, PolicyVerifier, RandomProposer, current_policy, evaluate,
                                        fleet_seeds, paired, policy_change, practice_digest)
from agent_harness.models import build_model
from agent_harness.storage.state import State

RUNS, ROUNDS, PRACTICE, VERIFICATION, HELD_OUT, MIN_GAIN = (1, 2, 3), 6, 30, 300, 1000, 0.005
VERIFICATION_START, HELD_OUT_START = 500000, 700000


def learn(arm, run, p0, proposer, verification, out, log):
    folder = out/f'{arm}-run{run}'
    state = State(folder/'state.db')
    practice = fleet_seeds('adaptation', PRACTICE, 100000*run)
    evidence = {f'practice-fleet-{s}': {'task_id': f'practice-fleet-{s}', 'split': 'adaptation', 'project': 'battleship'}
                for s in practice}
    ids = list(evidence)
    verifier = verification
    decisions = []
    try:
        first = verifier.apply(state, policy_change(state, p0, ids, 'initial policy, no feedback'), evidence, gate=False)
        decisions.append({'round': 0, 'accepted': first['accepted'], 'policy': p0, 'reason': 'initial harness'})
        history = []
        for round_index in range(1, ROUNDS+1):
            current = current_policy(state)
            digest = practice_digest(current, practice)
            try:
                candidate, raw = proposer.propose(current, digest, history, round_index)
            except (ValueError, KeyError, TypeError) as exc:
                decisions.append({'round': round_index, 'accepted': False, 'reason': f'invalid proposal: {type(exc).__name__}: {exc}'})
                log(f'  {arm} run{run} round{round_index}: invalid proposal ({exc})')
                continue
            practice_score = evaluate(candidate, practice)['mean_hits']
            history.append({'weights': candidate['weights'], 'practice_mean_hits': round(practice_score, 3)})
            decision = verifier.apply(state, policy_change(state, candidate, ids, f'{arm} revision {round_index}'), evidence, gate=True)
            decisions.append({'round': round_index, 'policy': candidate, 'practice_mean_hits': practice_score,
                              **{k: decision.get(k) for k in ('accepted', 'gain', 'baseline_score', 'candidate_score',
                                                               'improved_fleets', 'regressed_fleets', 'reason', 'version')}})
            log(f'  {arm} run{run} round{round_index}: gain {decision.get("gain", 0):+.4f} '
                f'{"ACCEPT" if decision["accepted"] else "reject"} practice hits {practice_score:.3f}')
        final = current_policy(state)
        (folder/'decisions.json').write_text(json.dumps(decisions, indent=2), encoding='utf-8')
        (folder/'harness_history.json').write_text(json.dumps(state.history(), indent=2), encoding='utf-8')
        return final, decisions, state.version
    finally:
        state.close()


def reference_controller(seeds):
    scores = []
    for seed in seeds:
        game = Battleship(seed)
        while not game.terminal:
            game.step(reference_salvo(game.observe()))
        scores.append(game.metrics())
    return {'mean_score': statistics.mean(m['game_score'] for m in scores), 'win_rate': statistics.mean(m['outcome'] == 'win' for m in scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/battleship-qwen-local.yaml'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error('choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    config = Config.load(args.config)
    model = build_model(config.model)
    verification_seeds = fleet_seeds('verification', VERIFICATION, VERIFICATION_START)
    held_out_seeds = fleet_seeds('held_out', HELD_OUT, HELD_OUT_START)
    verifier = PolicyVerifier(verification_seeds, config, MIN_GAIN)
    proposer = ModelProposer(model, config, out/'model-calls')
    started = time.time()
    try:
        p0, p0_raw = proposer.propose(None, None, [], 0)
        p0_source = 'model zero-shot'
    except (ValueError, KeyError, TypeError) as exc:
        p0, p0_raw, p0_source = NAIVE, f'invalid: {exc}', 'naive fallback (model zero-shot invalid)'
    log(f'initial policy ({p0_source}): {p0}')
    finals = {}
    for run in RUNS:
        for arm, prop in (('model', proposer), ('random', RandomProposer(run))):
            finals[(arm, run)] = learn(arm, run, p0, prop, verifier, out, log)
    log('--- learning finished; held-out policies frozen ---')
    held = {'initial': evaluate(p0, held_out_seeds)}
    results = {'protocol': {'runs': RUNS, 'rounds': ROUNDS, 'practice_fleets': PRACTICE, 'verification_fleets': VERIFICATION,
                            'held_out_fleets': HELD_OUT, 'min_gain': MIN_GAIN, 'model': config.model.model,
                            'temperature': config.model.temperature},
               'initial_policy': {'source': p0_source, 'policy': p0, 'raw': p0_raw},
               'held_out_initial': {k: v for k, v in held['initial'].items() if k != 'scores'}, 'runs': {}}
    for (arm, run), (final, decisions, version) in finals.items():
        final = final or p0
        result = evaluate(final, held_out_seeds)
        comparison = paired(held['initial'], result)
        results['runs'][f'{arm}-run{run}'] = {'final_policy': final, 'harness_version': version,
                                              'accepted_revisions': sum(1 for d in decisions[1:] if d['accepted']),
                                              'held_out': {k: v for k, v in result.items() if k != 'scores'},
                                              'paired_vs_initial': comparison}
        log(f'{arm}-run{run}: held-out {held["initial"]["mean_score"]:.4f} -> {result["mean_score"]:.4f} '
            f'(delta {comparison["mean_score_delta"]:+.4f}, 95% {comparison["bootstrap_95"]}), '
            f'wins {held["initial"]["win_rate"]:.3f} -> {result["win_rate"]:.3f}, accepted {results["runs"][f"{arm}-run{run}"]["accepted_revisions"]}')
    results['references_on_held_out'] = {'naive_row_major': {k: v for k, v in evaluate(NAIVE, held_out_seeds).items() if k != 'scores'},
                                         'handwritten_hunt_target': reference_controller(held_out_seeds)}
    results['metadata'] = metadata(config)
    results['elapsed_seconds'] = round(time.time()-started)
    (out/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (out/'log.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    log(f'references: {results["references_on_held_out"]}')


if __name__ == '__main__':
    main()
