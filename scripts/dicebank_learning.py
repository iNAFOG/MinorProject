"""Dice Bank policy-learning experiment: fixed local model vs random and hill-climb controls.

Protocol (declared before held-out evaluation; see docs/dicebank-learning-20261122.md):
  * Initial harness: the model's zero-shot policy (no feedback), installed as version 1.
  * Per arm and run: ROUNDS revisions. Each sees practice-game feedback only; the gate compares
    candidate vs current policy on the same VERIFICATION games (identical dice) and accepts iff
    mean paired score gain >= MIN_GAIN. Held-out games are played only after all learning ends.
  * Arms: model (fixed LLM reflects on feedback), random (uniform policies), hillclimb (small steps).
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
from agent_harness.games.dicebank import (NAIVE, DiceBankVerifier, HillClimbProposer, ModelProposer, RandomProposer,
                                          current_policy, evaluate, expected_turn_points, paired, policy_change,
                                          practice_digest)
from agent_harness.models import build_model
from agent_harness.storage.state import State

RUNS, ROUNDS, PRACTICE, VERIFICATION, HELD_OUT, MIN_GAIN = (1, 2, 3), 6, 50, 1000, 10000, 0.005
PRACTICE_START, VERIFICATION_START, HELD_OUT_START = 1_000_000, 5_000_000, 7_000_000


def exact(policy):
    return round(10*expected_turn_points(policy['bank_at'], policy['max_rolls']), 2)


def learn(arm, run, p0, proposer, verifier, out, log):
    folder = out/f'{arm}-run{run}'
    state = State(folder/'state.db')
    practice = list(range(PRACTICE_START*run, PRACTICE_START*run+PRACTICE))
    evidence = {f'practice-game-{s}': {'task_id': f'practice-game-{s}', 'split': 'adaptation', 'project': 'dicebank'}
                for s in practice}
    ids, decisions, history = list(evidence), [], []
    try:
        verifier.apply(state, policy_change(state, p0, ids, 'initial policy, no feedback'), evidence, gate=False)
        decisions.append({'round': 0, 'accepted': True, 'policy': p0, 'reason': 'initial harness'})
        for round_index in range(1, ROUNDS+1):
            current = current_policy(state)
            digest = practice_digest(current, practice)
            try:
                candidate, raw = proposer.propose(current, digest, history, round_index)
            except (ValueError, KeyError, TypeError) as exc:
                decisions.append({'round': round_index, 'accepted': False, 'raw': None,
                                  'reason': f'invalid proposal: {type(exc).__name__}: {exc}'})
                log(f'  {arm} run{run} round{round_index}: invalid proposal ({exc})')
                continue
            practice_total = evaluate(candidate, practice)['mean_total']
            history.append({**candidate, 'practice_mean_total': round(practice_total, 2)})
            decision = verifier.apply(state, policy_change(state, candidate, ids, f'{arm} revision {round_index}'), evidence, gate=True)
            decisions.append({'round': round_index, 'policy': candidate, 'practice_mean_total': practice_total,
                              **{k: decision.get(k) for k in ('accepted', 'gain', 'baseline_score', 'candidate_score', 'reason', 'version')}})
            log(f'  {arm} run{run} round{round_index}: {candidate} gain {decision.get("gain", 0)*100:+.2f} pts '
                f'{"ACCEPT" if decision["accepted"] else "reject"}')
        (folder/'decisions.json').write_text(json.dumps(decisions, indent=2), encoding='utf-8')
        (folder/'harness_history.json').write_text(json.dumps(state.history(), indent=2), encoding='utf-8')
        return current_policy(state), decisions, state.version
    finally:
        state.close()


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
    proposer = ModelProposer(build_model(config.model), config, out/'model-calls')
    verifier = DiceBankVerifier(list(range(VERIFICATION_START, VERIFICATION_START+VERIFICATION)), config, MIN_GAIN)
    held_out = list(range(HELD_OUT_START, HELD_OUT_START+HELD_OUT))
    started = time.time()
    try:
        p0, raw0 = proposer.propose(None, None, [], 0)
        source = 'model zero-shot'
    except (ValueError, KeyError, TypeError) as exc:
        p0, raw0, source = dict(NAIVE), f'invalid: {exc}', 'naive fallback (model zero-shot invalid)'
    log(f'initial policy ({source}): {p0}, exact expected total {exact(p0)}')
    finals = {}
    for run in RUNS:
        for arm, prop in (('model', proposer), ('random', RandomProposer(run)), ('hillclimb', HillClimbProposer(run))):
            finals[(arm, run)] = learn(arm, run, p0, prop, verifier, out, log)
    log('--- learning finished; policies frozen; held-out games played now ---')
    initial = evaluate(p0, held_out)
    results = {'protocol': {'runs': RUNS, 'rounds': ROUNDS, 'practice_games': PRACTICE, 'verification_games': VERIFICATION,
                            'held_out_games': HELD_OUT, 'min_gain': MIN_GAIN, 'model': config.model.model},
               'initial_policy': {'source': source, 'policy': p0, 'raw': raw0, 'exact_expected_total': exact(p0)},
               'held_out_initial': {k: v for k, v in initial.items() if k != 'scores'}, 'runs': {}}
    for (arm, run), (final, decisions, version) in finals.items():
        final = final or p0
        result = evaluate(final, held_out)
        comparison = paired(initial, result)
        accepted = sum(1 for d in decisions[1:] if d['accepted'])
        results['runs'][f'{arm}-run{run}'] = {'final_policy': final, 'exact_expected_total': exact(final),
                                              'harness_version': version, 'accepted_revisions': accepted,
                                              'held_out': {k: v for k, v in result.items() if k != 'scores'},
                                              'paired_vs_initial': comparison}
        log(f'{arm}-run{run}: {final} held-out total {initial["mean_total"]:.2f} -> {result["mean_total"]:.2f} '
            f'(delta {comparison["mean_score_delta"]*100:+.2f} pts, 95% [{comparison["bootstrap_95"][0]*100:+.2f}, '
            f'{comparison["bootstrap_95"][1]*100:+.2f}]), accepted {accepted}')
    optimum = {'bank_at': 20, 'max_rolls': 10}
    results['references_on_held_out'] = {
        'naive_bank_every_roll': {'policy': NAIVE, 'mean_total': evaluate(NAIVE, held_out)['mean_total']},
        'exact_optimum': {'policy': optimum, 'mean_total': evaluate(optimum, held_out)['mean_total'], 'exact_expected_total': exact(optimum)}}
    results['metadata'] = metadata(config)
    results['elapsed_seconds'] = round(time.time()-started)
    (out/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (out/'log.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    log(f'references: {json.dumps(results["references_on_held_out"])}')


if __name__ == '__main__':
    main()
