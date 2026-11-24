"""Dice Bank: a small push-your-luck game for testing policy learning.

Rules: 10 turns. In a turn you roll one six-sided die as often as you like, adding
the faces to your turn points. Rolling a 1 loses the turn points and ends the turn.
Banking ends the turn and adds the turn points to your total. Score = total / 100.

A policy is two integers: bank as soon as turn points reach ``bank_at``, or after
``max_rolls`` rolls in the turn, whichever comes first. Every game seed fixes the
dice for every (turn, roll) position, so all policies face exactly the same dice on
the same game: paired comparisons are not confounded by luck.

The policy interpreter, verifier and change records reuse the harness's existing
Verifier / State / HarnessChange machinery, as the Battleship policy skill does.
"""
import hashlib
import json
import random
import statistics
from dataclasses import asdict
from agent_harness.core.budget import Budget
from agent_harness.core.context import call_model
from agent_harness.core.trajectory import Trajectory, now
from agent_harness.environments.base import Task
from agent_harness.refinement.changes import HarnessChange
from agent_harness.verification.verifier import Verifier

TURNS, MAX_ROLLS_PER_TURN = 10, 30
RULES = ('Dice Bank: 10 turns. Each turn, roll one six-sided die as many times as you like, adding the faces to your '
         'turn points. Rolling a 1 loses all turn points and ends the turn. Banking ends the turn and adds the turn '
         'points to your total. Score = total after 10 turns. A policy banks as soon as turn points reach bank_at, or '
         'after max_rolls rolls in the turn, whichever comes first.')
LIMITS = {'bank_at': (1, 100), 'max_rolls': (1, MAX_ROLLS_PER_TURN)}
NAIVE = {'bank_at': 1, 'max_rolls': 1}      # always bank after the first safe roll
POLICY_PREFIX = 'dicebank-policy:'
POLICY_ID = hashlib.sha256(b'dicebank-policy').hexdigest()[:20]


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != set(LIMITS):
        raise ValueError('policy must be {"bank_at": int, "max_rolls": int}')
    clean = {}
    for name, (lo, hi) in LIMITS.items():
        value = policy[name]
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
        clean[name] = value
    return clean


def dice(seed):
    """Dice for every turn and roll position of one game, independent of decisions."""
    games = []
    for turn in range(TURNS):
        rng = random.Random(f'dicebank:{seed}:{turn}')
        games.append([rng.randint(1, 6) for _ in range(MAX_ROLLS_PER_TURN)])
    return games


def play(policy, seed):
    total, turns = 0, []
    for rolls in dice(seed):
        points, used, bust = 0, 0, False
        for face in rolls:
            used += 1
            if face == 1:
                points, bust = 0, True
                break
            points += face
            if points >= policy['bank_at'] or used >= policy['max_rolls']:
                break
        total += points
        turns.append({'rolls': rolls[:used], 'banked': points, 'bust': bust})
    return {'total': total, 'score': total/100, 'turns': turns}


def expected_turn_points(bank_at, max_rolls=MAX_ROLLS_PER_TURN):
    """Exact expected points from one turn under a policy (dynamic programming)."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def value(points, rolls):
        if rolls > 0 and (points >= bank_at or rolls >= max_rolls):
            return float(points)
        return sum(value(points+f, rolls+1) for f in range(2, 7))/6  # a 1 contributes 0
    return value(0, 0)


def evaluate(policy, seeds):
    rows = [play(policy, s) for s in seeds]
    return {'games': len(rows), 'mean_score': statistics.mean(r['score'] for r in rows),
            'mean_total': statistics.mean(r['total'] for r in rows), 'scores': [r['score'] for r in rows]}


def paired(before, after, iterations=2000, seed=713):
    diffs = [a-b for b, a in zip(before['scores'], after['scores'])]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(iterations))
    return {'mean_score_delta': statistics.mean(diffs), 'improved': sum(d > 0 for d in diffs),
            'regressed': sum(d < 0 for d in diffs), 'tied': sum(d == 0 for d in diffs),
            'bootstrap_95': [boot[int(0.025*iterations)-1], boot[int(0.975*iterations)-1]]}


def practice_digest(policy, seeds):
    rows = [play(policy, s) for s in seeds]
    turns = [t for r in rows for t in r['turns']]
    busts = [t for t in turns if t['bust']]
    lost = [sum(t['rolls'][:-1]) for t in busts]
    return {'practice_games': len(rows), 'mean_total': round(statistics.mean(r['total'] for r in rows), 2),
            'mean_banked_per_turn': round(statistics.mean(t['banked'] for t in turns), 2),
            'bust_rate_per_turn': round(len(busts)/len(turns), 3),
            'mean_points_lost_when_busting': round(statistics.mean(lost), 2) if lost else 0,
            'mean_rolls_per_turn': round(statistics.mean(len(t['rolls']) for t in turns), 2),
            'example_turns': [{'rolls': t['rolls'], 'banked': t['banked']} for t in rows[0]['turns'][:4]]}


def _find_policy(value):
    """First dict (depth-first) that holds both policy fields, e.g. under "improved_policy"."""
    if isinstance(value, dict):
        if all(k in value for k in LIMITS):
            return {k: value[k] for k in LIMITS}
        for child in value.values():
            found = _find_policy(child)
            if found is not None:
                return found
    return None


def parse_policy(raw):
    """Read the proposed policy from a model reply.

    Small models often wrap it (``{"improved_policy": {...}}``), add extra keys, or keep
    generating until the output limit, leaving invalid JSON after a valid policy. The
    first complete ``bank_at``/``max_rolls`` pair is used; the values are still validated.
    """
    import re
    text = raw.strip()
    if text.startswith('```'):
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('```'))
    if not text.startswith('{') and '{' in text:
        text = text[text.index('{'):]
    try:
        data = json.loads(text[:text.rindex('}')+1] if '}' in text else text)
        found = _find_policy(data)
        if found is not None:
            return validate_policy(found)
    except ValueError:
        pass
    numbers = {k: re.search(rf'"{k}"\s*:\s*(-?\d+(?:\.\d+)?)', text) for k in LIMITS}
    if all(numbers.values()):
        return validate_policy({k: float(m.group(1)) if '.' in m.group(1) else int(m.group(1)) for k, m in numbers.items()})
    raise ValueError('reply contains no bank_at/max_rolls policy')


class ModelProposer:
    def __init__(self, model, config, output):
        self.model, self.config, self.output = model, config, output

    def propose(self, current, digest, history, round_index):
        system = ('You design a policy for a dice game. ' + RULES + ' Reply with ONLY a JSON object '
                  '{"bank_at": integer 1-100, "max_rolls": integer 1-30}.')
        user = {'current_policy': current, 'practice_feedback_for_current_policy': digest,
                'previously_tried_policies': history[-6:],
                'task': 'Propose an improved policy that scores more points.' if current else
                        'Propose a good starting policy.'}
        trajectory = Trajectory(self.output, f'dicebank-round-{round_index}', 0, asdict(self.config))
        response = call_model(self.model, [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(user)}],
                              None, {'role': 'policy_proposer', 'max_output_tokens': self.config.model.max_output_tokens},
                              Budget(self.config.limits), trajectory)
        return parse_policy(response.content), response.content


class RandomProposer:
    """Control: uniform random policies, ignoring all feedback."""
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def propose(self, current, digest, history, round_index):
        policy = {name: self.rng.randint(lo, hi) for name, (lo, hi) in LIMITS.items()}
        return validate_policy(policy), json.dumps(policy)


class HillClimbProposer:
    """Control: small random steps from the current policy, ignoring the feedback text."""
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def propose(self, current, digest, history, round_index):
        base = current or NAIVE
        policy = {'bank_at': min(100, max(1, base['bank_at']+self.rng.choice([-1, 1])*self.rng.randint(1, 8))),
                  'max_rolls': min(MAX_ROLLS_PER_TURN, max(1, base['max_rolls']+self.rng.choice([-1, 1])*self.rng.randint(1, 3)))}
        return validate_policy(policy), json.dumps(policy)


def current_policy(state):
    for entry in state.entries('note'):
        if entry['id'] == POLICY_ID and entry.get('status') == 'active':
            return validate_policy(json.loads(entry['content'][len(POLICY_PREFIX):]))
    return None


def policy_change(state, policy, evidence, reason):
    old = next((e for e in state.entries('note') if e['id'] == POLICY_ID), None)
    entry = {'id': POLICY_ID, 'content': POLICY_PREFIX + json.dumps(policy, sort_keys=True),
             'source': 'policy_proposal_from_practice', 'provenance': list(evidence), 'status': 'active',
             'version': (old or {}).get('version', 0)+1, 'timestamp': now()}
    return HarnessChange('UPDATE_PROMPT_NOTE', POLICY_ID, old, entry, tuple(evidence), reason)


class DiceBankVerifier(Verifier):
    """Plays the stored policy on fixed verification games; aggregate paired gate."""
    def __init__(self, seeds, config, min_gain):
        tasks = [Task(f'dicebank-verification-{i}', 'Score points in Dice Bank.', 'verification', 'dicebank',
                      {'rules.txt': RULES}, ({'kind': 'game_outcome'},), target='rules.txt', writable_paths=(),
                      environment='game') for i in range(2)]
        super().__init__(tasks, lambda: None, config)
        self.seeds, self.min_gain = seeds, min_gain

    def evaluate(self, state, config=None):
        result = evaluate(current_policy(state) or NAIVE, self.seeds)
        return [{'task_id': f'dicebank-game-{s}', 'game_score': v, 'verified': True, 'model_calls': 0, 'tokens': 0,
                 'tool_calls': 0, 'estimated_cost': 0, 'latency': 0} for s, v in zip(self.seeds, result['scores'])]

    def compare(self, baseline, candidate):
        before, after = self.evaluate(baseline), self.evaluate(candidate)
        diffs = [a['game_score']-b['game_score'] for b, a in zip(before, after)]
        gain = statistics.mean(diffs)
        return {'accepted': gain >= self.min_gain, 'gain': gain, 'min_gain': self.min_gain,
                'baseline_score': statistics.mean(r['game_score'] for r in before),
                'candidate_score': statistics.mean(r['game_score'] for r in after),
                'improved_games': sum(d > 0 for d in diffs), 'regressed_games': sum(d < 0 for d in diffs),
                'verification_games': len(diffs), 'baseline': [], 'candidate': []}
