"""Learned Battleship firing policies: a bounded, model-proposed game skill.

A policy is a JSON object of feature weights. A local interpreter scores every
untested cell as a weighted sum of features computed from the PUBLIC observation
(board marks, sunk ships, fleet lengths) and fires at the three best cells. It
never reads the hidden fleet. Because play is local, one policy can be evaluated
on hundreds of fleets without a model request per move.

The fixed model proposes and revises policies from practice-game feedback. The
existing Verifier/State machinery stages each candidate, compares it with the
current policy on separate verification fleets, and promotes or rejects it.
"""
import hashlib
import json
import random
import statistics
from dataclasses import asdict
from agent_harness.core.budget import Budget
from agent_harness.core.context import call_model
from agent_harness.core.trajectory import Trajectory, now
from agent_harness.refinement.changes import HarnessChange
from agent_harness.verification.verifier import Verifier
from .battleship import Battleship
from .environment import game_task

FEATURES = {
    'placements': 'how many ways a remaining (unsunk) ship could still cover this cell, given misses and sunk ships (0..1)',
    'hit_placements': 'how many of those placements also pass through an unresolved hit (0..1)',
    'hit_neighbors': 'number of directly adjacent unresolved hits (0..1)',
    'line_extension': '1 if the cell extends a straight line of two or more unresolved hits, else 0',
    'checkerboard': '1 on cells where row+column is even, else 0',
    'center': '1 at the centre, falling to 0 at the corners',
}
WEIGHT_LIMIT = 5.0
POLICY_PREFIX = 'battleship-policy:'
NAIVE = {'weights': {name: 0.0 for name in FEATURES}}  # all ties -> fires cells in row-major order


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'weights'} or not isinstance(policy['weights'], dict):
        raise ValueError('policy must be {"weights": {...}}')
    weights = policy['weights']
    if set(weights) != set(FEATURES):
        raise ValueError('policy needs exactly these weights: ' + ', '.join(FEATURES))
    clean = {}
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -WEIGHT_LIMIT <= value <= WEIGHT_LIMIT:
            raise ValueError(f'weight {name} must be a number in [-{WEIGHT_LIMIT}, {WEIGHT_LIMIT}]')
        clean[name] = round(float(value), 3)
    return {'weights': clean}


def _placements(length):
    out = []
    for row in range(5):
        for col in range(5):
            if col + length <= 5:
                out.append(tuple(row*5+col+i for i in range(length)))
            if row + length <= 5:
                out.append(tuple((row+i)*5+col for i in range(length)))
    return out


PLACEMENTS = {length: _placements(length) for length in (2, 3)}


def features(observation):
    """Per-cell feature vectors from the public observation only."""
    board = observation['board']
    sunk = {c for ship in observation['sunk_ships'] for c in ship}
    remaining = list(observation['fleet_lengths'])
    for ship in observation['sunk_ships']:
        remaining.remove(len(ship))
    hits = {c for c, v in enumerate(board) if v == 'X'} - sunk
    blocked = {c for c, v in enumerate(board) if v == 'o'} | sunk
    cover, hit_cover = [0]*25, [0]*25
    for length in remaining:
        for cells in PLACEMENTS[length]:
            if blocked.intersection(cells):
                continue
            through_hit = bool(hits.intersection(cells))
            for c in cells:
                cover[c] += 1
                hit_cover[c] += through_hit
    max_cover, max_hit = max(cover) or 1, max(hit_cover) or 1
    rows = {}
    for cell in observation['legal_cells']:
        row, col = divmod(cell, 5)
        neighbors = [n for n, ok in ((cell-5, row > 0), (cell+5, row < 4), (cell-1, col > 0), (cell+1, col < 4)) if ok]
        line = 0
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            r1, c1, r2, c2 = row+dr, col+dc, row+2*dr, col+2*dc
            if 0 <= r2 < 5 and 0 <= c2 < 5 and r1*5+c1 in hits and r2*5+c2 in hits:
                line = 1
        rows[cell] = {'placements': cover[cell]/max_cover, 'hit_placements': hit_cover[cell]/max_hit,
                      'hit_neighbors': sum(n in hits for n in neighbors)/4, 'line_extension': line,
                      'checkerboard': float((row+col) % 2 == 0), 'center': 1-(abs(row-2)+abs(col-2))/4}
    return rows


def choose_salvo(observation, policy):
    weights = policy['weights']
    scored = features(observation)
    ranked = sorted(scored, key=lambda c: (-sum(weights[f]*v for f, v in scored[c].items()), c))
    return ranked[:3]


def play(policy, seed):
    game = Battleship(seed)
    while not game.terminal:
        game.step(choose_salvo(game.observe(), policy))  # policy sees public observations only
    return game.metrics() | {'history': game.history}


def layout_bucket(seed):
    ships = Battleship(seed)._ships  # evaluation-side split assignment only; never shown to policies or the model
    return int(hashlib.sha256(json.dumps(sorted(ships)).encode()).hexdigest(), 16) % 3


def fleet_seeds(split, count, start):
    """Disjoint fleet LAYOUTS across splits: practice=bucket 0, verification=1, held-out=2."""
    bucket = {'adaptation': 0, 'verification': 1, 'held_out': 2}[split]
    seeds, seed = [], start
    while len(seeds) < count:
        if layout_bucket(seed) == bucket:
            seeds.append(seed)
        seed += 1
    return seeds


def evaluate(policy, seeds):
    rows = [play(policy, s) for s in seeds]
    return {'fleets': len(rows), 'mean_score': statistics.mean(r['game_score'] for r in rows),
            'win_rate': statistics.mean(r['outcome'] == 'win' for r in rows),
            'mean_hits': statistics.mean(r['hits'] for r in rows), 'scores': [r['game_score'] for r in rows]}


def paired(before, after, iterations=2000, seed=713):
    diffs = [a-b for b, a in zip(before['scores'], after['scores'])]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(iterations))
    return {'mean_score_delta': statistics.mean(diffs), 'improved': sum(d > 0 for d in diffs),
            'regressed': sum(d < 0 for d in diffs), 'tied': sum(d == 0 for d in diffs),
            'bootstrap_95': [boot[int(0.025*iterations)-1], boot[int(0.975*iterations)-1]]}


def practice_digest(policy, seeds):
    """Feedback for the proposer, from public play records of the current policy."""
    rows = [play(policy, s) for s in seeds]
    salvos = [a for r in rows for a in r['history']]
    zero_hit = sum(all(x['result'] == 'miss' for x in a['results']) for a in salvos)
    unresolved = []
    for r in rows:
        last = r['history'][-1]['observation']
        sunk = {c for ship in last['sunk_ships'] for c in ship}
        unresolved.append(sum(v == 'X' and c not in sunk for c, v in enumerate(last['board'])))
    example = rows[0]
    return {'practice_games': len(rows), 'mean_hits_out_of_5': round(statistics.mean(r['hits'] for r in rows), 3),
            'win_rate': round(statistics.mean(r['outcome'] == 'win' for r in rows), 3),
            'salvos_with_no_hit_fraction': round(zero_hit/len(salvos), 3),
            'mean_unresolved_hits_at_end': round(statistics.mean(unresolved), 3),
            'example_game': [{'cells': a['cells'], 'results': [x['result'] for x in a['results']]} for a in example['history']]}


def parse_policy(raw):
    raw = raw.strip()
    if raw.startswith('```'):
        raw = '\n'.join(line for line in raw.splitlines() if not line.startswith('```'))
    if not raw.startswith('{') and '{' in raw:
        raw = raw[raw.index('{'):raw.rindex('}')+1]
    data = json.loads(raw)
    if isinstance(data, dict) and 'weights' not in data and set(data) == set(FEATURES):
        data = {'weights': data}  # tolerate a bare weight map
    return validate_policy({'weights': data.get('weights')} if isinstance(data, dict) else data)


class ModelProposer:
    """Asks the fixed model for a policy; records each call in a trajectory."""
    def __init__(self, model, config, output):
        self.model, self.config, self.output = model, config, output

    def propose(self, current, digest, history, round_index):
        system = ('You design firing policies for a 5x5 Battleship variant: two hidden ships of lengths 3 and 2, '
                  'five salvos of three shots, win by hitting all five ship cells. A policy gives one weight per feature; '
                  'each untested cell is scored as the weighted sum of its features and the three highest-scoring cells are fired. '
                  'Positive weight = prefer cells with that feature, negative = avoid. Reply with ONLY a JSON object '
                  '{"weights": {' + ', '.join(f'"{f}": number' for f in FEATURES) + '}} with every weight between -5 and 5.')
        user = {'features': FEATURES, 'current_policy': current,
                'practice_feedback_for_current_policy': digest,
                'previously_tried_policies': history[-6:],
                'task': 'Propose an improved policy that wins more practice games.' if current else
                        'Propose a good starting policy.'}
        trajectory = Trajectory(self.output, f'policy-round-{round_index}', 0, asdict(self.config))
        response = call_model(self.model, [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(user)}],
                              None, {'role': 'policy_proposer', 'max_output_tokens': self.config.model.max_output_tokens},
                              Budget(self.config.limits), trajectory)
        return parse_policy(response.content), response.content


class RandomProposer:
    """Control: proposals drawn uniformly from the same weight space, ignoring feedback."""
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def propose(self, current, digest, history, round_index):
        policy = {'weights': {f: round(self.rng.uniform(-WEIGHT_LIMIT, WEIGHT_LIMIT), 3) for f in FEATURES}}
        return validate_policy(policy), json.dumps(policy)


class PolicyVerifier(Verifier):
    """Rolls out the stored policy locally on verification fleets; aggregate paired gate."""
    def __init__(self, seeds, config, min_gain):
        tasks = [game_task('battleship', 'verification', s) for s in seeds[:2]]  # satisfies the split check
        super().__init__(tasks, lambda: None, config)
        self.seeds, self.min_gain = seeds, min_gain

    def evaluate(self, state, config=None):
        policy = current_policy(state)
        result = evaluate(policy or NAIVE, self.seeds)
        return [{'task_id': f'verification-fleet-{s}', 'game_score': v, 'verified': v == 1.0, 'model_calls': 0,
                 'tokens': 0, 'tool_calls': 0, 'estimated_cost': 0, 'latency': 0} for s, v in zip(self.seeds, result['scores'])]

    def compare(self, baseline, candidate):
        before, after = self.evaluate(baseline), self.evaluate(candidate)
        diffs = [a['game_score']-b['game_score'] for b, a in zip(before, after)]
        gain = statistics.mean(diffs)
        return {'accepted': gain >= self.min_gain, 'gain': gain, 'min_gain': self.min_gain,
                'baseline_score': statistics.mean(r['game_score'] for r in before),
                'candidate_score': statistics.mean(r['game_score'] for r in after),
                'improved_fleets': sum(d > 0 for d in diffs), 'regressed_fleets': sum(d < 0 for d in diffs),
                'verification_fleets': len(diffs), 'baseline': [], 'candidate': []}


POLICY_NOTE_ID = hashlib.sha256(b'battleship-policy').hexdigest()[:20]


def current_policy(state):
    for entry in state.entries('note'):
        if entry['id'] == POLICY_NOTE_ID and entry.get('status') == 'active':
            return validate_policy(json.loads(entry['content'][len(POLICY_PREFIX):]))
    return None


def policy_change(state, policy, evidence, reason):
    old = next((e for e in state.entries('note') if e['id'] == POLICY_NOTE_ID), None)
    entry = {'id': POLICY_NOTE_ID, 'content': POLICY_PREFIX + json.dumps(policy, sort_keys=True),
             'source': 'policy_proposal_from_practice', 'provenance': list(evidence), 'status': 'active',
             'version': (old or {}).get('version', 0)+1, 'timestamp': now()}
    return HarnessChange('UPDATE_PROMPT_NOTE', POLICY_NOTE_ID, old, entry, tuple(evidence), reason)
