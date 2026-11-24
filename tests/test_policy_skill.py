import pytest
from agent_harness.config import Config
from agent_harness.games.battleship import Battleship
from agent_harness.games.policy import (FEATURES, NAIVE, PolicyVerifier, RandomProposer, choose_salvo, current_policy,
                                        evaluate, features, fleet_seeds, layout_bucket, parse_policy, play, policy_change,
                                        validate_policy)
from agent_harness.storage.state import State

GOOD = {'weights': {'placements': 1, 'hit_placements': 4, 'hit_neighbors': 2, 'line_extension': 3, 'checkerboard': 0.5, 'center': 0}}


def test_policy_validation_is_strict():
    assert validate_policy(GOOD)['weights']['hit_placements'] == 4.0
    for bad in [{'weights': {}}, {'weights': {**GOOD['weights'], 'center': 9}}, {'weights': {**GOOD['weights'], 'extra': 1}},
                {'weights': {**GOOD['weights'], 'center': True}}, GOOD['weights']]:
        with pytest.raises(ValueError):
            validate_policy(bad)


def test_parse_policy_tolerates_fences_prose_and_bare_maps():
    import json
    assert parse_policy('```json\n' + json.dumps(GOOD) + '\n```') == validate_policy(GOOD)
    assert parse_policy('Sure: ' + json.dumps(GOOD['weights'])) == validate_policy(GOOD)


def test_features_use_public_observation_only_and_policy_is_always_legal():
    game = Battleship(3)
    observation = game.observe()
    assert set(features(observation)) == set(observation['legal_cells'])
    assert all(set(v) == set(FEATURES) for v in features(observation).values())
    for seed in range(20):
        result = play(GOOD, seed)
        assert result['illegal_moves'] == 0 and result['steps'] <= 5
    assert choose_salvo(observation, NAIVE) == [0, 1, 2]


def test_feature_based_policy_beats_naive_sweep():
    seeds = list(range(200))
    assert evaluate(GOOD, seeds)['mean_score'] > evaluate(NAIVE, seeds)['mean_score'] + 0.1


def test_fleet_layouts_are_disjoint_across_splits():
    splits = {s: fleet_seeds(s, 40, 1000) for s in ('adaptation', 'verification', 'held_out')}
    layouts = {s: {tuple(Battleship(x)._ships) for x in seeds} for s, seeds in splits.items()}
    assert not layouts['adaptation'] & layouts['verification']
    assert not layouts['verification'] & layouts['held_out'] and not layouts['adaptation'] & layouts['held_out']
    assert all(layout_bucket(x) == 2 for x in splits['held_out'])


def test_gate_accepts_improvement_and_rejects_regression_through_state():
    config = Config.load('configs/battleship-qwen-local.yaml')
    verifier = PolicyVerifier(fleet_seeds('verification', 60, 0), config, 0.005)
    evidence = {f'p{i}': {'task_id': f'p{i}', 'split': 'adaptation', 'project': 'battleship'} for i in range(2)}
    state = State()
    verifier.apply(state, policy_change(state, NAIVE, list(evidence), 'initial'), evidence, gate=False)
    better = verifier.apply(state, policy_change(state, GOOD, list(evidence), 'revision'), evidence, gate=True)
    assert better['accepted'] and current_policy(state) == validate_policy(GOOD) and state.version == 2
    worse = verifier.apply(state, policy_change(state, NAIVE, list(evidence), 'revision'), evidence, gate=True)
    assert not worse['accepted'] and current_policy(state) == validate_policy(GOOD)
    state.close()


def test_random_control_is_reproducible():
    assert RandomProposer(1).propose(None, None, [], 1) == RandomProposer(1).propose(None, None, [], 1)
