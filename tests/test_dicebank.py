import pytest
from agent_harness.config import Config
from agent_harness.games.dicebank import (NAIVE, DiceBankVerifier, HillClimbProposer, RandomProposer, current_policy,
                                          dice, evaluate, expected_turn_points, parse_policy, play, policy_change,
                                          validate_policy)
from agent_harness.storage.state import State


def test_rules_bust_and_bank():
    result = play({'bank_at': 100, 'max_rolls': 30}, 0)
    for turn in result['turns']:
        if turn['bust']:
            assert turn['rolls'][-1] == 1 and turn['banked'] == 0
        else:
            assert 1 not in turn['rolls'] and turn['banked'] == sum(turn['rolls'])
    assert result['total'] == sum(t['banked'] for t in result['turns'])


def test_same_seed_means_same_dice_for_every_policy():
    assert dice(5) == dice(5) and dice(5) != dice(6)
    a, b = play({'bank_at': 6, 'max_rolls': 30}, 5), play({'bank_at': 30, 'max_rolls': 30}, 5)
    for ta, tb in zip(a['turns'], b['turns']):
        shorter = min(len(ta['rolls']), len(tb['rolls']))
        assert ta['rolls'][:shorter] == tb['rolls'][:shorter]


def test_exact_values_and_simulation_agree():
    assert expected_turn_points(1, 1) == pytest.approx(10/3)
    # Banking at 20 and rolling once more at exactly 20 are worth the same (5/6 x 24 = 20).
    assert max((expected_turn_points(b), b) for b in range(1, 50))[1] in {20, 21}
    assert expected_turn_points(20) == pytest.approx(expected_turn_points(21), abs=1e-9)
    sim = evaluate({'bank_at': 20, 'max_rolls': 30}, range(4000))['mean_total']
    assert sim == pytest.approx(10*expected_turn_points(20), rel=0.03)


def test_policy_validation_and_parsing():
    assert validate_policy({'bank_at': 20.0, 'max_rolls': 5}) == {'bank_at': 20, 'max_rolls': 5}
    for bad in [{'bank_at': 0, 'max_rolls': 5}, {'bank_at': 20}, {'bank_at': 20, 'max_rolls': 31}, {'bank_at': True, 'max_rolls': 2}]:
        with pytest.raises(ValueError):
            validate_policy(bad)
    assert parse_policy('```json\n{"bank_at": 18, "max_rolls": 6, "why": "x"}\n```') == {'bank_at': 18, 'max_rolls': 6}
    assert parse_policy('Policy: {"policy": {"bank_at": 15, "max_rolls": 4}}') == {'bank_at': 15, 'max_rolls': 4}
    # Real replies from the run: wrapped, followed by invented feedback, or cut off at the output limit.
    wrapped = '```json\n{\n  "improved_policy": {\n    "bank_at": 7,\n    "max_rolls": 25\n  },\n  "practice_feedback": {"mean_total": 50.2}\n}\n```'
    assert parse_policy(wrapped) == {'bank_at': 7, 'max_rolls': 25}
    cut = '```json\n{\n  "improved_policy": {\n    "bank_at": 10,\n    "max_rolls": 25\n  },\n  "example_turns": [\n {"rolls": [1, 1, 1,'
    assert parse_policy(cut) == {'bank_at': 10, 'max_rolls': 25}
    with pytest.raises(ValueError):
        parse_policy('{"improved_policy": {"bank_at": 500, "max_rolls": 25}}')
    with pytest.raises(ValueError):
        parse_policy('I would bank sooner.')


def test_gate_accepts_better_policy_and_rejects_worse():
    verifier = DiceBankVerifier(list(range(300)), Config.load('configs/battleship-qwen-local.yaml'), 0.005)
    evidence = {f'p{i}': {'task_id': f'p{i}', 'split': 'adaptation', 'project': 'dicebank'} for i in range(2)}
    state = State()
    verifier.apply(state, policy_change(state, NAIVE, list(evidence), 'initial'), evidence, gate=False)
    assert verifier.apply(state, policy_change(state, {'bank_at': 20, 'max_rolls': 10}, list(evidence), 'r'), evidence)['accepted']
    assert not verifier.apply(state, policy_change(state, NAIVE, list(evidence), 'r'), evidence)['accepted']
    assert current_policy(state) == {'bank_at': 20, 'max_rolls': 10} and state.version == 2
    state.close()


def test_controls_are_reproducible_and_valid():
    for cls in (RandomProposer, HillClimbProposer):
        assert cls(3).propose(NAIVE, None, [], 1) == cls(3).propose(NAIVE, None, [], 1)
        validate_policy(cls(3).propose(NAIVE, None, [], 1)[0])
