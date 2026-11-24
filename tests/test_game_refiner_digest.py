import pytest
from agent_harness.games.battleship import Battleship
from agent_harness.games.benchmark import episode_digest, parse_strategy


def record_for(game, salvos, outcome, final=''):
    initial = game.observe()
    for cells in salvos:
        game.step(cells)
    metrics = game.metrics()
    return {'task_id': 't', 'game': 'battleship', 'outcome': outcome if not game.terminal else metrics['outcome'],
            'game_score': 0.0 if not game.terminal else metrics['game_score'], 'illegal_moves': metrics['illegal_moves'],
            'initial_observation': initial, 'game_actions': game.history, 'final': final}


def test_digest_reports_early_stop_and_wasted_salvos_from_public_data():
    game = Battleship(4)
    hit = sorted(game._occupied)[0]  # test setup only; the digest itself never reads hidden cells
    others = [c for c in range(25) if c != hit][:2]
    digest = episode_digest(record_for(game, [[hit, *others], [hit, 20, 21]], 'early_stop', 'I fired.'))
    text = ' '.join(digest['mistakes'])
    assert 'Stopped after 2 of 5 salvos' in text and '1 salvo(s) were invalid' in text
    assert 'untested neighbours' in text
    assert digest['agent_last_message'] == 'I fired.'
    assert '_ships' not in str(digest) and '_occupied' not in str(digest)


@pytest.mark.parametrize('raw,expected', [
    ('{"strategy": "Keep firing until the game ends."}', 'Keep firing until the game ends.'),
    ('```json\n{"strategy": "Keep firing."}\n```', 'Keep firing.'),
    ('Here you go:\n{"strategy": "Keep firing every turn."} Good luck', 'Keep firing every turn.'),
    ('{"strategy": ["Keep firing.", "Never repeat cells."]}', 'Keep firing. Never repeat cells.'),
    ('{"strategy": "Call fire every turn.", "call_fire_on_mistake": true}', 'Call fire every turn.')])
def test_parse_strategy_accepts_plain_rules(raw, expected):
    assert parse_strategy(raw) == expected


@pytest.mark.parametrize('raw', ['{"strategy": {"fire": "function(row) {}"}}', '{"strategy": "x"}',
                                 '{"strategy": "use def fire(): pass to play"}', 'no json', '{"rules": "keep firing always"}'])
def test_parse_strategy_rejects_code_and_bad_shapes(raw):
    with pytest.raises(ValueError):
        parse_strategy(raw)
