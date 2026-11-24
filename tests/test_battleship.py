from dataclasses import replace
import json
import pytest
from agent_harness.config import Config
from agent_harness.games.battleship import Battleship, reference_salvo
from agent_harness.games.benchmark import GameExperiment, make_suite, validate_suites
from agent_harness.games.environment import GameEnvironment, game_task


def test_hidden_fleet_has_correct_geometry_and_is_reproducible():
    fleets=set()
    for seed in range(100):
        game=Battleship(seed)
        assert [len(s) for s in game._ships]==[3,2]
        assert len(game._occupied)==5
        for ship in game._ships:
            assert len({c//5 for c in ship})==1 or len({c%5 for c in ship})==1
            assert max(ship)-min(ship) in {len(ship)-1,5*(len(ship)-1)}
        assert game._ships==Battleship(seed)._ships
        fleets.add(tuple(game._ships))
        assert game.observe()==Battleship(seed+1).observe()  # No seed or unhit ship information.
        assert not any(key.startswith('_') or key in {'seed','ships','occupied'} for key in game.observe())
    assert len(fleets)>80


@pytest.mark.parametrize('bad',[None,[0,0,1],[True,1,2],['0',1,2],[-1,0,1],[0,1,25],[0,1],{}])
def test_malformed_salvo_wastes_turn_and_never_repaired(bad):
    game=Battleship(0)
    obs=game.step(bad)
    assert obs['error'] and obs['shots_fired']==0 and game.steps==1 and game.illegal==1


def test_repeat_batch_is_atomic_and_five_bad_turns_lose():
    game=Battleship(2)
    game.step([0,1,2])
    assert game.step([2,3,4])['error']
    assert game.observe()['shots_fired']==3  # No partial execution before validation.
    for _ in range(3):
        game.step(None)
    assert game.outcome=='loss' and game.terminal
    assert game.metrics()['game_score']==len(game._shots & game._occupied)/5


def test_hits_sunk_feedback_and_win_agree_with_authoritative_fleet():
    game=Battleship(9)
    ship=list(game._ships[0])
    obs=game.step(ship)
    assert all(r['result']=='hit' for r in game.history[-1]['results'])
    assert obs['sunk_ships']==[ship]
    other=list(game._ships[1])
    water=next(c for c in obs['legal_cells'] if c not in game._occupied)
    obs=game.step(other+[water])
    assert obs['terminal'] and obs['outcome']=='win'
    assert game.metrics()['game_score']==1 and game.metrics()['ships_sunk']==2
    assert game.metrics()['shots_fired']==6


def test_game_tools_never_expose_hidden_board_and_reject_wrong_schema():
    task=game_task('battleship','held_out',2000)
    env=GameEnvironment(task)
    assert set(env.tool_definitions())=={'fire'}
    assert json.loads(task.files['observation.json'])==env.engine.observe()
    assert env.call_tool('fire',{'cells':[1,2,3],'secret':True})['error']
    assert env.engine.observe()['shots_fired']==0
    with pytest.raises(PermissionError):
        env.read('ships')


def test_hidden_fleet_split_disjointness_and_duplicate_detection():
    suites={split:make_suite('battleship',split,n) for split,n in [('adaptation',2),('verification',2),('held_out',3)]}
    validate_suites(suites)
    # Same layout with a different task ID is still a duplicate.
    suites['held_out'][0]=replace(suites['adaptation'][0],split='held_out',task_id='different-id',
                                  metadata=suites['adaptation'][0].metadata | {'size':999})
    with pytest.raises(ValueError,match='hidden fleet'):
        validate_suites(suites)


def test_model_task_labels_do_not_disclose_placement_seed():
    task=game_task('battleship','held_out',987654321)
    assert '987654321' not in json.dumps(task.visible())
    assert task.metadata['seed']==987654321
    assert task.task_id==game_task('battleship','held_out',987654321).task_id


def test_reference_controller_uses_public_feedback_and_valid_salvos():
    for seed in range(20):
        game=Battleship(seed)
        while not game.terminal:
            assert 'error' not in game.step(reference_salvo(game.observe()))
        assert game.steps<=5 and game.metrics()['shots_fired']<=15


def test_default_smoke_run_is_battleship_and_does_not_fake_learning(tmp_path):
    cfg=Config.load('configs/battleship-gemini.yaml')
    cfg=replace(cfg,model=replace(cfg.model,backend='game_smoke',model='handwritten-no-learning'))
    report=GameExperiment(cfg,tmp_path,max_requests=100,max_tokens=500000).run(
        adaptation=2,verification=2,held_out=3,ablate_notes=True)
    assert report['status']=='complete' and not report['is_real_model']
    assert report['accepted_changes']==0 and report['rejected_changes']==1
    result=report['results']['battleship']
    assert result['paired']['mean_score_delta']==0
    assert result['initial']['mean_game_score']==result['without_notes']['mean_game_score']
    assert 'battleship' in (tmp_path/'replay.html').read_text()
