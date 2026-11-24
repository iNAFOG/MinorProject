from dataclasses import replace
import json
from pathlib import Path
import pytest
from agent_harness.config import Config
from agent_harness.core.agent import Orchestrator
from agent_harness.games.engines import TicTacToe, KeyDoor, minimax_value, shortest_path, winner
from agent_harness.games.environment import GameEnvironment, game_task
from agent_harness.games.models import GameSmokeModel, GuardedModel, RequestGuard, RunStopped
from agent_harness.games.benchmark import GameExperiment, GameRefiner, GameVerifier, make_suite, validate_suites
from agent_harness.models.base import Action, ModelResponse
from agent_harness.refinement.changes import HarnessChange
from agent_harness.storage.state import State


def config():
    return Config.load('configs/games-smoke.yaml')


@pytest.mark.parametrize('line', [(0,1,2),(0,3,6),(0,4,8),(2,4,6)])
def test_winning_lines(line):
    board=['.']*9
    for cell in line:
        board[cell]='X'
    assert winner(board)=='X'


def test_illegal_moves_forfeit_without_board_repair():
    game=TicTacToe(0)
    original=list(game.board)
    for value in (-1,True,9):
        obs=game.step(value)
        assert obs['error']
    assert game.terminal and game.outcome=='illegal_move_forfeit' and game.board==original


def test_exact_opponent_never_loses_against_any_agent_play():
    # Enumerate all agent continuations for both player sides, one fixed opponent seed.
    import copy
    for mark in ('X','O'):
        stack=[TicTacToe(17,opponent='perfect',agent_mark=mark)]
        while stack:
            game=stack.pop()
            if game.terminal:
                assert game.outcome!='win'
                continue
            for cell in game.observe()['legal_cells']:
                child=copy.deepcopy(game)
                child.step(cell)
                stack.append(child)
    assert minimax_value('.'*9,'X')==0


@pytest.mark.parametrize('size',[7,9])
def test_maze_oracle_route_requires_key_and_reaches_goal(size):
    for seed in range(12):
        game=KeyDoor(seed,size=size)
        initial=game.observe()
        route=shortest_path(initial['grid'],initial['position'])
        assert route is not None and len(route)==game.optimal_steps
        for direction in route:
            game.step(direction)
        assert game.terminal and game.outcome=='goal' and game.has_key
        assert game.metrics()['path_efficiency']==1
        assert initial==KeyDoor(seed,size=size).observe()


def test_locked_door_and_time_limit():
    game=KeyDoor(1,max_steps=48)
    door=next((x,y) for y,row in enumerate(game.grid) for x,c in enumerate(row) if c=='D')
    game.position=(door[0]-1,door[1])
    initial=game.position
    assert game.step('right')['error'] and game.position==initial
    game.has_key=True
    assert 'error' not in game.step('right') and game.position==door
    game=KeyDoor(3)
    game.max_steps=1
    game.step(None)
    assert game.outcome=='step_limit'


def test_disjoint_splits_and_duplicate_rejection():
    suites={split:make_suite('tic_tac_toe',split,6) for split in ('adaptation','verification','held_out')}
    validate_suites(suites)
    assert {t.metadata['agent_mark'] for t in suites['held_out']}=={'X','O'}
    assert {t.metadata['opponent'] for t in suites['held_out']}=={'random','tactical','perfect'}
    suites['held_out'][0]=replace(suites['adaptation'][0],split='held_out')
    with pytest.raises(ValueError,match='duplicate'):
        validate_suites(suites)


@pytest.mark.parametrize('game',['tic_tac_toe','key_door','battleship'])
def test_game_uses_existing_orchestrator_and_freezes_state(tmp_path,game):
    state=State()
    before=state.fingerprint()
    task=game_task(game,'held_out',2021)
    env=GameEnvironment(task)
    result=Orchestrator(GameSmokeModel(),config(),state).run(task,env,tmp_path)
    assert env.terminal and result['error'] is None
    assert state.fingerprint()==before
    assert result['model_calls']>0 and result['tool_calls']>0
    if game=='key_door':
        assert result['verified']
    events=[json.loads(s) for s in (tmp_path/'trajectories.jsonl').read_text().splitlines()]
    assert any(e.get('name')==('fire' if game=='battleship' else 'move') for e in events)
    with pytest.raises(PermissionError):
        Orchestrator(GameSmokeModel(),config(),state).run(task,env,tmp_path,learning=True)
    state.close()


def test_invalid_moves_cannot_overwrite_game():
    env=GameEnvironment(game_task('tic_tac_toe','adaptation',2))
    assert env.call_tool('move',{'cell':'0'})['error']
    assert env.call_tool('move',{'surprise':'value'})['error']
    with pytest.raises(PermissionError):
        env.write('rules.txt','I win')


def test_request_cap_and_provider_failure_stop():
    guard=RequestGuard(max_requests=1,max_tokens=10000)
    model=GuardedModel(GameSmokeModel(),guard)
    messages=[{'role':'user','content':'reflect'}]
    model.generate(messages,context={'role':'game_refiner'})
    with pytest.raises(RunStopped,match='cap'):
        model.generate(messages,context={'role':'game_refiner'})
    assert guard.requests==1
    from urllib.error import HTTPError
    class RateLimited:
        def generate(self,*args):
            raise HTTPError('https://example.com',429,'quota',{},None)
    guard=RequestGuard()
    with pytest.raises(RunStopped,match='HTTP 429'):
        guard.generate(RateLimited(),messages,None,{})
    assert guard.requests==1


def test_remote_requires_local_key_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    monkeypatch.setattr('agent_harness.games.benchmark.load_key_from_environment',lambda name:False)
    with pytest.raises(ValueError,match='GEMINI_API_KEY'):
        GameExperiment(Config.load('configs/games-gemini.yaml'),tmp_path/'remote')
    assert not (tmp_path/'remote').exists()


def test_smoke_benchmark_reports_no_manufactured_improvement(tmp_path):
    exp=GameExperiment(config(),tmp_path/'smoke',max_requests=2000,max_tokens=2000000)
    report=exp.run(games=('tic_tac_toe','key_door'),adaptation=2,verification=2,held_out=2,ablate_notes=True)
    assert report['status']=='complete' and not report['is_real_model']
    assert report['accepted_changes']==0
    assert report['rejected_changes']==2
    for result in report['results'].values():
        assert result['paired']['mean_score_delta']==0
        assert result['initial']['mean_game_score']==result['without_notes']['mean_game_score']
    assert report['verification_overhead']['episodes']==8
    assert (tmp_path/'smoke/report.md').exists()


def test_request_exhaustion_is_incomplete_not_a_claimed_result(tmp_path):
    report=GameExperiment(config(),tmp_path/'partial',max_requests=1).run(games=('tic_tac_toe',),adaptation=2,verification=2,held_out=2)
    assert report['status']=='incomplete' and report['request_attempts']==1
    assert not report['results'] and report['stop_reason']


def test_game_gate_rejects_per_scenario_regression_and_promotes_good_note():
    state=State()
    tasks=make_suite('tic_tac_toe','verification',2)
    evidence={f'a{i}':{'task_id':f'a{i}','split':'adaptation','project':f'p{i}'} for i in range(2)}
    def runner(task,state,phase):
        content=' '.join(state.notes())
        value=(0 if task==tasks[0] else 1) if 'regress' in content else 1 if 'help' in content else .5
        return {'task_id':task.task_id,'game_score':value,'verified':value>0,'model_calls':1,'tokens':10,'tool_calls':1}
    verifier=GameVerifier(tasks,runner,config())
    def proposal(content):
        return HarnessChange('UPDATE_PROMPT_NOTE','strategy',None,{'id':'strategy','content':content,
            'source':'test_adaptation','provenance':['a0','a1'],'status':'active','version':1},('a0','a1'),'test strategy')
    rejected=verifier.apply(state,proposal('regress'),evidence)
    assert not rejected['accepted'] and rejected['regressions']==[tasks[0].task_id]
    accepted=verifier.apply(state,proposal('help'),evidence)
    assert accepted['accepted'] and state.version==1
    state.rollback(0)
    assert not state.notes()
    state.close()


def test_refiner_rejects_held_out_evidence(tmp_path):
    with pytest.raises(ValueError,match='adaptation'):
        GameRefiner().propose(State(),[{'task_id':'a','split':'held_out'},{'task_id':'b','split':'adaptation'}],GameSmokeModel(),config(),tmp_path)


def test_real_adapter_game_wire_protocol_uses_configured_model(tmp_path,monkeypatch):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from threading import Thread
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append(payload)
            pending=set()
            for message in payload['messages']:
                if message['role']=='assistant':
                    pending.update(call['id'] for call in message.get('tool_calls',[]))
                elif message['role']=='tool':
                    assert message['tool_call_id'] in pending
                    pending.remove(message['tool_call_id'])
            assert not pending
            tool_results=[json.loads(m['content']) for m in payload['messages'] if m['role']=='tool' or
                          (m['role']=='user' and m['content'].startswith('{"name": "move"'))]
            if tool_results:
                obs=tool_results[-1]['result']
            else:
                initial=json.loads(payload['messages'][0]['content'].split('\n',1)[1])['context']
                obs=json.loads(next(s for s in initial.splitlines() if s.startswith('observation.json: ')).split(': ',1)[1])
            message={'content':None,'tool_calls':[{'function':{'name':'move','arguments':json.dumps({'cell':obs['legal_cells'][0]})}}]}
            raw=json.dumps({'choices':[{'message':message}],'usage':{'prompt_tokens':100,'completion_tokens':20}}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('GAME_TEST_KEY','test-only-not-a-real-key')
    cfg=replace(config(),model=replace(config().model,backend='openai',model='fixed-test-model',
                    base_url=f'http://127.0.0.1:{server.server_port}',api_key_env='GAME_TEST_KEY'))
    try:
        exp=GameExperiment(cfg,tmp_path/'wire',max_requests=120)
        task=game_task('tic_tac_toe','held_out',2000)
        record=exp.run_episode(task,exp.state,'wire_test')
        assert record['error'] is None and record['outcome'] in {'win','draw','loss'}
        assert all(p['model']=='fixed-test-model' and p['tools'][0]['function']['name']=='move' for p in received)
        assert record['input_tokens']==100*record['model_calls']
        exp.state.close()
    finally:
        server.shutdown();server.server_close()


def test_replay_embeds_safe_json_and_plainly_labels_smoke(tmp_path):
    from agent_harness.games.reporting import write_replay
    summary={'model':{'model':'test'},'status':'complete','request_attempts':0,'request_cap':1,
             'interpretation':'handwritten controller','results':{},'limitations':[],'verification_overhead':{}}
    write_replay(tmp_path,summary,[],[{'reason':'</script><script>bad()</script>'}])
    html=(tmp_path/'replay.html').read_text()
    assert '</script><script>bad()' not in html
    import re
    data=json.loads(re.search(r'<script id="evidence" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert data['decisions'][0]['reason']=='</script><script>bad()</script>'
    assert data['summary']['interpretation']=='handwritten controller'
