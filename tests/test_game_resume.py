from dataclasses import replace
import hashlib
import json
from urllib.error import HTTPError
import pytest

from agent_harness.config import Config
from agent_harness.games.benchmark import GameExperiment, make_suite
from agent_harness.games.models import GameSmokeModel
from agent_harness.games.resume import PrefixVerifier, load_checkpoint, resume_pilot, resume_evaluation
from agent_harness.storage.state import State


@pytest.fixture
def checkpoint(tmp_path):
    cfg=Config.load('configs/battleship-gemini.yaml')
    cfg=replace(cfg,model=replace(cfg.model,backend='game_smoke',model='resume-test-no-learning'))
    shared={'fail':False}
    class InterruptedModel(GameSmokeModel):
        def generate(self,*args,**kwargs):
            if shared['fail']:
                raise HTTPError('https://example.com',503,'unavailable',{},None)
            return super().generate(*args,**kwargs)
    source=tmp_path/'source'
    exp=GameExperiment(cfg,source,max_requests=100,max_tokens=500000,model_factory=InterruptedModel)
    run_episode=exp.run_episode
    def interrupt_second_verification(task,state,phase):
        shared['fail']=phase=='candidate_verification' and task.metadata['seed']==1001
        return run_episode(task,state,phase)
    exp.run_episode=interrupt_second_verification
    summary=exp.run(adaptation=2,verification=2,held_out=3)
    assert summary['status']=='incomplete' and '503' in summary['stop_reason']
    return source,cfg,summary


def fingerprint_files(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def test_resume_preserves_source_reuses_prefix_and_does_not_reflect_again(checkpoint,tmp_path):
    source,cfg,saved=checkpoint
    original=fingerprint_files(source)
    calls=[]
    class NoNewReflection(GameSmokeModel):
        def generate(self,messages,tools=None,context=None):
            assert context.get('role')!='game_refiner'
            calls.append(1)
            return super().generate(messages,tools,context)
    result=resume_pilot(cfg,source,tmp_path/'continued',41,0,NoNewReflection)
    assert result['status']=='complete' and result['accepted_changes']==0 and result['rejected_changes']==1
    assert result['results']['battleship']['paired']['mean_score_delta']==0
    assert result['continuation']['reused_baseline_verification_games']==1
    assert result['continuation']['additional_request_attempts']==len(calls)<=30
    assert result['request_attempts']==saved['request_attempts']+len(calls)
    assert original==fingerprint_files(source)
    assert (source/'manifest.json').read_bytes()==(tmp_path/'continued/manifest.json').read_bytes()
    records=[json.loads(s) for s in (tmp_path/'continued/episodes.jsonl').read_text().splitlines()]
    assert sum(bool(r['error']) for r in records)==1  # Original infrastructure failure is retained.


def test_resume_stops_on_another_provider_failure_without_retry(checkpoint,tmp_path):
    source,cfg,saved=checkpoint
    class Unavailable:
        def generate(self,*args,**kwargs):
            raise HTTPError('https://example.com',503,'unavailable',{},None)
    result=resume_pilot(cfg,source,tmp_path/'failed-again',41,0,Unavailable)
    assert result['status']=='incomplete' and '503' in result['stop_reason']
    assert result['request_attempts']==saved['request_attempts']+1
    assert result['continuation']['additional_request_attempts']==1
    assert result['accepted_changes']==result['rejected_changes']==0


def test_resume_rejects_changed_model_or_insufficient_remaining_budget(checkpoint,tmp_path):
    source,cfg,_=checkpoint
    with pytest.raises(ValueError,match='configuration mismatch'):
        resume_pilot(replace(cfg,model=replace(cfg.model,temperature=1)),source,tmp_path/'bad-config',41,0)
    with pytest.raises(ValueError,match='worst case'):
        resume_pilot(cfg,source,tmp_path/'too-small',1,0)
    assert not (tmp_path/'bad-config').exists() and not (tmp_path/'too-small').exists()


def test_resume_validates_saved_model_response_digest(checkpoint):
    source,cfg,_=checkpoint
    events=[json.loads(s) for s in (source/'refinement/trajectories.jsonl').read_text().splitlines()]
    ref=next(e['response_ref'] for e in events if e['event']=='model_call')
    (source/'refinement/objects'/f'{ref}.json').write_text('{}')
    with pytest.raises(ValueError,match='integrity'):
        load_checkpoint(source,cfg)


def test_completed_loss_is_reused_and_candidate_is_evaluated_fresh():
    cfg=Config.load('configs/battleship-gemini.yaml')
    state=State()
    tasks=make_suite('battleship','verification',2)
    cached=[{'task_id':tasks[0].task_id,'game_score':0.2,'outcome':'loss'}]
    calls=[]
    def runner(task,state,phase):
        calls.append(task.task_id)
        return {'task_id':task.task_id,'game_score':0.4}
    verifier=PrefixVerifier(tasks,runner,cfg,cached,state.fingerprint())
    assert verifier.evaluate(state)[0] is cached[0]
    assert calls==[tasks[1].task_id]
    verifier.evaluate(state)
    assert calls==[tasks[1].task_id,tasks[0].task_id,tasks[1].task_id]
    state.close()


@pytest.fixture
def frozen_checkpoint(checkpoint,tmp_path):
    source,cfg,_=checkpoint
    manifest=json.loads((source/'manifest.json').read_text())
    interrupted_id=manifest['battleship']['held_out'][1]['task_id']
    class TimeoutDuringEvaluation(GameSmokeModel):
        def generate(self,messages,tools=None,context=None):
            if json.loads(messages[1]['content'])['task_id']==interrupted_id:
                raise TimeoutError('test provider timeout')
            return super().generate(messages,tools,context)
    frozen=tmp_path/'frozen-checkpoint'
    summary=resume_pilot(cfg,source,frozen,41,0,TimeoutDuringEvaluation)
    assert summary['status']=='incomplete' and summary['rejected_changes']==1
    return frozen,cfg,summary


def test_finish_frozen_evaluation_retains_completed_loss_and_skips_selection(frozen_checkpoint,tmp_path):
    source,cfg,saved=frozen_checkpoint
    original=fingerprint_files(source)
    manifest=json.loads((source/'manifest.json').read_text())
    expected={task['task_id'] for task in manifest['battleship']['held_out'][1:]}
    seen=[]
    class OnlyUnfinishedGames(GameSmokeModel):
        def generate(self,messages,tools=None,context=None):
            assert context.get('role')!='game_refiner'
            task_id=json.loads(messages[1]['content'])['task_id']
            assert task_id in expected
            seen.append(task_id)
            return super().generate(messages,tools,context)
    result=resume_evaluation(cfg,source,tmp_path/'finished',10,0,90,OnlyUnfinishedGames)
    assert result['status']=='complete' and result['rejected_changes']==1
    assert result['results']['battleship']['paired']['mean_score_delta']==0
    assert result['continuation']['reused_completed_held_out_games']==1
    assert result['request_attempts']==saved['request_attempts']+len(seen)
    assert result['model']['request_timeout']==90
    assert original==fingerprint_files(source)
    assert (source/'changes.jsonl').read_bytes()==(tmp_path/'finished/changes.jsonl').read_bytes()


def test_frozen_continuation_stops_on_provider_failure(frozen_checkpoint,tmp_path):
    source,cfg,saved=frozen_checkpoint
    class StillUnavailable:
        def generate(self,*args,**kwargs):
            raise TimeoutError('test provider timeout')
    result=resume_evaluation(cfg,source,tmp_path/'timeout-again',10,0,90,StillUnavailable)
    assert result['status']=='incomplete' and 'TimeoutError' in result['stop_reason']
    assert result['request_attempts']==saved['request_attempts']+1
    assert not result['results']


def test_frozen_continuation_rejects_changed_model_and_insufficient_budget(frozen_checkpoint,tmp_path):
    source,cfg,_=frozen_checkpoint
    with pytest.raises(ValueError,match='configuration'):
        resume_evaluation(replace(cfg,epsilon=0.1),source,tmp_path/'altered',10,0)
    with pytest.raises(ValueError,match='insufficient'):
        resume_evaluation(cfg,source,tmp_path/'undersized',1,0)
