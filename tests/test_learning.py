from dataclasses import replace
import pytest
from agent_harness.config import condition
from agent_harness.core.agent import Orchestrator
from agent_harness.environments.coding import CodingEnvironment
from agent_harness.storage.state import State
from agent_harness.models import MockModel
from agent_harness.refinement.refiner import HarnessRefiner
from agent_harness.refinement.changes import HarnessChange
from agent_harness.verification.verifier import Verifier
from test_core import fixture_task


def experiences(tmp_path, state, cfg):
    records = []
    for task_id in ['a','b']:
        task = fixture_task(task_id=task_id)
        records.append(Orchestrator(MockModel(), cfg, state).run(task,
            CodingEnvironment(task,tmp_path/task_id/'workspace',cfg.limits),tmp_path/task_id/'logs',learning=True))
    return records


def test_verified_skill_learning_and_reuse(tmp_path):
    state, cfg = State(tmp_path/'state.db'), condition(8)
    records = experiences(tmp_path,state,cfg)
    proposals = HarnessRefiner().propose(state,records,cfg)
    assert {'CREATE_MEMORY','CREATE_SKILL','CREATE_AGENT','UPDATE_PROMPT_NOTE'} <= {p.change_type for p in proposals}
    verifier = Verifier([fixture_task('verification','v')], MockModel, cfg)
    evidence = {r['task_id']:r for r in records}
    skill_change = next(p for p in proposals if p.kind=='skill')
    decision = verifier.apply(state,skill_change,evidence)
    assert decision['accepted'] and decision['gain'] > 0
    assert state.entries('skill')[0]['status']=='active' and state.version==1
    held = fixture_task('held_out','h')
    before = state.fingerprint()
    result = Orchestrator(MockModel(),cfg,state).run(held,
        CodingEnvironment(held,tmp_path/'held/workspace',cfg.limits),tmp_path/'held/logs')
    assert result['skill_reuse'] and result['verified']
    assert before == state.fingerprint()
    with pytest.raises(PermissionError):
        Orchestrator(MockModel(),cfg,state).run(held,
            CodingEnvironment(held,tmp_path/'held2/workspace',cfg.limits),tmp_path/'held2/logs',learning=True)


def test_specialist_requires_measured_gain(tmp_path):
    state, cfg = State(), condition(5)
    records = experiences(tmp_path,state,cfg)
    proposal = next(p for p in HarnessRefiner().propose(state,records,cfg) if p.kind=='agent')
    verifier = Verifier([fixture_task('verification','v')],MockModel,cfg)
    decision = verifier.apply(state,proposal,{r['task_id']:r for r in records},gate=False)
    assert decision['accepted']
    assert state.entries('agent')[0]['performance_history'][0]['marginal_contribution'] > 0


def test_rejected_updates_immutable_targets_and_leakage(tmp_path):
    state, cfg = State(), condition(8)
    records = experiences(tmp_path,state,cfg)
    verifier = Verifier([fixture_task('verification','v')],MockModel,cfg)
    evidence = {r['task_id']:r for r in records}
    memory = next(p for p in HarnessRefiner().propose(state,records,cfg) if p.kind=='memory')
    before = state.fingerprint()
    decision = verifier.apply(state,memory,evidence)
    assert not decision['accepted'] # no measured benefit from adding prompt text
    assert state.fingerprint()==before and state.version==0
    evil = replace(memory,change_type='UPDATE_SECURITY')
    assert not verifier.apply(state,evil,evidence)['accepted']
    leaked = replace(memory,evidence=('held',))
    assert not verifier.apply(state,leaked,{'held':{'split':'held_out'}})['accepted']
    with pytest.raises(ValueError):
        Verifier([fixture_task('held_out','h')],MockModel,cfg)


def test_actual_regression_rejected(tmp_path):
    from agent_harness.models.base import Action,ModelResponse
    class PoisonSensitiveModel(MockModel):
        def generate(self,messages,tools=None,context=None):
            if 'poison-pill' in messages[0]['content']:
                return ModelResponse('',(Action('write',{'path':'solution.py','content':'def solve(values):\n    return 0\n'}),),1,1)
            return super().generate(messages,tools,context)
    cfg,state=condition(8),State()
    records=experiences(tmp_path,state,cfg)
    note=next(p for p in HarnessRefiner().propose(state,records,cfg) if p.kind=='note')
    note=replace(note,new_state=note.new_state | {'content':'poison-pill'})
    verifier=Verifier([fixture_task('verification','v')],PoisonSensitiveModel,cfg)
    decision=verifier.apply(state,note,{r['task_id']:r for r in records})
    assert not decision['accepted'] and decision['regressions']==['v']
    assert state.version==0 and not state.entries('note')


@pytest.mark.parametrize('index', range(9))
def test_all_conditions_end_to_end(tmp_path,index):
    cfg,state,task=condition(index),State(),fixture_task()
    result=Orchestrator(MockModel(),cfg,state).run(task,
        CodingEnvironment(task,tmp_path/'workspace',cfg.limits),tmp_path/'logs')
    assert result['verified']
    assert bool(result['subagent_calls']) == (index in (3,4))
