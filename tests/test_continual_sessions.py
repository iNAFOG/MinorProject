import pytest
from agent_harness.config import condition
from agent_harness.storage.state import State
from agent_harness.evaluation.benchmark import Experiment
from test_core import fixture_task


def test_learning_combines_distinct_tasks_across_restarts(tmp_path):
    path=tmp_path/'state.db'
    cfg=condition(8)
    verification=[fixture_task('verification','v')]
    state=State(path)
    Experiment(cfg,state,tmp_path/'first',verification).run_task(fixture_task(task_id='a'),learning=True,phase='adaptation')
    assert not state.entries('skill')
    state.close()
    state=State(path)
    Experiment(cfg,state,tmp_path/'second',verification).run_task(fixture_task(task_id='b'),learning=True,phase='adaptation')
    assert state.entries('skill')[0]['status']=='active'
    assert [r['task_id'] for r in state.experiences()]==['a','b']
    held=fixture_task('held_out','h')
    with pytest.raises(PermissionError):
        state.record_experience({'task_id':held.task_id,'split':held.split})
    frozen=state.fingerprint()
    experience_before=state.experiences()
    result=Experiment(cfg,state,tmp_path/'held',verification).run_task(held)
    assert result['skill_reuse'] and result['verified']
    assert frozen==state.fingerprint() and state.experiences()==experience_before
