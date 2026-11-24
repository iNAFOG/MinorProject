import pytest
from agent_harness.storage.state import State


def add_memory(state, **kwargs):
    return state.memory.create('Run targeted tests', type='semantic', scope='project',
                              source='verified:a', task_id='a', project='p', provenance=['a'], **kwargs)


def test_memory_restart_scope_and_provenance(tmp_path):
    path = tmp_path/'state.db'
    state = State(path)
    entry = add_memory(state)
    state.close()
    state = State(path)
    assert state.memory.retrieve('targeted', 'p', 'b')[0]['source']=='verified:a'
    assert not state.memory.retrieve('targeted', 'other', 'b')
    assert state.get('memory', entry['id'])['confidence']==0.5
    state.memory.update(entry['id'], content='Run focused tests')
    assert state.get('memory', entry['id'])['version']==2
    state.memory.delete(entry['id'])
    assert not state.entries('memory')


def test_confidence_contradiction_and_task_scope():
    state = State()
    entry = add_memory(state, claim_key='test-command')
    result = state.memory.feedback(entry['id'], 0)
    assert result['confidence']==pytest.approx(0.4)
    state.memory.create('Do not run targeted tests', type='semantic', scope='project', source='verified:b',
                        task_id='b', project='p', provenance=['b'], claim_key='test-command')
    assert not state.memory.retrieve('targeted', 'p', 'b')
    state.memory.create('task secret', type='working', scope='task', source='a', task_id='a', provenance=['a'])
    assert state.memory.retrieve('secret', 'p', 'a')
    assert not state.memory.retrieve('secret', 'p', 'b')
    with pytest.raises(ValueError):
        state.memory.create('global shortcut', type='semantic', scope='global', source='a', task_id='a', provenance=['a'])


def test_skill_lifecycle_and_agent_evidence():
    state = State()
    skill = state.skills.candidate('reader','Read target',[{'tool':'read','arguments':{'path':'$path'}}],['a','b'])
    with pytest.raises(ValueError):
        state.skills.activate(skill['id'], {'success':True})
    state.skills.activate(skill['id'], {'success':True,'independent':True,'fixture_digest':'digest'})
    for _ in range(3):
        state.skills.feedback(skill['id'], False)
    assert state.get('skill', skill['id'])['status']=='degraded'
    assert state.skills.prune('1900')==[skill['id']]
    with pytest.raises(ValueError):
        state.agents.candidate('reader','debug','inspect',['a'])
    agent = state.agents.candidate('reader','debug','inspect',['a','b'])
    with pytest.raises(ValueError):
        state.agents.activate(agent['id'], {'independent':True,'marginal_contribution':0})
    state.agents.activate(agent['id'], {'independent':True,'marginal_contribution':0.1})
    assert state.agents.revise(agent['id'], 'inspect carefully')['status']=='candidate'
    assert state.agents.retire(agent['id'])['status']=='retired'


def test_snapshot_rollback_and_audit(tmp_path):
    state = State(tmp_path/'state.db')
    candidate = state.clone()
    add_memory(candidate)
    assert state.promote(candidate, [{'kind':'memory'}], {'accepted':True})==1
    assert state.entries('memory')
    state.rollback(0)
    assert not state.entries('memory')
    assert len(state.history())==2
    assert [row['event'] for row in state.audit_rows()]==['promote','rollback']
    assert (tmp_path/'harness/v0001/state.json').exists()
