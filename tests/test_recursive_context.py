import pytest
from agent_harness.config import Limits
from agent_harness.core.budget import Budget,BudgetExceeded
from agent_harness.core.context import ContextEngine
from agent_harness.core.trajectory import Trajectory
from agent_harness.core.runtime import Runtime
from agent_harness.models import MockModel
from agent_harness.config import condition
from agent_harness.environments.coding import CodingEnvironment
from test_core import fixture_task


def test_recursive_partition_and_aggregation(tmp_path):
    limits=Limits(max_context_chars=1200,max_subagents=6,max_model_calls=8,max_depth=3)
    budget=Budget(limits)
    trajectory=Trajectory(tmp_path,'large',0,{})
    engine=ContextEngine(MockModel(),budget,trajectory)
    result=engine.prepare({'first.py':'alpha '*300,'second.py':'beta '*120},'beta','recursive')
    assert len(result)<=1200
    assert budget.recursive_calls > 0 and budget.model_calls==budget.recursive_calls
    children=[e for e in trajectory.events if e['event']=='recursive_child']
    assert all(len(trajectory.objects.get(e['selected_ref']))<=1200 for e in children)
    assert all(e['depth']<=limits.max_depth for e in children)
    assert any(e['event']=='context_selection' and e['aggregation_ref'] for e in trajectory.events)


def test_depth_and_external_reference_access(tmp_path):
    limits=Limits(max_context_chars=300,max_depth=1,max_model_calls=10,max_subagents=10)
    engine=ContextEngine(MockModel(),Budget(limits),Trajectory(tmp_path/'recursive','large',0,{}))
    with pytest.raises(BudgetExceeded,match='depth'):
        engine.prepare({'long':'x'*900},'x','recursive')
    cfg=condition(1)
    trajectory=Trajectory(tmp_path/'runtime','a',0,{})
    runtime=Runtime(CodingEnvironment(fixture_task(),tmp_path/'workspace',cfg.limits),Budget(cfg.limits),trajectory,cfg)
    ref=runtime.objects.put('abcdef')
    runtime.allowed_refs={ref}
    assert runtime.call('retrieve',{'ref':ref,'start':2,'end':4})['result']=='cd'
    denied=runtime.call('retrieve',{'ref':'a'*64,'start':0,'end':2})
    assert 'PermissionError' in denied['error']
