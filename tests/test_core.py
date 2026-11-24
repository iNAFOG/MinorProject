from pathlib import Path
import json
import pytest
from agent_harness.config import condition, Limits
from agent_harness.models import MockModel
from agent_harness.core.agent import Orchestrator
from agent_harness.core.budget import Budget, BudgetExceeded
from agent_harness.core.context import ContextEngine
from agent_harness.core.trajectory import Trajectory
from agent_harness.environments.base import Task
from agent_harness.environments.coding import CodingEnvironment
from agent_harness.environments.safe_python import SafePython, UnsafeCode


def fixture_task(split='adaptation', task_id='a'):
    return Task(task_id, 'Return the sum of values', split, 'project-a',
                {'solution.py': 'def solve(values):\n    return 0\n'},
                ({'kind':'function','path':'solution.py','function':'solve',
                  'cases':[{'args':[[1,2,3]],'expected':6}, {'args':[[]],'expected':0}]},))


@pytest.mark.parametrize('index', [0, 1])
def test_core_end_to_end(tmp_path, index):
    cfg = condition(index)
    task = fixture_task()
    env = CodingEnvironment(task, tmp_path/'workspace', cfg.limits)
    result = Orchestrator(MockModel(), cfg).run(task, env, tmp_path/'logs')
    assert result['verified'] and result['success']
    assert result['model_calls'] > 0 and result['tokens'] > 0
    assert bool(result['tool_calls']) == bool(index)
    events = [json.loads(line) for line in (tmp_path/'logs/trajectories.jsonl').read_text().splitlines()]
    assert any(e['event']=='model_call' and e['prompt_ref'] for e in events)
    assert events[-1]['event'] == 'outcome'


def test_runtime_boundaries(tmp_path):
    env = CodingEnvironment(fixture_task(), tmp_path/'workspace', Limits())
    for name in ['../config.json', '/etc/passwd', 'C:/secret', '.harness/state.db']:
        with pytest.raises(PermissionError):
            env.read(name)
    with pytest.raises(PermissionError):
        env.write('tests.py', 'pass')
    with pytest.raises((RuntimeError, ValueError)):
        env.execute('python -c dangerous')


@pytest.mark.parametrize('source', ['import os', 'def solve(x):\n return open(x)',
                                      'def solve(x):\n return x.__class__',
                                      'def solve(x):\n while True: pass'])
def test_ast_rejects_unsafe_code(source):
    with pytest.raises((UnsafeCode, KeyError)):
        engine = SafePython(source)
        engine.call('solve', ['a'])


def test_budget_and_recursion(tmp_path):
    budget = Budget(Limits(max_model_calls=1, max_subagents=1, max_context_chars=20))
    trajectory = Trajectory(tmp_path, 'a', 0, {})
    engine = ContextEngine(MockModel(), budget, trajectory)
    with pytest.raises(BudgetExceeded):
        engine.prepare({'large': 'a'*100}, 'a', 'recursive')
    assert budget.model_calls <= 1 and budget.subagent_calls <= 1


def test_compaction_preserves_complete_native_tool_exchanges(tmp_path):
    from dataclasses import replace
    class PairCheckingMock(MockModel):
        def generate(self,messages,tools=None,context=None):
            pending=set()
            for message in messages:
                if message['role']=='assistant':
                    assert not pending
                    pending.update(call['id'] for call in message.get('tool_calls',[]))
                elif message['role']=='tool':
                    assert message['tool_call_id'] in pending
                    pending.remove(message['tool_call_id'])
            assert not pending
            return super().generate(messages,tools,context)
    cfg=replace(condition(1),limits=Limits(max_context_chars=250,max_tokens=50000))
    result=Orchestrator(PairCheckingMock(),cfg).run(fixture_task(),
            CodingEnvironment(fixture_task(),tmp_path/'workspace',cfg.limits),tmp_path/'logs')
    assert result['verified'] and result['error'] is None
    assert any(e['event']=='context_compaction' for e in result['events'])
