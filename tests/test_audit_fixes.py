"""Regression tests for defects found in the November 2026 repository audit."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import pytest
from agent_harness.config import Limits, ModelConfig, condition
from agent_harness.core.agent import Orchestrator
from agent_harness.environments.base import Task
from agent_harness.environments.coding import CodingEnvironment
from agent_harness.environments.safe_python import SafePython, UnsafeCode
from agent_harness.evaluation import benchmark
from agent_harness.evaluation.benchmark import Experiment
from agent_harness.models.base import Action, ModelResponse
from agent_harness.models.mock import MockModel
from agent_harness.models.openai_compatible import INVALID_ARGUMENTS, OpenAICompatibleModel, parse_arguments
from agent_harness.storage.state import State

SUM = 'def solve(values):\n    return sum(values)\n'


def task_with(source, cases, files=None):
    return Task('t', 'Return the sum of values', 'adaptation', 'p', files or {'solution.py': source},
                ({'kind': 'function', 'path': 'solution.py', 'function': 'solve', 'cases': cases},))


def test_each_check_case_gets_its_own_step_budget(tmp_path):
    loop = 'def solve(values):\n    total = 0\n    for v in values:\n        total = total + v\n    return total\n'
    cases = [{'args': [list(range(200))], 'expected': sum(range(200))} for _ in range(12)]
    assert CodingEnvironment(task_with(loop, cases), tmp_path, Limits()).verify().success
    # A single runaway case is still stopped.
    with pytest.raises(UnsafeCode):
        SafePython('def solve(values):\n    for i in range(9999):\n        for j in range(9999):\n            pass\n').invoke('solve', [[]])


def test_verification_reads_whole_file_beyond_agent_read_limit(tmp_path):
    padding = '\n'.join(f'def helper{i}(values):\n    return {i}\n' for i in range(700))
    source = padding + '\n' + SUM
    env = CodingEnvironment(task_with(source, [{'args': [[1, 2]], 'expected': 3}]), tmp_path, Limits())
    assert len(source) > Limits().max_output_chars
    assert len(env.read('solution.py')) == Limits().max_output_chars  # agent view stays bounded
    assert env.verify().success


@pytest.mark.parametrize('body', ["return b'ab' * 200000000", "return 'x' * 200000000",
                                  "return '%0300000000d' % 1", "return b'%0300000000d' % 1"])
def test_sandbox_blocks_large_allocations_before_they_happen(body):
    with pytest.raises(UnsafeCode):
        SafePython(f'def solve(values):\n    {body}\n').invoke('solve', [[]])


def test_numeric_modulo_still_supported():
    assert SafePython('def solve(values):\n    return values[0] % 3\n').invoke('solve', [[10]]) == 1


@pytest.mark.parametrize('raw,expected', [(None, {}), ('', {}), ('  ', {}), ('{"a": 1}', {'a': 1}),
                                          ({'a': 1}, {'a': 1}), ('{bad', {INVALID_ARGUMENTS: '{bad'}),
                                          ('[1, 2]', {INVALID_ARGUMENTS: '[1, 2]'})])
def test_tool_argument_parsing_is_total(raw, expected):
    assert parse_arguments(raw) == expected


def test_adapter_tolerates_null_fields_from_servers():
    bodies = [{'choices': [{'message': {'content': 'done', 'tool_calls': None}, 'finish_reason': 'stop'}], 'usage': None},
              {'choices': [{'message': {'content': None, 'tool_calls': [
                  {'id': 'a', 'type': 'function', 'function': {'name': 'verify', 'arguments': ''}}]}}],
               'usage': {'prompt_tokens': None, 'completion_tokens': 3}}]
    served = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            raw = json.dumps(bodies[len(served)]).encode()
            served.append(1)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = OpenAICompatibleModel(ModelConfig(backend='local', base_url=f'http://127.0.0.1:{server.server_port}/v1'))
        first = model.generate([{'role': 'user', 'content': 'x'}])
        second = model.generate([{'role': 'user', 'content': 'x'}])
    finally:
        server.shutdown()
    assert first.actions == () and first.content == 'done' and first.input_tokens > 0
    assert second.actions == (Action('verify', {}),) and second.output_tokens == 3 and second.input_tokens > 0


def test_unknown_tool_is_an_observation_not_an_episode_crash(tmp_path):
    task = task_with('def solve(values):\n    return -1\n', [{'args': [[1, 2]], 'expected': 3}])
    model = MockModel(responses=[ModelResponse('', (Action('read_file', {'path': 'solution.py'}),)),
                                 ModelResponse('', (Action('write', {'path': 'solution.py', 'content': SUM}),)),
                                 ModelResponse('done')])
    env = CodingEnvironment(task, tmp_path/'w', Limits())
    record = Orchestrator(model, condition(1)).run(task, env, tmp_path/'o')
    assert record['verified'] and record['error'] is None
    assert record['tool_calls'] == 2  # the rejected request still consumed budget
    errors = [e for e in record['events'] if e['event'] == 'error']
    assert errors and 'tool disabled: read_file' in errors[0]['error']


def test_repeated_runs_never_reuse_a_workspace(tmp_path, monkeypatch):
    base = Task.load('benchmarks/example_task.json')
    task = replace(base, files={'README.md': 'Implement solve in solution.py'})  # target is a new file
    scripts = [[ModelResponse('', (Action('write', {'path': 'solution.py', 'content': SUM}),)), ModelResponse('done')],
               [ModelResponse('I give up')]]
    state = State(tmp_path/'s.db')
    outcomes = []
    try:
        for script in scripts:
            monkeypatch.setattr(benchmark, 'build_model', lambda cfg, s=script: MockModel(responses=list(s)))
            outcomes.append(Experiment(condition(1), state, tmp_path/'out', []).run_task(task)['verified'])
    finally:
        state.close()
    assert outcomes == [True, False]
    assert len(list((tmp_path/'out'/'workspaces').iterdir())) == 2
