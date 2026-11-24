"""Deterministic tool-policy fixture. It is not an LLM or a capability benchmark."""
import json
import time
from .base import Action, BaseModel, ModelResponse, token_estimate


class MockModel(BaseModel):
    def __init__(self, config=None, responses=None):
        self.config = config
        self.responses = list(responses) if responses is not None else None

    def generate(self, messages, tools=None, context=None):
        started = time.monotonic()
        ctx = context or {}
        if self.responses is not None:
            if not self.responses:
                raise RuntimeError('mock response script exhausted')
            return self.responses.pop(0)
        if ctx.get('role') == 'context':
            content, actions = str(ctx.get('selected', ''))[:800], ()
        elif ctx.get('role') == 'specialist':
            content, actions = 'Inspect the target source before changing it, then verify.', ()
        else:
            task = json.loads(messages[1]['content'])
            observations = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
            names = [o['name'] for o in observations if not o.get('error')]
            available = {t['name'] for t in tools or []}
            target = task.get('target', 'solution.py')
            if not tools:
                content, actions = self.solution(task), ()
            elif 'write' in names:
                if 'verify' in available and 'verify' not in names:
                    content, actions = '', (Action('verify'),)
                else:
                    content, actions = 'Completed the source change.', ()
            elif 'skill' not in names and 'skill' in available and ctx.get('skills'):
                content, actions = '', (Action('skill', {'skill_id': ctx['skills'][0]['id'], 'inputs': {'path': target}}),)
            elif ctx.get('specialist_inspected'):
                content, actions = '', (Action('write', {'path': target, 'content': self.solution(task)}),)
            elif 'search' not in names and 'skill' not in names:
                content, actions = '', (Action('search', {'query': target}),)
            elif 'read' not in names and 'skill' not in names:
                content, actions = '', (Action('read', {'path': target}),)
            else:
                content, actions = '', (Action('write', {'path': target, 'content': self.solution(task)}),)
        out = token_estimate({'content': content, 'actions': [a.__dict__ for a in actions]})
        if out > ctx.get('max_output_tokens', 100000):
            raise RuntimeError('mock output exceeds allowance')
        return ModelResponse(content, actions, token_estimate(messages)+token_estimate(tools), out,
                             time.monotonic() - started, 0, 'mock:fixture-policy-v1')

    @staticmethod
    def solution(task):
        goal = task.get('goal', '').lower()
        function = task.get('function', 'solve')
        # Only these explicitly documented fixture families are supported.
        if 'sum' in goal:
            return f'def {function}(values):\n    return sum(values)\n'
        if 'maximum' in goal:
            return f'def {function}(values):\n    return max(values) if values else 0\n'
        if 'count' in goal:
            return f'def {function}(values):\n    return len(values)\n'
        return 'Unable to solve this fixture family.'
