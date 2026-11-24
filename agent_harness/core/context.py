import json
from .budget import BudgetExceeded
from agent_harness.models.base import token_estimate


def call_model(model, messages, tools, context, budget, trajectory):
    allowance = budget.model_allowance(token_estimate(messages) + token_estimate(tools),
                                       context.get('max_output_tokens', 1000))
    budget.reserve('model_calls')
    context = context | {'max_output_tokens': allowance,
                         'timeout': max(0.01, budget.limits.max_execution_time - ( __import__('time').monotonic() - budget.started))}
    prompt_ref = trajectory.objects.put(messages)
    response = model.generate(messages, tools, context)
    trajectory.log('model_call', prompt_ref=prompt_ref, response_ref=trajectory.objects.put({
        'content': response.content, 'actions': [a.__dict__ for a in response.actions],
        'tool_calls': response.tool_calls, 'finish_reason': response.finish_reason}),
        backend=response.backend, input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        latency=response.latency, estimated_cost=response.estimated_cost, role=context.get('role', 'main'))
    budget.usage(response)
    return response


class ContextEngine:
    def __init__(self, model, budget, trajectory):
        self.model, self.budget, self.trajectory = model, budget, trajectory

    def prepare(self, objects, query, mode='retrieved', depth=0):
        limit = self.budget.limits.max_context_chars
        terms = set(query.lower().split())
        ranked = sorted(objects.items(), key=lambda p: (-sum(t in (p[0] + p[1]).lower() for t in terms), p[0]))
        total = sum(len(v) for _, v in ranked)
        references = {name: self.trajectory.objects.put(text) for name, text in ranked}
        if mode == 'recursive' and total > limit:
            result = self._recursive(ranked, query, depth)
        elif mode == 'summarized':
            per = max(1, limit // max(1, len(ranked)))
            result = '\n'.join(f'{name}: {text[:per]}' for name, text in ranked)[:limit]
        else:
            ordered = list(objects.items()) if mode == 'full' else ranked
            result = '\n'.join(f'{name}: {text}' for name, text in ordered)[:limit]
        self.trajectory.log('context_selection', mode=mode, depth=depth, context_refs=references,
                            selected_chars=len(result), discarded_chars=max(0, total-len(result)),
                            aggregation_ref=self.trajectory.objects.put(result))
        return result

    def _recursive(self, objects, query, depth):
        if depth >= self.budget.limits.max_depth:
            raise BudgetExceeded('recursive depth limit')
        limit = self.budget.limits.max_context_chars
        chunks = []
        for name, text in objects:
            chunks.extend((name, text[i:i+limit]) for i in range(0, len(text), limit))
        outputs = []
        child_chars = max(100, min(800, limit//2-32))
        for name, text in chunks:
            self.budget.reserve('subagent_calls')
            self.budget.recursive_calls += 1
            messages = [{'role': 'system', 'content': f'Select relevant evidence from this subset only. Keep the result within {child_chars} characters.'},
                        {'role': 'user', 'content': json.dumps({'query': query, 'source': name, 'subset': text})}]
            response = call_model(self.model, messages, None, {'role': 'context', 'selected': text},
                                  self.budget, self.trajectory)
            outputs.append(response.content[:child_chars])
            self.trajectory.log('recursive_child', depth=depth+1, source=name, selected_ref=self.trajectory.objects.put(text))
        combined = '\n'.join(outputs)
        if len(combined) > limit:
            return self.prepare({'child_outputs': combined}, query, 'recursive', depth+1)
        return combined
