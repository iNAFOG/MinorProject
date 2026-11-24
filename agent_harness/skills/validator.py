import hashlib
import json


def validate_structure(implementation):
    if not isinstance(implementation, list) or not 1 <= len(implementation) <= 8:
        raise ValueError('skills require 1–8 bounded steps')
    allowed = {'read': {'path'}, 'search': {'query'}, 'verify': set()}
    for step in implementation:
        if set(step) != {'tool', 'arguments'} or step['tool'] not in allowed:
            raise ValueError('skill DSL allows read/search/verify only')
        if set(step['arguments']) != allowed[step['tool']]:
            raise ValueError('invalid skill arguments')
        for value in step['arguments'].values():
            if not isinstance(value, str) or len(value) > 500:
                raise ValueError('invalid skill argument value')
            if '$' in value and value != '$path':
                raise ValueError('unknown skill input')


class SkillValidator:
    def validate(self, skill, runtimes):
        from .executor import execute
        validate_structure(skill['implementation'])
        if not runtimes:
            raise ValueError('no validation fixtures')
        rows, success = [], True
        for runtime in runtimes:
            inputs = {'path': runtime.environment.task.target}
            try:
                result = execute(skill, inputs, runtime)
                ok = all(not r.get('error') for r in result)
            except Exception as exc:
                ok = False
                result = str(exc)
            success &= ok
            rows.append({'task_id': runtime.environment.task.task_id, 'result': result})
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        return {'success': bool(success), 'independent': True, 'fixture_digest': digest,
                'fixtures': [r['task_id'] for r in rows],
                'overhead':{key:sum(r.budget.metrics().get(key,0) for r in runtimes)
                            for key in ('model_calls','tokens','tool_calls','estimated_cost','latency')}}
