from .validator import validate_structure


def execute(skill, inputs, runtime):
    validate_structure(skill['implementation'])
    if set(inputs) != {'path'} or not isinstance(inputs['path'], str):
        raise ValueError('skill requires a path input')
    runtime.environment.path(inputs['path'])
    results = []
    for step in skill['implementation']:
        args = {key: inputs['path'] if value == '$path' else value
                for key, value in step['arguments'].items()}
        row = runtime.call(step['tool'], args)
        results.append(row)
        if row.get('error'):
            raise RuntimeError(row['error'])
    return results
