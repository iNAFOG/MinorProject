from agent_harness.core.trajectory import now


class SkillRegistry:
    def __init__(self, state):
        self.state = state

    def candidate(self, name, description, implementation, provenance, family='coding', project=None):
        from .validator import validate_structure
        validate_structure(implementation)
        return self.state.put('skill', dict(name=name, description=description,
            trigger_conditions=[family], input_schema={'path': 'string'},
            implementation=implementation, validation_procedure='paired fixture replay',
            provenance=list(provenance), source='verified_trajectory',
            creation_time=now(), last_used=None, usage_count=0, success_count=0,
            failure_count=0, status='candidate', project=project))

    def activate(self, entry_id, validation):
        if not validation.get('success') or not validation.get('independent') or not validation.get('fixture_digest'):
            raise ValueError('skill activation requires independent replay validation')
        entry = self.state.get('skill', entry_id)
        entry.update(status='active', validation=validation, version=entry['version']+1)
        return self.state.put('skill', entry)

    def available(self, family, project):
        return [e for e in self.state.entries('skill') if e['status']=='active'
                and family in e['trigger_conditions'] and e.get('project') in (None, project)]

    def execute(self, entry_id, inputs, runtime):
        from .executor import execute
        entry = self.state.get('skill', entry_id)
        if entry['status'] != 'active':
            raise PermissionError('unvalidated/degraded skill cannot execute')
        task = runtime.environment.task
        if task.family not in entry['trigger_conditions'] or entry.get('project') not in (None, task.project):
            raise PermissionError('skill outside its validated scope')
        return execute(entry, inputs, runtime)

    def feedback(self, entry_id, success):
        entry = self.state.get('skill', entry_id)
        entry['usage_count'] += 1
        entry['success_count' if success else 'failure_count'] += 1
        entry['last_used'] = now()
        if entry['failure_count'] >= 3 and entry['failure_count'] > entry['success_count']:
            entry['status'] = 'degraded'
        entry['version'] += 1
        return self.state.put('skill', entry)

    def prune(self, before):
        removed = []
        for entry in self.state.entries('skill'):
            if entry['status']=='degraded' or not entry['usage_count'] and entry['creation_time'] < before:
                self.state.delete('skill', entry['id'])
                removed.append(entry['id'])
        return removed
