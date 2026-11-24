class AgentRegistry:
    def __init__(self, state):
        self.state = state

    def candidate(self, name, role, instructions, provenance, family='coding'):
        if len(set(provenance)) < 2:
            raise ValueError('dynamic specialists require repeated independent task evidence')
        return self.state.put('agent', dict(name=name, role=role, instructions=instructions,
            capabilities=[family], tools=['read', 'search'], provenance=list(provenance),
            source='repeated_verified_trajectories', created_from_task=provenance[-1],
            performance_history=[], status='candidate'))

    def activate(self, entry_id, evaluation):
        if not evaluation.get('independent') or evaluation.get('marginal_contribution', 0) <= 0:
            raise ValueError('specialist requires measured positive marginal contribution')
        entry = self.state.get('agent', entry_id)
        entry['performance_history'].append(evaluation)
        entry.update(status='active', version=entry['version']+1)
        return self.state.put('agent', entry)

    def revise(self, entry_id, instructions):
        entry = self.state.get('agent', entry_id)
        entry.update(instructions=instructions, status='candidate', version=entry['version']+1)
        return self.state.put('agent', entry)

    def retire(self, entry_id):
        entry = self.state.get('agent', entry_id)
        entry.update(status='retired', version=entry['version']+1)
        return self.state.put('agent', entry)

    @staticmethod
    def fixed():
        return {'id': 'fixed-repository-reader', 'name': 'Repository reader',
                'role': 'repository_debugger', 'instructions': 'Inspect the target source and identify the smallest useful change.',
                'capabilities': ['coding'], 'tools': ['read'], 'status':'active', 'fixed':True}
