import re


class SQLiteMemory:
    TYPES = {'working', 'episodic', 'semantic', 'procedural'}

    def __init__(self, state):
        self.state = state

    def create(self, content, *, type, scope, source, task_id, provenance,
               confidence=0.5, tags=(), project=None, claim_key=None, status='active'):
        if type not in self.TYPES or scope not in {'task', 'project', 'global'}:
            raise ValueError('invalid memory type/scope')
        if scope == 'project' and not project or scope == 'task' and not task_id:
            raise ValueError('scope requires its owner')
        if not 0 <= confidence <= 1:
            raise ValueError('confidence must be in [0,1]')
        if scope == 'global' and len(set(provenance)) < 2:
            raise ValueError('global knowledge needs independent task evidence')
        if claim_key:
            for old in self.state.entries('memory'):
                if (old.get('claim_key'),old['scope'],old.get('project'),old['task_id'] if scope=='task' else None) == (
                        claim_key,scope,project,task_id if scope=='task' else None) and old['content'] != content:
                    old['confidence'] *= 0.5
                    old['status'] = 'needs_reverification'
                    old['version'] += 1
                    self.state.put('memory', old)
                    status = 'needs_reverification'
                    confidence *= 0.5
        return self.state.put('memory', dict(content=content, type=type, scope=scope,
            source=source, task_id=task_id, project=project, confidence=confidence,
            tags=list(tags), provenance=list(provenance), claim_key=claim_key,
            usage_count=0, success_count=0, failure_count=0, status=status))

    def retrieve(self, query, project, task_id, limit=5):
        terms = set(re.findall(r'\w+', query.lower()))
        scored = []
        for entry in self.state.entries('memory'):
            if entry['status'] != 'active' or entry['confidence'] < 0.3:
                continue
            if entry['scope'] == 'task' and entry['task_id'] != task_id:
                continue
            if entry['scope'] == 'project' and entry['project'] != project:
                continue
            words = set(re.findall(r'\w+', (entry['content'] + ' ' + ' '.join(entry['tags'])).lower()))
            score = len(terms & words) * entry['confidence']
            if score:
                scored.append((score, entry))
        return [entry for _, entry in sorted(scored, key=lambda p: (-p[0], p[1]['id']))[:limit]]

    def feedback(self, entry_id, evidence, decay=0.8):
        if not 0 <= evidence <= 1 or not 0 <= decay <= 1:
            raise ValueError('invalid confidence update')
        entry = self.state.get('memory', entry_id)
        entry['confidence'] = decay*entry['confidence'] + (1-decay)*evidence
        entry['usage_count'] += 1
        entry['success_count' if evidence >= 0.5 else 'failure_count'] += 1
        entry['version'] += 1
        if entry['failure_count'] >= 3:
            entry['status'] = 'needs_reverification'
        return self.state.put('memory', entry)

    def update(self, entry_id, **updates):
        entry = self.state.get('memory', entry_id)
        forbidden = {'id', 'scope', 'project', 'task_id', 'source', 'provenance'}
        if forbidden & updates.keys():
            raise ValueError('memory provenance/ownership is immutable')
        entry.update(updates)
        entry['version'] += 1
        return self.state.put('memory', entry)

    def delete(self, entry_id):
        self.state.delete('memory', entry_id)
