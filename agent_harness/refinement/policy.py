from agent_harness.storage.state import KINDS
from agent_harness.skills.validator import validate_structure


TYPES = {f'{op}_{kind.upper()}' for op in ('CREATE','UPDATE','DELETE') for kind in ('memory','skill','agent')}
TYPES.add('UPDATE_PROMPT_NOTE')
FIELDS = {
    'memory': {'id','content','type','scope','source','task_id','project','confidence','tags','provenance',
               'claim_key','usage_count','success_count','failure_count','status','version','timestamp'},
    'skill': {'id','name','description','trigger_conditions','input_schema','implementation','validation_procedure',
              'provenance','source','creation_time','last_used','usage_count','success_count','failure_count',
              'status','project','version','timestamp','validation'},
    'agent': {'id','name','role','instructions','capabilities','tools','provenance','source','created_from_task',
              'performance_history','status','version','timestamp'},
    'note': {'id','content','source','provenance','status','version','timestamp'},
}


def validate_change(change, state, evidence, config):
    if change.change_type not in TYPES or change.kind not in KINDS:
        raise PermissionError('change cannot target security, evaluation, model, audit or code')
    if not change.evidence or not change.reason or not 0 <= change.confidence <= 1:
        raise ValueError('change needs evidence/reason/confidence')
    records = [evidence.get(task_id) for task_id in change.evidence]
    if any(r is None or r['split']!='adaptation' for r in records):
        raise PermissionError('persistence evidence must come only from adaptation')
    if change.kind == 'memory' and not config.memory or change.kind == 'skill' and not config.skills:
        raise PermissionError('change type disabled by experiment')
    if change.kind == 'agent' and not config.dynamic_agents or change.kind == 'note' and not config.self_refinement:
        raise PermissionError('change type disabled by experiment')
    exists = any(e['id']==change.target_id for e in state.entries(change.kind))
    if change.old_state is not None:
        if not exists or state.get(change.kind, change.target_id) != change.old_state:
            raise ValueError('stale change: old state does not match')
    elif exists:
        raise ValueError('create target already exists')
    if change.change_type.startswith('DELETE'):
        if not exists or change.new_state is not None:
            raise ValueError('invalid delete')
        return
    new = change.new_state
    if not new or set(new)-FIELDS[change.kind] or new.get('id') != change.target_id:
        raise ValueError('invalid external state schema')
    if not new.get('source') or set(new.get('provenance', [])) != set(change.evidence):
        raise ValueError('provenance must match verified evidence')
    if len(str(new)) > 12000:
        raise ValueError('change too large')
    if change.kind in {'skill','agent','note'} and len(set(change.evidence)) < 2:
        raise ValueError('reusable changes require repeated task evidence')
    if change.kind == 'memory':
        if new.get('scope') not in {'task','project','global'} or new.get('type') not in {'working','episodic','semantic','procedural'}:
            raise ValueError('invalid memory scope/type')
        if not 0 <= new.get('confidence', -1) <= 1 or not new.get('content'):
            raise ValueError('invalid memory confidence/content')
        if new['scope']=='project' and (not new.get('project') or any(r['project']!=new['project'] for r in records)):
            raise ValueError('project memory evidence must match owner')
        if new['scope']=='task' and new.get('task_id') not in change.evidence:
            raise ValueError('task memory owner missing')
        if new['scope']=='global' and len({r['project'] for r in records}) < 2:
            raise ValueError('global memory requires cross-project evidence')
        if change.old_state and any(new.get(k)!=change.old_state.get(k) for k in ('scope','project','task_id')):
            raise ValueError('memory ownership cannot be rewritten')
    elif change.kind == 'skill':
        validate_structure(new.get('implementation'))
        if new.get('status') != 'candidate':
            raise ValueError('skill proposals must enter candidate status')
        if new.get('input_schema') != {'path':'string'} or not new.get('trigger_conditions'):
            raise ValueError('invalid skill trigger/input schema')
    elif change.kind == 'agent':
        if new.get('status') != 'candidate' or set(new.get('tools', []))-{'read','search'}:
            raise ValueError('agent proposals must be bounded candidates')
        if not new.get('instructions') or not new.get('capabilities'):
            raise ValueError('agent definition incomplete')
    elif not new.get('content') or new.get('status') != 'active':
        raise ValueError('strategy note incomplete')
