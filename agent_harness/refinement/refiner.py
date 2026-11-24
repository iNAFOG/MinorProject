"""Trajectory-derived procedure induction; no stored task answers or source patches."""
import hashlib
from agent_harness.core.trajectory import now
from .changes import HarnessChange


def identifier(text):
    return hashlib.sha256(text.encode()).hexdigest()[:20]


class HarnessRefiner:
    def model_proposals(self, state, records, config, model, trajectory, budget):
        """Optional fixed-model proposal generation, still subject to controller policy."""
        import json
        from agent_harness.core.context import call_model
        from .policy import FIELDS,TYPES
        adaptation=[r for r in records if r['split']=='adaptation']
        summaries=[]
        for record in adaptation[-8:]:
            summaries.append({k:record[k] for k in ('task_id','project','family','verified','error')} | {
                'operations':[{'event':e['event'],**{k:e[k] for k in ('name','error','skill_id') if k in e}}
                              for e in record['events'] if e['event'] in {'tool_call','error','skill_invoked'}][-20:]})
        prompt={
            'instruction':'Suggest bounded external-harness improvements from adaptation evidence. Return JSON {"changes": [...]} only. '
                'Never store task solutions, expected outputs, source patches, or security/evaluation edits. '
                'Each item has change_type, target_id, new_state, evidence (task IDs), reason, confidence. '
                'Skills are read/search/verify workflows (tool, arguments); parameter $path only. '
                'New agents and skills must have status candidate. Notes use status active. '
                'Memory type is working/episodic/semantic/procedural and scope task/project/global. '
                'Reuse needs two independent adaptation task IDs; global facts need different projects. '
                'A delete has new_state null. Return no changes if the evidence is insufficient.',
            'allowed_change_types':sorted(TYPES),'allowed_entry_fields':{k:sorted(v) for k,v in FIELDS.items()},
            'state':state.export(),'adaptation_experience':summaries}
        messages=[{'role':'system','content':'You are the harness refiner. Model weights, code, verification and experiment configuration are fixed.'},
                  {'role':'user','content':json.dumps(prompt)}]
        if len(messages[1]['content']) > config.limits.max_context_chars*2:
            # Select recent experience and applicable state instead of truncating structured JSON.
            prompt['adaptation_experience']=summaries[-3:]
            prompt['state']={k:[{field:e[field] for field in ('id','name','content','status') if field in e}
                                for e in entries[:5]] for k,entries in state.export().items()}
            messages[1]['content']=json.dumps(prompt)
        response=call_model(model,messages,None,{'role':'refiner','max_output_tokens':config.model.max_output_tokens},budget,trajectory)
        data=json.loads(response.content)
        if not isinstance(data,dict) or not isinstance(data.get('changes'),list) or len(data['changes'])>6:
            raise ValueError('invalid refiner response')
        proposals=[]
        for item in data['changes']:
            change_type=item['change_type']
            kind='note' if change_type=='UPDATE_PROMPT_NOTE' else change_type.split('_')[-1].lower()
            entry=item.get('new_state')
            target=item.get('target_id') or identifier(json.dumps(entry,sort_keys=True))
            evidence=tuple(item['evidence'])
            existing=next((e for e in state.entries(kind) if e['id']==target),None)
            if entry is not None:
                entry=dict(entry,id=target,source='model_proposal_from_adaptation',provenance=list(evidence),
                           version=(existing or {}).get('version',0)+1,timestamp=now())
                if kind=='skill':
                    entry={'input_schema':{'path':'string'},'validation_procedure':'independent tool replay',
                           'creation_time':now(),'last_used':None,'usage_count':0,'success_count':0,'failure_count':0,
                           'project':None,**entry,'status':'candidate'}
                elif kind=='agent':
                    entry={'performance_history':[],'tools':['read','search'],'created_from_task':evidence[-1],
                           **entry,'status':'candidate'}
                elif kind=='memory':
                    entry={'usage_count':0,'success_count':0,'failure_count':0,'tags':[],
                           'task_id':evidence[-1],'project':None,'confidence':item.get('confidence',0.5),'status':'active',**entry}
                elif kind=='note':
                    entry['status']='active'
            proposals.append(HarnessChange(change_type,target,existing,entry,evidence,item['reason'],item.get('confidence',0.5)))
        return proposals

    def propose(self, state, records, config):
        adaptation = [r for r in records if r['split']=='adaptation']
        successful = [r for r in adaptation if r['verified']]
        changes = []
        for entry in state.entries('memory'):
            uses = [r for r in adaptation if any(e['event']=='memory_read' and entry['id'] in e['ids'] for e in r['events'])]
            if not uses:
                continue
            observed = len(uses)
            if observed <= entry.get('usage_count', 0):
                continue
            success_count = sum(r['verified'] for r in uses)
            evidence_ids = tuple(dict.fromkeys([*entry['provenance'],*(r['task_id'] for r in uses)]))
            updated = entry | {'usage_count':observed, 'success_count':success_count,
                'failure_count':observed-success_count, 'confidence':0.8*entry['confidence']+0.2*(success_count/observed),
                'provenance':list(evidence_ids), 'version':entry['version']+1}
            if observed-success_count >= 3:
                updated['status']='needs_reverification'
            changes.append(HarnessChange('UPDATE_MEMORY',entry['id'],entry,updated,evidence_ids,
                                         'Update confidence from independently measured reuse outcomes'))
        for entry in state.entries('skill'):
            uses = [r for r in adaptation if any(e['event']=='skill_invoked' and e['skill_id']==entry['id'] for e in r['events'])]
            if len([r for r in uses if not r['verified']]) >= 3:
                changes.append(HarnessChange('DELETE_SKILL', entry['id'], entry, None,
                    tuple(r['task_id'] for r in uses), 'Repeated independently measured skill failures', 0.9))
            elif uses and len(uses) > entry.get('usage_count',0):
                evidence_ids=tuple(dict.fromkeys([*entry['provenance'],*(r['task_id'] for r in uses)]))
                updated=entry | {'usage_count':len(uses), 'success_count':sum(r['verified'] for r in uses),
                    'failure_count':sum(not r['verified'] for r in uses), 'last_used':now(),
                    'version':entry['version']+1, 'status':'candidate', 'provenance':list(evidence_ids)}
                changes.append(HarnessChange('UPDATE_SKILL',entry['id'],entry,updated,evidence_ids,
                                             'Record observed skill reuse with renewed validation'))
        if config.memory and successful:
            recent = successful[-1]
            target_id = identifier('memory:'+recent['project'])
            if not any(e['id']==target_id for e in state.entries('memory')):
                content = 'Inspect the relevant source, make a focused change, and verify it before completing the coding task.'
                entry = dict(id=target_id, content=content, type='procedural', scope='project',
                    source='verified_trajectory', task_id=recent['task_id'], project=recent['project'],
                    confidence=0.6, tags=['coding','inspect','verify'], provenance=[recent['task_id']],
                    usage_count=0, success_count=0, failure_count=0, version=1, status='active', timestamp=now())
                changes.append(HarnessChange('CREATE_MEMORY', target_id, None, entry,
                    (recent['task_id'],), 'A verified coding workflow observed in this project'))
        read_successes = []
        for record in successful:
            names = [e['name'] for e in record['events'] if e['event']=='tool_call']
            if 'search' in names and 'read' in names and names.index('search') < names.index('read'):
                read_successes.append(record)
        evidence = tuple(dict.fromkeys(r['task_id'] for r in read_successes))
        if len(evidence) >= 2:
            if config.skills and not state.entries('skill'):
                entry = dict(id=identifier('inspect-target'), name='inspect_target',
                    description='Locate and read the task target in one reusable workflow.',
                    trigger_conditions=['coding'], input_schema={'path':'string'},
                    implementation=[{'tool':'search','arguments':{'query':'$path'}},
                                    {'tool':'read','arguments':{'path':'$path'}}],
                    validation_procedure='independent tool replay', provenance=list(evidence),
                    source='verified_trajectory', creation_time=now(), last_used=None,
                    usage_count=0, success_count=0, failure_count=0, version=1, status='candidate', project=None)
                changes.append(HarnessChange('CREATE_SKILL', entry['id'], None, entry, evidence,
                    'Repeated search/read sequence abstracted without task answers', 0.8))
            if config.dynamic_agents and not state.entries('agent'):
                entry = dict(id=identifier('source-inspector'), name='Source inspector', role='repository_debugger',
                    instructions='Inspect the supplied target source. Identify a focused correction and verification needs. Return concise evidence.',
                    capabilities=['coding'], tools=['read','search'], provenance=list(evidence),
                    source='verified_trajectory', created_from_task=evidence[-1], performance_history=[],
                    version=1, status='candidate')
                changes.append(HarnessChange('CREATE_AGENT', entry['id'], None, entry, evidence,
                    'Repeated useful source inspection may justify a specialist', 0.7))
            if config.self_refinement and not state.entries('note'):
                entry = dict(id=identifier('inspect-then-verify'),
                    content='Inspect a relevant source subset before editing. Use independent verification; do not treat fluent completion text as success.',
                    source='verified_trajectory', provenance=list(evidence), status='active', version=1)
                changes.append(HarnessChange('UPDATE_PROMPT_NOTE', entry['id'], None, entry, evidence,
                    'Promote repeated verified workflow to a persistent strategy', 0.8))
        return changes
