import json
from agent_harness.core.context import call_model


class Coordinator:
    def __init__(self, model, registry, config):
        self.model, self.registry, self.config = model, registry, config
        self.selected = {}

    def select(self, family):
        entries = []
        if self.config.fixed_agents:
            entries.append(self.registry.fixed())
        if self.config.dynamic_agents:
            entries.extend(e for e in self.registry.state.entries('agent')
                           if e['status']=='active' and family in e['capabilities'])
        if self.config.temporary_agents:
            entries.append(self.registry.fixed() | {'id': 'temporary-reader', 'temporary':True})
        entries = entries[:1]
        self.selected.update({e['id']:e for e in entries})
        return entries

    def run(self, agent_id, query, runtime):
        if agent_id in self.selected:
            entry = self.selected[agent_id]
        else:
            entry = self.registry.state.get('agent', agent_id)
            if not self.config.dynamic_agents or entry['status'] != 'active':
                raise PermissionError('agent is not available in this condition')
        runtime.budget.reserve('subagent_calls')
        runtime.mailbox.setdefault(agent_id, [])
        task = runtime.environment.task
        # Child receives only the target source; no whole parent prompt or persistent DB.
        runtime.budget.reserve('tool_calls')
        source = runtime.environment.read(task.target)[:runtime.budget.limits.max_context_chars]
        runtime.trajectory.log('tool_call', name='read', agent_id=agent_id,
                               arguments_ref=runtime.objects.put({'path':task.target}))
        runtime.trajectory.log('observation', name='read', agent_id=agent_id,
                               observation_ref=runtime.objects.put({'result':source}))
        messages = [{'role':'system','content':entry['instructions']},
                    {'role':'user','content':json.dumps({'query':query, 'path':task.target, 'source':source,
                                                       'messages':runtime.mailbox[agent_id]})}]
        response = call_model(self.model, messages, None, {'role':'specialist'}, runtime.budget, runtime.trajectory)
        row = {'agent_id':agent_id,'advice':response.content, 'inspected':task.target}
        runtime.trajectory.log('subagent_call', **row)
        return row
