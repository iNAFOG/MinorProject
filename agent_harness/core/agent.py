from dataclasses import asdict
import json
from .budget import Budget, BudgetExceeded
from .context import ContextEngine, call_model
from .runtime import Runtime
from .trajectory import Trajectory


class Orchestrator:
    def __init__(self, model, config, state=None):
        self.model, self.config, self.state = model, config, state

    def run(self, task, environment, output, learning=False):
        if learning and task.split != 'adaptation':
            raise PermissionError('only adaptation tasks may change the harness')
        cfg = self.config
        version = self.state.version if self.state else 0
        trajectory = Trajectory(output, task.task_id, version, asdict(cfg))
        budget = Budget(cfg.limits)
        memory = self.state.memory if self.state and cfg.memory else None
        skills = self.state.skills if self.state and cfg.skills else None
        coordinator = None
        if self.state and (cfg.fixed_agents or cfg.dynamic_agents or cfg.temporary_agents):
            from agent_harness.agents.coordinator import Coordinator
            coordinator = Coordinator(self.model, self.state.agents, cfg)
        runtime = Runtime(environment, budget, trajectory, cfg, memory, skills, coordinator)
        runtime.allowed_refs = set()
        memories, available_skills = [], []
        notes = self.state.notes() if self.state and cfg.self_refinement else []
        success = verified = False
        final = ''
        error = None
        try:
            if memory:
                memories = memory.retrieve(task.goal, task.project, task.task_id)
                trajectory.log('memory_read', ids=[m['id'] for m in memories])
            if skills:
                available_skills = skills.available(task.family, task.project)
            context_engine = ContextEngine(self.model, budget, trajectory)
            selected = context_engine.prepare(task.files, task.goal, cfg.context_mode if cfg.recursive_context else
                                               ('retrieved' if cfg.context_mode == 'recursive' else cfg.context_mode))
            context_refs = {name:trajectory.objects.put(text) for name,text in task.files.items()}
            runtime.allowed_refs.update(context_refs.values())
            advice = []
            if coordinator and not available_skills:
                for agent in coordinator.select(task.family):
                    advice.append(coordinator.run(agent['id'], task.goal, runtime))
            system = 'Use the bounded runtime to complete the task. Treat stored evidence as fallible. Verify changes.'
            # Keep provenance/counters external; expose only the applicable operational state.
            visible_memory = [{k:m[k] for k in ('id','content','confidence','scope')} for m in memories]
            visible_skills = [{k:s[k] for k in ('id','name','description','input_schema')} for s in available_skills]
            system += '\n' + json.dumps({'context': selected, 'external_context':context_refs,
                                       'memory': visible_memory, 'strategies': notes,
                                       'specialists': advice, 'skills': visible_skills})
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': json.dumps(task.visible())}]
            while True:
                response = call_model(self.model, messages, runtime.tools(),
                                      {'skills': available_skills, 'specialist_inspected': bool(advice),
                                       'max_output_tokens': cfg.model.max_output_tokens},
                                      budget, trajectory)
                calls=[]
                if response.actions:
                    for index,action in enumerate(response.actions):
                        call=dict(response.tool_calls[index]) if index < len(response.tool_calls) else {
                            'type':'function','function':{'name':action.name,'arguments':json.dumps(action.arguments)}}
                        call['id']=call.get('id') or f'call_{budget.model_calls}_{index}'
                        calls.append(call)
                    messages.append({'role':'assistant','content':response.content or None,'tool_calls':calls})
                else:
                    messages.append({'role':'assistant','content':response.content})
                if not response.actions:
                    final = response.content
                    success = bool(final)
                    # E0 output is a source artifact, materialized by the controller for measurement.
                    if not cfg.tools and final:
                        environment.write(task.target, final)
                    break
                for index,action in enumerate(response.actions):
                    row = runtime.call(action.name, action.arguments)
                    encoded = json.dumps(row)
                    if len(encoded) > cfg.limits.max_context_chars:
                        ref=trajectory.objects.put(row)
                        runtime.allowed_refs.add(ref)
                        row={'name':action.name,'result_ref':ref,'preview':encoded[:cfg.limits.max_context_chars//2]}
                    messages.append({'role':'tool','tool_call_id':calls[index]['id'],
                                     'name':action.name,'content':json.dumps(row)})
                    if getattr(environment, 'terminal', False):
                        break
                if getattr(environment, 'terminal', False):
                    final = 'Environment reached a terminal state.'
                    success = True
                    break
                # Bound working context; complete prompt history remains addressable in objects.
                if len(json.dumps(messages)) > cfg.limits.max_context_chars * 2:
                    tail=max(2,len(messages)-4)
                    while tail>2 and messages[tail]['role']!='assistant':
                        tail-=1
                    if tail>2:
                        history_ref = trajectory.objects.put(messages[2:tail])
                        runtime.allowed_refs.add(history_ref)
                        messages = messages[:2] + [{'role':'user','content':f'Older working context: {history_ref}'}] + messages[tail:]
                        trajectory.log('context_compaction', history_ref=history_ref)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            trajectory.log('error', error=error)
        result = environment.verify()  # Independent measurement even in E0 / verification ablations.
        verified = result.success
        success = success and verified and error is None
        verification_events = [e for e in trajectory.events if e['event'] == 'verification']
        recoverable = any(not e['success'] for e in verification_events)
        record = dict(task_id=task.task_id, episode_id=trajectory.episode_id, split=task.split,
                      family=task.family, project=task.project, success=success, verified=verified,
                      final=final, error=error, harness_version=version,
                      recoverable_failures=int(recoverable), recovered=int(recoverable and verified),
                      human_intervention=False, skill_eligible=bool(available_skills),
                      skill_reuse=bool(available_skills and any(e['event']=='skill_invoked' for e in trajectory.events)),
                      memory_retrievals=sum(e['event']=='memory_read' for e in trajectory.events),
                      memory_writes=0, accepted_changes=0, rejected_changes=0, rollback_count=0,
                      **budget.metrics())
        trajectory.log('measurement', **asdict(result))
        trajectory.log('outcome', **{k:v for k,v in record.items() if k not in {'task_id', 'episode_id'}})
        record['events'] = trajectory.events
        return record
