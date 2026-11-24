from dataclasses import asdict


class Runtime:
    def __init__(self, environment, budget, trajectory, config, memory=None, skills=None, coordinator=None):
        self.environment, self.budget, self.trajectory, self.config = environment, budget, trajectory, config
        self.memory, self.skills, self.coordinator = memory, skills, coordinator
        self.objects = trajectory.objects
        self.mailbox = {}
        self.observations = []

    def tools(self):
        if not self.config.tools:
            return []
        definitions = dict(self.environment.tool_definitions()) if hasattr(self.environment, 'tool_definitions') else {
            'search': {'query': {'type': 'string'}},
            'read': {'path': {'type': 'string'}},
            'write': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
            'execute': {'argv': {'type': 'array', 'items': {'type': 'string'}}},
            'retrieve': {'ref': {'type': 'string'}, 'start': {'type':'integer'}, 'end': {'type':'integer'}},
        }
        if self.config.basic_verification:
            definitions['verify'] = {}
        if self.config.memory:
            definitions['memory'] = {'query': {'type': 'string'}}
        if self.config.skills:
            definitions['skill'] = {'skill_id': {'type': 'string'}, 'inputs': {'type': 'object'}}
        if self.coordinator:
            definitions['spawn_agent'] = {'agent_id': {'type': 'string'}, 'query': {'type': 'string'}}
            definitions['message'] = {'agent_id': {'type': 'string'}, 'content': {'type': 'string'}}
        return [{'name': name, 'description': f'Bounded runtime operation: {name}',
                 'parameters': {'type': 'object', 'properties': props,
                                'required': list(props), 'additionalProperties': False}} for name, props in definitions.items()]

    def call(self, name, arguments):
        # Every requested action, including a disabled/unknown one, consumes the tool budget.
        self.budget.reserve('tool_calls')
        self.trajectory.log('tool_call', name=name, arguments_ref=self.objects.put(arguments))
        try:
            if name not in {t['name'] for t in self.tools()}:
                # Returned as an observation so the model can correct itself; never executed.
                raise PermissionError(f'tool disabled: {name}')
            if not isinstance(arguments, dict):
                raise TypeError('tool arguments must be an object')
            if hasattr(self.environment, 'tool_definitions') and name in self.environment.tool_definitions():
                result = self.environment.call_tool(name, arguments)
            elif name in {'read', 'write', 'search', 'execute'}:
                result = getattr(self.environment, name)(**arguments)
            elif name == 'verify':
                result = asdict(self.environment.verify())
                self.trajectory.log('verification', **result)
            elif name == 'retrieve':
                if arguments['ref'] not in getattr(self, 'allowed_refs', set()):
                    raise PermissionError('reference is not addressable in this task')
                start, end = arguments['start'], arguments['end']
                if not isinstance(start,int) or not isinstance(end,int) or not 0 <= start <= end or end-start > self.config.limits.max_context_chars:
                    raise ValueError('retrieve requires a bounded character range')
                value = self.objects.get(arguments['ref'])
                text = value if isinstance(value,str) else __import__('json').dumps(value)
                result = text[start:end]
            elif name == 'memory':
                result = self.memory.retrieve(arguments['query'], self.environment.task.project,
                                              self.environment.task.task_id)
                self.trajectory.log('memory_read', ids=[r['id'] for r in result])
            elif name == 'skill':
                result = self.skills.execute(arguments['skill_id'], arguments['inputs'], self)
                self.trajectory.log('skill_invoked', skill_id=arguments['skill_id'], success=True)
            elif name == 'spawn_agent':
                result = self.coordinator.run(arguments['agent_id'], arguments['query'], self)
            elif name == 'message':
                if arguments['agent_id'] not in self.mailbox:
                    raise ValueError('unknown active child')
                self.mailbox[arguments['agent_id']].append(arguments['content'][:1000])
                self.trajectory.log('agent_message', agent_id=arguments['agent_id'])
                result = {'delivered': True}
            else:
                raise PermissionError(f'tool has no handler: {name}')
            self.budget.check()
            row = {'name': name, 'result': result}
        except Exception as exc:
            from .budget import BudgetExceeded
            if isinstance(exc, BudgetExceeded):
                raise
            row = {'name': name, 'error': f'{type(exc).__name__}: {exc}'}
            self.trajectory.log('error', **row)
        ref = self.objects.put(row)
        self.trajectory.log('observation', name=name, observation_ref=ref)
        self.observations.append(row)
        return row
