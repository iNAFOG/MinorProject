from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from agent_harness.core.agent import Orchestrator
from agent_harness.environments.coding import CodingEnvironment
from agent_harness.environments.mock import MockEnvironment
from agent_harness.core.budget import Budget
from agent_harness.core.trajectory import Trajectory
from agent_harness.core.runtime import Runtime
from agent_harness.skills.validator import SkillValidator
from agent_harness.refinement.policy import validate_change


def score(records):
    if not records:
        raise ValueError('empty verification suite')
    # Explicit cost-penalized objective, same inference budget on both sides.
    return sum(float(r['verified']) - 0.001*r['model_calls'] - 0.0000001*r['tokens']
               - 0.0001*r['tool_calls'] for r in records)/len(records)


class ContributionRejected(ValueError):
    def __init__(self, evaluation):
        super().__init__('specialist has no positive marginal contribution')
        self.evaluation=evaluation


class Verifier:
    def __init__(self, tasks, model_factory, config, allow_local_execution=False):
        if not tasks or any(t.split != 'verification' for t in tasks):
            raise ValueError('verifier requires its own non-held-out verification split')
        # Snapshot task definitions so mutable caller objects cannot change verification.
        from agent_harness.environments.base import Task
        self._tasks = tuple(Task.from_dict(asdict(t)) for t in tasks)
        self._model_factory = model_factory
        self._config = config
        self._allow_local_execution=allow_local_execution

    def evaluate(self, state, config=None):
        config = config or self._config
        before = state.fingerprint()
        results = []
        with tempfile.TemporaryDirectory(prefix='harness-verification-') as root:
            for index, task in enumerate(self._tasks):
                env_cls = MockEnvironment if task.environment == 'mock' else CodingEnvironment
                env = env_cls(task, Path(root)/str(index)/'workspace', config.limits,
                              allow_local_execution=self._allow_local_execution)
                result = Orchestrator(self._model_factory(), config, state).run(task, env, Path(root)/str(index)/'logs')
                results.append({k:v for k,v in result.items() if k!='events'})
        if before != state.fingerprint():
            raise RuntimeError('verification modified persistent harness state')
        return results

    def validate_skill(self, state, entry_id):
        with tempfile.TemporaryDirectory(prefix='harness-skill-validation-') as root:
            runtimes = []
            for index, task in enumerate(self._tasks):
                env = CodingEnvironment(task, Path(root)/str(index)/'workspace', self._config.limits,
                                        allow_local_execution=self._allow_local_execution)
                trajectory = Trajectory(Path(root)/str(index)/'logs', task.task_id, state.version, asdict(self._config))
                runtimes.append(Runtime(env, Budget(self._config.limits), trajectory, self._config,
                                        state.memory, state.skills))
            validation = SkillValidator().validate(state.get('skill', entry_id), runtimes)
            state.skills.activate(entry_id, validation)
            return validation

    def stage(self, state, change, evidence):
        validate_change(change, state, evidence, self._config)
        candidate = state.clone()
        try:
            if change.change_type.startswith('DELETE'):
                candidate.delete(change.kind, change.target_id)
            else:
                candidate.put(change.kind, change.new_state)
            component_validation = None
            if change.kind == 'skill' and change.new_state:
                component_validation = self.validate_skill(candidate, change.target_id)
            if change.kind == 'agent' and change.new_state:
                # Provisional activation exists only in the isolated in-memory candidate.
                provisional = candidate.get('agent', change.target_id)
                provisional['status']='active'
                candidate.put('agent', provisional)
                baseline = self.evaluate(state)
                result = self.evaluate(candidate)
                gain = score(result)-score(baseline)
                if any(b['verified'] and not c['verified'] for b,c in zip(baseline,result)):
                    gain = min(gain, 0)
                component_validation = dict(independent=True, marginal_contribution=gain,
                    baseline=baseline, candidate=result, equal_budget=asdict(self._config.limits))
                if gain <= 0:
                    raise ContributionRejected(component_validation)
                candidate.agents.activate(change.target_id, component_validation)
            return candidate, component_validation
        except Exception:
            candidate.close()
            raise

    def compare(self, baseline, candidate):
        before = self.evaluate(baseline)
        after = self.evaluate(candidate)
        delta = score(after)-score(before)
        regressions = [b['task_id'] for b,c in zip(before,after) if b['verified'] and not c['verified']]
        return dict(accepted=not regressions and delta >= self._config.epsilon,
                    baseline_score=score(before), candidate_score=score(after), gain=delta,
                    epsilon=self._config.epsilon, regressions=regressions,
                    baseline=before, candidate=after, equal_budget=asdict(self._config.limits))

    def apply(self, state, change, evidence, gate=True):
        try:
            candidate, component = self.stage(state, change, evidence)
        except Exception as exc:
            decision = {'accepted':False, 'reason':f'{type(exc).__name__}: {exc}'}
            if isinstance(exc,ContributionRejected):
                decision['component_validation']=exc.evaluation
                measured=[*exc.evaluation['baseline'],*exc.evaluation['candidate']]
                decision['verification_overhead']={key:sum(r.get(key,0) for r in measured)
                                                  for key in ('model_calls','tokens','tool_calls','estimated_cost','latency')}
            state.audit('rejected_change', {'change':change.to_dict(), 'decision':decision})
            return decision
        try:
            if gate:
                decision = self.compare(state, candidate)
            else:
                decision = {'accepted':True, 'reason':'comparative gate disabled by condition',
                            'structural_validation':True}
            decision['component_validation'] = component
            measured = [*decision.get('baseline', []), *decision.get('candidate', [])]
            if component:
                measured += [*component.get('baseline', []), *component.get('candidate', [])]
            decision['verification_overhead'] = {key:sum(r.get(key,0) for r in measured)
                                                  +(component or {}).get('overhead',{}).get(key,0)
                                                  for key in ('model_calls','tokens','tool_calls','estimated_cost','latency')}
            if decision['accepted']:
                decision['version'] = state.promote(candidate, [change.to_dict()], decision)
            else:
                state.audit('rejected_change', {'change':change.to_dict(), 'decision':decision})
            return decision
        finally:
            candidate.close()
