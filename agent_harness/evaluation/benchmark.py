from dataclasses import asdict, replace
from pathlib import Path
import csv
import hashlib
import json
import platform
import random
import subprocess
from agent_harness.core.agent import Orchestrator
from agent_harness.core.trajectory import now
from agent_harness.environments.base import Task
from agent_harness.environments.coding import CodingEnvironment
from agent_harness.environments.mock import MockEnvironment
from agent_harness.models import build_model
from agent_harness.refinement.refiner import HarnessRefiner
from agent_harness.storage.state import State
from agent_harness.verification.verifier import Verifier
from .metrics import aggregate


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        self.version = data['version']
        self.tasks = [Task.from_dict(t) for t in data['tasks']]
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError('task IDs must be unique across all splits')
        # Catch identical tasks under different IDs, even if split labels differ.
        seen = {}
        for task in self.tasks:
            content = json.dumps({k:getattr(task,k) for k in ('goal','files','checks','project')},sort_keys=True)
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in seen and seen[digest] != task.split:
                raise ValueError('identical task leaked across splits')
            seen[digest] = task.split

    def split(self, name):
        return [t for t in self.tasks if t.split==name]


def metadata(config, dataset=None):
    source_root=Path(__file__).resolve().parents[1]
    try:
        # Ask the repository containing this source, not whatever directory the command ran from.
        result = subprocess.run(['git','rev-parse','HEAD'],cwd=source_root,capture_output=True,text=True,timeout=3)
        commit = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    source_digest=hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        source_digest.update(path.relative_to(source_root).as_posix().encode())
        source_digest.update(path.read_bytes())
    return {'timestamp':now(),'git_commit':commit,'source_digest':source_digest.hexdigest(),
            'config':asdict(config),'config_digest':config.fingerprint,
            'model_configuration':asdict(config.model),'seed':config.seed,
            'benchmark_version':dataset.version if dataset else None,
            'benchmark_digest':dataset.digest if dataset else None,
            'environment':{'python':platform.python_version(),'system':platform.platform()}}


class Experiment:
    def __init__(self, config, state, output, verification_tasks, allow_local_execution=False):
        self.config,self.state,self.output = config,state,Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.allow_local_execution = allow_local_execution
        self.verifier = Verifier(verification_tasks,lambda:build_model(config.model),config,allow_local_execution) if verification_tasks else None
        self.records = []

    def run_task(self, task, learning=False, phase='task'):
        if learning and task.split != 'adaptation':
            raise PermissionError('held-out/verification data is measurement only')
        env_cls = MockEnvironment if task.environment=='mock' else CodingEnvironment
        # Never reuse a workspace: files left by an earlier run would be measured as this run's output.
        index = len(self.records)
        while (folder := self.output/'workspaces'/f'{phase}-{index:04d}').exists():
            index += 1
        env = env_cls(task,folder,self.config.limits,allow_local_execution=self.allow_local_execution)
        before = self.state.fingerprint()
        record = Orchestrator(build_model(self.config.model),self.config,self.state).run(task,env,self.output,learning=learning)
        record['phase']=phase
        record['model_configuration']=asdict(self.config.model)
        self.records.append(record)
        if learning:
            self.state.record_experience(record)
            # E2/E3/E4 acquire memory; E5 adds evaluated roles, E6 validated skills.
            if self.config.memory or self.config.skills or self.config.dynamic_agents or self.config.self_refinement:
                if not self.verifier:
                    raise ValueError('learning requires independent verification tasks')
                experiences = [r for r in self.state.experiences()
                               if r.get('model_configuration')==asdict(self.config.model)]
                proposals = HarnessRefiner().propose(self.state,experiences,self.config)
                if self.config.self_refinement and self.config.model.backend != 'mock':
                    from agent_harness.core.budget import Budget
                    from agent_harness.core.trajectory import Trajectory
                    import time
                    # Refinement shares this episode's remaining inference/resource budget.
                    budget=Budget(self.config.limits)
                    for key in ('model_calls','tool_calls','subagent_calls','recursive_calls','input_tokens','output_tokens','estimated_cost'):
                        setattr(budget,key,record[key])
                    budget.started=time.monotonic()-record['latency']
                    trajectory=Trajectory(self.output,task.task_id,self.state.version,asdict(self.config))
                    try:
                        proposals += HarnessRefiner().model_proposals(self.state,experiences,self.config,
                                        build_model(self.config.model),trajectory,budget)
                    except Exception as exc:
                        trajectory.log('refinement_error',error=f'{type(exc).__name__}: {exc}')
                    record.update(budget.metrics())
                evidence = {r['task_id']:r for r in experiences}
                for proposal in proposals:
                    decision = self.verifier.apply(self.state,proposal,evidence,gate=self.config.verification_gate)
                    record['accepted_changes' if decision['accepted'] else 'rejected_changes'] += 1
                    if decision['accepted'] and proposal.kind=='memory':
                        record['memory_writes'] += 1
                    for key,value in decision.get('verification_overhead',{}).items():
                        record['verification_'+key]=record.get('verification_'+key,0)+value
                    with (self.output/'changes.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'task_id':task.task_id,'proposal':proposal.to_dict(),'decision':decision})+'\n')
                    with (self.output/'trajectories.jsonl').open('a',encoding='utf-8') as f:
                        from agent_harness.core.trajectory import Objects
                        ref=Objects(self.output/'objects').put({'proposal':proposal.to_dict(),'decision':decision})
                        f.write(json.dumps({'timestamp':now(),'task_id':task.task_id,'episode_id':record['episode_id'],
                                            'event':'harness_change','change_ref':ref,'accepted':decision['accepted']})+'\n')
                self.state.record_experience(record)
        elif self.state.fingerprint() != before:
            raise RuntimeError('measurement run modified harness state')
        return record

    def write_artifacts(self, info, comparisons=None):
        clean = [{k:v for k,v in r.items() if k!='events'} for r in self.records]
        metrics = aggregate(clean)
        metrics.update({key:sum(r.get(key,0) for r in clean) for key in (
            'verification_model_calls','verification_tokens','verification_tool_calls',
            'verification_estimated_cost','verification_latency')})
        summary = info | {'harness_version':self.state.version,'harness_digest':self.state.fingerprint(),
                          'episodes':clean,'metrics':metrics,
                          'interpretation':'Offline mock runs validate mechanisms; they do not establish LLM capability gains.'
                          if self.config.model.backend=='mock' else 'Measure transfer with fixed backend/settings and matched budgets.'}
        for name,value in [('experiment_summary.json',summary),('metrics.json',metrics),
                           ('harness_history.json',self.state.history())]:
            (self.output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        rows = comparisons or [{'condition':self.config.condition,**metrics}]
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (self.output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return summary

    def benchmark(self, dataset):
        if (self.output/'experiment_summary.json').exists():
            raise FileExistsError('benchmark output already exists; use a fresh output directory')
        if not dataset.split('adaptation') or not dataset.split('held_out'):
            raise ValueError('benchmark requires adaptation and held-out tasks')
        initial = self.state.clone()
        try:
            baseline = Experiment(self.config,initial,self.output/'baseline',dataset.split('verification'),self.allow_local_execution)
            baseline_adaptation = [baseline.run_task(t,phase='initial_adaptation') for t in dataset.split('adaptation')]
            baseline_test = [baseline.run_task(t,phase='initial_held_out') for t in dataset.split('held_out')]
            adaptation_order=dataset.split('adaptation')
            random.Random(self.config.seed).shuffle(adaptation_order)
            for task in adaptation_order:
                self.run_task(task,learning=True,phase='adaptation')
            frozen = self.state.fingerprint()
            adaptation_eval = [self.run_task(t,phase='learned_adaptation') for t in dataset.split('adaptation')]
            held_out = [self.run_task(t,phase='learned_held_out') for t in dataset.split('held_out')]
            if frozen != self.state.fingerprint():
                raise RuntimeError('adaptation evaluation or held-out measurement changed state')
            groups = [('initial_adaptation',baseline_adaptation),('initial_held_out',baseline_test),
                      ('learned_adaptation',adaptation_eval),('learned_held_out',held_out)]
            rows = [{'condition':self.config.condition,'seed':self.config.seed,'phase':phase,**aggregate(records)} for phase,records in groups]
            initial_adapt,initial_test,learned_adapt,learned_test = [aggregate(records) for _,records in groups]
            gains = {'learning_gain':learned_adapt['verified_completion_rate']-initial_adapt['verified_completion_rate'],
                     'transfer_gain':learned_test['verified_completion_rate']-initial_test['verified_completion_rate'],
                     'held_out_model_call_delta':learned_test['model_calls']-initial_test['model_calls'],
                     'held_out_token_delta':learned_test['tokens']-initial_test['tokens'],
                     'paired_task_ids':[t.task_id for t in dataset.split('held_out')],
                     'adaptation_order':[t.task_id for t in adaptation_order],
                     'groups':{phase:aggregate(records) for phase,records in groups}}
            baseline.write_artifacts(metadata(self.config,dataset))
            summary=self.write_artifacts(metadata(self.config,dataset)|gains,rows)
            return summary
        finally:
            initial.close()
