"""Continue a pre-decision Battleship pilot after a provider interruption.

Retains the original task labels, model configuration, proposal, and completed
episode prefix. Completed losses are reused just like wins. No new reflection,
seed selection, or rule change is performed.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from agent_harness.core.trajectory import Objects, now
from agent_harness.environments.base import Task
from agent_harness.refinement.changes import HarnessChange
from .benchmark import GameExperiment, GameVerifier, aggregate, paired_comparison, validate_suites
from .environment import GameEnvironment
from .models import RunStopped


def _read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def load_checkpoint(source, config):
    source=Path(source).resolve()
    summary=json.loads((source/'game_summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if digest!=summary['manifest_digest'] or asdict(config)!=summary['metadata']['config']:
        raise ValueError('checkpoint manifest or model/configuration mismatch')
    if (summary['status']!='incomplete' or 'provider ' not in (summary['stop_reason'] or '')
            or summary['accepted_changes'] or summary['rejected_changes'] or summary['harness_version']!=0
            or set(manifest)!={'battleship'}):
        raise ValueError('resume requires an interrupted, undecided, version-zero Battleship pilot')
    suites={split:[Task.from_dict(t) for t in tasks] for split,tasks in manifest['battleship'].items()}
    validate_suites(suites)
    if len(suites['adaptation'])!=2:
        raise ValueError('resume supports one proposal from exactly two adaptation games')
    tasks={t.task_id:t for group in suites.values() for t in group}
    records=_read_lines(source/'episodes.jsonl')
    held=[r for r in records if r['phase']=='initial_held_out']
    adaptation=[r for r in records if r['phase']=='adaptation']
    verification=[r for r in records if r['phase']=='candidate_verification']
    expected=[('initial_held_out',t.task_id) for t in suites['held_out']]
    expected += [('adaptation',t.task_id) for t in suites['adaptation']]
    expected += [('candidate_verification',t.task_id) for t in suites['verification'][:len(verification)]]
    if ([(r['phase'],r['task_id']) for r in records]!=expected
            or len(held)!=len(suites['held_out']) or len(adaptation)!=2
            or not 1<=len(verification)<=len(suites['verification'])):
        raise ValueError('checkpoint is not an ordered baseline-verification prefix')
    if not records[-1].get('error') or 'provider ' not in records[-1]['error']:
        raise ValueError('only the last provider-failed episode may be restarted')
    for index, record in enumerate(records):
        task=tasks[record['task_id']]
        env=GameEnvironment(task)
        if (record['scenario']!=task.metadata or record['split']!=task.split
                or record['harness_version']!=0 or record['initial_observation']!=env.engine.observe()
                or task.files['rules.txt']!=env.engine.rules):
            raise ValueError('saved scenario/rules/state differs from the checkpoint')
        for action in record['game_actions']:
            env.engine.step(action['cells'])
        if index!=len(records)-1:
            if record['error'] or not env.terminal:
                raise ValueError('completed prefix contains an unfinished or failed episode')
            metrics=env.engine.metrics()
            if any(record[key]!=metrics[key] for key in metrics):
                raise ValueError('saved outcome does not match authoritative action replay')
    events=_read_lines(source/'refinement/trajectories.jsonl')
    calls=[e for e in events if e['event']=='model_call' and e.get('role')=='game_refiner']
    if len(calls)!=1:
        raise ValueError('checkpoint needs exactly one saved adaptation reflection')
    objects=Objects(source/'refinement/objects')
    response=objects.get(calls[0]['response_ref'])
    prompt=objects.get(calls[0]['prompt_ref'])
    # Object digests protect the saved model response; validate its original input too.
    messages=prompt['messages'] if isinstance(prompt,dict) else prompt
    payload=json.loads(messages[-1]['content'])
    if (payload['game']!='battleship' or payload['previous_strategy'] is not None
            or [e['task_id'] for e in payload['adaptation_episodes']]!=[r['task_id'] for r in adaptation]):
        raise ValueError('reflection provenance does not match adaptation episodes')
    data=json.loads(response['content'])
    if set(data)!={'strategy'} or not isinstance(data['strategy'],str) or not 8<=len(data['strategy'])<=2500:
        raise ValueError('saved reflection is not a valid bounded strategy')
    target=hashlib.sha256(b'game-strategy:battleship').hexdigest()[:20]
    evidence=tuple(r['task_id'] for r in adaptation)
    entry={'id':target,'content':'For battleship only:\n'+data['strategy'],
           'source':'game_adaptation_reflection','provenance':list(evidence),
           'status':'active','version':1,'timestamp':calls[0]['timestamp']}
    proposal=HarnessChange('UPDATE_PROMPT_NOTE',target,None,entry,evidence,
                           'Original adaptation strategy recovered from saved model response')
    return summary,manifest,suites,records,held,adaptation,verification[:-1],proposal


class PrefixVerifier(GameVerifier):
    def __init__(self,tasks,runner,config,prefix,baseline_digest):
        super().__init__(tasks,runner,config)
        self.prefix,self.baseline_digest=prefix,baseline_digest
        self.evaluations=0

    def evaluate(self,state,config=None):
        self.evaluations+=1
        if self.evaluations!=1:
            return super().evaluate(state,config)
        if state.fingerprint()!=self.baseline_digest:
            raise ValueError('cannot reuse verification with a different baseline state')
        if [r['task_id'] for r in self.prefix]!=[t.task_id for t in self._tasks[:len(self.prefix)]]:
            raise ValueError('cached verification is not the original ordered prefix')
        before=state.fingerprint()
        result=list(self.prefix)+[self.runner(t,state,'candidate_verification') for t in self._tasks[len(self.prefix):]]
        if state.fingerprint()!=before:
            raise RuntimeError('verification changed persistent state')
        return result


def resume_pilot(config,source,output,additional_requests=41,request_interval=6,model_factory=None):
    source=Path(source).resolve()
    saved,manifest,suites,records,before,adaptation,prefix,proposal=load_checkpoint(source,config)
    remaining_tasks=len(suites['verification'])-len(prefix)+len(suites['verification'])+len(suites['held_out'])
    cap=min(saved['request_cap'],saved['request_attempts']+additional_requests)
    if additional_requests<1 or cap-saved['request_attempts']<remaining_tasks*config.limits.max_model_calls:
        raise ValueError('remaining request cap cannot cover the continuation worst case')
    experiment=GameExperiment(config,output,cap,saved['token_cap'],request_interval,model_factory)
    output=experiment.output
    results={}
    try:
        # SQLite backup reads the original database without modifying it.
        with sqlite3.connect((source/'state.db').as_uri()+'?mode=ro',uri=True) as original:
            original.backup(experiment.state.db)
        if (experiment.state.version!=0 or experiment.state.fingerprint()!=saved['final_state_digest']
                or experiment.state.notes() or experiment.state.experiences()!=adaptation):
            raise ValueError('checkpoint state or adaptation archive does not match the saved experiment')
        for name in ('episodes','refinement','harness'):
            shutil.copytree(source/name,output/name,dirs_exist_ok=True)
        for name in ('manifest.json','episodes.jsonl'):
            shutil.copyfile(source/name,output/name)
        experiment.records=list(records)
        experiment.manifest_digest=saved['manifest_digest']
        experiment.guard.requests=saved['request_attempts']
        experiment.guard.tokens=saved['total_tokens']
        continuation={'source':str(source),'source_summary_digest':hashlib.sha256((source/'game_summary.json').read_bytes()).hexdigest(),
                      'source_code_digest':saved['metadata']['source_digest'],'started_at':now(),
                      'prior_request_attempts':saved['request_attempts'],'prior_tokens':saved['total_tokens'],
                      'reused_initial_games':len(before),'reused_adaptation_games':len(adaptation),
                      'reused_baseline_verification_games':len(prefix),
                      'reused_original_reflection':True,'additional_request_cap':cap-saved['request_attempts'],
                      'task_labels':'Original manifest preserved, including its original labels.',
                      'failed_attempt':'Original provider failure retained as overhead; restart its unfinished game.'}
        (output/'continuation.json').write_text(json.dumps(continuation,indent=2)+'\n',encoding='utf-8')
        try:
            verifier=PrefixVerifier(suites['verification'],experiment.run_episode,config,prefix,experiment.state.fingerprint())
            decision=verifier.apply(experiment.state,proposal,{r['task_id']:r for r in adaptation},gate=True)
            decision={'game':'battleship','proposal':proposal.to_dict(),**decision}
            experiment.decisions.append(decision)
            (output/'changes.jsonl').write_text(json.dumps(decision)+'\n',encoding='utf-8')
            frozen=experiment.state.fingerprint()
            after=[experiment.run_episode(t,experiment.state,'learned_held_out') for t in suites['held_out']]
            if frozen!=experiment.state.fingerprint():
                raise RuntimeError('held-out evaluation modified learned state')
            results['battleship']={'initial':aggregate(before),'learned':aggregate(after),'paired':paired_comparison(before,after)}
            experiment.status='complete'
        except RunStopped as exc:
            experiment.reason=str(exc)
        summary=experiment.write_report(results,saved['seed'])
        continuation.update(additional_request_attempts=experiment.guard.requests-saved['request_attempts'],
                            additional_tokens=experiment.guard.tokens-saved['total_tokens'])
        summary['continuation']=continuation
        (output/'continuation.json').write_text(json.dumps(continuation,indent=2)+'\n',encoding='utf-8')
        (output/'game_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        with (output/'report.md').open('a',encoding='utf-8') as report:
            report.write('\n## Continuation accounting\n\nCompleted source episodes and the original reflection were reused. '
                         f'This continuation used {continuation["additional_request_attempts"]} additional requests; '
                         'the totals above include the original attempt and its failed provider request. '
                         'The original manifest, task labels, model configuration and gate were preserved.\n')
        from .reporting import write_replay
        write_replay(output,summary,experiment.records,experiment.decisions)
        return summary
    finally:
        experiment.state.close()


def resume_evaluation(config,source,output,additional_requests=10,request_interval=6,request_timeout=90,model_factory=None):
    """Finish a frozen evaluation, retaining every completed outcome, including losses.

    An infrastructure-interrupted game restarts with its normal five-call budget.
    Its earlier attempt remains in the cumulative usage/episode evidence. Only
    transport timeout and cumulative attempt cap may change; no selection repeats.
    """
    source=Path(source).resolve()
    saved=json.loads((source/'game_summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if (asdict(config)!=saved['metadata']['config'] or saved['status']!='incomplete'
            or 'provider ' not in (saved['stop_reason'] or '') or set(manifest)!={'battleship'}
            or hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()!=saved['manifest_digest']):
        raise ValueError('frozen checkpoint configuration, manifest or provider interruption mismatch')
    suites={split:[Task.from_dict(t) for t in tasks] for split,tasks in manifest['battleship'].items()}
    validate_suites(suites)
    decisions=_read_lines(source/'changes.jsonl')
    if len(decisions)!=1 or not decisions[0].get('baseline') or not decisions[0].get('candidate'):
        raise ValueError('frozen checkpoint requires one completed comparative decision')
    decision=decisions[0]
    if bool(decision['accepted'])!=(saved['harness_version']==1):
        raise ValueError('frozen version and verification decision disagree')
    records=_read_lines(source/'episodes.jsonl')
    before=[r for r in records if r['phase']=='initial_held_out']
    after_attempts=[r for r in records if r['phase']=='learned_held_out']
    if ([r['task_id'] for r in before]!=[t.task_id for t in suites['held_out']]
            or not after_attempts or not after_attempts[-1].get('error')
            or 'provider ' not in after_attempts[-1]['error']
            or [r['task_id'] for r in after_attempts]!=[t.task_id for t in suites['held_out'][:len(after_attempts)]]):
        raise ValueError('frozen evaluation must be a completed prefix followed by one provider interruption')
    prefix=after_attempts[:-1]
    for record,task in zip(before,suites['held_out']):
        if record['error'] or record['scenario']!=task.metadata:
            raise ValueError('initial held-out checkpoint is incomplete or mismatched')
    for record,task in zip(after_attempts,suites['held_out']):
        env=GameEnvironment(task)
        if (record['scenario']!=task.metadata or record['harness_version']!=saved['harness_version']
                or record['initial_observation']!=env.engine.observe() or task.files['rules.txt']!=env.engine.rules):
            raise ValueError('frozen scenario/rules/version mismatch')
        for action in record['game_actions']:
            env.engine.step(action['cells'])
        if record in prefix and (record['error'] or not env.terminal
                                 or any(record[k]!=v for k,v in env.engine.metrics().items())):
            raise ValueError('completed frozen prefix fails authoritative replay')
    remaining=suites['held_out'][len(prefix):]
    if additional_requests<config.limits.max_model_calls*len(remaining) or request_timeout<config.model.request_timeout:
        raise ValueError('insufficient continuation requests or reduced transport timeout')
    effective=replace(config,model=replace(config.model,request_timeout=request_timeout))
    exp=GameExperiment(effective,output,saved['request_attempts']+additional_requests,
                       saved['token_cap'],request_interval,model_factory)
    results={}
    try:
        with sqlite3.connect((source/'state.db').as_uri()+'?mode=ro',uri=True) as original:
            original.backup(exp.state.db)
        if exp.state.version!=saved['harness_version'] or exp.state.fingerprint()!=saved['final_state_digest']:
            raise ValueError('frozen persistent state does not match its saved digest')
        for name in ('episodes','refinement','harness'):
            shutil.copytree(source/name,exp.output/name,dirs_exist_ok=True)
        for name in ('manifest.json','episodes.jsonl','changes.jsonl'):
            shutil.copyfile(source/name,exp.output/name)
        exp.records,exp.decisions=list(records),decisions
        exp.manifest_digest=saved['manifest_digest']
        exp.guard.requests,exp.guard.tokens=saved['request_attempts'],saved['total_tokens']
        continuation={'kind':'finish frozen held-out evaluation after provider timeout','source':str(source),
                      'prior_request_attempts':saved['request_attempts'],'prior_tokens':saved['total_tokens'],
                      'source_code_digest':saved['metadata']['source_digest'],
                      'reused_completed_held_out_games':len(prefix),'additional_request_cap':additional_requests,
                      'original_transport_timeout':config.model.request_timeout,'transport_timeout':request_timeout,
                      'selection':'Original completed verification decision and frozen state retained; no new reflection or selection.',
                      'restarted_game':after_attempts[-1]['task_id'],
                      'accounting':'Infrastructure-failed attempt retained as overhead; completed losses are never replayed.'}
        (exp.output/'continuation.json').write_text(json.dumps(continuation,indent=2)+'\n',encoding='utf-8')
        frozen=exp.state.fingerprint()
        try:
            after=list(prefix)+[exp.run_episode(t,exp.state,'learned_held_out') for t in remaining]
            if exp.state.fingerprint()!=frozen:
                raise RuntimeError('continuation changed frozen learned state')
            results['battleship']={'initial':aggregate(before),'learned':aggregate(after),'paired':paired_comparison(before,after)}
            exp.status='complete'
        except RunStopped as exc:
            exp.reason=str(exc)
        summary=exp.write_report(results,saved['seed'])
        continuation.update(additional_request_attempts=exp.guard.requests-saved['request_attempts'],
                            additional_tokens=exp.guard.tokens-saved['total_tokens'])
        summary['continuation']=continuation
        (exp.output/'continuation.json').write_text(json.dumps(continuation,indent=2)+'\n',encoding='utf-8')
        (exp.output/'game_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        with (exp.output/'report.md').open('a',encoding='utf-8') as report:
            report.write('\n## Infrastructure continuation\n\nThe previous provider timeout interrupted evaluation. '
                         'Completed held-out games, including losses, were retained. Only the interrupted and unstarted games ran. '
                         f'This stage used {continuation["additional_request_attempts"]} additional requests. '
                         f'HTTP timeout increased from {config.model.request_timeout} to {request_timeout} seconds; '
                         'model ID, prompts, temperature, output limits, game budgets, seeds, selected note and gate were preserved. '
                         'Cumulative usage includes all earlier failures. Final score comparisons exclude the infrastructure-interrupted attempt that was restarted.\n')
        from .reporting import write_replay
        write_replay(exp.output,summary,exp.records,exp.decisions)
        return summary
    finally:
        exp.state.close()
