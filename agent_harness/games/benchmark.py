"""Game evaluation using the existing orchestrator, state, policy and verifier."""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json
import random
import statistics
from agent_harness.core.agent import Orchestrator
from agent_harness.core.budget import Budget
from agent_harness.core.context import call_model
from agent_harness.core.trajectory import Trajectory, now
from agent_harness.evaluation.benchmark import metadata
from agent_harness.models import build_model
from agent_harness.refinement.changes import HarnessChange
from agent_harness.storage.state import State
from agent_harness.verification.verifier import Verifier
from .environment import GameEnvironment, game_task
from .engines import make_game
from .models import GameSmokeModel, GuardedModel, RequestGuard, RunStopped, load_key_from_environment


def make_suite(game,split,count,seed=0):
    if split not in {'adaptation','verification','held_out'} or not 1 <= count <= 500:
        raise ValueError('valid split and 1..500 episodes per split required')
    offset={'adaptation':0,'verification':1000,'held_out':2000}[split]
    return [game_task(game,split,seed*10000+offset+i,
                      opponent=('random','tactical','perfect')[i%3],agent_mark='XO'[i%2],size=(7,9)[i%2])
            for i in range(count)]


def validate_suites(suites):
    seen_ids,seen_scenarios,seen_layouts={},{},{}
    for split,tasks in suites.items():
        for task in tasks:
            if task.split!=split:
                raise ValueError('task split mismatch')
            identity=json.dumps(task.metadata,sort_keys=True)
            for mapping,key in ((seen_ids,task.task_id),(seen_scenarios,identity)):
                if key in mapping:
                    raise ValueError('duplicate game scenario across task splits')
                mapping[key]=split
            if task.metadata['game']=='key_door':
                layout=task.files['observation.json']
                if layout in seen_layouts:
                    raise ValueError('duplicate maze layout across task splits')
                seen_layouts[layout]=split
            if task.metadata['game']=='battleship':
                # Evaluation-only check: hidden fleet coordinates never enter model observations.
                fleet=tuple(make_game(task.metadata)._ships)
                if fleet in seen_layouts:
                    raise ValueError('duplicate hidden fleet across task splits')
                seen_layouts[fleet]=split


def game_score(records):
    if not records:
        raise ValueError('empty game verification suite')
    # Fixed declared objective. No per-episode game-score regression is permitted.
    return statistics.mean(r['game_score']-0.0001*r['model_calls']-0.00000001*r['tokens'] for r in records)


def aggregate(records):
    if not records:
        return {'episodes':0}
    n=len(records)
    total_steps=sum(r['steps'] for r in records)
    solved=[r for r in records if r['outcome']=='goal']
    return {'episodes':n,'mean_game_score':statistics.mean(r['game_score'] for r in records),
            'wins':sum(r['outcome'] in {'win','goal'} for r in records),
            'draws':sum(r['outcome']=='draw' for r in records),
            'losses':sum(r['outcome']=='loss' for r in records),
            'failures':sum(r['outcome'] not in {'win','draw','loss','goal'} for r in records),
            'win_or_goal_rate':sum(r['outcome'] in {'win','goal'} for r in records)/n,
            'non_loss_rate':sum(r['outcome'] in {'win','draw','goal'} for r in records)/n,
            'model_calls':sum(r['model_calls'] for r in records),
            'tokens':sum(r['tokens'] for r in records),'tool_calls':sum(r['tool_calls'] for r in records),
            'recursive_calls':sum(r['recursive_calls'] for r in records),
            'illegal_moves':sum(r['illegal_moves'] for r in records),
            'illegal_move_rate':sum(r['illegal_moves'] for r in records)/max(1,total_steps),
            'mean_steps':statistics.mean(r['steps'] for r in records),
            'mean_latency':statistics.mean(r['latency'] for r in records),
            'path_efficiency_on_solved':statistics.mean(r['path_efficiency'] for r in solved) if solved else None,
            'mean_hits':statistics.mean(r['hits'] for r in records) if all('hits' in r for r in records) else None,
            'mean_ships_sunk':statistics.mean(r['ships_sunk'] for r in records) if all('ships_sunk' in r for r in records) else None,
            'shots_fired':sum(r.get('shots_fired',0) for r in records),
            'shot_hit_rate':sum(r.get('hits',0) for r in records)/max(1,sum(r.get('shots_fired',0) for r in records))}


def paired_comparison(before,after):
    if [r['task_id'] for r in before] != [r['task_id'] for r in after] or not before:
        raise ValueError('paired comparison requires identical ordered scenario IDs')
    diffs=[a['game_score']-b['game_score'] for b,a in zip(before,after)]
    rng=random.Random(713)
    boot=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(2000))
    return {'mean_score_delta':statistics.mean(diffs),'improved_episodes':sum(d>0 for d in diffs),
            'regressed_episodes':sum(d<0 for d in diffs),'tied_episodes':sum(d==0 for d in diffs),
            'exploratory_paired_episode_bootstrap_95_percent':[boot[49],boot[1949]],
            'model_call_delta':sum(a['model_calls']-b['model_calls'] for b,a in zip(before,after)),
            'token_delta':sum(a['tokens']-b['tokens'] for b,a in zip(before,after)),
            'caution':'Episode bootstrap describes this fixed suite and one learning trajectory, not independent adaptation-seed uncertainty.'}


class GameVerifier(Verifier):
    def __init__(self,tasks,runner,config):
        super().__init__(tasks,lambda:None,config)
        self.runner=runner

    def evaluate(self,state,config=None):
        before=state.fingerprint()
        records=[self.runner(task,state,'candidate_verification') for task in self._tasks]
        if state.fingerprint()!=before:
            raise RuntimeError('game verification changed persistent state')
        return records

    def compare(self,baseline,candidate):
        before,after=self.evaluate(baseline),self.evaluate(candidate)
        gain=game_score(after)-game_score(before)
        regressions=[b['task_id'] for b,a in zip(before,after) if a['game_score']<b['game_score']]
        return {'accepted':not regressions and gain>=self._config.epsilon,
                'gain':gain,'baseline_score':game_score(before),'candidate_score':game_score(after),
                'epsilon':self._config.epsilon,'regressions':regressions,
                'baseline':before,'candidate':after,'equal_budget':asdict(self._config.limits)}


def episode_digest(record):
    """Compact, factual summary of one practice game built only from public observations.

    Lists concrete mistakes (stopping early, wasted salvos, unexplored hits) so a small
    fixed model can write rules that address them. Hidden fleet data is never used.
    """
    actions=record.get('game_actions',[])
    digest={'task_id':record['task_id'],'outcome':record['outcome'],'score':record['game_score'],
            'illegal_moves':record['illegal_moves'],'moves_used':len(actions)}
    mistakes=[]
    if record.get('game')=='battleship':
        digest['salvos']=[{'cells':a['cells'],'results':[r['result'] for r in a['results']],'error':a['error']} for a in actions]
        last=actions[-1]['observation'] if actions else record['initial_observation']
        digest['final_board_rows']=last['rows']
        if not last['terminal']:
            mistakes.append(f'Stopped after {len(actions)} of {last["max_salvos"]} salvos by replying with text instead of '
                            'calling fire. An unfinished game scores 0 even after hits.')
        if record['illegal_moves']:
            mistakes.append(f'{record["illegal_moves"]} salvo(s) were invalid (not three distinct, untested cells 0..24) and were wasted.')
        sunk={c for ship in last['sunk_ships'] for c in ship}
        tested={c for a in actions if not a['error'] for c in a['cells']}
        open_hits=[c for c,v in enumerate(last['board']) if v=='X' and c not in sunk
                   and any(n not in tested for n in (c-5,c+5,c-1 if c%5 else None,c+1 if c%5<4 else None) if n is not None and 0<=n<25)]
        if open_hits:
            mistakes.append(f'{len(open_hits)} hit cell(s) of an unsunk ship still had untested neighbours when the game ended.')
        if last['terminal'] and record['outcome']!='win':
            mistakes.append(f'Used every salvo and hit {last["hits"]} of 5 ship cells.')
    else:
        digest['actions']=actions[-12:]
        if record['outcome'] in {'early_stop','agent_error'}:
            mistakes.append('Stopped before the game ended; an unfinished game scores 0.')
        if record['illegal_moves']:
            mistakes.append(f'{record["illegal_moves"]} illegal move request(s).')
    digest['mistakes']=mistakes or ['No rule violation recorded.']
    if record.get('final'):
        digest['agent_last_message']=record['final'][:300]
    return digest


def parse_strategy(raw):
    """Accept one JSON object {"strategy": text}, tolerating code fences or surrounding prose."""
    raw=raw.strip()
    if raw.startswith('```'):
        raw='\n'.join(line for line in raw.splitlines() if not line.startswith('```'))
    if not raw.startswith('{') and '{' in raw and '}' in raw:
        raw=raw[raw.index('{'):raw.rindex('}')+1]
    data=json.loads(raw)
    if not isinstance(data,dict) or 'strategy' not in data:
        raise ValueError('refiner must return one JSON object with a "strategy" field')
    # Any other keys are discarded, never stored.
    strategy=data['strategy']
    if isinstance(strategy,list) and all(isinstance(s,str) for s in strategy):
        strategy=' '.join(s.strip() for s in strategy)
    if not isinstance(strategy,str) or not 8<=len(strategy)<=2500:
        raise ValueError('refiner must return one bounded strategy string')
    if any(token in strategy for token in ('function(','def ','=>','{','}')):
        raise ValueError('strategy must be plain-language rules, not code')
    return strategy


class GameRefiner:
    def propose(self,state,records,model,config,output):
        if len({r['task_id'] for r in records})<2 or any(r['split']!='adaptation' for r in records):
            raise ValueError('game refinement requires two distinct adaptation episodes')
        games={r['game'] for r in records}
        if len(games)!=1:
            raise ValueError('refine one game at a time')
        game=next(iter(games))
        recent=records[-3:]
        evidence=tuple(r['task_id'] for r in recent)
        target=hashlib.sha256(f'game-strategy:{game}'.encode()).hexdigest()[:20]
        old=next((e for e in state.entries('note') if e['id']==target),None)
        examples=[episode_digest(r) for r in recent]
        messages=[{'role':'system','content':
                   'You improve the written playing instructions of a fixed game-playing model, using only its practice games below. '
                   'Each game lists the mistakes that were recorded. Reply with ONLY a JSON object of the form '
                   '{"strategy": "rule one. rule two. ..."} whose value is plain English: 3 to 8 short imperative rules, '
                   'at most 1200 characters in total. Fix the listed mistakes first. The rules must apply to new games of the '
                   'same kind: no code, no task IDs, no positions copied from these particular games, no test or scoring changes. '
                   'If a previous strategy is given, revise it.'},
                  {'role':'user','content':json.dumps({'game':game,'previous_strategy':old['content'] if old else None,
                                                       'adaptation_episodes':examples})}]
        trajectory=Trajectory(output,'refine-'+game,state.version,asdict(config))
        response=call_model(model,messages,None,{'role':'game_refiner','max_output_tokens':config.model.max_output_tokens},
                            Budget(config.limits),trajectory)
        strategy=parse_strategy(response.content)
        entry={'id':target,'content':f'For {game} only:\n'+strategy,'source':'game_adaptation_reflection',
               'provenance':list(evidence),'status':'active','version':(old or {}).get('version',0)+1,'timestamp':now()}
        return HarnessChange('UPDATE_PROMPT_NOTE',target,old,entry,evidence,
                             'General strategy extracted from adaptation game trajectories')


class GameExperiment:
    def __init__(self,config,output,max_requests=120,max_tokens=200000,request_interval=0,model_factory=None):
        if config.fixed_agents or config.dynamic_agents or config.temporary_agents or config.skills or config.basic_verification:
            raise ValueError('game pilot uses strategy notes; disable coding specialists, workflow skills and in-task verify')
        if not config.tools or not config.self_refinement or not config.verification_gate:
            raise ValueError('game pilot requires tools and strategy notes enabled')
        keyless=config.model.backend in {'game_smoke','transformers'}
        if not keyless and model_factory is None and not load_key_from_environment(config.model.api_key_env):
            raise ValueError(f'Set {config.model.api_key_env} locally before remote experiments; do not put the key in configuration')
        self.output=Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError('choose a fresh game experiment output directory')
        self.output.mkdir(parents=True,exist_ok=True)
        self.config=config
        self.guard=RequestGuard(max_requests,max_tokens,request_interval)
        self.factory=model_factory or (GameSmokeModel if config.model.backend=='game_smoke' else lambda:build_model(config.model))
        self.records,self.decisions=[],[]
        self.state=State(self.output/'state.db')
        self.status='incomplete'
        self.reason=None

    def model(self):
        return GuardedModel(self.factory(),self.guard)

    def run_episode(self,task,state,phase):
        before=state.fingerprint()
        env=GameEnvironment(task)
        index=len(self.records)
        record=Orchestrator(self.model(),self.config,state).run(task,env,self.output/'episodes'/f'{index:05d}')
        record={k:v for k,v in record.items() if k!='events'} | env.engine.metrics()
        if not env.terminal:
            record['outcome']='agent_error' if record['error'] else 'early_stop'
            record['game_score']=0.0
        record.update(phase=phase,scenario=task.metadata,initial_observation=json.loads(task.files['observation.json']),
                      game_actions=env.engine.history)
        if state.fingerprint()!=before:
            raise RuntimeError('episode changed learned state')
        self.records.append(record)
        with (self.output/'episodes.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        if self.guard.fatal_reason:
            raise RunStopped(self.guard.fatal_reason)
        if record['error'] and any(x in record['error'] for x in ('HTTPError','URLError')):
            raise RunStopped('inference provider failure')
        return record

    def run(self,games=('battleship',),adaptation=4,verification=4,held_out=12,seed=0,ablate_notes=False):
        if not games or len(set(games))!=len(games) or set(games)-{'tic_tac_toe','key_door','battleship'}:
            raise ValueError('choose distinct supported games')
        if adaptation<2 or verification<2 or held_out<2:
            raise ValueError('use at least two episodes in each split')
        suites={game:{split:make_suite(game,split,n,seed) for split,n in
                      (('adaptation',adaptation),('verification',verification),('held_out',held_out))} for game in games}
        for split_suites in suites.values():
            validate_suites(split_suites)
        manifest={game:{split:[asdict(t) for t in tasks] for split,tasks in split_suites.items()}
                  for game,split_suites in suites.items()}
        (self.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        self.manifest_digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        results={}
        try:
            for game,split_suites in suites.items():
                # Preserve each game's initial and frozen state; selection never sees held-out outcomes.
                initial=self.state.clone()
                try:
                    before=[self.run_episode(t,initial,'initial_held_out') for t in split_suites['held_out']]
                finally:
                    initial.close()
                experiences=[]
                verifier=GameVerifier(split_suites['verification'],self.run_episode,self.config)
                for index,task in enumerate(split_suites['adaptation']):
                    record=self.run_episode(task,self.state,'adaptation')
                    self.state.record_experience(record)
                    experiences.append(record)
                    if (index+1)%2==0:
                        try:
                            proposal=GameRefiner().propose(self.state,experiences,self.model(),self.config,self.output/'refinement')
                        except RunStopped:
                            raise
                        except (ValueError,KeyError,TypeError) as exc:
                            decision={'game':game,'accepted':False,'reason':f'invalid proposal: {type(exc).__name__}'}
                            self.state.audit('rejected_game_proposal',decision)
                        else:
                            decision=verifier.apply(self.state,proposal,{r['task_id']:r for r in experiences},gate=True)
                            decision={'game':game,'proposal':proposal.to_dict(),**decision}
                        self.decisions.append(decision)
                        with (self.output/'changes.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(decision)+'\n')
                frozen=self.state.fingerprint()
                after=[self.run_episode(t,self.state,'learned_held_out') for t in split_suites['held_out']]
                results[game]={'initial':aggregate(before),'learned':aggregate(after),'paired':paired_comparison(before,after)}
                # Subgroups expose opponent strength, side and maze-size effects.
                strata=sorted({str(r['scenario'].get('opponent',r['scenario'].get('size'))) for r in before})
                results[game]['strata']={key:{'initial':aggregate([r for r in before if str(r['scenario'].get('opponent',r['scenario'].get('size')))==key]),
                                                    'learned':aggregate([r for r in after if str(r['scenario'].get('opponent',r['scenario'].get('size')))==key])} for key in strata}
                results[game]['sides']={mark:{'initial':aggregate([r for r in before if r.get('agent_mark')==mark]),
                                                     'learned':aggregate([r for r in after if r.get('agent_mark')==mark])}
                                             for mark in ('X','O')} if game=='tic_tac_toe' else {}
                if ablate_notes:
                    ablated=self.state.clone()
                    try:
                        for entry in ablated.entries('note'):
                            ablated.delete('note',entry['id'])
                        no_notes=[self.run_episode(t,ablated,'ablated_notes_held_out') for t in split_suites['held_out']]
                        results[game]['without_notes']=aggregate(no_notes)
                    finally:
                        ablated.close()
                if self.state.fingerprint()!=frozen:
                    raise RuntimeError('held-out evaluation modified the harness')
            self.status='complete'
        except RunStopped as exc:
            self.reason=str(exc)
        finally:
            summary=self.write_report(results,seed)
            self.state.close()
        return summary

    def write_report(self,results,seed):
        overhead=aggregate([r for r in self.records if r['phase']=='candidate_verification'])
        summary={'status':self.status,'stop_reason':self.reason,'timestamp':now(),'seed':seed,
                 'benchmark':'local-games-v2','manifest_digest':self.manifest_digest,
                 'model':asdict(self.config.model),'metadata':metadata(self.config),
                 'is_real_model':self.config.model.backend!='game_smoke',
                 'interpretation':('Handwritten plumbing controller. Ignores learned notes. No LLM or learning-performance claim.'
                                   if self.config.model.backend=='game_smoke' else
                                   'Exploratory fixed-model strategy-note adaptation. Report improvement only when measured on frozen held-out scenarios.'),
                 'results':results,'accepted_changes':sum(d['accepted'] for d in self.decisions),
                 'rejected_changes':sum(not d['accepted'] for d in self.decisions),
                 'request_attempts':self.guard.requests,'total_tokens':self.guard.tokens,
                 'request_cap':self.guard.max_requests,'token_cap':self.guard.max_tokens,
                 'verification_overhead':overhead,'harness_version':self.state.version,
                 'final_state_digest':self.state.fingerprint(),
                 'game_order':list(results),
                 'limitations':['One adaptation trajectory per game and run seed. Repeat independent seeds before drawing a research conclusion.',
                               'No outcome oracle or shortest-path solver is exposed to a real agent.',
                               'These small observations primarily test strategy notes and task execution, not recursive-context benefit.',
                               'Configured zero cost rates do not verify free-tier eligibility or measure monetary cost.',
                               'In a multi-game run, earlier accepted game-specific notes remain in both conditions of the later game. Run each game separately for independent experiments.']}
        (self.output/'game_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (self.output/'harness_history.json').write_text(json.dumps(self.state.history(),indent=2),encoding='utf-8')
        with (self.output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            fields=['game','phase','episodes','mean_game_score','win_or_goal_rate','non_loss_rate','model_calls','tokens','illegal_moves','mean_steps']
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for game,groups in results.items():
                for phase in ('initial','learned','without_notes'):
                    if phase in groups:
                        writer.writerow({'game':game,'phase':phase,**groups[phase]})
        lines=['# Game benchmark results','',f'Run status: **{self.status}**', '',summary['interpretation'],'',
               f'Model: `{self.config.model.model}`. Run requests: {self.guard.requests}/{self.guard.max_requests}.',
               f'Accepted updates: {summary["accepted_changes"]}. Rejected updates: {summary["rejected_changes"]}.','']
        if self.reason:
            lines.extend([f'Stopped: {self.reason}. This is an incomplete experiment.',''])
        lines+=['| Game | Phase | Episodes | Mean score | Win/goal rate | Model calls | Illegal moves |',
                '|---|---|---:|---:|---:|---:|---:|']
        for game,groups in results.items():
            for phase in ('initial','learned','without_notes'):
                if phase in groups:
                    r=groups[phase]
                    lines.append(f'| {game} | {phase} | {r["episodes"]} | {r["mean_game_score"]:.3f} | {r["win_or_goal_rate"]:.1%} | {r["model_calls"]} | {r["illegal_moves"]} |')
            pair=groups['paired']
            lines+=['',f'{game}: paired mean-score change {pair["mean_score_delta"]:+.3f}, '
                    f'{pair["improved_episodes"]} improved / {pair["regressed_episodes"]} regressed / {pair["tied_episodes"]} tied.','']
        lines+=['## Interpretation','',
                'Tic-tac-toe scores win=1, draw=0.5 and loss/failure=0. KeyDoor scores goal=1 and failure=0.',
                'Battleship is a custom 5x5 hidden-fleet salvo variant: mean score is occupied cells hit / 5, not win rate. A win requires all five hits within five salvos.',
                'Verification uses the declared score minus small call/token penalties and rejects any per-scenario score regression.',
                'Paired episode intervals in the JSON are exploratory, not uncertainty across independent adaptation runs.',
                'Total requests include task execution, reflection and candidate verification. Inference savings alone do not prove lower total cost.',
                'See manifest.json for fixed splits, episodes.jsonl and episodes/*/trajectories.jsonl for replayable evidence, and changes.jsonl for proposals and decisions.','',
                '## Limitations','']+['- '+s for s in summary['limitations']]
        (self.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        from .reporting import write_replay
        write_replay(self.output,summary,self.records,self.decisions)
        return summary
