import argparse
from dataclasses import asdict,replace
from pathlib import Path
import json
from agent_harness.config import Config,condition
from agent_harness.environments.base import Task
from agent_harness.evaluation.benchmark import Dataset,Experiment,metadata
from agent_harness.storage.state import State


def parser():
    root=argparse.ArgumentParser(description='Local RLM context and adaptive external harness')
    commands=root.add_subparsers(dest='command',required=True)
    for name in ('run','evaluate','benchmark'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--config',default='configs/e8.yaml')
        cmd.add_argument('--state',default='.harness/state.db')
        cmd.add_argument('--output',default='runs/latest')
        cmd.add_argument('--seed',type=int)
        cmd.add_argument('--allow-local-execution',action='store_true',help='Enable process execution for trusted repository code')
        for flag in ('memory','skills','dynamic-agents','recursive-context','verification','self-refinement'):
            cmd.add_argument('--disable-'+flag,action='store_true')
        if name=='run':
            cmd.add_argument('--task',required=True)
            cmd.add_argument('--learn',action='store_true',help='Adapt from this adaptation task')
            cmd.add_argument('--verification-dataset',default='benchmarks/coding.json')
        else:
            cmd.add_argument('--dataset',default='benchmarks/coding.json')
        if name=='benchmark':
            cmd.add_argument('--conditions',help='Comma-separated condition numbers, e.g. 0,1,8')
            cmd.add_argument('--seeds',help='Comma-separated integer seeds')
    for name in ('inspect-memory','inspect-skills','inspect-agents','inspect-harness','history','rollback'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--state',default='.harness/state.db')
        if name=='rollback':
            cmd.add_argument('--version',type=int,required=True)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command.startswith('inspect') or args.command in {'history','rollback'}:
        state=State(args.state)
        try:
            if args.command=='rollback':
                state.rollback(args.version)
                result={'restored_version':state.version}
            elif args.command=='history':
                result=state.history()
            elif args.command=='inspect-harness':
                result={'version':state.version,'digest':state.fingerprint(),'state':state.export()}
            else:
                kind={'inspect-memory':'memory','inspect-skills':'skill','inspect-agents':'agent'}[args.command]
                result=state.entries(kind)
            print(json.dumps(result,indent=2))
        finally:
            state.close()
        return
    cfg=Config.load(args.config)
    if args.seed is not None:
        cfg=replace(cfg,seed=args.seed)
    names=[name for name in ('memory','skills','dynamic_agents','recursive_context','verification','self_refinement')
           if getattr(args,'disable_'+name)]
    cfg=cfg.ablate(*names)
    if args.command=='benchmark' and (args.conditions or args.seeds):
        dataset=Dataset(args.dataset)
        indices=[int(x) for x in args.conditions.split(',')] if args.conditions else [int(cfg.condition[1:])]
        seeds=[int(x) for x in args.seeds.split(',')] if args.seeds else [cfg.seed]
        if any(i not in range(9) for i in indices):
            raise ValueError('conditions must be E0–E8')
        rows=[]
        for index in indices:
            for seed in seeds:
                selected=replace(condition(index),model=cfg.model,limits=cfg.limits,seed=seed,
                                 epsilon=cfg.epsilon).ablate(*names)
                output=Path(args.output)/f'e{index}-seed{seed}'
                state=State(output/'state.db')
                try:
                    result=Experiment(selected,state,output,dataset.split('verification'),args.allow_local_execution).benchmark(dataset)
                    # 'metrics' pools every episode of the learned run (adaptation + frozen evaluation);
                    # the held-out columns isolate the paired before/after test comparison.
                    held_out=result['groups']
                    rows.append({'condition':selected.condition,'seed':seed,'transfer_gain':result['transfer_gain'],
                                 'learning_gain':result['learning_gain'],
                                 'held_out_initial_verified_rate':held_out['initial_held_out']['verified_completion_rate'],
                                 'held_out_learned_verified_rate':held_out['learned_held_out']['verified_completion_rate'],
                                 'held_out_initial_model_calls':held_out['initial_held_out']['model_calls'],
                                 'held_out_learned_model_calls':held_out['learned_held_out']['model_calls'],
                                 'held_out_model_call_delta':result['held_out_model_call_delta'],
                                 'held_out_token_delta':result['held_out_token_delta'],**result['metrics']})
                finally:
                    state.close()
        import csv
        with (Path(args.output)/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps([{k:row[k] for k in ('condition','seed','held_out_initial_verified_rate',
                    'held_out_learned_verified_rate','learning_gain','transfer_gain','held_out_model_call_delta',
                    'accepted_changes','rejected_changes')}
                    for row in rows],indent=2))
        return
    state=State(args.state)
    try:
        if args.command=='run':
            dataset=Dataset(args.verification_dataset) if args.learn else None
            exp=Experiment(cfg,state,args.output,dataset.split('verification') if dataset else [],args.allow_local_execution)
            result=exp.run_task(Task.load(args.task),learning=args.learn,phase='adaptation' if args.learn else 'task')
            exp.write_artifacts(metadata(cfg,dataset))
            print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
        else:
            dataset=Dataset(args.dataset)
            exp=Experiment(cfg,state,args.output,dataset.split('verification'),args.allow_local_execution)
            if args.command=='benchmark':
                result=exp.benchmark(dataset)
            else:
                for task in dataset.split('held_out'):
                    exp.run_task(task,phase='held_out')
                result=exp.write_artifacts(metadata(cfg,dataset))
            compact={k:result[k] for k in ('learning_gain','transfer_gain','held_out_model_call_delta',
                                          'held_out_token_delta','harness_version') if k in result}
            compact.update(condition=cfg.condition,metrics=result['metrics'],artifacts=str(Path(args.output).resolve()))
            print(json.dumps(compact,indent=2))
    finally:
        state.close()
