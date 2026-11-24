"""Regenerate research figures from immutable experiment summary data."""
from pathlib import Path
import argparse
import json


def report(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'experiment_summary.json').read_text(encoding='utf-8'))
    episodes=summary['episodes']
    if not episodes:
        raise ValueError('no episodes to plot')
    x=list(range(1,len(episodes)+1))
    success,verified,failures=[],[],[]
    for i in x:
        success.append(sum(r['success'] for r in episodes[:i])/i)
        verified.append(sum(r['verified'] for r in episodes[:i])/i)
        failures.append(sum(not r['verified'] for r in episodes[:i]))
    series=[('completion','Cumulative completion',{'success':success,'verified':verified}),
            ('tokens','Task inference tokens',{'tokens':[r['tokens'] for r in episodes]}),
            ('calls','Task inference model calls',{'calls':[r['model_calls'] for r in episodes]}),
            ('cost_adjusted','Verified outcomes per 1,000 tokens',{
                'efficiency':[1000*int(r['verified'])/max(1,r['tokens']) for r in episodes]}),
            ('skill_reuse','Verified tasks using learned skills',{'reuse':[int(r['verified'] and r['skill_reuse']) for r in episodes]}),
            ('changes','Harness decisions',{'accepted':[r['accepted_changes'] for r in episodes],
                                          'rejected':[r['rejected_changes'] for r in episodes]}),
            ('failures','Cumulative failures',{'failures':failures}),
            ('rollback','Rollback events recorded in episodes',{'rollbacks':[r['rollback_count'] for r in episodes]})]
    paths=[]
    for name,title,values in series:
        fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
        for label,y in values.items():
            ax.plot(x,y,marker='o',label=label)
        ax.set(title=title,xlabel='Episode')
        ax.grid(alpha=0.25)
        ax.legend()
        path=output/f'{name}.svg'
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    if 'transfer_gain' in summary:
        fig,ax=plt.subplots(figsize=(5,3.5),layout='constrained')
        ax.bar(['Learning gain','Transfer gain'],[summary['learning_gain'],summary['transfer_gain']])
        ax.axhline(0,color='black',linewidth=0.8)
        ax.set(ylabel='Verified completion rate difference',title='Paired capability gains')
        fig.savefig(output/'gains.svg')
        plt.close(fig)
        paths.append(str(output/'gains.svg'))
    (output/'figure_manifest.json').write_text(json.dumps({'source_config_digest':summary['config_digest'],
        'source_benchmark_digest':summary['benchmark_digest'],'figures':paths},indent=2),encoding='utf-8')
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print('\n'.join(report(args.input,args.output)))
