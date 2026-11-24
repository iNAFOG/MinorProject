import json
import hashlib
from agent_harness.environments.base import Environment, Task, VerificationResult
from .engines import make_game


def game_task(game, split, seed, *, opponent='tactical', agent_mark='X', size=7, max_steps=48):
    metadata={'game':game,'seed':seed}
    if game=='tic_tac_toe':
        metadata.update(opponent=opponent,agent_mark=agent_mark)
    elif game=='battleship':
        metadata.update(size=5,max_steps=5)
    else:
        metadata.update(size=size,max_steps=max_steps)
    engine=make_game(metadata)
    identity=f'{game}-{seed}-{opponent}-{agent_mark}' if game=='tic_tac_toe' else f'{game}-{seed}-{metadata["size"]}'
    project=f'{game}-scenario-{seed}'
    if game=='battleship':
        # Seeds belong in the reproducibility manifest, not in the model's task label.
        identity='battleship-'+hashlib.sha256(f'battleship:{seed}'.encode()).hexdigest()[:20]
        project=identity
    return Task(task_id=identity,goal=f'Play {game} legally and maximize the game outcome.',split=split,
                project=project,family=f'game:{game}',environment='game',
                files={'rules.txt':engine.rules,'observation.json':json.dumps(engine.observe(),sort_keys=True)},
                checks=({'kind':'game_outcome'},),writable_paths=(),target='observation.json',metadata=metadata)


class GameEnvironment(Environment):
    def __init__(self,task):
        self.task=task
        self.engine=make_game(task.metadata)

    @property
    def terminal(self):
        return self.engine.terminal

    def tool_definitions(self):
        if self.engine.name=='battleship':
            return {'fire':{'cells':{'type':'array','items':{'type':'integer','minimum':0,'maximum':24},
                                    'minItems':3,'maxItems':3}}}
        if self.engine.name=='tic_tac_toe':
            return {'move':{'cell':{'type':'integer','minimum':0,'maximum':8}}}
        return {'move':{'direction':{'type':'string','enum':['up','down','left','right']}}}

    def call_tool(self,name,arguments):
        if self.engine.name=='battleship':
            if name!='fire':
                raise PermissionError('unknown game action')
            return self.engine.step(arguments['cells'] if isinstance(arguments,dict) and set(arguments)=={'cells'} else None)
        if name!='move':
            raise PermissionError('unknown game action')
        expected={'cell'} if self.engine.name=='tic_tac_toe' else {'direction'}
        if not isinstance(arguments,dict) or set(arguments)!=expected:
            # Malformed tool schemas remain failed agent actions, not repaired oracle moves.
            return self.engine.step(None)
        value=next(iter(arguments.values()))
        if self.engine.name=='key_door' and not isinstance(value,str):
            value=None
        return self.engine.step(value)

    def verify(self):
        metrics=self.engine.metrics()
        success=self.terminal and (metrics['outcome'] in {'win','draw','goal'})
        return VerificationResult(success,int(success),1,() if success else (metrics['outcome'],))

    def read(self,path):
        if path=='observation.json':
            return json.dumps(self.engine.observe())
        if path=='rules.txt':
            return self.engine.rules
        raise PermissionError('game exposes only rules and observation')

    def write(self,*args,**kwargs):
        raise PermissionError('game rules and board cannot be overwritten')

    def search(self,*args,**kwargs):
        raise PermissionError('file search is unavailable in games')

    def execute(self,*args,**kwargs):
        raise PermissionError('process execution is unavailable in games')
