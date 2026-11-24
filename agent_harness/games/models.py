"""Offline plumbing controller and a run-wide request guard for real inference."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from agent_harness.models.base import Action, BaseModel, ModelResponse, token_estimate
from .engines import shortest_path
from .battleship import reference_salvo


def load_key_from_environment(name):
    """Read only the explicitly configured key variable, never credential files.

    A running Windows app may have an old process environment after the user sets
    a User variable. Read that named variable directly without exposing its value.
    """
    if os.environ.get(name):
        return True
    if os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
                value,_=winreg.QueryValueEx(key,name)
            if isinstance(value,str) and value.strip():
                os.environ[name]=value
                return True
        except OSError:
            pass
    return False


class RunStopped(RuntimeError):
    pass


class RequestGuard:
    def __init__(self,max_requests=120,max_tokens=200000,interval=0):
        if max_requests < 1 or max_tokens < 1 or interval < 0:
            raise ValueError('positive request/token limits and nonnegative interval required')
        self.max_requests,self.max_tokens,self.interval=max_requests,max_tokens,interval
        self.requests,self.tokens,self.last=0,0,0.0
        self.fatal_reason=None

    def stop(self,reason):
        self.fatal_reason=reason
        raise RunStopped(reason)

    def generate(self,model,messages,tools,context):
        if self.fatal_reason:
            raise RunStopped(self.fatal_reason)
        estimate=token_estimate(messages)+token_estimate(tools)
        if self.requests >= self.max_requests:
            self.stop('run-wide request cap reached')
        if self.tokens+estimate >= self.max_tokens:
            self.stop('run-wide token cap reached')
        delay=max(0,self.interval-(time.monotonic()-self.last))
        if delay:
            time.sleep(delay)
        self.requests+=1
        self.last=time.monotonic()
        context=dict(context or {})
        context['max_output_tokens']=min(context.get('max_output_tokens',1000),self.max_tokens-self.tokens-estimate)
        try:
            result=model.generate(messages,tools,context)
        except HTTPError as exc:
            self.stop(f'provider HTTP {exc.code}; no automatic retry or provider fallback')
        except (URLError,TimeoutError) as exc:
            self.stop(f'provider network failure ({type(exc).__name__}); run stopped')
        self.tokens+=result.input_tokens+result.output_tokens
        if self.tokens > self.max_tokens:
            self.stop('provider-reported usage exceeded the remaining run token allowance')
        return result


class GuardedModel(BaseModel):
    def __init__(self,model,guard):
        self.model,self.guard=model,guard

    def generate(self,messages,tools=None,context=None):
        return self.guard.generate(self.model,messages,tools,context or {})


class GameSmokeModel(BaseModel):
    """Handwritten controller, NOT an LLM, and deliberately ignores learned notes.

    Tic-tac-toe chooses the first legal cell. KeyDoor uses an exact BFS controller.
    Battleship uses a public-observation hunt/target heuristic.
    This tests both failing and successful paths without manufacturing learning gains.
    """
    def generate(self,messages,tools=None,context=None):
        context=context or {}
        if context.get('role')=='game_refiner':
            content=json.dumps({'strategy':'Check legal actions and plan a route before choosing the next move.'})
            actions=()
        elif context.get('role')=='context':
            content=str(context.get('selected',''))[:800]
            actions=()
        else:
            observations=[json.loads(m['content']) for m in messages if m['role']=='tool']
            if observations:
                obs=observations[-1]['result']
            else:
                system=json.loads(messages[0]['content'].split('\n',1)[1])
                line=next(line for line in system['context'].splitlines() if line.startswith('observation.json: '))
                obs=json.loads(line[len('observation.json: '):])
            if obs['terminal']:
                content,actions='Game complete.',()
            elif obs['game']=='tic_tac_toe':
                content,actions='',(Action('move',{'cell':obs['legal_cells'][0]}),)
            elif obs['game']=='battleship':
                content,actions='',(Action('fire',{'cells':reference_salvo(obs)}),)
            else:
                path=shortest_path(obs['grid'],obs['position'],obs['has_key'])
                content,actions='',(Action('move',{'direction':path[0]}),) if path else ()
        return ModelResponse(content,actions,token_estimate(messages)+token_estimate(tools),
                             token_estimate({'content':content,'actions':[a.__dict__ for a in actions]}),
                             backend='game-smoke-v1:handwritten-no-learning')
