import json
import os
import time
from urllib.request import Request, urlopen
from .base import Action, BaseModel, ModelResponse, token_estimate


INVALID_ARGUMENTS = 'invalid_tool_arguments'


def parse_arguments(raw):
    """Decode tool-call arguments without letting one malformed call abort the episode.

    Empty arguments mean a no-argument call. Malformed or non-object JSON is passed
    through under a reserved key, so the runtime/game records a failed action and the
    model sees the error, rather than the harness silently repairing the request.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {INVALID_ARGUMENTS: str(raw)[:500]}
    return value if isinstance(value, dict) else {INVALID_ARGUMENTS: str(raw)[:500]}


class OpenAICompatibleModel(BaseModel):
    """Chat Completions wire adapter; credentials never enter trajectories."""
    def __init__(self, config):
        self.config = config

    def generate(self, messages, tools=None, context=None):
        cfg, ctx = self.config, context or {}
        # Preserve native call/response pairing, including provider-specific metadata.
        # Legacy standalone observations without IDs remain ordinary user messages.
        wire_messages = [dict(m, role='user') if m['role'] == 'tool' and not m.get('tool_call_id') else m for m in messages]
        payload = {'model': cfg.model, 'messages': wire_messages,
                   'temperature': cfg.temperature,
                   'max_tokens': min(cfg.max_output_tokens, ctx.get('max_output_tokens', cfg.max_output_tokens))}
        if tools:
            payload['tools'] = [{'type': 'function', 'function': t} for t in tools]
        if cfg.reasoning_effort is not None:
            payload['reasoning_effort'] = cfg.reasoning_effort
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get(cfg.api_key_env)
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = Request(cfg.base_url.rstrip('/') + '/chat/completions',
                          data=json.dumps(payload).encode(), headers=headers)
        started = time.monotonic()
        with urlopen(request, timeout=max(0.01, min(cfg.request_timeout,ctx.get('timeout', cfg.request_timeout)))) as response:
            data = json.load(response)
        message = data['choices'][0]['message']
        # Some servers send null instead of omitting fields; treat both as absent.
        calls = message.get('tool_calls') or []
        actions = tuple(Action(c['function']['name'], parse_arguments(c['function'].get('arguments'))) for c in calls)
        usage = data.get('usage') or {}
        inp = usage.get('prompt_tokens')
        out = usage.get('completion_tokens')
        inp = token_estimate(wire_messages) if inp is None else inp
        out = token_estimate(message) if out is None else out
        cost = (inp * cfg.input_cost_per_million + out * cfg.output_cost_per_million) / 1e6
        return ModelResponse(message.get('content') or '', actions, inp, out,
                             time.monotonic() - started, cost, f'{cfg.backend}:{cfg.model}',
                             tuple(calls),data['choices'][0].get('finish_reason',''))
