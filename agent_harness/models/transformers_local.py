"""In-process inference with a locally cached Hugging Face chat model.

Weights are loaded read-only from the local cache (``local_files_only``); this
backend never downloads files and never changes model parameters. Tool calls use
the model's own chat template (Qwen-style ``<tool_call>{json}</tool_call>``).
"""
import json
import re
import time
from .base import Action, BaseModel, ModelResponse
from .openai_compatible import INVALID_ARGUMENTS, parse_arguments

_LOADED = {}
_CALL = re.compile(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|$)', re.DOTALL)


def parse_tool_calls(text):
    """Split generated text into plain content and OpenAI-style tool calls.

    A block that is not valid JSON with a string ``name`` still becomes a call, named
    ``invalid_tool_call``, so the runtime records a failed action instead of the
    harness guessing what the model meant.
    """
    calls = []
    for index, match in enumerate(_CALL.finditer(text)):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            name = data['name']
            if not isinstance(name, str) or not name:
                raise ValueError('missing name')
            arguments = data.get('arguments', {})
            arguments = parse_arguments(json.dumps(arguments) if isinstance(arguments, dict) else arguments)
        except (ValueError, TypeError, KeyError):
            name, arguments = 'invalid_tool_call', {INVALID_ARGUMENTS: raw[:500]}
        calls.append({'id': f'call_{index}', 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(arguments)}})
    content = _CALL.sub('', text).strip()
    return content, calls


def to_chat(messages):
    """Harness (OpenAI wire) messages -> chat-template messages."""
    chat = []
    for message in messages:
        role = message['role']
        if role == 'assistant':
            entry = {'role': 'assistant', 'content': message.get('content') or ''}
            if message.get('tool_calls'):
                entry['tool_calls'] = [{'type': 'function', 'function': {
                    'name': c['function']['name'],
                    'arguments': parse_arguments(c['function'].get('arguments'))}}
                    for c in message['tool_calls']]
            chat.append(entry)
        elif role == 'tool':
            chat.append({'role': 'tool' if message.get('tool_call_id') else 'user', 'content': message['content']})
        else:
            chat.append({'role': role, 'content': message.get('content') or ''})
    return chat


class TransformersModel(BaseModel):
    def __init__(self, config):
        self.config = config

    def _load(self):
        cfg = self.config
        key = (cfg.model, cfg.cache_dir, cfg.device)
        if key not in _LOADED:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if device.startswith('cuda') else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.cache_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cfg.model, cache_dir=cfg.cache_dir, local_files_only=True,
                                                         dtype=dtype).to(device).eval()
            _LOADED[key] = (tokenizer, model, device)
        return _LOADED[key]

    def generate(self, messages, tools=None, context=None):
        import torch
        cfg, ctx = self.config, context or {}
        tokenizer, model, device = self._load()
        specs = [{'type': 'function', 'function': t} for t in tools] if tools else None
        encoded = tokenizer.apply_chat_template(to_chat(messages), tools=specs, add_generation_prompt=True,
                                                return_tensors='pt', return_dict=True).to(device)
        prompt_tokens = encoded['input_ids'].shape[1]
        limit = max(1, min(cfg.max_output_tokens, int(ctx.get('max_output_tokens', cfg.max_output_tokens))))
        options = {'max_new_tokens': limit, 'pad_token_id': tokenizer.eos_token_id}
        if cfg.temperature > 0:
            options.update(do_sample=True, temperature=cfg.temperature)
        else:  # Greedy decoding: identical prompts give identical outputs.
            options.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        started = time.monotonic()
        with torch.no_grad():
            output = model.generate(**encoded, **options)
        generated = output[0, prompt_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        content, calls = parse_tool_calls(text) if tools else (text.strip(), [])
        actions = tuple(Action(c['function']['name'], json.loads(c['function']['arguments'])) for c in calls)
        finish = 'length' if len(generated) >= limit else ('tool_calls' if calls else 'stop')
        return ModelResponse(content, actions, prompt_tokens, int(len(generated)), time.monotonic() - started, 0.0,
                             f'transformers:{cfg.model}', tuple(calls), finish)
