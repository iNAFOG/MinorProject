from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import yaml


@dataclass(frozen=True)
class Limits:
    max_depth: int = 2
    max_model_calls: int = 24
    max_subagents: int = 6
    max_tokens: int = 24000
    max_execution_time: float = 60
    max_tool_calls: int = 40
    max_context_chars: int = 6000
    max_output_chars: int = 12000

    def __post_init__(self):
        if any(v <= 0 for v in asdict(self).values()):
            raise ValueError('limits must be positive')


@dataclass(frozen=True)
class ModelConfig:
    backend: str = 'mock'
    model: str = 'fixture-policy-v1'
    base_url: str = 'https://api.openai.com/v1'
    api_key_env: str = 'OPENAI_API_KEY'
    temperature: float = 0
    max_output_tokens: int = 1000
    input_cost_per_million: float = 0
    output_cost_per_million: float = 0
    request_timeout: float = 45
    reasoning_effort: str | None = None
    # Local Hugging Face backend only: read-only weight cache and torch device.
    cache_dir: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class Config:
    condition: str = 'E8'
    tools: bool = True
    memory: bool = True
    temporary_agents: bool = False
    recursive_context: bool = False
    fixed_agents: bool = False
    dynamic_agents: bool = True
    skills: bool = True
    self_refinement: bool = True
    verification_gate: bool = True
    basic_verification: bool = True
    versioning: bool = True
    rollback: bool = True
    context_mode: str = 'retrieved'
    seed: int = 0
    epsilon: float = 0.00001
    limits: Limits = Limits()
    model: ModelConfig = ModelConfig()

    def __post_init__(self):
        if self.context_mode not in {'full', 'summarized', 'retrieved', 'recursive'}:
            raise ValueError('invalid context mode')
        if self.epsilon <= 0:
            raise ValueError('epsilon must be positive')

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        data['limits'] = Limits(**data.get('limits', {}))
        data['model'] = ModelConfig(**data.get('model', {}))
        return cls(**data)

    def ablate(self, *names):
        allowed = {'memory', 'skills', 'dynamic_agents', 'recursive_context',
                   'verification', 'self_refinement'}
        if set(names) - allowed:
            raise ValueError('unknown ablation')
        updates = {name: False for name in names if name != 'verification'}
        if 'verification' in names:
            updates.update(basic_verification=False, verification_gate=False)
        return replace(self, **updates)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def condition(index):
    return Config(condition=f'E{index}', tools=index >= 1, memory=index >= 2,
                  temporary_agents=index == 3, recursive_context=index >= 3,
                  fixed_agents=index == 4, dynamic_agents=index >= 5,
                  skills=index >= 6, self_refinement=index >= 7,
                  verification_gate=index == 8, basic_verification=index >= 1,
                  versioning=index == 8, rollback=index == 8,
                  context_mode='recursive' if index >= 3 else 'retrieved')
