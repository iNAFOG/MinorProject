from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str = ''
    actions: tuple[Action, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0
    estimated_cost: float = 0
    backend: str = ''
    tool_calls: tuple[dict, ...] = ()
    finish_reason: str = ''


def token_estimate(value):
    return max(1, (len(json.dumps(value, ensure_ascii=False)) + 2) // 3)


class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages, tools=None, context=None) -> ModelResponse:
        """Inference only. context carries a per-call output budget and timeout."""
