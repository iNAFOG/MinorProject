from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Task:
    task_id: str
    goal: str
    split: str
    project: str
    files: dict
    checks: tuple
    target: str = 'solution.py'
    function: str = 'solve'
    family: str = 'coding'
    writable_paths: tuple = ('solution.py',)
    environment: str = 'coding'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.split not in {'adaptation', 'verification', 'held_out'}:
            raise ValueError('task split must be explicit')
        for name in [*self.files, *self.writable_paths, self.target]:
            path = PurePosixPath(name.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in name or name.startswith('.'):
                raise ValueError('unsafe task path')
        if not self.checks:
            raise ValueError('task requires independent checks')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_dict(cls, data):
        return cls(**(data | {'checks': tuple(data['checks']),
                             'writable_paths': tuple(data.get('writable_paths', [data.get('target', 'solution.py')]))}))

    def visible(self):
        return {k: getattr(self, k) for k in ('task_id', 'goal', 'project', 'target', 'function', 'family')}


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    passed: int
    total: int
    errors: tuple[str, ...] = ()


class Environment(ABC):
    @abstractmethod
    def read(self, path): ...

    @abstractmethod
    def write(self, path, content): ...

    @abstractmethod
    def search(self, query): ...

    @abstractmethod
    def execute(self, argv): ...

    @abstractmethod
    def verify(self): ...
