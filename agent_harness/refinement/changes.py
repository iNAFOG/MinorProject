from dataclasses import asdict, dataclass, field
from uuid import uuid4
from agent_harness.core.trajectory import now


@dataclass(frozen=True)
class HarnessChange:
    change_type: str
    target_id: str
    old_state: dict | None
    new_state: dict | None
    evidence: tuple[str, ...]
    reason: str
    confidence: float = 0.7
    change_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=now)

    def to_dict(self):
        return asdict(self)

    @property
    def kind(self):
        suffix = self.change_type.split('_')[-1].lower()
        return 'note' if self.change_type == 'UPDATE_PROMPT_NOTE' else suffix
