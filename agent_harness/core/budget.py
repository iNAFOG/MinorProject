from dataclasses import dataclass
import time
from agent_harness.config import Limits


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    limits: Limits
    model_calls: int = 0
    tool_calls: int = 0
    subagent_calls: int = 0
    recursive_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0

    def __post_init__(self):
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started >= self.limits.max_execution_time:
            raise BudgetExceeded('execution time limit')

    def reserve(self, kind, amount=1):
        self.check()
        maximum = {'model_calls': self.limits.max_model_calls,
                   'tool_calls': self.limits.max_tool_calls,
                   'subagent_calls': self.limits.max_subagents}[kind]
        if getattr(self, kind) + amount > maximum:
            raise BudgetExceeded(f'{kind} limit')
        setattr(self, kind, getattr(self, kind) + amount)

    def model_allowance(self, estimated_input, desired_output):
        self.check()
        remaining = self.limits.max_tokens - self.input_tokens - self.output_tokens
        if estimated_input >= remaining:
            raise BudgetExceeded('token limit')
        return min(desired_output, remaining - estimated_input)

    def usage(self, response):
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.estimated_cost += response.estimated_cost
        if self.input_tokens + self.output_tokens > self.limits.max_tokens:
            raise BudgetExceeded('backend exceeded token allowance')
        self.check()

    def metrics(self):
        return {k: getattr(self, k) for k in ('model_calls', 'tool_calls', 'subagent_calls',
                'recursive_calls', 'input_tokens', 'output_tokens', 'estimated_cost')} | {
                'tokens': self.input_tokens + self.output_tokens,
                'latency': time.monotonic() - self.started}
