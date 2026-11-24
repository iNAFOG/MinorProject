# Audit and implementation plan

The initial workspace contains only `Adaptive_Agent_Synopsis_8_Page_Simplified.pages`.
There is no Git repository, source code, dependency manifest, test suite, sandbox,
model integration, evaluation infrastructure, or storage to preserve. The original
synopsis is preserved. `synopsis-extracted.txt` contains its body and table strings;
`scripts/extract_synopsis.py` reproduces extraction from the IWA archive.

The synopsis specifies H=(strategy, agents, skills, memory), fixed model weights,
bounded programmatic context, evidence-based persistence, rollback and controlled
E0–E8 comparisons. The pasted implementation brief adds concrete CLI and tests.

Implement sequential increments:
1. Model protocol, budgets, content-addressed trajectories, constrained coding
   environment, E0/E1 and offline tests.
2. SQLite storage, scoped retrieval, confidence/contradictions, E2.
3. External context selection, recursive aggregation and shared budgets, E3.
4. Fixed and candidate specialists with paired marginal evaluation, E4/E5.
5. Declarative executable workflows, isolated validation and lifecycle, E6.
6. Structured proposals, evidence policy, immutable audit and snapshots, E7.
7. Independent paired replay, regression rejection, atomic versions/rollback, E8.
8. Disjoint adaptation/verification/test splits, frozen evaluation, repeated seeds,
   ablations, metrics, reports, CLI and reproducible example.

Defaults: Python 3.12+, standard library runtime, PyYAML for config, pytest for
tests. No training APIs. Skills use a bounded tool DSL rather than unrestricted
Python. Agent-visible tools cannot reach state/audit/evaluation files through
the file and object APIs. Offline fixture checks use an AST interpreter with a
whitelist and step limit.
User clarification: simplify to a local Python harness; no Docker infrastructure.
Local command execution is opt-in for trusted repositories and is not an OS sandbox.
Offline checks continue to use the bounded AST interpreter.
The mock model is a documented deterministic fixture solver, not research evidence.
Verification tasks are separate from final held-out tasks and inaccessible to
the refiner. E7 bypasses comparative gating but retains structural/security checks.
