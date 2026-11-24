# Implementation boundaries

`core.agent.Orchestrator` owns one task. `Budget` is shared with context and
specialist children. `Runtime` exposes only configured operations. `ContextEngine`
supports full (bounded), summarized, retrieved and recursive contexts. Full mode
fails to preserve text beyond the explicit context budget; discarded characters
are logged rather than hidden.

A requested tool that is disabled or unknown is never executed. It consumes one tool
call and returns an error observation, so the model can correct itself. Controller-side
checks read whole files; agent reads are truncated at `max_output_chars`. Each task run
gets a fresh workspace folder.

`State` stores four registries in SQLite. Adaptive state contains JSON entries,
not Python code, model parameters, verification definitions or experiment config.
File APIs resolve paths beneath a per-task workspace and respect immutable
writable manifests. Content-addressed retrieval is limited to task-approved refs.
The local execution opt-in operates outside this application-level boundary and
assumes trusted code. No claim of protection against malicious local processes is
made. Source, SQLite and experiment data should not be exposed to those processes.

`HarnessRefiner` derives proposals from adaptation evidence. Optional model-assisted
proposals are bounded JSON and undergo the same validation. `policy.validate_change`
checks allowed mutation categories, source/provenance, task split, scope and stale
old state. Candidate entries are isolated in an in-memory SQLite clone.

`SkillValidator` executes a restricted workflow on independent fixtures.
`Verifier` performs paired fixed-backend runs under the same budgets. Specialists
must first demonstrate positive marginal contribution, even in ungated conditions.
E8 also applies the improvement threshold and rejects individual regressions.
Model assertions about quality are never an acceptance signal.

SQLite promotion writes current entries, version pointer and version snapshot in
one transaction. Exported `harness/vNNNN` folders are convenient copies; the SQLite
snapshot and digest are authoritative for rollback. Audit hashes detect accidental
tampering; they are not an externally signed adversarial integrity guarantee.

Held-out execution performs no registry writes, including usage counters.
Feedback is derived only from adaptation outcomes through proposals. Trajectories,
reports and task workspace outputs are measurements outside the mutable harness.

Model calls are inference-only. The local backend is a loopback wire adapter;
inference server installation and weights are deliberately outside the prototype.
