# Local validation results

> **Re-run on 22 November 2026 (current code, Python 3.12.1):** completion, model calls (10 → 8 held-out), tool calls, and decisions (1 accepted, 8 rejected, 154 verification calls) match the original. Estimated tokens are higher: held-out 12,590 → 10,515 (−16.5%), verification 197,441, recursive task 13,708. The runs below predate the native tool-call message format, which adds characters to every prompt. Artifacts and figures are in `runs/audit-20261122/`; details are in the [audit](audit-20261122.md).

The offline example was executed on Python 3.12.13 on Windows. The mock foundation
policy remained fixed. The benchmark used three adaptation tasks, two independent
verification tasks and two held-out tasks in new projects.

E8 accepted one parameterized search/read skill and rejected eight proposals
without sufficient measured marginal benefit. Held-out verified completion was
100% before and after learning. Held-out model calls changed from 10 to 8; task
inference tokens changed from 11,849 to 10,071. Transfer capability gain was zero.
The gate spent 154 additional verification model calls. These figures demonstrate
functionality and the cost tradeoff, not a statistical LLM research result.

Artifacts are in `runs/demo/`; the comparison across E0–E8 and seeds 0/1 is in
`runs/comparison/comparison.csv`. Generated figures are in `runs/demo/figures/`.
These directories are ignored by Git and can be recreated with the README commands.

The separate large-context task completed with three recursive child calls, eight
total model calls and 13,336 tokens. Its logs are in `runs/recursive/`.

Usage-counter-only proposals can be rejected in E8 because they have no positive
performance gain. Episode-level skill reuse remains recorded in trajectories and
metrics. Counter history is not silently mutated during frozen evaluation.

For these original coding results, external inference backends were tested against
a loopback wire-protocol server, without a real-model coding experiment. A separate
[real Gemini game pilot](gemini-game-pilot-20261017.md)
was completed on 17 October 2026. Model-assisted arbitrary-role proposals
were tested using a scripted response and are subject to independent evaluation.
