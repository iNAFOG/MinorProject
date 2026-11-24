# Adaptive Agent Harness

A local Python research prototype for **recursive context processing and
self-improvement around a fixed model**. The model makes inference calls;
the harness learns scoped memory, executable workflows, specialist definitions
and strategy notes. No model training, Docker, or agent framework is required.

The original Pages synopsis is preserved. Its extracted text and repository audit
are in `docs/`. This implementation follows the synopsis with the user's requested
simplification to a local harness.

## Start here

Requires Python 3.12 or newer. From this directory:

```powershell
python -m pip install -e ".[test]"
python -m pytest
python scripts/reproduce.py --output runs/my-demo
```

The last command runs adaptation, independent candidate verification, and frozen
held-out evaluation with the offline model. It creates a SQLite harness and
experiment artifacts in `runs/my-demo`. A completed example is already saved in
`runs/demo`. Use a new output directory for each experiment.

Inspect or restore what was learned:

```powershell
python -m agent_harness inspect-skills --state runs/my-demo/state.db
python -m agent_harness inspect-harness --state runs/my-demo/state.db
python -m agent_harness history --state runs/my-demo/state.db
python -m agent_harness rollback --state runs/my-demo/state.db --version 0
```

Run one task without adapting, or explicitly learn from an adaptation task:

```powershell
python -m agent_harness run --task benchmarks/example_task.json --output runs/task
python -m agent_harness run --task benchmarks/example_task.json --learn --output runs/adapt
```

`run` uses `.harness/state.db` by default. Learning requires a separate verification
split; the supplied benchmark provides it. The installed `agent-harness` command
accepts the same arguments as `python -m agent_harness`. Repeating `run` with the
same `--output` is safe: each task gets a new workspace folder, so files left by an
earlier run are never counted as the new run's output.

To exercise recursive processing of an input larger than the active context:

```powershell
python -m agent_harness run --task benchmarks/recursive_task.json --state runs/recursive/state.db --output runs/recursive
```

This supplied example runs three child calls before completing the coding task.

## How it works

```text
fixed model → orchestrator → bounded tools / recursive context
                    ↕
           SQLite memory, skills, agents, strategy
                    ↓
         adaptation trajectories → change proposals
                    ↓
       isolated candidate → independent paired verification
                    ↓
            accept + version / reject + audit
```

**Recursive context (RLM).** This is a harness mechanism, not a separately trained
model. Files stay in an external content-addressed store. Relevant regions are
selected or partitioned; child model calls receive only one subset. Child outputs
are combined, recursively summarized when necessary, then used by the main agent.
The model can retrieve bounded ranges by object reference. Calls share token,
depth, sub-agent, tool and time limits. Small inputs do not need extra child calls.

**Memory.** Working, episodic, semantic and procedural entries have scope,
provenance, confidence and versions. Task facts stay task-local, project facts stay
project-local, and global knowledge needs evidence from different projects.
Contradictions mark entries for reverification. Retrieval does not silently modify
the harness during evaluation.

**Skills.** Repeated verified search/read sequences become candidate tool workflows.
The executable DSL supports bounded `search`, `read` and `verify` steps with a
`$path` parameter. It stores a procedure, never a benchmark answer or source patch.
Independent replay must validate a candidate before activation. Usage feedback,
degradation and pruning are supported.

**Specialists.** E4 uses a fixed repository reader. Later conditions can propose
new specialist roles from repeated experience. Isolated paired runs must show a
positive marginal contribution before persistence. Roles can be revised and
retired. A model-assisted refiner can propose additional roles; it is not limited
to the offline example's source inspector.

**Self-improvement.** The refiner extracts repeated procedures and failure/reuse
evidence from adaptation trajectories. With a real backend, it also asks the same
fixed model for structured changes. A controller validates provenance, scope and
allowed fields. In E8, it compares the baseline and candidate on a separate
verification suite under equal task budgets, rejecting any per-task regression.
The declared objective is verified completion minus model-call, token and action
penalties. Accepted changes get atomic SQLite snapshots; rollback restores a
previous snapshot and appends an audit event.

## Connect a fixed model

The default `MockModel` is a deterministic fixture policy for sum, count and maximum
functions. It deliberately makes inspection tool calls, allowing offline tests to
measure reuse and reduced calls. It is **not evidence of an LLM capability gain**.

Copy `configs/e8.yaml` and change its `model` section to an already-running local
OpenAI-compatible inference server:

```yaml
model:
  backend: local
  model: your-server-model-id
  base_url: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_MODEL_API_KEY
  temperature: 0
  max_output_tokens: 1500
  input_cost_per_million: 0
  output_cost_per_million: 0
```

For a remote compatible server use `backend: openai`, its `/v1` base URL, and an
environment variable for credentials. Keep the same model ID and settings across
comparisons. The adapter supports function calls, server-reported usage, latency,
timeouts and configured cost estimates. The local wire protocol is tested with a
loopback test server, including servers that send `null` for absent fields. A
malformed tool call becomes a failed action that the model can see; it does not
abort the episode, and the harness never repairs it. Real Gemini tic-tac-toe
(17 October 2026) and Battleship (18 October 2026) pilots have been completed. The
coding results remain offline mock results.

Run the copied configuration with `--config configs/my-model.yaml`. The harness
does not start a model server, download weights, or modify model parameters.

## Coding tasks

`benchmarks/example_task.json` shows the task format. Each task declares its ID,
goal, split, project, input files, writable paths and independent checks. The agent
sees the goal and source context; expected outputs and check definitions remain
controller-side. Work occurs in an isolated task directory with an explicit write
manifest. Multiple source files and editable paths are supported.

Offline checks use a bounded Python AST interpreter supporting simple functions,
loops, arithmetic and selected built-ins. It never executes arbitrary source with
Python `eval` or `exec`. Each test case gets its own step budget. Checks read the
whole file, while agent `read` calls stay limited to `max_output_chars`.
Comprehensions, `while`, attribute access and string `%` formatting are rejected. For trusted real repositories, use a `command` check with
an argv list and enable local process execution:

```json
{"kind": "command", "argv": ["python", "-m", "pytest", "-q"]}
```

```powershell
python -m agent_harness run --task my-task.json --allow-local-execution --output runs/repo
```

Local process execution has a timeout and bounded captured output. It is a
convenience for trusted code, **not an operating-system sandbox**. It is disabled
by default. The default AST checks and file APIs need no external infrastructure.

## Experiments and ablations

All conditions use the same orchestrator. E3 and later retain recursive context;
only large contexts trigger recursive calls.

| Config | Added mechanism |
|---|---|
| E0 | Direct model output, independently measured |
| E1 | Tools and in-task verification |
| E2 | Scoped persistent memory |
| E3 | Recursive context and temporary child agents |
| E4 | Fixed specialist registry |
| E5 | Evaluated dynamic specialists |
| E6 | Validated reusable skills |
| E7 | Continual strategy/refinement, comparative gate disabled |
| E8 | Comparative verification gate and rollback |

All persistence retains structural checks and provenance. E7 bypasses the
comparative improvement gate; executable skills and dynamic specialists still
require their component validation. Snapshots remain available in lower conditions
for audit/recovery even when versioning is not the experimental treatment.

```powershell
python -m agent_harness benchmark --conditions 0,1,2,3,4,5,6,7,8 --seeds 0,1 --output runs/comparison
python -m agent_harness benchmark --disable-skills --state runs/no-skills/state.db --output runs/no-skills
python -m agent_harness evaluate --state runs/demo/state.db --output runs/held-out
```

With `--conditions`/`--seeds`, the top-level `comparison.csv` has one row per run.
Its `held_out_*` columns give the paired before/after test comparison. Its other
metric columns pool every episode of the learned run, adaptation included. Each
`eN-seedS/comparison.csv` splits the four phases.

Other ablations: `--disable-memory`, `--disable-dynamic-agents`,
`--disable-recursive-context`, `--disable-verification`, `--disable-self-refinement`.
Disabling in-task/gated verification retains independent outcome measurement.
`evaluate` freezes learned state and never invokes refinement.

Datasets explicitly separate `adaptation`, `verification` and `held_out` tasks.
Only adaptation can drive updates. Candidate selection uses verification, never
final held-out results. Duplicate IDs and identical cross-split fixtures are
rejected. Each seed deterministically varies adaptation order; deterministic
mock repeats do not constitute statistical evidence about a stochastic LLM.

## Outputs and reproduction

Every experiment writes `experiment_summary.json`, `metrics.json`,
`trajectories.jsonl`, `harness_history.json` and `comparison.csv`. Proposal decisions
are in `changes.jsonl`; immutable prompt/context objects are stored once under
`objects/`. The SQLite DB includes snapshots and a hash-chained audit log.

Adaptation experience is archived separately in SQLite, so distinct `run --learn`
calls can induce reusable procedures across process restarts. The archive alone
does not count as learning; only accepted external-harness changes do. Held-out
and verification trajectories never enter this learning archive.

Metrics include success and verified completion, action/token efficiency,
recovery, intervention, skill reuse, learning and transfer gains, costs, calls,
latency, accepted/rejected updates and failures. Candidate-verification overhead
is reported separately from task inference. Config, source and dataset hashes,
seed/order, model settings, Python/platform metadata and Git commit (when present)
are saved. Rollback events remain in the SQLite audit after a report is produced.

Generate figures from stored artifacts:

```powershell
python -m pip install -e ".[plots]"
python scripts/report.py --input runs/demo --output runs/demo/figures
```

The offline example should show zero capability transfer gain because the mock
already solves all fixtures, with fewer later model calls after a skill is accepted.
Real long-horizon claims require larger repository tasks, a real fixed model,
several seeds and analysis of negative transfer and total adaptation cost.

## Game-agent experiments

The main game benchmark is now **hidden-information Battleship**, a compact
single-player salvo variant with randomly placed fleets and hit/miss feedback.
See [the Battleship benchmark](docs/battleship-benchmark.md) for its rules, bounded
Gemini pilot and 2,000-game reference comparison. Tic-tac-toe and the generated
key-and-door grid puzzle remain optional integration checks.
The [completed Gemini Battleship experiment](docs/gemini-battleship-results-20261018.md)
accepted a strategy after verification improved from 50% to 80% target coverage.
Held-out coverage then fell from 73.3% to 60%; wins stayed at one of three.
The pipeline works, but this small pilot found no playing-strength improvement.
The game pilot tests verified strategy-note adaptation with a fixed model,
separate adaptation/verification/held-out scenarios, frozen evaluation and an
optional notes-removal ablation. It reuses the orchestrator, runtime, budgets,
SQLite state and change policy. No game solver is exposed to a real agent.

See [the game benchmark guide](docs/game-benchmarks.md) for Gemini AI Studio setup,
bounded pilot commands, metrics, limitations and replayable results. The saved
offline game run is plumbing validation with a handwritten controller, not an LLM
performance result. Real-model improvement must be measured rather than assumed.

The [first real Gemini pilot](docs/gemini-game-pilot-20261017.md) used 40 requests.
Both held-out evaluations recorded one draw and one loss. The strategy candidate
was rejected because it added tokens without improving verification outcomes.
This establishes real-model integration, not an improvement in playing strength.

## Audit, 22 November 2026

A full audit fixed seven defects. Two made verification fail correct code, one
counted leftover files from earlier runs as task output, and two let the AST sandbox
allocate hundreds of megabytes. Afterwards, every offline experiment was re-run.
Outcomes, model and tool calls, and accept/reject decisions matched the saved runs
exactly. Estimated token counts are 1–12% higher than in runs saved before
17 October, because the message format switched to native tool calls then. Current
results are in `runs/audit-20261122/`; see [the audit report](docs/audit-20261122.md).

## Learning-gain search with a local model, 22 November 2026

With a fixed local `Qwen2.5-1.5B-Instruct` (`backend: transformers`, cached weights, no
network), a **learned Battleship policy skill** raised frozen held-out target coverage from
0.509 to 0.767 (+0.258, 95% CI [0.244, 0.271]) on 1,000 unseen fleet layouts. Wins rose
from 0% to 19.6%. A random-proposal control under the same gate did as well or better,
averaging 0.800. The gain therefore comes from the verified skill loop, not from
better-than-chance model reflection. Strategy notes had no effect on this model, and its
coding baseline completed nothing, so neither produced a gain. See
[the report](docs/policy-learning-20261122.md); reproduce with
`python scripts/policy_learning.py --output runs/new-dir`.

A second, self-designed game, **Dice Bank** (push-your-luck dice, with an exact optimum
of banking at 20 points), repeated the test. Once the parser was fixed to read the
model's wrapped replies, all three model runs improved held-out score: +14.2, +30.4
and +33.2 points from 48.3. One run reached the exact optimum. Every valid model
proposal improved verification (7/7), against 6/18 for random proposals and 8/18 for
hill-climbing. Both controls still reached about 81 points, because the model stopped
producing parseable replies after 2–3 rounds. See
[the Dice Bank report](docs/dicebank-learning-20261122.md); reproduce with
`python scripts/dicebank_learning.py --output runs/new-dir`.

## Extend the prototype

Implement `BaseModel.generate()` to add a model backend and register it in
`models.build_model`. Implement the `Environment` interface to add a research or
interactive environment, with its own controller-side checks and task factory.
Coding and local game environments are implemented; other domains remain extension
interfaces. See `docs/architecture.md` and `docs/experiments.md` for the boundaries.
