# Testing harness adaptation through games

The default benchmark is now **hidden-information Battleship**; see the [Battleship guide](battleship-benchmark.md) for current rules, protocol and results. **Tic-tac-toe** and a **key-and-door grid puzzle** remain available for integration checks. All compare a fixed model with its initial external state against the same model after verified strategy-note adaptation.

Improvement is a measured outcome, not a condition the benchmark forces. A run may improve, remain unchanged, or fail to produce an accepted update. Report all of those outcomes.

## What each game tests

| Game | Task | Measures |
|---|---|---|
| Battleship (default) | Find two randomly placed hidden ships using five three-shot salvos | Target coverage, win rate, shots, sunk ships, illegal actions, calls and tokens |
| Tic-tac-toe | Play X and O against seeded random, tactical, and exact minimax opponents | Wins/draws/losses, non-loss rate, illegal moves, calls and tokens |
| KeyDoor | Collect a key, cross the locked door, and reach the goal on a generated 7×7 or 9×9 map | Goal rate, path efficiency relative to an evaluation-only shortest path, illegal moves, calls and tokens |

Tic-tac-toe is an easy correctness check with a low ceiling: against an exact opponent, avoiding defeat is the meaningful target. The second game adds an ordered prerequisite and a longer action sequence. The map is fully visible; this version does not claim to test hidden-state exploration or spatial memory.

KeyDoor is our own small Python environment inspired by the task structure in [MiniGrid DoorKey](https://minigrid.farama.org/v2.4.0/environments/minigrid/DoorKeyEnv/). It does not use MiniGrid or claim score comparability with it. The board-game API follows the general action/state idea described in [OpenSpiel](https://openspiel.readthedocs.io/en/latest/concepts.html), without adding that dependency.

## How it uses the existing project

```text
Game task and current board
    -> existing ContextEngine and Orchestrator
    -> fixed model requests move(...)
    -> existing Runtime and Budget
    -> authoritative local game rules
    -> observation returned to the model
    -> episode outcome and trajectory recorded

Adaptation episodes only
    -> same fixed model proposes a general strategy note
    -> existing change policy validates provenance and allowed fields
    -> isolated SQLite candidate
    -> GameVerifier compares baseline and candidate on separate scenarios
    -> existing State.promote / audit / rollback
    -> frozen held-out evaluation
```

`GameEnvironment` supplies a `move` tool. The runtime still enforces budgets and records observations. Terminal game states stop the episode without needing an extra model request to declare victory. Rules, opponent policy, legal moves, rewards, and maze solutions cannot be edited through the game tool.

The first experiment deliberately studies **strategy-note adaptation**. Coding workflow skills and coding specialists are disabled because their search/read procedures do not directly represent game actions. The original memory, skill, and specialist implementations remain available to coding experiments. These small game observations do not yet establish an RLM benefit; that requires a separate large-context treatment and ablation.

## Experimental protocol

1. Declare the games, model ID, settings, split sizes, and run seed before execution.
2. Generate disjoint adaptation, verification, and held-out scenario seeds. Maze layouts must also differ across splits.
3. Measure the initial harness on the fixed held-out scenarios. Those results never enter reflection or selection.
4. Play adaptation games and archive only adaptation experience.
5. After every two adaptation episodes, ask the fixed model for a concise general strategy using recent adaptation traces. It must not store task-specific boards, coordinates, or test answers.
6. Validate the proposal, stage it in an isolated SQLite candidate, and compare candidate and baseline on the verification split under equal per-task limits.
7. Accept only a positive objective gain with **no per-scenario game-score regression** in the verification suite.
8. Freeze learned state and evaluate the same held-out scenarios again.
9. Optionally evaluate a cloned state with learned notes removed. This ablation helps test whether the external notes explain a difference.

The objective is mean game score minus `0.0001 × model_calls` and `0.00000001 × tokens`. Tic-tac-toe scores win=1, draw=0.5, loss/failure=0. KeyDoor scores reaching the goal=1 and failure=0. These fixed coefficients are prototype choices and appear in the code/report.

An invalid action is recorded and penalized by the game. The controller does not replace it with an expert move. Three illegal tic-tac-toe requests forfeit the game. Maze requests, including illegal ones, consume the step budget.

Opponent random seeds, agent sides, maps, and settings stay paired across before/after evaluations. The opponent algorithm does not learn or become weaker. Different play can lead to different board trajectories, which is a normal consequence of the agent's decisions.

In a multi-game run, notes accepted for the earlier game remain in both baseline and adapted conditions of the later game. They are explicitly labelled for their game. For independent experiments, run one game per output directory.

## What counts as evidence

Report completion and costs together. The JSON includes:

- Win/goal rate, draw/loss/failure counts, and mean game score.
- Non-loss rate for tic-tac-toe, split by opponent and player side.
- Goal rate and path efficiency for each maze size.
- Illegal-action rate, steps, model/tool calls, tokens, latency, and recursive-call counts.
- Accepted/rejected updates, version history, verification overhead, and total request attempts.
- Paired improved/regressed/tied episode counts and an exploratory episode-bootstrap interval.

The bootstrap describes one fixed suite and one adaptation trajectory. It does not establish uncertainty across independently repeated learning runs. For a research result, predeclare several run seeds, run each in a fresh directory, and report their variation. Do not select the best held-out seed or revise notes using final evaluation outcomes.

Both protocols can produce ceiling effects. If the initial real model already solves the tasks, report that result and predeclare a harder follow-up benchmark, rather than weakening the baseline to manufacture a gain.

## Gemini setup

The configuration uses `gemini-3.1-flash-lite` through [Google's compatible endpoint](https://ai.google.dev/gemini-api/docs/openai). Google's [pricing page](https://ai.google.dev/gemini-api/docs/pricing) listed free-tier input/output for that model when checked on 17 October 2026. Actual availability and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) depend on your AI Studio project. Confirm the model and free-tier status there. No paid-model fallback is implemented.

Keep the key out of chat, configuration files, and Git. From the project folder, run this script yourself in PowerShell:

```powershell
.\scripts\set_gemini_key.ps1
```

It asks for the key through a hidden prompt and stores the named `GEMINI_API_KEY` Windows user environment variable. The game runner checks that variable without printing it, including the user variable when a running app has an older process environment.

Check local setup without making an API request:

```powershell
python scripts/game_benchmark.py --config configs/games-gemini.yaml --check-setup
```

The output indicates only whether the key is available. It cannot verify billing tier or remaining provider quota.

### First real-model pilot

Start with tic-tac-toe only and a small explicit cap:

```powershell
python scripts/game_benchmark.py --config configs/games-gemini.yaml --games tic_tac_toe --adaptation 2 --verification 2 --held-out 2 --max-requests 120 --max-total-tokens 200000 --request-interval 6 --output runs/gemini-ttt-pilot
```

This is a connectivity and protocol pilot, not enough data for a research conclusion. With only two held-out scenarios it also omits the exact-opponent stratum. The larger suite below covers all opponent types and both sides.

The request cap includes gameplay, reflection, and candidate verification. The six-second interval is configurable pacing, not a guarantee that your quota permits that rate. On provider errors, including quota exhaustion, the run stops without automatic retries or a provider/model switch. An incomplete report stays labelled incomplete, and saved episode logs remain available.

The Gemini configuration uses minimal reasoning effort and a 45-second per-request timeout, held constant across the experiment. The conversation preserves native assistant tool calls, their matching tool-result IDs, and provider metadata. This avoids treating a requested game move as an ordinary completion message. History compaction retains complete tool exchanges.

If your free project offers a different supported model, select its exact ID explicitly:

```powershell
python scripts/game_benchmark.py --config configs/games-gemini.yaml --model YOUR_MODEL_ID --check-setup
```

Use the same `--model` argument for the entire subsequent experiment. Zero configured monetary rates mean monetary cost is not measured; they do not enforce free billing.

### Larger experiments after the pilot

Choose fresh output directories, confirm sufficient quota, and explicitly set suitable request/token caps:

```powershell
python scripts/game_benchmark.py --config configs/games-gemini.yaml --games tic_tac_toe --adaptation 4 --verification 6 --held-out 24 --ablate-notes --seed 0 --max-requests 1200 --max-total-tokens 2000000 --request-interval 6 --output runs/gemini-ttt-seed0

python scripts/game_benchmark.py --config configs/games-gemini.yaml --games key_door --adaptation 4 --verification 4 --held-out 12 --ablate-notes --seed 0 --max-requests 2400 --max-total-tokens 4000000 --request-interval 6 --output runs/gemini-keydoor-seed0
```

These are proposed commands and have not been run with a real model. A maze episode may require many calls, so the larger commands can take substantial time and exceed a free project's daily quota. The runner does not resume an incomplete run in place; keep it as evidence and start a fresh run when the full planned experiment fits the available quota. Provider tokens are used where supplied, otherwise the adapter falls back to an estimate.

## Offline validation already executed

The [first real Gemini tic-tac-toe pilot](gemini-game-pilot-20261017.md) is now complete. It confirms live model/tool execution, but accepted no strategy update and showed no held-out score gain. The results below describe the separate handwritten control run.

This command exercises the full pipeline without any API key or network requests:

```powershell
python scripts/game_benchmark.py --config configs/games-smoke.yaml --adaptation 4 --verification 4 --held-out 12 --ablate-notes --max-requests 3000 --max-total-tokens 5000000 --output runs/games-smoke-NEW
```

The offline controller is deliberately labelled **handwritten**, not an LLM:

- Tic-tac-toe selects the first legal cell.
- KeyDoor follows a hardcoded shortest-path controller.
- It ignores strategy notes. Consequently, reflection cannot manufacture a performance gain.

The saved run `runs/games-smoke-20261017` completed with:

| Measure | Before | After |
|---|---:|---:|
| Tic-tac-toe wins over 12 held-out episodes | 2 | 2 |
| Tic-tac-toe losses | 10 | 10 |
| Maze goals over 12 held-out episodes | 12 | 12 |
| Illegal moves in either held-out suite | 0 | 0 |
| Accepted strategy changes | — | 0 |

Four unhelpful proposals were rejected. This validates execution, comparison, rejection, frozen evaluation, and reporting. It is **not evidence of improved real-agent game performance**. The maze smoke controller has access to an exact search algorithm; a real agent never receives that solver or its answer. The oracle is used for evaluation metrics and the explicitly handwritten smoke controller only.

## Outputs for a professor demonstration

Each run creates:

| File | Purpose |
|---|---|
| `report.md` | Readable before/after table and interpretation |
| `game_summary.json` | Metrics, paired results, limits, model configuration, and experiment metadata |
| `comparison.csv` | Plot-ready before/after measurements |
| `replay.html` | Offline viewer that reconstructs recorded board moves and maze steps |
| `manifest.json` | Declared tasks and scenario splits |
| `episodes.jsonl` | Complete episode records |
| `episodes/*/trajectories.jsonl` and `objects/` | Exact requests, responses, observations, and context references |
| `changes.jsonl` | Proposed strategy notes and verification decisions |
| `state.db` and `harness_history.json` | Learned state and version history |

Open `replay.html` directly in a browser. Select an initial or learned episode and step through the recorded actions. It does not call an API. In tic-tac-toe, frames include opponent moves as well as agent moves; illegal requests are counted in metrics and exact trajectory logs. The maze viewer includes failed move attempts.

Suggested explanation: “We keep the model fixed, propose a general strategy from adaptation games, verify that strategy on a separate suite, and then compare frozen before/after performance on paired held-out scenarios. The game engine independently decides every outcome. The offline control validates the machinery. Our first real Gemini pilot completed, but its candidate did not improve verification outcomes and was rejected. Larger experiments are still needed to assess useful adaptation.”

## Code entry points

- [Game engines](../agent_harness/games/engines.py)
- [Environment adapter](../agent_harness/games/environment.py)
- [Benchmark, reflection, and game verifier](../agent_harness/games/benchmark.py)
- [Request guard and offline controller](../agent_harness/games/models.py)
- [Command-line runner](../scripts/game_benchmark.py)
- [Game tests](../tests/test_games.py)
