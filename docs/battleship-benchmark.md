# Hidden-information Battleship benchmark

Battleship is now the default game benchmark. Tic-tac-toe remains available as a small integration check. A deterministic game can still be difficult, but tic-tac-toe has a small, solved state space and little room to assess adaptation.

**Latest result, 18 October 2026:** the full Gemini comparison is complete. A strategy note passed verification, but held-out target coverage fell from 73.3% to 60%, with wins unchanged at one of three. See the [completed results and retry accounting](gemini-battleship-results-20261018.md). No learning gain is established.

Our replacement is a **custom single-player Battleship salvo variant**. It follows the hidden-ship, hit/miss search idea in [Hasbro's Battleship description](https://instructions.hasbro.com/en-us/instruction/battleship-classic-board-game-strategy-game-for-kids-ages-7-and-up). It is not the official full-size two-player game and its scores are not comparable with published Battleship benchmarks.

## What the agent plays

- A 5×5 board contains two hidden straight ships of lengths three and two.
- Ships are randomly placed horizontally or vertically, cannot overlap, and may touch. Placement is sequential: uniformly choose a legal length-three placement, then a legal non-overlapping length-two placement. This is not uniform sampling of all complete fleets.
- A move commits three distinct, untested cells simultaneously. The engine then returns hit/miss feedback. Sunk-ship cells are disclosed only after the whole ship has been hit.
- Five salvos allow at most 15 shots. An invalid batch wastes one salvo and fires no shots.
- A win requires all five occupied cells to be hit. The continuous score is targets hit divided by five; partial score is reported separately from win rate.

The fleet is sampled once. Shots are deterministic given that private fleet: this game tests uncertainty from hidden random placement, not ongoing random action outcomes. Search, inference from feedback, and deciding between investigating hits and searching elsewhere matter. The observation retains previous hits and misses, so this version does not isolate long-term memory or prove recursive-context benefits.

## Flow through the existing harness

```text
Private seeded fleet -> engine exposes public observation only
                              |
                              v
Existing ContextEngine + Orchestrator + fixed Gemini model
                              |
                         fire([a,b,c])
                              |
                     Runtime + equal budgets
                              |
                    engine validates the salvo
                              |
                  hit / miss / sunk-ship feedback
                              |
                   repeat until win or five salvos

Adaptation traces -> model proposes general strategy note
 -> existing provenance policy -> isolated SQLite candidate
 -> separate verification fleets -> accept/reject and audit
 -> freeze state -> paired unseen-fleet evaluation
```

No fleet coordinates, ship-placement seed, simulator source, or solver tool are supplied as game observations. Evaluation records retain scenario seeds for reproducibility. The engine owns legality and scoring. The model cannot edit the board, scores, or rules. The baseline and candidate are the same model with the same settings and limits; external strategy notes are the adaptive component.

The initial remote pilot used task identifiers containing scenario seeds, although the simulator implementation and execution tools were unavailable. After that pilot, model-visible labels were changed to opaque identifiers; seeds remain in private evaluation metadata and the reproducibility manifest. See the [pilot report](gemini-battleship-pilot-20261017.md) for this implementation distinction.

For a simple explanation to a professor: **we keep the AI model fixed, let it play games, ask it to extract general lessons from its practice games, test those lessons separately, and retain only lessons that pass the gate. Then we check whether it plays unseen games better.** This version tests one game-playing agent and external strategy refinement; it does not measure collaboration between specialist agents.

## Pilot protocol declared before remote execution — 17 October 2026

Use `gemini-3.1-flash-lite`, temperature zero, the existing Gemini compatible endpoint, and a fresh SQLite state. Keep all other model settings fixed. Run seed zero generates adaptation seeds 0–1, verification seeds 1000–1001, and held-out seeds 2000–2002. The generator rejects duplicate fleet layouts across these splits.

1. Evaluate the initial harness on the three held-out fleets. These traces are saved for reporting and never supplied to the refiner or verifier.
2. Play two adaptation fleets and propose one general strategy note using those traces only.
3. Evaluate the unchanged baseline and isolated candidate on the two verification fleets.
4. Keep the existing conservative gate: no individual verification-score decrease, and gain of at least 0.00001 in mean score minus `0.0001 × calls + 0.00000001 × tokens`. This tiny prototype gate does not establish statistically significant improvement.
5. Accept or reject through the existing state/version mechanism, then freeze state and evaluate the three original held-out fleets again.

Each episode permits five model calls and five tool calls. Twelve possible episodes × five calls, plus one reflection, gives a **61-request maximum**. Use a 100,000-token run cap, six-second request spacing, and no automatic provider retry or paid-model fallback. Earlier Gemini trials conservatively consumed 49 of the originally allowed 120 requests; this pilot stays below the remaining 71. Local handwritten runs do not consume API requests.

**Original attempt:** stopped after 30 attempts on provider HTTP 503 during baseline verification. Two later continuations completed the experiment using 32 additional attempts, retaining every completed outcome. One HTTP timeout required restarting an interrupted held-out game with a longer transport timeout. The final cumulative cap was 63 and actual usage was 62 attempts, within the original overall allowance. The conservative remaining overall allowance is now nine requests. The command below records the original invocation; repeating it would exceed that remainder. See the [completed report](gemini-battleship-results-20261018.md) and [original interruption report](gemini-battleship-pilot-20261017.md).

This is an integration pilot with one adaptation trajectory and three paired held-out fleets. Neither a positive nor a negative result is strong research evidence. Do not change the rules, seeds, gate, or baseline after seeing held-out results to obtain a gain.

```powershell
python scripts/game_benchmark.py --config configs/battleship-gemini.yaml --games battleship --adaptation 2 --verification 2 --held-out 3 --max-requests 61 --max-total-tokens 100000 --request-interval 6 --output runs/gemini-battleship-pilot-20261017
```

## Local evidence that strategy matters

Run `python scripts/battleship_controls.py --episodes 2000`. It plays two fixed, handwritten public-observation controllers on the same seed range, 50000–51999. These samples are independent of the remote pilot's scenario seeds. Fleet layouts can repeat in this Monte Carlo control run.

| Non-learning controller | Games | Mean targets hit / 5 | Win rate | Mean shots |
|---|---:|---:|---:|---:|
| Random untested shots | 2,000 | 3.010 | 5.1% | 14.940 |
| Hunt/target heuristic | 2,000 | 4.597 | 67.3% | 13.895 |

The paired mean normalized-score difference is +0.3174, with an exploratory paired-episode bootstrap interval of approximately [0.3072, 0.3281]. The heuristic prioritizes extensions of unresolved hits and otherwise searches a checkerboard. It cannot inspect hidden ships. This establishes **benchmark sensitivity to strategy**, not model learning or harness improvement. The Gemini baseline is Gemini's own decisions, not this weak random controller.

The full harness smoke run also completes with a handwritten controller that ignores notes: before and after scores are equal and its proposed note is rejected. This checks the orchestration, split separation, action schema, state isolation, gate, and reports without manufacturing improvement.

## Files and next experiments

- `agent_harness/games/battleship.py`: private fleet, observations, legality, scoring, reference controllers.
- `agent_harness/games/environment.py`: game task and `fire` tool.
- `agent_harness/games/benchmark.py`: existing orchestration, adaptation, verification, frozen evaluation, statistics.
- `configs/battleship-gemini.yaml`: fixed pilot model and five-call episode budget.
- `scripts/battleship_controls.py`: reproducible local reference comparison.
- `scripts/resume_battleship.py`: bounded continuation of pre-decision or frozen-evaluation checkpoints.
- `agent_harness/games/resume.py`: source integrity, exact proposal reuse, prefix reuse and failure accounting.
- `tests/test_battleship.py`: hidden-state boundaries, salvo validation, feedback, replay inputs, and complete no-learning control run.
- `runs/battleship-controls-20261017/report.md`: reference comparison.
- `runs/battleship-smoke-20261017/report.md`: offline integration evidence.

Every real experiment saves a split manifest, model/config/source metadata, prompt/response objects, episode actions, candidate decisions, state history, a report and an HTML replay. The replay shows only publicly observed hits and misses.

A stronger research experiment should predeclare several independent adaptation seeds and many verification and held-out fleets, report win rate and target coverage alongside all adaptation/verification costs, and include a notes-removed ablation. Larger boards or fleets can be a separately declared follow-up if the initial model reaches a ceiling. Memory and recursive-context ablations require additional treatments; do not infer those benefits from a short board observation.
