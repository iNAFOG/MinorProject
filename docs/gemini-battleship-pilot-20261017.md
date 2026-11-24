# Gemini Battleship pilot — incomplete

Historical first-attempt report. The later retry completed the comparison on 18 October 2026; see the [completed experiment](gemini-battleship-results-20261018.md). Its held-out coverage fell from 73.3% to 60%, with no win-rate increase.

The new hidden-information Battleship benchmark works with the real `gemini-3.1-flash-lite` model through the existing harness. The remote experiment stopped on **HTTP 503 during baseline verification**. It is not a completed before/after comparison and supplies **no evidence of a learning gain**.

The run identifier uses its UTC start date, 17 October 2026. It finished after midnight on 18 October in India.

## What completed

| Phase | Completed games | Wins | Targets hit | Model calls | Tokens |
|---|---:|---:|---:|---:|---:|
| Initial held-out evaluation | 3 | 1 | 11 / 15 | 14 | 19,202 |
| Adaptation practice | 2 | 0 | 5 / 10 | 10 | 14,068 |
| Baseline verification, first fleet only | 1 | 1 | 5 / 5 | 4 | 5,071 |
| Strategy reflection | — | — | — | 1 | 4,145 |
| Failed provider request on second verification fleet | 0 | — | — | 1 attempted | No response usage available |

There were **30 request attempts**, **42,486 reported tokens**, six completed games, and no illegal shots in those completed games. The initial held-out score was 11/15 = **0.7333**, with a **33.3% win rate**. This is an initial baseline, not an improvement result.

The model proposed a general strategy using only the two adaptation games: checkerboard search, investigating neighbors of unresolved hits, following an inferred ship axis, and updating priorities after feedback. The original response is preserved under `refinement/objects/`; a readable copy is saved as `proposed-strategy.json`. These are model-generated suggestions, not verified claims about optimal play.

The existing policy accepted the proposal's structure and adaptation provenance for staging. The baseline/candidate comparison did not complete: the second baseline verification request returned 503 before the candidate played any verification games. Consequently **zero changes were accepted and zero were rejected**, persistent harness version remains **0**, and no final learned held-out evaluation exists. The generic phase name `candidate_verification` in the episode log covers both sides of the comparison; the two saved records in that phase are baseline attempts.

## What the local experiments establish

The separate 2,000-fleet reference comparison found 5.1% wins for random firing and 67.3% for a fixed hunt/target heuristic. Those are **handwritten non-learning controls**, on a different seed range. They show that the game rewards strategy. They do not establish Gemini or harness improvement, and they cannot be directly compared with the three-fleet Gemini baseline as though the samples were matched.

The offline full-harness control completes adaptation, staging, verification, frozen before/after evaluation and notes removal. Its handwritten controller ignores notes, produces equal before/after results, and accepts no changes.

## Scope and reproducibility

The declared rules, seed ranges, model settings, gate, call limits and manifest were not changed in response to this pilot's scores. No provider retry or model fallback was used. Earlier trials conservatively consumed 49 requests; this run consumed 30, leaving **41 of the original 120-request allowance**. A fresh run with the original 61-request maximum would exceed that remaining allowance.

The pilot used readable task labels containing scenario seeds. The agent had no simulator source or execution tool and received only hit/miss observations. After the pilot, task construction was tightened to use opaque labels for subsequent runs; actual seeds remain in evaluation metadata. This label change was tested offline and is not represented as having been used in this saved pilot. The recorded source digest and manifest identify the pilot implementation.

This small game tests strategy-note adaptation, not recursive-context scaling, learned model weights, or collaboration among specialist agents. A full research result needs completed paired evaluations over many fleets and independent adaptation runs, with notes-removal ablations and all verification costs included.

## Evidence

- [Declared Battleship rules and protocol](battleship-benchmark.md)
- [Original run report](../runs/gemini-battleship-pilot-20261017/report.md)
- [Recorded-game replay](../runs/gemini-battleship-pilot-20261017/replay.html)
- [Machine-readable status and usage](../runs/gemini-battleship-pilot-20261017/game_summary.json)
- [Episode actions and outcomes](../runs/gemini-battleship-pilot-20261017/episodes.jsonl)
- [Local reference comparison](../runs/battleship-controls-20261017/report.md)
