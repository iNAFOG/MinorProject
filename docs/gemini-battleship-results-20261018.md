# Completed Gemini Battleship experiment — 18 October 2026

**The complete before/after test finished. The accepted strategy note improved verification scores but did not improve held-out performance.** Target coverage fell from 73.3% to 60%, with the win rate unchanged at one win in three games. This is a negative transfer result in a small exploratory pilot.

## Held-out results

The model ID remained `gemini-3.1-flash-lite`, temperature zero, with the same five-salvo game budget. The same three hidden fleets were used before and after adaptation. The strategy note was frozen before final evaluation.

| Metric | Initial harness | After accepted strategy |
|---|---:|---:|
| Completed held-out games | 3 | 3 |
| Wins | 1 / 3 | 1 / 3 |
| Targets hit | 11 / 15 | 9 / 15 |
| Mean normalized score | 0.7333 | 0.6000 |
| Model calls | 14 | 14 |
| Reported tokens | 19,202 | 23,206 |
| Illegal shots | 0 | 0 |

The mean score change was **−0.1333**, or **−13.3 percentage points** of target coverage. The note added **4,004 tokens** to held-out inference without reducing calls. No game improved, one regressed, and two tied on target coverage.

| Evaluation fleet seed | Initial hits | Final hits | Initial outcome | Final outcome |
|---|---:|---:|---|---|
| 2000 | 4 / 5 | 2 / 5 | Loss | Loss |
| 2001 | 5 / 5 | 5 / 5 | Win | Win |
| 2002 | 2 / 5 | 2 / 5 | Loss | Loss |

The exploratory paired-episode bootstrap interval for mean-score change is [−0.4, 0.0]. It describes these three fleets and one adaptation trajectory; it is not uncertainty across independent learning runs. This small sample does not establish a general degradation rate or rule out improvements in another properly declared experiment.

## Why the note was accepted

Two adaptation games supplied the original model-generated strategy: checkerboard search, investigating unresolved hits, following an inferred ship axis, and updating search priorities after feedback. No new reflection was requested during either continuation.

On the separate verification fleets:

| Verification fleet seed | Baseline hits | Candidate hits |
|---|---:|---:|
| 1000 | 5 / 5 | 5 / 5 |
| 1001 | 0 / 5 | 3 / 5 |

Mean verification score increased from **0.50 to 0.80**, with no per-fleet regression. Both sides used nine model calls. Baseline tokens were 12,097; candidate tokens were 14,679. The original cost-penalized objective gain was **+0.29998709**, so the unchanged gate promoted the note to **harness version 1**.

The positive verification result did not transfer to the held-out fleets. Two verification examples were insufficient to ensure generalization in this run. The final artifact retains the accepted version and its failed held-out result; no note was revised or selected using the held-out outcomes.

## Retry and usage accounting

The original attempt stopped on HTTP 503 during verification. On the user's retry request, the runner reused the original initial evaluation, adaptation traces, exact model-generated proposal, completed verification prefix, and manifest. It preserved the original task labels and configuration.

The first continuation used 23 additional requests and completed the verification gate, then timed out during the second held-out game. A final continuation reused the completed held-out loss, restarted only the infrastructure-interrupted game, and ran the last unstarted game. It used nine additional requests. HTTP transport timeout increased from 45 to 90 seconds; prompts, model ID, temperature, output limits, game budgets, selected note and fleet seeds stayed fixed.

The final cumulative attempt cap was 63 to accommodate infrastructure restarts. The completed experiment used **62 attempts and 91,623 reported tokens**, including the original attempt and both interruptions. **32 attempts were new in this retry turn.** Combined with the conservative 49-attempt allowance for earlier trials, this accounts for 111 of the original 120 requests, leaving **nine**. Usage for a provider-failed request may not be reported; all such attempts are counted in request totals.

The two interrupted episode attempts remain in the logs and usage totals. Their censored scores are excluded from the final game comparison because their games were restarted after infrastructure failures. Every completed game outcome, including losses, was retained. No completed loss was replayed to obtain a better result.

## What this demonstrates

The real-model pipeline completed game execution, adaptation-only reflection, candidate isolation, comparative verification, promotion/versioning, frozen evaluation and reproducible reporting. The new continuation tests verify source preservation, unchanged model settings, saved-response integrity, reuse of completed losses, budget checks, and stopping on another provider failure. **All 84 code tests pass.**

The game experiment provides no evidence of a playing-strength or inference-efficiency improvement. It also does not establish benefits from recursive context, specialist collaboration, or weight training: those treatments were not studied, and recursive calls were zero.

A follow-up should predeclare larger verification suites and independent adaptation runs, then use fresh unseen evaluation fleets. These observed fleets should not be reused as an unseen test after tuning the strategy or gate. A notes-removed ablation would also help separate note effects from provider variability. No additional model calls were made for those follow-ups.

## Saved evidence

- [Completed run report](../runs/gemini-battleship-final-20261018/report.md)
- [Machine-readable results and usage](../runs/gemini-battleship-final-20261018/game_summary.json)
- [Game replay, including interrupted attempts](../runs/gemini-battleship-final-20261018/replay.html)
- [Verification decision and original proposal](../runs/gemini-battleship-final-20261018/changes.jsonl)
- [Episode observations/actions](../runs/gemini-battleship-final-20261018/episodes.jsonl)
- [Version history](../runs/gemini-battleship-final-20261018/harness_history.json)
- [Original interrupted run](gemini-battleship-pilot-20261017.md)

Reproduction code is in `scripts/resume_battleship.py` and `agent_harness/games/resume.py`. Continuation metadata is saved alongside each corresponding run. Existing output folders are preserved; a new run requires a fresh output path and an explicit remaining request allowance.
