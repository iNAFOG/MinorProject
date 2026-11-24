# First real Gemini game-agent pilot

**Date:** 17 October 2026  
**Model:** `gemini-3.1-flash-lite` through Google's compatible API endpoint  
**Status:** Complete integration/protocol pilot  
**Conclusion:** Real game-agent execution works. This pilot demonstrates no performance improvement.

## Setup

The agent played tic-tac-toe using the existing orchestrator, bounded runtime, game rule engine, and SQLite state. Model weights, model ID, temperature, reasoning effort, tools, and task limits stayed fixed across comparisons.

Configuration: temperature 0, minimal reasoning effort, maximum 1,024 output tokens per request, a 45-second request timeout, and a six-second minimum request interval. The saved completed run has a 111-request cap. Together with the conservatively reserved requests from the interrupted diagnostic run, this stays within the 120-request session allowance.

The split contained:

- Two adaptation games supplying experience.
- Two verification scenarios, played once with the baseline and once with the candidate, giving four verification episodes.
- Two held-out scenarios, evaluated before and after adaptation, giving four evaluation episodes.

That totals **10 game episodes**, plus one reflection request. The pilot's two held-out scenarios cover a random opponent as X and a tactical opponent as O. This small pilot does not include the exact-opponent stratum in held-out evaluation.

## Results

| Measure | Initial evaluation | Post-adaptation evaluation |
|---|---:|---:|
| Held-out games | 2 | 2 |
| Wins | 0 | 0 |
| Draws | 1 | 1 |
| Losses | 1 | 1 |
| Non-loss rate | 50% | 50% |
| Mean score, win=1 / draw=0.5 / loss=0 | 0.25 | 0.25 |
| Model calls | 8 | 8 |
| Provider-reported task tokens | 6,940 | 6,940 |
| Illegal moves | 0 | 0 |

The two adaptation games produced one win and one loss. One illegal request occurred during adaptation. No illegal moves occurred in the verification or held-out games.

The complete valid pilot used **40 requests** and **35,853 reported tokens**, including gameplay, reflection, and verification. Verification itself used 16 requests and 15,577 tokens. Monetary cost was not measured, and the harness does not independently verify account billing tier.

## Why no strategy change was accepted

The fixed model proposed a general strategy emphasizing immediate wins, blocking threats, center/corner play, and legal-cell checks. Only adaptation traces supplied its evidence.

On the separate verification suite, both baseline and candidate produced one win and one draw with eight model calls each. The candidate added 2,035 verification tokens without improving those outcomes. Its cost-penalized score decreased by approximately 0.000010175, so the gate rejected it.

**Accepted updates: 0. Rejected updates: 1. Final harness version: 0.**

The generated table's `learned` phase name therefore means post-adaptation evaluation of the unchanged harness. It does not mean that a learned strategy was activated. No performance gain or successful adaptation can be claimed from this result.

## Integration issue found and repaired

The first diagnostic attempt used a legacy conversation format that represented requested actions as assistant JSON text and returned observations as ordinary user messages. Gemini sometimes continued with JSON text instead of a native tool call, ending an episode early.

We stopped that run and marked it unusable for performance comparison. Eight completed requests were recorded there, with one additional possibly in-flight request conservatively reserved in the session cap.

The adapter and orchestrator now preserve native assistant tool calls, their matching tool-result IDs, and provider-specific metadata. Conversation compaction retains complete tool exchanges. A per-request timeout prevents a slow request from consuming the full episode timeout. Local protocol and regression tests pass: **59 tests total**.

The completed run started from a fresh harness state after this repair. It is stored separately from the interrupted diagnostic attempt.

## What this pilot establishes

- The locally stored key can authenticate real Gemini requests without entering configuration or experiment logs.
- The model can select game moves through the tool interface and receive authoritative game observations.
- Adaptation reflection and paired candidate verification execute with a real model.
- The gate rejects an unhelpful strategy, and final evaluation leaves learned state unchanged.
- Saved trajectories, outcomes, proposal decisions, and board replays can be inspected.

It does not establish improved playing strength, broad generalization, better recursive reasoning, or lower total computation. Two held-out scenarios and one adaptation trajectory are insufficient for a research conclusion.

## Next experiment

Predeclare a larger tic-tac-toe suite with all opponent strengths and both player sides, repeat independent run seeds, and evaluate notes-removal controls if updates are accepted. Then run the key-and-door planning pilot separately, with its own request allowance and sufficient provider quota. Report ceiling effects and failed updates rather than selecting only successful runs.

## Evidence

- [Completed pilot report](../runs/gemini-ttt-pilot-20261017-v2/report.md)
- [Full metrics and configuration](../runs/gemini-ttt-pilot-20261017-v2/game_summary.json)
- [Episode outcomes and actions](../runs/gemini-ttt-pilot-20261017-v2/episodes.jsonl)
- [Proposed strategy and verification decision](../runs/gemini-ttt-pilot-20261017-v2/changes.jsonl)
- [Recorded board replay](../runs/gemini-ttt-pilot-20261017-v2/replay.html)
- [Interrupted diagnostic-run annotation](../runs/gemini-ttt-pilot-20261017/interruption.json)
