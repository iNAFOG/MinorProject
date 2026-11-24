# Improving the game-learning harness

Diagnosis and proposed next experiment, 18 October 2026. This document proposes changes; no new model calls or new learning experiment were run for it.

## What the completed experiment actually tested

The game configuration enabled strategy-note refinement and disabled coding skills, memory retrieval and specialist agents. Two practice games produced one paragraph of advice. Two verification fleets selected that paragraph. Three held-out fleets showed no win-rate gain and lower target coverage.

The action traces reveal a specific execution problem: every initial and final held-out game opened with cells `[2, 7, 12]`, a single vertical column. That opening persisted after the note recommended checkerboard search. A textual lesson therefore did not reliably translate into the corresponding action rule. This observation does not, by itself, explain every mistake or prove that the note caused the score decrease.

The verifier also had very little evidence. One of its two fleets improved from zero to three hits, while the other stayed at five. This was enough for the declared gate, but the positive result did not transfer. Finally, the refiner currently reads only the last three practice traces; collecting more games should be accompanied by better evidence aggregation.

## Priority 1: learn a reusable game-playing procedure

Add a game-specific skill type: a bounded policy description that a local interpreter executes. The fixed model proposes and revises the description from practice feedback. A candidate can specify search/target modes, ranking weights and conditions for changing modes. The interpreter should accept public observations and return three legal cells; it should have no simulator-private fields or hidden-board access.

For example, a candidate could learn to prioritize the ends of an unresolved hit line, then resume searching once that ship is sunk. The policy should express such conditions explicitly, so execution can be checked against the stored procedure. This requires a new game-skill adapter; enabling the existing coding-skills flag alone is insufficient.

Executable skill storage and reuse have a research precedent in [Voyager](https://arxiv.org/abs/2305.16291). That work motivates the design; its Minecraft results are not evidence that this Battleship implementation will improve.

## Priority 2: provide a clear public-state summary

Compute remaining ship lengths, unresolved hits, sunk-ship cells, available cells and placements consistent with public feedback. Keep these calculations independent of private simulator state. Candidate policies can use the same features to rank actions rather than repeatedly asking the language model to reconstruct spatial relationships.

Give identical features and tools to the initial and adapted conditions. Measure the effect of adding the helper separately from the effect of learning a better policy. A handwritten expert controller remains a reference control, not the learned agent.

## Priority 3: improve reflection and acceptance

Use varied practice games and aggregate recurring action-level errors across the archive. Ask for small, testable revisions tied to those errors. Keep a short history of policies and failure categories, and preserve the previous policy when the available evidence is inconclusive.

Test candidates on substantially more verification fleets and repeat independent adaptation runs. Predeclare sample sizes, acceptance criteria and any sequential-testing rules. Account for repeated candidate selection; repeatedly trying prompts on one tiny validation set can overfit it. Reflecting on trajectories and testing evolving candidates is also central to [GEPA](https://arxiv.org/abs/2507.19457), although our exact gate and budgets must be designed and evaluated separately.

For stochastic-fleet population performance, use a declared aggregate objective with uncertainty and checks across scenario groups. Do not require zero loss on every individual sampled fleet as a substitute for population evidence. Retain legality and state-isolation checks as hard requirements. Any change from the original gate belongs to a new declared experiment.

## Priority 4: make large evaluations affordable

In the proposed skill-execution mode, one model call produces a policy that the local interpreter can play across hundreds of fleets without a model request per move. A few model-driven revisions can therefore be evaluated much more broadly than the current per-move protocol. This is a new experiment mode, with an initial policy and refined policy compared within that same mode.

Generate enough distinct fleet layouts across the practice, verification and held-out splits; different numeric seeds alone do not guarantee different fleets. Keep all evaluation-only fleet data outside the model's observations and refinement evidence.

## Comparison required for a credible learning claim

1. Measure the initial policy using the same interpreter and features as the adaptive condition.
2. Learn using practice feedback; select using verification only.
3. Freeze the selected policy and evaluate fresh, untouched fleets.
4. Repeat independent learning runs and report coverage, wins, actions, model requests, tokens and total adaptation/selection overhead.
5. Include a learned-skill-disabled comparison to measure the policy's contribution. Keep the handwritten reference separate.

The already inspected fleets are development evidence for this redesign. They must not be presented as an untouched test after tuning against their failures. The original completed result remains unchanged.

The model can stay fixed. Recursive context can be evaluated later on a sufficiently large practice archive with a matched context ablation; the present short board observations triggered no recursive calls and establish no RLM advantage.

## Implementation targets

- Add a bounded game-policy schema, interpreter and game-skill validation adapter.
- Add public board features and tests showing independence from hidden fleet state.
- Replace the last-three-only reflection input with bounded summaries and retrievable evidence from the practice archive.
- Extend candidate evaluation to local policy rollouts and a predeclared statistical gate.
- Add unique-layout split generation and matched initial/adapted/skill-disabled experiments.

The first implementation milestone should be the skill runner and its offline comparisons. The nine remaining API requests can be preserved until that path is ready; improving performance is an experimental outcome, not a guaranteed deliverable.
