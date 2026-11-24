# Learning gain search with a fixed local model — 22 November 2026

## Protocol, declared before any held-out evaluation

**Question:** can the harness show a learning gain with a real, fixed model: better frozen held-out performance after adaptation than before it?

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`, read from the local Hugging Face cache, fp16 on a GTX 1650, greedy decoding (temperature 0). Weights never change. No network requests and no API keys.

### Experiment A — strategy notes (existing game pipeline)

These results come from development runs on throwaway seed 9 only. They are reported as findings, not as held-out results.

- The model fires one salvo, then answers in prose instead of calling `fire` again. Every game ends early and scores 0.
- With the original refiner prompt, the model wrote JavaScript instead of a strategy, which the policy correctly rejected. The refiner was then changed: it now gets a factual per-game mistake digest built from public observations, and its JSON parsing is more lenient. With that change it produced a valid note, which the gate rejected (gain +5×10⁻⁹, behavior unchanged).
- **Note-channel check:** a strong handwritten note ("after every fire result, call fire again…; never reply with text while the game is in progress") left all six diagnostic games identical.
- **Conclusion:** with this model, strategy notes cannot change Battleship play, so this channel cannot produce a learning gain. No held-out note experiment was run.

### Experiment B — coding benchmark E8 with the real model

`configs/e8-qwen-local.yaml` is `e8.yaml` with the local model and a 600 s per-task time limit. It uses the unchanged `benchmarks/coding.json` splits (3 adaptation, 2 verification, 2 held-out).

### Experiment C — learned policy skill (main test)

This implements Priority 1 of the [improvement plan](battleship-improvement-plan.md).

**Skill:** a policy is a JSON object of six bounded feature weights. A local interpreter scores each untested cell from the public board and fires at the top three cells (`agent_harness/games/policy.py`). It cannot read the hidden fleet.

**Initial harness:** the model's zero-shot policy, written without any practice feedback. If that is invalid, the all-zero row-major sweep is used instead.

**Learning, per run:** 6 rounds. In each round:
1. The current policy plays 30 practice fleets locally.
2. The model sees the summary (hits, wins, salvos with no hit, unresolved hits, one example game) and the practice scores of policies it already tried.
3. The model proposes a revision.
4. The change goes through the existing `Verifier.apply` → `State.promote` path.

**Gate:** accept if the mean paired score gain on 300 fixed verification fleets is ≥ 0.005. The aggregate gate replaces the per-fleet no-regression rule, which is unusable with hundreds of random fleets.

**Arms × runs:**
- **model arm:** the fixed model proposes revisions.
- **random arm (control):** uniform random weights, which ignore feedback, with the same rounds, gate and start policy.
- Runs 1, 2 and 3 use different practice fleets. Verification and held-out fleets are fixed.

**Held-out:** 1,000 fleets, evaluated only after all learning finishes. Fleet *layouts* are disjoint across the three splits (layout-hash buckets), and the seed ranges are new (100000+ practice, 500000+ verification, 700000+ held-out).

**Primary outcome:** paired held-out mean score (ship cells hit ÷ 5), final policy vs initial policy, per run, with a paired bootstrap 95% interval. Win rate is secondary. **Learning gain is claimed only if the model arm improves held-out score with an interval above zero in a majority of runs.** Whether the gain comes from the model's reflection, rather than gated search alone, is judged against the random arm.

References on the same held-out fleets give scale: the naive row-major sweep and the handwritten hunt/target controller.

Constants live in `scripts/policy_learning.py` and were not changed after the run.

## Results

### B — coding E8 with Qwen: no gain possible

At baseline, held-out completion was 0/2 and adaptation completion 0/3. The model called the disabled `execute` tool, or invented tools named after the target files (`aggregate`, `highest`), then stopped. Because no adaptation task succeeded, the rule-based refiner had nothing to propose, and the model's own refiner did not return JSON. The result was 0 accepted and 0 rejected changes, with learned completion identical to the initial (0/2). Artifacts: `runs/qwen-coding-e8/`.

### C — learned policy skill: a real held-out gain, but not from better-than-chance reflection

Artifacts are in `runs/qwen-policy-learning-20261122/` (`results.json`, `log.txt`, per-arm `decisions.json`, SQLite state and version history). The run used 19 model calls in total and took 337 s.

Held-out results, 1,000 fleets whose layouts never appeared in practice or verification. Score is ship cells hit ÷ 5.

| Policy on held-out fleets | Mean score | Win rate |
|---|---:|---:|
| Initial harness: model zero-shot policy | 0.509 | 0.0% |
| **Model arm, after learning** (runs 1, 2, 3 identical) | **0.767** | **19.6%** |
| Random-proposal control, run 1 / 2 / 3 | 0.717 / 0.761 / 0.921 | 7.0% / 23.0% / 70.6% |
| Reference: naive row-major sweep | 0.643 | 22.3% |
| Reference: handwritten hunt/target controller | 0.927 | 70.2% |

**Model arm:** paired held-out change **+0.258** (95% bootstrap [0.244, 0.271]) in every run; 722 fleets improved, 41 regressed, 237 tied. Each run accepted exactly one revision.

**By the declared criterion, a learning gain is established.** The fixed model's harness state, the stored policy, changed only through practice feedback and a verification gate, and frozen play on unseen fleet layouts improved by 25.8 percentage points of target coverage.

That gain needs careful interpretation:

1. **It starts from a poor zero-shot policy.** The model first gave *negative* weights to every hit-related feature, so it avoided cells next to its own hits. That policy is worse than the naive sweep. The learned policy beats the naive sweep by +0.124, but stays well below the handwritten controller.
2. **The model made one useful correction, then stalled.** In round 1 it read the practice feedback and flipped the hit-feature signs to +1. That was accepted. In rounds 2–6 it re-proposed the same ±1 variants, all rejected. With greedy decoding and an identical start, the three model runs followed the same path, so they are effectively **one** trajectory, not three independent replications.
3. **Random proposals did as well or better.** With the same rounds and gate, uniform random weights averaged 0.800 against the model's 0.767. One random run reached 0.921, about the handwritten controller's level. So the gain comes from the harness loop: a bounded skill, practice feedback, a verification gate and versioned state. **The model's reflection was not better than chance** at finding good policies here, although the model's first revision reached a good region immediately and every random run accepted at least one revision too.
4. **The feature set is handwritten.** Initial and learned policies use the same interpreter and features, so the comparison is fair. Still, the space of possible policies was designed by people.

**Bottom line:** the adaptive harness *can* produce a measured, held-out learning gain with a fixed real model, through a verified executable skill. In this setup a 1.5B model adds no measurable value over random proposals. Strategy notes did not work for this model, because it ignores them, and coding produced no gain, because the baseline completed nothing. A credible next step: a stronger fixed model, more rounds, and a perturbation-based control, to test whether model reflection beats search.

### Development history (for transparency)

- The refiner prompt and parser changes in Experiment A were made after the seed-9 debug failures, before any held-out run. Debug artifacts: `runs/qwen-debug-seed9*`, `runs/qwen-diagnostic-note-channel`.
- The policy experiment's script was dry-run on CPU with the mock model, smaller sizes and throwaway seed ranges (900000+). None of its fleets overlap the reported runs.
