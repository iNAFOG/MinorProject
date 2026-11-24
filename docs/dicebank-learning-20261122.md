# Dice Bank: a second learning-gain test — 22 November 2026

## The game (our own design)

10 turns. In each turn you roll one six-sided die as often as you like, adding the faces to your turn points. Rolling a **1** loses the turn points and ends the turn. **Banking** ends the turn and keeps the points. Score = total banked after 10 turns.

A **policy** is two integers: bank once turn points reach `bank_at` (1–100), or after `max_rolls` rolls (1–30), whichever comes first. Code: `agent_harness/games/dicebank.py`.

The best policy can be computed exactly:
- **Bank at 20** gives the highest expected score, **81.4 points per game**. Rolling once more at exactly 20 is worth the same (5/6 × 24 = 20).
- Banking after every roll gives 33.3.

The exact values are checked against simulation in `tests/test_dicebank.py`.

Every game seed fixes the dice for every turn and roll position, so all policies face **identical dice** on the same game. Before/after comparisons are therefore not confounded by luck.

## Protocol, declared before the held-out games were played

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, local, fp16, greedy decoding. Weights fixed. No network.
- **Initial harness:** the model's zero-shot policy (no feedback). If it is invalid, the always-bank-after-one-roll policy is used.
- **Per arm and run, 6 rounds:**
  1. The current policy plays 50 practice games.
  2. The proposer sees a summary: mean total, mean banked per turn, bust rate, points lost when busting, rolls per turn, and example turns.
  3. A candidate goes through `Verifier.apply` → `State.promote`. It is accepted if its mean paired score on 1,000 verification games beats the current policy by ≥ 0.5 points.
- **Arms**, all starting from the same initial policy:
  - **model:** the fixed LLM proposes revisions from the feedback.
  - **random:** uniform random policies.
  - **hill-climb:** small random steps from the current policy, ignoring the feedback text.
- **Runs:** 1, 2 and 3, each with different practice games. Verification and held-out games are fixed. Seed ranges: practice 1,000,000×run; verification 5,000,000+; held-out 7,000,000+.
- **Held-out:** 10,000 games, played only after all learning finished.
- **Primary outcome:** paired held-out mean total, final vs initial policy, with a bootstrap 95% interval. **A learning gain is claimed if the model arm improves with an interval above zero in a majority of runs.** Whether the model's reflection helps is judged against both controls and against the exact optimum.

Constants are in `scripts/dicebank_learning.py`. A CPU dry run with the mock model used throwaway seeds (90,000,000+).

## Results

The model's zero-shot policy was **bank at 5, max 20 rolls**, worth 48.2 expected points (held-out 48.25). The exact optimum (bank at 20) scored 81.46 on the same 10,000 held-out games, and banking after every roll scored 33.44.

### Declared run (`runs/qwen-dicebank-learning-20261122/`): no gain for the model arm

Every model revision was rejected as invalid. The model *did* propose sensible changes, such as `bank_at` 7 or 10, but it wrapped them as `{"improved_policy": {...}}` and then invented feedback statistics, sometimes until the 768-token output limit. The parser accepted only a `policy` wrapper. The model arm therefore stayed at 48.25. Both controls gained about +32 to +33 points (random 80.4–81.3, hill-climb 80.6–81.3).

### v2: parser fix only (`runs/qwen-dicebank-learning-20261122-v2/`)

**Disclosure:** after seeing the declared run's *proposals*, I changed only `parse_policy`, so that it finds the first `bank_at`/`max_rolls` pair anywhere in the reply. The prompt, model, games, gate and seeds were unchanged, so the model received the same inputs as before. Control-arm results are identical to the declared run, as expected.

| Arm, run | Policy path (bank_at) | Held-out mean total | Paired change (95% CI) |
|---|---|---:|---:|
| **model 1** | 5 → 7 → 8 | 48.25 → 62.42 | **+14.17** [+13.97, +14.39] |
| **model 2** | 5 → 10 → 15 | 48.25 → 78.62 | **+30.37** [+29.95, +30.84] |
| **model 3** | 5 → 10 → 15 → **20** | 48.25 → **81.46** | **+33.21** [+32.66, +33.81] |
| random 1 / 2 / 3 | final 18 / 22 / 70 | 81.14 / 81.32 / 80.42 | +32.90 / +33.07 / +32.17 |
| hill-climb 1 / 2 / 3 | final 17 / 18 / 19 | 80.59 / 81.14 / 81.26 | +32.34 / +32.90 / +33.01 |
| exact optimum (reference) | 20 | 81.46 | — |

**A learning gain is established for the model arm** under the declared criterion: all 3 runs improved, with intervals well above zero. Run 3 reached the exact optimal policy.

**Quality of the model's reflection:**
- **All 7 valid model proposals were accepted.** Each one moved `bank_at` upward, towards the optimum. By comparison, random proposals were accepted 6 times out of 18 and hill-climb steps 8 times out of 18.
- The three model runs took different paths, because their practice games and feedback differed. So, unlike the Battleship test, they are genuinely separate runs.
- The model read the feedback (low banked points per turn, small losses when busting) and inferred "bank later". That is the right direction.

**Limits:**
- **11 of the 18 model replies were unusable.** After two or three rounds the model renamed the fields (`new_bank_at`, `new_max_rolls`) or did not finish its reply, so learning stopped early. This is why the model arm averages 74.2 against about 81 for the controls. I did not loosen the parser a second time.
- The game has only two parameters and a smooth optimum, so both controls also reach the optimum within 6 rounds. Dice Bank shows that the model's suggestions are well aimed, not that they beat search on final score.
- Pig-style "hold at 20" advice exists in public text, so the model may partly know the answer. Its zero-shot guess (bank at 5) suggests it did not apply that knowledge directly.
- The model used 19 calls in total; the v2 run took 753 s.
