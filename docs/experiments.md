# Experimental protocol

1. Choose a fixed model, temperature, token/output limits and cost estimates.
2. Build disjoint adaptation, candidate-verification and final held-out task sets.
3. Start from a fresh state and output directory. Record the source/dataset digests.
4. Measure initial adaptation and held-out performance without updating state.
5. Run adaptation; propose, independently validate and optionally gate each change.
6. Freeze the learned state. Measure adaptation and held-out tasks again.
7. Report paired completion gains and call/token changes, plus verification overhead.
8. Repeat conditions/seeds; use the same datasets, backend and resource ceilings.

The benchmark CLI runs this protocol. `comparison.csv` stores the initial/learned
groups. In a multi-condition run, the top-level `comparison.csv` uses its `held_out_*`
columns for the paired test comparison. Its other metric columns pool all episodes
of the learned run. `experiment_summary.json` records task IDs, adaptation order, gains and
configuration. The final test split is never sent to the refiner or candidate gate.

The default gate score is

`mean(verified - 0.001*model_calls - 1e-7*tokens - 0.0001*tool_calls)`.

Accept only when score gain is at least epsilon and no baseline-passing task fails
with the candidate. This is a declared engineering objective, not a universal
research utility. Cost estimates are zero unless provider prices are configured.
Matched limits alone do not mean matched actual compute; report both actual usage
and limits. Candidate verification has its own independently reported overhead.

The fixture mock deliberately follows search → read → write → verify. Its learned
search/read workflow avoids one decision call while still executing both actions.
The mock's specialist can replace redundant inspection decisions. These are tests
of adaptation machinery. Do not present fixture results as long-horizon LLM gains.

Task schemas support multiple files and custom trusted command checks. For larger
experiments add real repositories and fix dependencies, source revisions and test
commands. The AST fixture checker supports a restricted Python subset; it is not a
replacement for pytest, compilers or independent software-engineering benchmarks.
