# Development

Install with `python -m pip install -e ".[test]"`. Run `python -m pytest`.
Tests exercise offline execution, restart persistence, scope/provenance,
confidence/contradictions, skill and specialist lifecycles, recursive limits,
proposal policy, independent acceptance/rejection, actual regressions, rollback,
evaluation freezing, configurations, transfer artifacts and a loopback API adapter.

The workspace originally had no source files and no Git metadata. The supplied
synopsis remains untouched. Add Git tracking when desired; until then experiment
metadata records a source digest and null Git commit rather than inventing one.

To add workflow operations, update the validator and executor together and test
their boundaries. Do not let workflows call arbitrary Python or rewrite checker
definitions. To add a persistent state backend, preserve the `MemoryStore` protocol
and registry operations plus atomic candidate promotion and rollback semantics.

Standalone task JSON files embed source files and private checks. Agent prompts
receive only the public task view and selected source. New environment factories
must similarly separate observations from checker authority.
