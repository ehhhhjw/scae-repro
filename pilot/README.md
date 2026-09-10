# `pilot/` -- the SCAE code itself

This directory is the paper's implementation. The package name stays `pilot`, so the commands recorded in
the paper's Appendix E.1 (e.g. `python3 -m pilot.analyze.entropy_scan`) can be run verbatim.

For top-level docs see `../README.md` (overview) and `../REPRODUCE.md` (execution steps).

## Module layout

```
config/env_pin.sh        environment noise pinning (U_env pinning / CRN), equivalent to appserver.PINNED_ENV
harness/appserver.py     JSON-RPC/stdio client for the codex app-server (experimentalApi)
harness/rollout.py       read/write/truncate/action-rewrite for rollout JSONL -- the low-level primitives of prefix replay
harness/runner.py        batch file localization (S1), forcing a reset to base_commit for each instance
harness/parallel.py      parallelism sharded by repo, merging the per-shard JSONL
harness/action_exec.py   real execution of read-only substitute actions (the source of o' on the intervention branch)
probes/task.py           instance loading, prompt, outcome determination Y, repository precondition checks
attribution/causal.py    v_t estimation, Δ_t, hybrid estimator (hybrid_delta_at, the τ_H branch)
attribution/judge.py     full-trace / prefix-only LLM judge baselines
attribution/run_attribution.py  attribution orchestration (equally spaced valuation points)
uq/predictor.py          Bayesian logistic + Laplace posterior + three-way decomposition + calibration metrics
uq/features.py           pre-execution features for v_0 (7 covariates, does not read the trace)
channels/parsed_command.py  Python re-implementation of codex ParsedCommand (action semantics layer)
channels/code_channel.py    code dependency graph channel U^obj
channels/tool_model.py      tool channel Markov model U^tool
graph/ast_index.py       pure-stdlib AST index (no tree-sitter dependency)
smoke/smoke.py           S0 feasibility verification (V1-V4)
smoke/probe_v5.py        action entropy probe (positivity measurement)
analyze/                 the per-RQ offline analyses (see ../docs/SCRIPT_INVENTORY.md)
analyze/legacy/          exploratory scripts that were superseded or did not make the final draft
```

## Dependencies

Python 3.10+ standard library + `numpy` (only some analyze scripts). **No GPU.**
The data collection stages additionally require a logged-in `codex` CLI and an API key.

## Data paths

No data is bundled. Point at a local SWE-bench-Verified snapshot via environment variables:

```bash
export SCAE_CSV=/path/to/instance_summary.csv
export SCAE_PROJECTS=/path/to/swebench_verified_projects/projects
```

When unset, defaults to `<repo>/dataset/instance_summary.csv` and `<repo>/dataset/projects`.

## The five feasibility criteria of S0

Run `../scripts/00_smoke.sh` before the real experiments. These five decide whether the method is feasible:

| Item | Question | Criterion | Consequence if it fails |
|---|---|---|---|
| **V1a** | Can `capabilities.experimentalApi=true` expose `Thread.path` and fork by path | `thread.path` non-empty | prefix replay degrades to turn granularity → redesign needed |
| **V2** | Can the tool surface be narrowed to read-only localization | no `apply_patch/write/edit` style calls | the action space is not discrete |
| **V2b** | Were any server-initiated reverse requests (approvals) received | count is 0 | `approvalPolicy=never` / `sandbox=read-only` did not take effect |
| **V3** | Which item types are the bare minimum for forking a hand-built rollout | at least one ablation variant forks successfully | blocking, must be resolved |
| **V4** | After a fork, does `Continue.` resume or restart | the first k−1 step action buckets match one by one, and total steps ≥ k | all causal estimates are invalid |
| **V5** | Action entropy of repeated forks from the same prefix | at least 2 distinct actions (H > 0) | positivity fails locally → take the intervention branch |

If V5 gives H ≈ 0, the policy is nearly deterministic on that prefix and the "effect relative to the policy's own
baseline" term in Theorem 1 degenerates. This is not an implementation bug but one of the paper's core findings:
the hybrid estimator of main-text §4.3 (Eq. 8) is designed exactly for this, routing low-entropy points to the
**soft intervention** branch and flagging in the report that the estimand is relative to the perturbed policy π̃
rather than the original policy π.

## Key implementation conventions (do not break these when changing code)

Corresponding to the paper's Appendix C.1, "two implementation invariants are load-bearing and easy to violate":

1. **The continuation prompt `CONTINUE_PROMPT` must be byte-identical across all arms.** It enters the context and
   is therefore part of the estimand's definition; any difference between arms becomes an undeclared intervention
   that contaminates Δ_t.
2. **The `new_output` of `substitute_action` must be the result of actually executing `new_args`.** Filling in a
   fabricated output is not a faithful do(A_t = a') but intervening on an observation -- the two have completely
   different causal semantics.
3. **After truncation, `assert_no_dangling_calls` must pass.** A dangling tool call is rejected on the model side.
   Both truncation operators are safe by construction, but rewrite paths need an explicit check.
4. **`check_repo` must pass before running a task** (clean workspace + HEAD == base_commit). Otherwise the agent
   reads already-fixed code = answer leakage, and replays are not comparable. See `../docs/PITFALLS.md`.
5. **Do not silently ignore compaction events.** A non-empty `compaction_lines()` means Assumption 1 (observable
   context) fails on that trace and identifiability is broken. Paper §5.2 reports 0 compaction events; if your run
   has a non-zero count, the attribution conclusions for that trace cannot be used directly.
