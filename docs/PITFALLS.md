# Pitfalls That Silently Ruin Reproduction Results

Ordered by severity. **The first few do not raise any error, they just make the numbers look better** -- this is the hardest class of failure to notice.

---

## 1. ⚠️ Dataset workspace has the gold patch applied (answer leakage)

**Symptom**: localization success rate clearly above the paper's 93.99%, with no error at all.

**Cause**: the distributed SWE-bench repository snapshots often have the workspace in an **already-fixed** state (gold patch applied),
some even carry a `procbench-*` stash. The localization task must run on the buggy `base_commit`,
otherwise the agent reads the answer straight out of its own working tree.

**The three gates in this package**:

1. `setup/prepare_dataset.py --verify-only` -- full check of whether HEAD and the workspace are clean
2. `setup/check_env.py` -- spot-check the `base_commit` of 20 instances
3. `pilot/probes/task.py::reset_repo` -- before every instance, force
   `checkout -f` + `reset --hard` + `clean -fdq`

```python
def reset_repo(inst):
    _git(repo, "checkout", "-f", inst.base_commit)
    _git(repo, "reset", "--hard", inst.base_commit)
    _git(repo, "clean", "-fdq")
    return check_repo(inst)   # assert: workspace clean AND HEAD == base_commit
```

**Do not bypass reset to save time.** Paper Appendix E.1 records this specifically:
"Running the same pipeline on the as-distributed snapshot measures an agent that can read the answer
out of its own working tree, and reports an inflated localization accuracy with no visible symptom."

---

## 2. ⚠️ Environment noise not pinned down → replays are not comparable

**Symptom**: `replay_check` consistency rate < 1.00, or traces replayed from the same prefix inexplicably diverge.

**Cause**: locale / timezone / terminal width / color / hash seed / git pager all enter tool output,
hence enter the context, hence change the policy. Paper Assumption 3 (replayable environment) requires all of these to be fixed.

**Practice**: every entry point must first `source pilot/config/env_pin.sh` (`scripts/common.sh` already does this for you),
whose Python equivalent is `PINNED_ENV` in `pilot/harness/appserver.py`.

```bash
LC_ALL=C  LANG=C  TZ=UTC  NO_COLOR=1  TERM=dumb  COLUMNS=200  LINES=50
PYTHONHASHSEED=0  GIT_PAGER=cat  PAGER=cat  RIPGREP_CONFIG_PATH=/dev/null  CI=1
```

**Change one place, you must change both**: if `env_pin.sh` and `PINNED_ENV` disagree, processes started by the shell and
processes started by the app-server see different environments, and that kind of skew is extremely hard to localize.

---

## 3. ⚠️ Continuation prompt inconsistent across arms

**Symptom**: unexplained systematic offset in Δ_t.

**Cause**: `CONTINUE_PROMPT` ("Continue.") enters the context, and therefore **is part of the estimand's definition**.
Any literal difference across arms becomes an undeclared intervention.

Paper Appendix C.1 lists this as one of two "load-bearing and easily violated" implementation invariants:
the continuation prompt must be **byte-for-byte identical** across all arms.

---

## 4. ⚠️ Fabricating the output of a substitute action

**Symptom**: Δ from the intervention branch looks clean but is not reproducible.

**Cause**: the `output` in `substitute_action(k, tool, args, output)` must be the result of **actually executing** `args`
(`pilot/harness/action_exec.py::exec_readonly_action`).
Filling in a plausible-looking output is no longer `do(A_t = a')`, it is intervening on the **observation** --
these are different causal quantities.

This is the second invariant in Appendix C.1. Algorithm 1, line 16, says
"**execute** a' for real in the reset repository ▷ never a fabricated output".

---

## 5. `rg` in the shell may be a function rather than a binary

**Symptom**: `action_exec`'s read-only allowlist check passes, but actual execution behavior differs from the agent's at the time.

**Cause**: `rg`/`grep` may be shadowed by a shell function or alias. The `action_exec.py` allowlist matches on the command head
(`READONLY_CMDS`), but actual execution goes through `shell=True`.

**Practice**: `RIPGREP_CONFIG_PATH=/dev/null` in `env_pin.sh` removes ripgrep's user config;
execution uses `PINNED_ENV` rather than the inherited interactive shell environment.

---

## 6. `shlex` fails on 26.6% of command segments (defect disclosed in the paper)

**Symptom**: query channel top-3 abnormally low (0.029).

**Cause**: command parsing uses `shlex` to recover the search pattern; on unbalanced quotes (e.g. `grep "a\b"`) it throws,
then falls back to whitespace splitting, keeping quote characters and truncating. **75.3% of query objects are affected**,
of which 249 collapse into the single fragment `q: "def`.

**How the paper handles it**: not by silently fixing it, but by reimplementing a parser that does not depend on `shlex` and rerunning the 1661 query steps.
The result moves the number **down** (0.067 → 0.031) -- i.e. the defect makes query look **more predictable** than it actually is,
so it is conservative and does not threaten the "query is the bottleneck" conclusion. See Appendix D.5 and
`pilot.analyze.x1d_query_channel`.

**When reproducing**: do not "just fix" `parsed_command.py` and then compare directly against the paper's numbers,
the two definitions would differ.

---

## 7. Context compaction breaks identifiability (not harder to estimate, but not identifiable)

**Symptom**: `compaction_lines()` returns non-empty.

**Cause**: compaction maps h_t to φ(h_t) irreversibly. The discarded context becomes an **unobserved confounder**,
paper Assumption 1 fails, and step 1 of the Proposition 1 proof does not hold.

**The paper reports 0 compaction events** (§5.2). If your run yields non-zero:
the attribution conclusions for that trace are **not directly usable** -- not "slightly less precise" but "that quantity is not identifiable".

---

## 8. The attribution stage requires shard directories staged in advance

**Symptom**: `parallel --cmd attr` exits without doing anything.

**Cause**: attr mode does not pick instances itself, it requires `shard*/` to already exist under `--out`,
with `results.jsonl` and `rollouts/` inside each shard. In the original experiment this step was done by hand.

**Practice**: use `setup/stage_attr_shards.py` (`scripts/02_attribution.sh` already calls it automatically).

---

## 9. Single-file instances have nothing to say under attribution

**Symptom**: Δ_t almost all 0, attribution conclusions vacuous.

**Cause**: with binary Y as the outcome, single-target instances are almost always solved, and most substitute actions do not change the indicator,
so Δ_t ≡ 0. The paper therefore restricts attribution to the **multi-file subset** (`--multifile-only`,
hard-coded in `parallel.py`), and switches the outcome variable to **recall over F\***
(Appendix C.6: intervention sensitivity 2.505 vs 1.643 for binary).

Even so, **22.9% of episodes fully degenerate** and **54.7% of evaluation points have Δ exactly 0** --
this is itself one of the paper's conclusions (RQ4), not a bug.

---

## 10. The answer must be parsed from the rollout, not from the final text

**Symptom**: localization results low or missing.

**Cause**: the more reliable source of the `<answer>` block is the rollout JSONL, not the text at the end of the session.
`pilot/probes/task.py` provides both `parse_answer(text)` and `answer_from_rollout(items)`,
the latter being the main path.

---

## 11. There are three verdict definitions, do not mix them

The paper's primary definition is **`gold ⊆ answer` (recall == 1)**, with two sensitivity definitions also reported:

| Definition | Meaning | Value |
|---|---|---|
| `recall == 1` (primary) | F\* ⊆ F̂, over-reporting allowed | **93.99%** |
| Strict set equality | F\* == F̂ | 15.23% |
| Equality after filtering test files | drop test paths, then compare | ≈ primary |

The huge gap between 93.99% and 15.23% is **entirely caused by the agent additionally naming test files** (the gold patch contains no tests).
The paper therefore treats strict equality as a diagnostic for "the agent's naming habits", not as a measure of localization ability.

---

## 12. After truncation you must check for dangling tool calls

**Symptom**: the forked rollout is rejected by the model endpoint.

**Cause**: truncation may leave a dangling call that has a `function_call` with no corresponding `function_call_output`.
`truncate_before_step` / `truncate_including_step` are safe by construction, but any **rewriting** path
(e.g. `substitute_action`, `mask_output_lines`) must afterwards pass
`rollout.assert_no_dangling_calls()`.

---

## 13. Disk: the full artifacts are about 17GB

| Directory | Size | Needed for offline analysis |
|---|---|---|
| `v4prime/` | 5.3GB | only `attribution.jsonl` within it (140KB) |
| `graphs/` | 3.4GB | needed for RQ2 |
| `full_attr/` | 3.3GB | only `attribution.jsonl` (156KB) |
| `m8/` | 1.9GB | no (m robustness check, moved into legacy) |
| `full_locate/` | 1.4GB | **needs `rollouts/` (85MB) and `results.jsonl` (240KB)** |

The vast majority of the size is fork intermediates (`attr_work/`), **not needed for offline analysis**.
For route B only, about 90MB plus `graphs/` suffices.

---

## 14. Statistical definitions: conventions fixed up front, no picking after seeing results

Paper Appendix C.5 fixes the statistical conventions; do not substitute your own when reproducing:

- Confidence intervals = episode-level or instance-level **bootstrap**
- Null hypothesis = **within-episode permutation** or **closed-form expectation**, never a sampled "random baseline"
- Ranking metrics use the **closed-form expectation under random tie-breaking** (count-based scorers have many ties; breaking them arbitrarily is an undeclared design choice)
- Predictors with different candidate-set sizes must be compared on a **common denominator** (falling outside the candidate set counts as a miss,
  otherwise abstention buys accuracy)
- Multiple comparisons **BH-FDR, q = 0.10**, distinguishing exploratory from confirmatory findings
- Wilcoxon refuses to give a p-value when n < 6, rather than giving an unreliable one

---

## 15. Four of the paper's pre-registrations did not hold; do not treat them as bugs to "fix"

Table 9 in Appendix E.2 records them faithfully (reproduction will reproduce these "not met" outcomes):

| Pre-registration | Outcome |
|---|---|
| Entropy **decreases monotonically** with step position (ρ ≤ −0.2) | **Not supported**: ρ = +0.072, non-monotone ("certain-random-certain") |
| LORO AUROC of v₀ ≥ 0.60 | **Not met**: 0.481 (at chance), reported as a negative result |
| Degenerate rate 30-40% | **Better than expected**: 22.9% |
| D₁ top-1 disagreement rate falls in [40%,75%] | **Criterion retired**: observed 97.1%, but the null hypothesis itself is also around 97.1% ⇒ the criterion itself does not hold |

All three directional predictions (D₃ displacement, collider isolation, U_model inflation) hold **above** their pre-registered intervals.
