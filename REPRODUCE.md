# Reproduction Steps

This file is the operating manual for this repository. Read in order until you reach the route you want to take.

> **Read [docs/PITFALLS.md](docs/PITFALLS.md) first.** The pitfalls listed there (especially gold
> patch leakage) will destroy every result without raising a single error, and the way they destroy
> them is by **making the numbers look better**.

---

## 0. Pick a route

| Route | Prerequisites | Time | Output |
|---|---|---|---|
| **A Self-check** | python 3.10+, numpy | ~10 s | Appendix D.7 (synthetic data, ground truth known) |
| **B Offline analysis** | one recorded `pilot/runs/` artifact set | ~5-30 min | all tables and statistics in the paper |
| **C Full re-run** | codex CLI + API key + dataset + 30h+ | 30h+ | re-run from scratch, including agent execution |

This package **does not ship the recorded data** (17GB in full). Route B needs you to supply the
artifacts; route C generates them from scratch.

---

## 1. Environment and dataset

### 1.1 Dependencies

```bash
python3 --version          # needs >= 3.10 (the code uses X | Y type annotations)
pip install -r requirements.txt
```

Only `numpy`. **No** scipy / sklearn / torch -- the estimators and tests (IRLS + Laplace,
permutation test, bootstrap, Spearman, BH-FDR) are all hand-written, so reproduction is immune to
version drift.

### 1.2 Configuration

```bash
cp env.example.sh env.sh
vim env.sh          # at minimum fill in IDEALAB_API_KEY (needed for collection only) and the dataset path
source env.sh
```

### 1.3 Dataset

The paper's corpus is the 500 instances of SWE-bench-Verified. The code needs two things:

1. `instance_summary.csv` -- fields `instance_id, base_commit, problem_statement, patch_file`,
   where `patch_file` is the file paths touched by the gold patch (newline-separated), **this is
   exactly the paper's F\***
2. `projects/<instance_id>/` -- one git repository snapshot per instance, HEAD parked at `base_commit`

```bash
# Generate the CSV only (about 900KB, takes seconds)
python3 setup/prepare_dataset.py --from-hf --skip-clone
# Or use a local SWE-bench-Verified JSON
python3 setup/prepare_dataset.py --from-json /path/to/instance_questions.json

# Clone the repository snapshots (tens of GB, several hours; required for route C, not for A/B)
python3 setup/prepare_dataset.py --from-hf --clone --workers 4

# Already have snapshots? Just check whether they are parked at base_commit (strongly recommended)
python3 setup/prepare_dataset.py --verify-only
```

`--verify-only` reports instances that are `dirty` / `wrong_commit`. **Any single non-ok instance
means leakage risk**, because distributed snapshots often have the gold patch already applied
(= already-fixed state).

### 1.4 Self-check

```bash
python3 setup/check_env.py            # offline-analysis profile
python3 setup/check_env.py --collect  # collection profile: additionally checks git / codex CLI / API key
```

---

## 2. Route A: self-check (needs no data at all)

```bash
python3 -m pilot.analyze.uq_selftest
```

Reproduces Appendix D.7 of the paper -- validating the three-way decomposition on synthetic data
with a **known generating process**:

| Quantity | Measured here | Paper |
|---|---|---|
| Parameter recovery corr(w_true, MAP) | 0.9935 | 0.994 |
| In-support probability MAE | 0.022 | 0.022 |
| Predictive entropy decomposition | 0.4638 = 0.3075 + 0.1458 + 0.0124 | 0.464 = 0.308/0.146/0.012 |
| Epistemic off-support inflation | 0.0124 → 0.1002 (×8.08) | ×8.1 |

The **boundary** of this validation must be stated clearly: it shows the three-way decomposition
machinery works when ground truth is known, it does **not** show the three components are
identifiable on real traces. Paper D.7 explicitly states the latter is not established.

---

## 3. Route C: full re-run

Four stages. Every stage script can be re-run independently, and each validates its upstream
artifacts.

### Stage 0: feasibility validation (~10-30 min)

```bash
bash scripts/00_smoke.sh
```

Clears the five unknowns that decide whether the method is feasible, and produces the two
prerequisite validations reported in §5.2 of the paper:

- **Replay consistency 1.00** (100 actions byte-identical) → assumption 3 "replayable environment"
- **Compression events 0** → assumption 1 "context observable"

At the same time `probe_v5` measures action entropy = a **direct measurement of positivity**. One
of the paper's key findings is that positivity is a **local** property: 50% of probe points have
action randomness, and its rank correlation with step position is only +0.072 (**not** a monotone
decay -- this is one pre-registered item that was not supported, and the paper reports it
faithfully).

> If any criterion fails, **do not proceed**. Replay inconsistency means the environment is not
> pinned down (see `pilot/config/env_pin.sh`), and every causal estimate downstream is
> untrustworthy.

### Stage 1: full file localization (6-way parallel, ~10-20 hours)

```bash
bash scripts/01_locate.sh
# Validate the pipeline on a small sample first:
SCAE_LIMIT=10 bash scripts/01_locate.sh
```

Produces the corpus base for every conclusion:

- `pilot/runs/full_locate/results.jsonl` -- per-instance Y / recall / n_steps / predicted file set
- `pilot/runs/full_locate/shard*/rollouts/*.jsonl` -- full traces (the input to prefix replay)

At the end the script prints per-repository localization success rates; compare against Table 7 of
the paper (overall **93.99%**).

**Leakage gate**: before each instance runs, `git reset --hard <base_commit>` + `clean -fdq` is
forced (`pilot/probes/task.py::reset_repo`). This is the last gate; do not bypass it.

### Stage 2: prefix-replay attribution (~12.7 hours, the most expensive)

```bash
bash scripts/02_attribution.sh          # run both profiles
bash scripts/02_attribution.sh judge    # RQ5 profile only
bash scripts/02_attribution.sh power    # RQ4 profile only
```

Two profiles are run because the paper uses them to answer two different questions:

| Profile | Scale | m | Purpose |
|---|---|---|---|
| `full_attr` | 70 episodes / 10 repos / 19 failures | 3 | RQ5 judge comparison (D.10 / D.11) |
| `v4prime` | 38 episodes → **190 valuation points** | **8** | RQ4 step-level power (§5.6) |

The "1901 continuations over 12.7 hours" in §5.2 of the paper is the `v4prime` profile.

The script first calls `setup/stage_attr_shards.py` to arrange the localization results into the
shape `parallel --cmd attr` expects (this step was done by hand in the original experiment; here it
is fixed in code). At the end it prints the episode count, valuation-point count, and degenerate
rate, which can be compared directly against the paper (the `full_attr` profile should be
**22.9% = 16/70** degenerate, and valuation points with Δ exactly 0 should be **54.7%**).

### Stage 3: judge cross-model replication (~11 hours, dominated by rate limiting)

```bash
bash scripts/03_judge_crossmodel.sh
```

Reproduces Table 2 of the paper. The core manipulation: **the model and rubric are held completely
fixed, only the information set the judge can see is changed**.

- `full-trace`: conditions on prefix + subsequent steps + realized outcome Y ⇒ conditions on
  descendants of A_t ⇒ collider bias
- `prefix-only`: conditions on a fixed 60% prefix, outcome Y hidden

If the full-trace judge were approximating a prefix-conditional step-local quantity, this
manipulation **should not** move the blame location. Measured shift is **+0.537**, sign test
p = 1.1×10⁻¹⁶, positive for all three judges.

> **Rate limiting**: default is 10 calls per 60-minute window, 25 episodes × 2 conditions = 50
> calls ≈ 5.5 hours per model. If a run does not finish you do not have to start over:
> `x9_judge_crossmodel` supports `--analyze-only`, recomputing on existing JSONL.

### Stage 5: K=4 repeated rollouts for U_policy (~2-4 hours)

```bash
bash scripts/05_upolicy.sh
```

In the three-way decomposition `H[Y|q,G] = U_task + U_policy + U_model`, only `U_policy` can be
estimated **directly** from real traces: hold q fixed, run K times, and take the mean of the
per-instance outcome variances. The paper reports **0.049** (23 K=4 instances).

The main pipeline uses K=1 (500 × 4 is too expensive), so this profile is run separately on a small
subset.

> **Honest note on the reproduction definition**: the 23 instance ids the paper actually used are a
> data artifact and **cannot be derived from the code**. The script reselects them with a hard-coded
> sampling rule (localization succeeded + step count ≥ 5 + stratified round-robin by repository), so
> your `U_policy` will be close to but not necessarily equal to 0.0489. If you have the original
> list, point `SCAE_UPOLICY_IDS` at it.
> Verified: feeding the original `upolicy/results.jsonl` into this package's `t3_uq_decomposition`
> produces output **field-for-field identical** to the paper's artifact (0.0489) -- i.e. the
> estimator implementation has no deviation, the difference comes only from instance sampling.

### Stage 4: offline analysis (minutes, zero API)

```bash
bash scripts/04_analysis_offline.sh            # run everything
bash scripts/04_analysis_offline.sh rq1        # one group: rq1|rq2|rq3|rq4|rq5|report|selftest
```

Every number in the paper is written into some JSON by one of the scripts here. See
[docs/PAPER_MAP.md](docs/PAPER_MAP.md) for the item-by-item mapping. When an upstream artifact is
missing the script **reports the skip explicitly**; it will not silently emit a truncated
conclusion.

---

## 4. Route B: offline analysis only

If you already have a set of recorded artifacts:

```bash
export SCAE_RUNS=/path/to/existing/runs
bash scripts/04_analysis_offline.sh
```

The minimum structure `pilot/runs/` needs:

```
runs/
├── full_locate/
│   ├── results.jsonl              localization results (merged)
│   └── shard*/rollouts/*.jsonl    traces (RQ1-RQ3 depend entirely on these)
├── full_attr/
│   ├── attribution.jsonl          70 episodes, m=3
│   └── shard*/attribution.jsonl
├── v4prime/
│   ├── attribution.jsonl          38 episodes, m=8, includes ys_t detail
│   └── shard*/attribution.jsonl
├── v5_scan/*/v5_report.json       entropy probe (C.4 / Table 6)
├── graphs/<instance_id>/          graph construction artifacts (RQ2, 71 instances)
└── tool_model_official.json       tool channel (generated by the rq1 group in stage 4)
```

Size reference: the `rollouts` under `full_locate` are about 85MB, the `attribution.jsonl` files
total about 300KB, derived JSON about 3.6MB. What actually takes up space are the fork intermediate
artifacts (`attr_work/`) and `graphs/`. Offline analysis alone **does not need** `attr_work/`.

---

## 5. External dependencies of the graph channel (known gap in RQ2)

`pilot/runs/graphs/<instance_id>/` is generated by `pilot/channels/graph_builder.py`, which calls
two upstream tools that are **not distributed with this package**:

```
tree-sitter-fold/AST_Tree_Graph      →  entities.json, dependency_graph.json
graph_leidan_fold/rag-python-tools   →  cluster_entities.json, cluster_communities.json
```

Full reproduction of RQ2 therefore requires these two tools. Handling when they are missing:

- `scripts/04_analysis_offline.sh rq2` explicitly reports the missing `graphs/` and degrades to
  reporting only the parsable subset
- `pilot/graph/ast_index.py` is a **pure stdlib** AST index (symbol definition sites, import edges,
  graph distance) and can serve as an alternative source for the dependency graph, but the graph it
  produces is not the same one the paper used, so the numbers are not directly comparable

This gap does not affect the provenance half of RQ1: the §5.3 conclusion "execution provenance
beats code structure" actually expands from 509 decision points to 7616 once the code graph is
dropped (Appendix D.2), because provenance only needs string normalization.

---

## 6. Budget summary

| Stage | Time | API | Disk |
|---|---|---|---|
| 0 smoke | 10-30 min | small | < 1GB |
| 1 localization | 10-20 h (6-way parallel) | ~500 long sessions | ~1.4GB + traces |
| 2 attribution | ~12.7 h | ~3000 continuations | ~8.6GB (incl. fork intermediates) |
| 3 judge | ~11 h (rate-limit dominated) | ~96 calls | < 100MB |
| 4 analysis | minutes | **0** | ~4MB |

Parallelism is constrained both by API rate limits and by disk IO. `SCAE_WORKERS=6` is the value
the paper actually used: sharded by repo, guaranteeing that the same repository directory is reset
by only one worker at a time.

---

## 7. How to check reproduction results

The paper claims every number can be regenerated by CPU scripts. How to check:

1. Run `scripts/04_analysis_offline.sh`; artifact JSON lands in `$SCAE_RUNS/`
2. Compare JSON fields against the paper's numbers item by item following
   [docs/PAPER_MAP.md](docs/PAPER_MAP.md)
3. Mind the sources of randomness: bootstrap CIs and permutation tests will fluctuate in the last
   digit; the point estimates and the order of magnitude of p values should be stable

**Places where deviation is known to occur** (recorded faithfully in the paper):

- Attribution estimates carry Monte Carlo error, which attenuates regression coefficients toward 0
  (Appendix E.1)
- `v_t` is estimated at equally spaced points rather than every step, so Σ Δ_t is not exactly equal
  to v_T − v_0; the paper therefore does **not** report additivity as evidence (Appendix E.1)
- Entropy falls on a coarse grid (with 4 replays the only nonzero attainable values are 0.56 / 1.04
  / 1.39 nats, Table 6)

---

## 8. When things go wrong

| Symptom | Most likely |
|---|---|
| Localization success rate clearly above 93.99% | **gold patch leakage**, the repository was not reset clean → `--verify-only` |
| Replay consistency < 1.00 | environment not pinned → confirm `source pilot/config/env_pin.sh` took effect |
| Nonzero compression events appear | assumption 1 fails on that trace; attribution for that trace cannot be used directly |
| The judge stage hangs | rate limiting. Use `--analyze-only` to draw conclusions from the data you already have |
| Analysis script reports missing files | an upstream stage did not finish; the script names which artifact is missing |
| Query channel top-3 abnormally low | the known `shlex` parsing defect (26.6% of command segments), see Appendix D.5 |

More in [docs/PITFALLS.md](docs/PITFALLS.md).
