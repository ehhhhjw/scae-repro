# Script Inventory and Status

All code is split into **core** and **legacy** by whether the paper cites it.
The legacy scripts are kept because the paper's chain of argument needs them (mostly the link that
was "refuted/superseded by later analysis"), but they do not produce final-draft numbers.

In artifact paths, `$RUNS` = `pilot/runs` (or `SCAE_RUNS`).

---

## Execution layer `pilot/harness/`

| File | Purpose | Called by |
|---|---|---|
| `appserver.py` | JSON-RPC/stdio client for codex app-server; `thread/start`, `thread/fork`, `turn/start`, three-way message demultiplexing; `PINNED_ENV` | all collection scripts |
| `rollout.py` | read/write/truncate/rewrite of rollout JSONL -- **the low-level primitive for prefix replay**. `truncate_before_step`, `substitute_action`, `mask_output_lines`, `action_bucket`, `assert_no_dangling_calls`, `compaction_lines` | all |
| `runner.py` | batch file localization; writes `config.toml`, forced reset, parses `<answer>`, resume from checkpoint | `parallel.py` |
| `parallel.py` | parallelism sharded by repo; locate and attr modes; merges per-shard JSONL | `scripts/01`, `scripts/02` |
| `action_exec.py` | **real execution** of read-only substitute actions (source of o' in the intervention branch); allowlist + deny set | `attribution/causal.py` |

## Causal estimation layer `pilot/attribution/`

| File | Purpose |
|---|---|
| `causal.py` | core estimators: `estimate_v` (estimates v_t over m replays), `causal_attribution` (Δ_t = v_t − v_{t−1}), `estimate_policy_at` (estimates π̂_t), `hybrid_delta_at` (**τ_H branch: observational replay vs soft intervention**) |
| `judge.py` | full-trace / prefix-only LLM judge baselines (the arms under attack) |
| `run_attribution.py` | attribution orchestration: equally spaced valuation points, three outcome definitions (binary/per_file/recall) |

## Uncertainty layer `pilot/uq/`

| File | Purpose | Status |
|---|---|---|
| `predictor.py` | Bayesian logistic (IRLS + Laplace posterior), three-way decomposition, ECE/Brier/NLL/AUROC/AURC | core |
| `features.py` | **pre-execution** features for v₀ (7 covariates, **does not read the trace**, which would leak execution-time information) | core |
| `nextstep.py` | next-action prediction (8 features + LORO) | early version, superseded by `analyze/x1c` |
| `anytime.py` | anytime AUROC(t)/ECE(t) curves | did not make the final draft |

## Channel layer `pilot/channels/`

| File | Purpose | Notes |
|---|---|---|
| `parsed_command.py` | Python reimplementation of codex `ParsedCommand`, classifies shell commands into (kind, path/query) | **the definition of the action semantics layer**; known `shlex` defect: see PITFALLS #6 |
| `tool_model.py` | tool channel Markov models (M0-M3), produces H_tool | key implementation detail: `exec_command` accounts for 8159/8200 calls, **the semantic tool is the shell command head, not the nominal API** |
| `code_channel.py` | code dependency graph channel U^obj, three-layer structural prior (W=0.62/0.30/0.08) | requires `$RUNS/graphs/` |
| `graph_builder.py` | graph construction orchestration | **depends on two external tools not shipped with this package**, see REPRODUCE.md §5 |

## Task layer `pilot/probes/` and `pilot/graph/`

| File | Purpose |
|---|---|
| `probes/task.py` | **single source of truth for the dataset interface**: instance loading, prompt, outcome decision Y, `reset_repo` (prevents answer leakage) |
| `graph/ast_index.py` | pure stdlib AST index (symbol definition sites, import edges, graph distance), no tree-sitter dependency |
| `graph/ast_probe.py` | whether AST ambiguity features separate success from failure (Mann-Whitney U) |
| `graph/step_semantics.py` | semantic classification of steps (E5) and AST distance distribution over adjacent steps (E3) |

## Feasibility probes `pilot/smoke/`

| File | Purpose |
|---|---|
| `smoke.py` | V1-V4 of S0: fork feasibility, read-only tool surface, no approval requests, `Continue.` semantics |
| `probe_v5.py` | action entropy probe = **direct measurement of positivity** (paper C.4 / Figure 5) |

---

## Analysis layer `pilot/analyze/` -- core

Grouped by research question. All zero-API (except the annotator).

### RQ1 next-action prediction (§5.3 / Tables 1, 10, 11)

| Script | Purpose | Artifact |
|---|---|---|
| `v3_nextstep_dual.py` | **shared episode loading library** (imported by the 7 scripts below, do not move) | -- |
| `v3g_regime.py` | 509 decision points on a shared denominator + first-visit/revisit dual regime decomposition | `v3g_regime.json` |
| `v3j_trace_graph.py` | **execution provenance graph prediction (main row of Table 1)** | `v3j_trace_graph.json` |
| `v3b_distribution_quality.py` | top-k curves / MRR / ECE / NLL / exp(H) | `v3b_distribution.json` |
| `x1_joint_l1.py` | H_tool distribution and stratification | `x1_joint_l1.json` |
| `x1b_joint_independence.py` | tool × object conditional independence (I = 0.159 nats) + per-stratum reliability (Table 11) | `x1b_joint_independence.json` |
| `x1c_tool_routed.py` | candidate set routed by predicted tool (Table 10), sample grows from 71 to ~496 instances | `x1c_tool_routed.json` |
| `x1d_query_channel.py` | query channel diagnostics + quantification of the `shlex` defect (D.5) | `x1d_query_channel.json` |

### RQ2 code dependency graph (§5.4 / Figure 7)

| Script | Purpose | Artifact |
|---|---|---|
| `v3i_graph_probe.py` | multi-angle graph evidence: adjacency vs permutation null, k-hop coverage, gold ranking, visited-set clustering | `v3i_graph_probe.json` |

### RQ3 reuse patterns (§5.5 / D.4)

| Script | Purpose | Artifact |
|---|---|---|
| `v3h_reuse.py` | first shows the reuse **rate** is a permutation invariant (zero information), then tests temporal **structure** | `v3h_reuse.json` |

### RQ4 uncertainty hierarchy (§5.6 / Figure 8 / D.6, D.7)

| Script | Purpose | Artifact |
|---|---|---|
| `x4_uexec_corr.py` | corr(H_struct, U^exec); partial correlation controlling for v_t | `x4_uexec_corr.json` |
| `x4b_uexec_validity.py` | whether U^exec has **non-mechanical** structure (residualized ICC is the key) | `x4b_uexec_validity.json` |
| `x4c_instance_level.py` | instance-level aggregate cross-check | `x4c_instance_level.json` |
| `x5_step_power.py` | **step-level power + BH-FDR (the paper's core negative result)** | `x5_step_power.json` |
| `x6_budget_alloc.py` | entropy-guided budget allocation (reports 5×, later refuted by x6b) | `x6_budget_alloc.json` |
| `x6b_effect_criterion.py` | shows the 5× above is a binary-threshold artifact (α sweep 3.36→1.71) | `x6b_effect_criterion.json` |
| `v4_structural_vs_exec.py` | Bernoulli approximation U^exec = 4v(1−v); kept in paper C.3 as a **negative control** | `v4_structural_vs_exec.json` |
| `process_uncertainty.py` | entropy traces, fragile successes, convergence point t*, degenerate rate | `process_uncertainty.json` |
| `estimability.py` | three-state accounting INESTIMABLE / NO_EFFECT / EFFECT | `estimability.json` |
| `x8_outcome_sensitivity.py` | intervention sensitivity screening over six outcome variables (C.6) | `x8_outcome_sensitivity.json` |
| `uq_selftest.py` | **synthetic self-check of the three-way decomposition (D.7, the only positive verification)** | `uq_selftest.json` |
| `entropy_scan.py` | entropy structure summary (C.4 / Table 6 / Figure 5) | `entropy_scan_summary.json` |

### RQ5 attribution and collider bias (§5.7 / Tables 2, 12 / Figure 4)

| Script | Purpose | Artifact |
|---|---|---|
| `attribution_compare.py` | D₁ / D₃ and **collider isolation** (+0.537) | `d1d3_report.json` |
| `v3k_trace_causal.py` | whether \|Δ_t\| concentrates on gold-provenance; landing points of judge/causal top-1 | `v3k_trace_causal.json` |
| `x2_judge_mechanism.py` | whether zero-LLM structural rules can reproduce the judge (Table 12) + selection-space correction | `x2_judge_mechanism.json` |
| `judge_http.py` | direct `/chat/completions` connection (qwen streaming is incompatible with `/responses`) **requires API key** | -- |
| `x9_judge_crossmodel.py` | cross-model replication (Table 2) **requires API key**, supports `--analyze-only` | `x9_*.json` |
| `x9b_crossmodel_pool.py` | summary: displacement should be homogeneous, enrichment should be heterogeneous | `x9b_crossmodel_pool.json` |

### Infrastructure

| Script | Purpose | Artifact |
|---|---|---|
| `full_report.py` | aggregate of paper tables (localization Table 7, v₀ LORO D.8) | `analysis.json` |
| `x_synthesis.py` | aggregates the individual JSONs for use by the paper's macro definitions | `synthesis.json` |
| `replay_check.py` | replay consistency (empirical verification of Assumption 3) | `replay_check.json` |
| `recover_v5_report.py` | crash recovery tool: rebuilds `v5_report.json` from leftover forks | -- |

---

## Analysis layer `pilot/analyze/legacy/` -- superseded / did not make the final draft

Reason for keeping: the paper argues by "report a result first, then refute/supersede it with a
stricter test". Deleting these would leave the reader unable to check the refuted link.

| Script | Original purpose | Why legacy |
|---|---|---|
| `v3f_ablation.py` | ablation of the L1/L2/L3 three-layer structural prior | superseded by `v3g_regime` + `v3i_graph_probe` |
| `m_robustness.py` | argues m=3 suffices (Spearman agreement of m3 vs m8) | the final draft's main-text definition is **m=8**, so this argument is no longer needed |
| `x7_fragile_l1.py` | fragile success ↔ H_tool association | did not make the final draft; the related conclusion is carried by `process_uncertainty` |
| `x3_prov_mask.py` | do-calculus experiment on provenance masking (3-arm PSE) **requires API key** | the paper does not report PSE; this design was judged infeasible at the current budget |
| `repair_runner.py` | four-arm repair experiment E1 (formerly `attribution/repair.py`) **requires API key** | the paper does not report RepairSuccess@1 |
| `repair_analysis.py` | analysis side of the above | same as above |

Invocation becomes `python3 -m pilot.analyze.legacy.<name>`.
The `plan` / `analyze` modes of `x3_prov_mask` are zero-API; only `run` mode needs a key.

---

---

## Orchestration entry points `scripts/` and `setup/`

| Script | Stage | Needs API | Runtime |
|---|---|---|---|
| `setup/check_env.py` | pre-flight self-check (two tiers) | no | seconds |
| `setup/prepare_dataset.py` | build CSV + repo snapshots | no | seconds ~ hours |
| `setup/stage_attr_shards.py` | attribution shard preparation | no | seconds |
| `scripts/00_smoke.sh` | feasibility V1-V5 + replay consistency | **yes** | 10-30 min |
| `scripts/01_locate.sh` | localization over 499 instances | **yes** | 10-20 h |
| `scripts/02_attribution.sh` | prefix-replay attribution (two tiers) | **yes** | ~12.7 h |
| `scripts/03_judge_crossmodel.sh` | cross-model judge (Table 2) | **yes** | ~11 h (rate-limited) |
| `scripts/05_upolicy.sh` | K=4 repetitions for U_policy (D.7) | **yes** | 2-4 h |
| `scripts/04_analysis_offline.sh` | all offline analysis | **no** | a few minutes |
| `scripts/common.sh` | common preamble (environment pinning + pre-flight checks) | -- | -- |
| `scripts/run_all.sh` | full pipeline | **yes** | 30h+ |

`pilot/legacy_pipelines/` holds the original orchestration scripts (`run_pilot.sh`, `run_remaining.sh`),
kept only for provenance reference -- they reference data artifacts not shipped with the package, so
running them directly will fail. See that directory's README for the correspondence.

---

## One pitfall in the dependency structure

`v3_nextstep_dual.py` sounds like "early exploration", but it is the **episode loading library
imported by 7 core scripts** (`v3b`, `v3g`, `v3h`, `v3i`, `x1b`, `x1c`, plus legacy `v3f`).
Moving or deleting it breaks all of the RQ1-RQ3 analysis at once.

Other key import edges:

```
v3j_trace_graph  ← v3k, x1b, x1c, x1d, x4b, x5, x6b, x8
v3k_trace_causal ← x2, x4b, x4c, x5, x6b, x8, x9, x9b
v3g_regime       ← v3j, x1b, x1c, x1d
```

Changing the public function signatures in these three files ripples across a large swath of downstream code.
