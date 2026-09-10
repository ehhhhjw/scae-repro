# Paper numbers → scripts → artifacts map

Every number reported in the paper is written into some JSON by some CPU script. This table gives the
correspondence, for cross-checking reproduction results. `$RUNS` = `pilot/runs` (or `SCAE_RUNS`).

Annotation key: **✅ verified by actual run** = this package's code actually produced the number and it
matches the paper; the rest are mappings established from code paths and paper text (checking them
requires the corresponding recorded artifacts).

---

## Main-text claims

### §5.2 Experimental setup and pre-flight verification

| Paper number | Script | Artifact |
|---|---|---|
| 499 episodes / 12 repositories | `scripts/01_locate.sh` | `$RUNS/full_locate/results.jsonl` |
| localization success rate 93.99% (Table 7) | `pilot.analyze.full_report` | `$RUNS/analysis.json` |
| replay consistency 1.00 / 100 actions | `pilot.analyze.replay_check` | `$RUNS/replay_check.json` |
| 0 compaction events | `rollout.compaction_lines()`, produced by the above | `$RUNS/replay_check.json` |
| movable fraction: recall 98.6% vs binary 73.2% | `pilot.analyze.x8_outcome_sensitivity` | `$RUNS/x8_outcome_sensitivity.json` |
| 190 prefixes × 8 continuations / 38 instances / 12.7h | `scripts/02_attribution.sh power` | `$RUNS/v4prime/attribution.jsonl` |

> **500 vs 499**: the CSV has 500 rows; `django__django-15278` is missing its repo snapshot, so the corpus is 499.

### §5.3 RQ1 what predicts the next action (Table 1)

All predictors share the same denominator: points where the realized object falls outside the candidate set are counted as a miss rather than discarded.

| Paper number | Script | Artifact field |
|---|---|---|
| 341 decision points at first-visit steps | `pilot.analyze.v3g_regime` | `$RUNS/v3g_regime.json` |
| provenance (last, position-ordered) top-3 **0.326** CI[0.276,0.375] | `pilot.analyze.v3j_trace_graph` | `$RUNS/v3j_trace_graph.json` |
| provenance (recency-weighted) top-3 0.308, top-10 0.499 | same as above | same as above |
| code graph (enhanced) top-3 **0.058** CI[0.039,0.078] | `pilot.analyze.v3i_graph_probe` | `$RUNS/v3i_graph_probe.json` |
| repository frequency (marginal) top-3 0.072 | `pilot.analyze.v3g_regime` | same as above |
| visit-history frequency coverage 0.000 (empty by construction at first-visit steps) | same as above | same as above |
| paired improvement top-3 +0.136 CI[+0.102,+0.173] | `pilot.analyze.v3j_trace_graph` | same as above |
| effective candidate count exp(H): provenance **3.46** vs graph 1719.13 | same as above / `v3i_graph_probe` | same as above |
| candidate efficiency 0.689 of first-visit targets already appeared in earlier output | `pilot.analyze.v3j_trace_graph` | same as above |
| handler 27.3% top-3 0.582 / path 51.1% 0.185 / query 21.6% **0.029** | `pilot.analyze.x1c_tool_routed` | `$RUNS/x1c_tool_routed.json` |
| tool × object mutual information I = **0.159** nats, p = 5×10⁻⁴ | `pilot.analyze.x1b_joint_independence` | `$RUNS/x1b_joint_independence.json` |
| routing top-1 0.109 → 0.150; exp(H) 39.93 → 12.85 (Table 10) | `pilot.analyze.x1c_tool_routed` | same as above |
| after search top-3 0.469 ECE 0.013; after read ECE 0.160 (Table 11) | `pilot.analyze.x1b_joint_independence` | same as above |

### §5.4 RQ2 what the code dependency graph is good for (Figure 7)

The null hypothesis is **within-episode permutation**: keep the multiset of visited items fixed, shuffle only the order.

| Paper number | Script | Artifact field |
|---|---|---|
| one-hop lift **0.94×**, three-hop 1.02×, same-community 1.01× | `pilot.analyze.v3i_graph_probe` | `$RUNS/v3i_graph_probe.json` |
| distance 0 (immediate self-repeat) lift 1.47× | same as above | same as above |
| internal edge density of the visited set **0.269** vs null 0.0087 (**31×**) | same as above | same as above |
| rank quantile of unseen gold files 0.049 CI[0.021,0.084] | same as above | same as above |
| 65% of gold files were already visited at ranking time (leakage control) | same as above | same as above |
| 46.2% of agent objects resolve to a graph node | same as above | same as above |
| raw one-hop rate 0.284 vs permutation expectation 0.301 (D.3) | same as above | same as above |

### §5.5 RQ3 which reuse patterns are real ✅ verified by actual run

| Paper number | Measured in this package | Artifact field |
|---|---|---|
| tool reuse rate 79.1% | **0.7912** CI[0.7815,0.8005] | `$RUNS/v3h_reuse.json` |
| object reuse rate 41.6% | **0.4159** CI[0.4001,0.4313] | same as above |
| permutation-null std of the reuse rate is **exactly 0** | **0.00e+00** | same as above |
| object immediate repeat 1.83×, p ~ 10⁻¹⁴¹ | **1.83×**, p ~ 1e-141 | same as above |
| tool self-repeat 1.02×, p = 0.51 (**not significant**) | **1.02×**, p = **0.511** | same as above |
| tool channel mutual information 1.13×, p ~ 10⁻²² | **1.13×**, p ~ 1e-22 | same as above |
| visit-count preference lift 1.70× CI[1.63,1.76] | **1.6965** CI[1.6335,1.7618] | same as above |
| recency-rank preference lift 1.74× CI[1.67,1.81] | **1.739** CI[1.6702,1.809] | same as above |
| 487 episodes | **n=487 ep** (491 loaded) | same as above |

Script: `pilot.analyze.v3h_reuse --runs-root $RUNS/full_locate`

### §5.6 RQ4 which layer execution uncertainty lives in (Figure 8)

| Paper number | Script | Artifact field |
|---|---|---|
| step-level ρ = **−0.088** CI[−0.244,+0.062] | `pilot.analyze.x4_uexec_corr` | `$RUNS/x4_uexec_corr.json` |
| instance-level ρ = −0.219 | `pilot.analyze.x4c_instance_level` | `$RUNS/x4c_instance_level.json` |
| positive control ρ = **−0.425** CI[−0.695,−0.102] | `pilot.analyze.x4_uexec_corr` | same as above |
| instance ICC **0.640** vs permutation null 0.196, p = 5×10⁻⁴ | `pilot.analyze.x4b_uexec_validity` | `$RUNS/x4b_uexec_validity.json` |
| step position ρ = −0.131; tool type p = 0.65 (neither significant) | same as above | same as above |
| **5.3% of 190 effects reach uncorrected p<0.05, 0 survive BH-FDR** | `pilot.analyze.x5_step_power` | `$RUNS/x5_step_power.json` |
| at m=8 the median replay variance is 0; degenerate 22.9% (16/70) | `pilot.analyze.process_uncertainty` | `$RUNS/process_uncertainty.json` |
| threshold sweep ratio 3.36 → 1.71 (the artifact in D.6) | `pilot.analyze.x6b_effect_criterion` | `$RUNS/x6b_effect_criterion.json` |
| the refuted original 5× result | `pilot.analyze.x6_budget_alloc` | `$RUNS/x6_budget_alloc.json` |
| Bernoulli approximation of U^exec (negative control in C.3) | `pilot.analyze.v4_structural_vs_exec` | `$RUNS/v4_structural_vs_exec.json` |

### §5.7 RQ5 what observational attribution actually measures (Figure 4 / Table 2 / Table 12)

| Paper number | Script | Artifact field |
|---|---|---|
| collider displacement **+0.537** CI[+0.459,+0.610], sign test p = 1.1×10⁻¹⁶ | `pilot.analyze.attribution_compare` | `$RUNS/d1d3_report.json` |
| 59/70 episodes shift the verdict later; 0.214 → 0.751 | same as above | same as above |
| judge top-1 lands on a gold-provenance step **0.900** vs baseline 0.413 | `pilot.analyze.v3k_trace_causal` | `$RUNS/v3k_trace_causal.json` |
| within-class late bias 0.746 vs uniform null 0.5 CI[+0.172,+0.311] | `pilot.analyze.x2_judge_mechanism` | `$RUNS/x2_judge_mechanism.json` |
| best zero-LLM rule "the last gold-provenance step" 0.243 vs within-class 0.139 (1.7×, Table 12) | same as above | same as above |
| displacement across three judges +0.533 / +0.434 / +0.394 (Table 2) | `pilot.analyze.x9_judge_crossmodel` | `$RUNS/x9_{general,coder}.json` |
| heterogeneity across judges: displacement p = 0.181 (none); enrichment p = 0.044 (present) | `pilot.analyze.x9b_crossmodel_pool` | `$RUNS/x9b_crossmodel_pool.json` |
| D₁ disagreement rate 97.1% (**the paper has retired this criterion**, see D.12) | `pilot.analyze.attribution_compare` | `$RUNS/d1d3_report.json` |
| interventional baseline top-1 mean position 0.333 | same as above | same as above |

---

## Appendices

| Appendix | Content | Script | Artifact |
|---|---|---|---|
| **C.4** | entropy at 10 probe points: 50% non-zero, ρ = +0.072 (Table 6 / Figure 5) | `pilot.analyze.entropy_scan` + `pilot.smoke.probe_v5` | `$RUNS/v5_scan/entropy_scan_summary.json` |
| **C.6** | outcome variable screening: recall 2.505 > F₁ 2.165 > binary 1.643 | `pilot.analyze.x8_outcome_sensitivity` | `$RUNS/x8_outcome_sensitivity.json` |
| **D.1** | localization success rate by repository (Table 7) | `pilot.analyze.full_report` | `$RUNS/analysis.json` |
| **D.2** | 50.5 files mentioned per episode, 6.5 actually visited (18.0% conversion); 7616 decision points | `pilot.analyze.v3j_trace_graph` / `x1c_tool_routed` | `$RUNS/v3j_trace_graph.json` |
| **D.5** | `shlex` fails on 26.6% of command segments; 75.3% of query objects affected | `pilot.analyze.x1d_query_channel` | `$RUNS/x1d_query_channel.json` |
| **D.6** | the artifact manufactured by a binary criterion + α ∈ {0.01,0.05,0.10,0.20} sweep | `pilot.analyze.x6b_effect_criterion` | `$RUNS/x6b_effect_criterion.json` |
| **D.7** ✅ | synthetic self-check of the three-way decomposition: 0.994 / 0.022 / 0.308-0.146-0.012 / ×8.1 | `pilot.analyze.uq_selftest` | `$RUNS/uq_selftest.json` |
| **D.8** | LORO AUROC of v₀ **0.481** (at chance), ECE 0.015, Brier 0.058, AURC 0.058 | `pilot.analyze.full_report` | `$RUNS/analysis.json` |
| **D.10** | attribution sample 70 episodes / 10 repositories / 19 failures; judge in-window 32.9% (n=23) | `pilot.analyze.v3k_trace_causal` | `$RUNS/v3k_trace_causal.json` |
| **D.11** | \|N\|=54 non-degenerate episodes; Δ is exactly 0 at 54.7% of valuation points | `pilot.analyze.estimability` | `$RUNS/estimability.json` |
| **D.12** | why D₁ was retired | `pilot.analyze.attribution_compare` | `$RUNS/d1d3_report.json` |
| **E.2** | pre-registration record vs results (Table 9) | summary across scripts | `$RUNS/synthesis.json` |

### Commands explicitly recorded in the paper's Appendix E.1

```bash
python3 -m pilot.analyze.uq_selftest      # D.7 three-way decomposition self-check
python3 -m pilot.analyze.entropy_scan     # C.4 entropy structure
python3 -m pilot.analyze.full_report      # localization and prediction tables
```

(There are also two plotting scripts and two paper validation scripts, excluded from this package as requested.)

---

## Artifact structure

```
$RUNS/
├── full_locate/
│   ├── results.jsonl                  ★ localization results (RQ1-RQ3 + v₀ corpus)
│   └── shard*/rollouts/*.jsonl        ★ traces
├── full_attr/
│   ├── attribution.jsonl              ★ 70 episodes, m=3 (RQ5)
│   └── shard*/{attribution.jsonl,attr_work/}
├── v4prime/
│   ├── attribution.jsonl              ★ 38 episodes, m=8, includes per-replay ys_t (RQ4)
│   └── shard*/
├── v5_scan/<instance>/v5_report.json  entropy probe
├── graphs/<instance>/                 graph artifacts (RQ2, 71 instances, produced by external tools)
├── tool_model_official.json           tool channel (produced by the stage-4 rq1 group)
└── *.json                             outputs of the individual analysis scripts (~3.6MB)
```

One episode per line in `attribution.jsonl`; key fields:

| Field | Meaning |
|---|---|
| `Y`, `recall`, `n_gold`, `missed_files` | outcome variables (the paper uses `recall`) |
| `n_steps` | trace length T (used to normalize position t/T) |
| `causal_v0` | pre-execution prediction v₀ = E[Y \| q, G] |
| `causal_segments[]` | per valuation point: `t`, `delta`(Δ_t), `v_t`, `v_prev`, `pos`, `action` |
| `causal_segments[].ys_t` | per-replay outcomes over m replays (**only in the m=8 tier**, the source of U^exec) |
| `causal_top1_step/pos/delta` | causal top-1 attributed step |
| `judge_full_top1_step/pos` | attribution by the full-trace judge (collider-bias arm) |
| `judge_prefix_top1_step/pos` | attribution by the prefix-only judge (control arm) |
| `degenerate` | whether all Δ are identically 0 for this episode |

Collider displacement is defined as the per-episode mean of `pos(judge_full_top1) − pos(judge_prefix_top1)`.
