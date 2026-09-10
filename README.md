# SCAE -- Reproducibility code package for the paper's experiments

**What Process Evaluation of Coding Agents Actually Measures: Action, Task, and Step Are Three Different Levels** (ICLR 2027 submission)

This repository contains the complete experimental code for the paper.
(Full artifacts are ~17GB) -- only the code, execution steps, and metric definitions needed to produce every number in the paper.

---

## The paper in one sentence

Process evaluation conflates three different things: predicting what the agent does next (**action**), how much uncertainty remains in the task outcome
(**task**), and whether a given step actually changed the outcome (**step**). The three require different **conditioning sets**, so what a "process score"
actually measures depends on what it conditions on, not on who does the scoring. The paper formalizes the third one with SCAE (a replay-based estimator
derived from a structural causal model of agent execution), and uses it as an instrument to measure the first two.

Across 499 file-localization episodes on SWE-bench-Verified, the three levels yield asymmetric conclusions:

| Level | Conclusion | Key numbers |
|---|---|---|
| **action** | Strong, but driven by the agent's own **execution provenance**, not the code dependency graph | first-visit step top-3: provenance 0.326 vs graph 0.058 |
| **task** | Execution uncertainty lives at the **task** level, not the step level | instance ICC 0.640 (permutation null 0.196); step position and tool type both insignificant |
| **step** | **Not certifiable** under a feasible replay budget | 190 effects, **0** survive BH-FDR (q=0.10) |
| **judge** | full-trace judge shows systematic **collider bias** | blame-position displacement **+0.537**, sign test p = 1.1×10⁻¹⁶ |

Conclusion: what a full-trace judge measures looks more like "semantic relevance shaped by downstream information" than "certified causal contribution".

---

## Quick start

```bash
# 1) Dependencies (pure CPU, no GPU)
pip install -r requirements.txt

# 2) Configuration
cp env.example.sh env.sh && vim env.sh && source env.sh

# 3) Self-check
python3 setup/check_env.py            # offline-analysis tier
python3 setup/check_env.py --collect  # data-collection tier (additionally checks codex CLI / API key)

# 4) The one thing that needs no data at all: the UQ three-way decomposition self-check from Appendix D.7
python3 -m pilot.analyze.uq_selftest
```

Step 4 should output (verified to reproduce with this package):

```
Parameter recovery corr(w_true, MAP) = 0.9935   paper: 0.994
In-support probability MAE = 0.022               paper: 0.022
Predictive entropy 0.4638 = 0.3075 + 0.1458 + 0.0124  paper: 0.464 = 0.308/0.146/0.012
epistemic inflation 0.0124 -> 0.1002 (x8.08)     paper: 8.1x
```

For the full execution steps (including compute/time/API budget) see **[REPRODUCE.md](REPRODUCE.md)**.

---

## Directory layout

```
scae-repro/
├── README.md                  this file
├── REPRODUCE.md               ★ step-by-step execution + budget + tiered reproduction strategy
├── requirements.txt           numpy (datasets optional)
├── env.example.sh             all environment variables
├── setup/
│   ├── check_env.py           environment self-check (two tiers: offline analysis / data collection)
│   ├── prepare_dataset.py     build instance_summary.csv + repository snapshots from SWE-bench-Verified
│   └── stage_attr_shards.py   shard preparation for the attribution stage
├── scripts/
│   ├── common.sh              shared preamble (pin down environment noise, check prerequisites)
│   ├── 00_smoke.sh            stage 0 feasibility checks V1-V5 + replay consistency
│   ├── 01_locate.sh           stage 1 file localization over 499 instances (the corpus underlying all conclusions)
│   ├── 02_attribution.sh      stage 2 prefix-replay attribution (most expensive: ~12.7h)
│   ├── 03_judge_crossmodel.sh stage 3 cross-model judge replication (Table 2)
│   ├── 04_analysis_offline.sh stage 4 ★ all offline analysis, zero API
│   ├── 05_upolicy.sh          stage 5 K=4 repeated rollouts for U_policy (D.7)
│   └── run_all.sh             the whole pipeline chained together
├── pilot/                     ★ the code itself (package name stays pilot, see below)
│   ├── harness/               agent execution layer: app-server driving, rollout rewriting, prefix replay
│   ├── attribution/           causal estimation layer: v_t / Δ_t / hybrid estimator / judge baselines
│   ├── uq/                    uncertainty layer: Bayesian logistic + Laplace + three-way decomposition
│   ├── channels/              channel models: tool channel, code graph channel, command parsing
│   ├── graph/                 AST index and graph probes (pure stdlib)
│   ├── probes/                task definition: instance loading, prompt, outcome criterion Y
│   ├── smoke/                 feasibility and positivity probes
│   ├── analyze/               offline analysis per RQ (+ legacy/ superseded scripts)
│   └── legacy_pipelines/      original orchestration scripts, kept for provenance only (superseded by scripts/)
└── docs/
    ├── PAPER_MAP.md           ★ every number in the paper → script → output JSON field
    ├── SCRIPT_INVENTORY.md    status and supersession relations of all scripts
    ├── PITFALLS.md            ★ pitfalls that silently ruin results (read this first)
    └── METHOD_NOTES.md        implementation notes on estimators and metric definitions
```

### Why the package is still called `pilot`

The reproduction commands recorded in Appendix E.1 of the paper look like `python3 -m pilot.analyze.entropy_scan`.
Keeping the package name lets those commands **run verbatim**, and avoids rewriting imports across 70 files.

---

## Three reproduction routes

Pick based on what you have:

| Route | Requires | Time | What it reproduces |
|---|---|---|---|
| **A self-check** | just python + numpy | seconds | the UQ three-way decomposition of Appendix D.7 (synthetic data with known ground truth) |
| **B offline analysis** | recorded artifacts from stages 1-3 | minutes | **all** tables and statistics in the paper |
| **C full re-run** | codex CLI + API key + dataset + 30h+ | 30h+ | re-run from scratch, including agent execution |

**This package does not ship the recorded data that route B needs.** To take route B, prepare your own `pilot/runs/`
(structure described in the "artifact structure" section of [docs/PAPER_MAP.md](docs/PAPER_MAP.md)), or run route C first to generate it.

---

## Experimental setup (paper Appendix C.2, aligned item by item in the code)

| Item | Value | Location |
|---|---|---|
| agent under test | Codex CLI, JSON-RPC app-server, `capabilities.experimentalApi=true` | `pilot/harness/appserver.py` |
| sandbox | `read-only`, `approvalPolicy=never`, and approval-request count verified to be 0 | `pilot/harness/runner.py` |
| backing model | `gpt-5.5-0424-global`, 900s per-turn timeout | `SCAE_MODEL` |
| replay count m | **8** (RQ4 step-level power tier) / 3 (RQ5 judge tier) | `SCAE_M_POWER` / `SCAE_M_JUDGE` |
| valuation points | **5 equally spaced points** per episode | `--window-points 5` |
| entropy threshold τ_H | **0** (most permissive: take the observational branch as soon as ≥2 distinct actions appear in the replays) | `causal.hybrid_delta_at` |
| outcome variable | **recall over F\*** (intervention sensitivity 2.505, better than F₁ 2.165 and binary 1.643) | `--outcome recall` |
| prefix-only judge | fixed **60%** prefix, outcome Y hidden | `--prefix-frac 0.6` |
| attribution samples | multi-file instances only (single-file ones almost always succeed, Δ≡0) | `--multifile-only` |
| the three judges | gpt-5.5-0424-global / qwen3.6-plus / qwen3-coder-plus | `SCAE_JUDGE_*` |
| multiple comparisons | BH-FDR, q = 0.10 | inside each analysis script |

---

## Verified reproduction results

The following were obtained by actually running this package's code, and match the paper digit for digit (details in [docs/PAPER_MAP.md](docs/PAPER_MAP.md)):

- **Appendix D.7** UQ three-way decomposition self-check: 0.9935 / 0.022 / 0.308+0.146+0.012 / ×8.08
- **§5.5 + D.4** reuse structure: tool reuse rate 0.7912, object 0.4159, permutation-null std 0.00e+00,
  tool adjacent-step repetition 1.02× (p=0.511), object adjacent-step repetition 1.83× (p~1e-141),
  high-frequency lift 1.6965 CI[1.63,1.76], recency lift 1.739 CI[1.67,1.81], n=487 episodes
- **§5.7 + D.10 collider bias (the paper's headline result)**: displacement **0.5368** (paper +0.537),
  D₁ disagreement rate **97.1%**, judge top-1 falling in gold-provenance **0.900 CI[0.829,0.971]**,
  all-step baseline **0.4131** (paper 0.413), judge falling inside the valuation window **32.9%** (paper D.10: 32.9%), 70 episodes
- **D.7 U_policy**: feeding the original K=4 results into this package's estimator outputs **0.0489** (paper 0.049), identical field by field
- **Dataset interface**: the `instance_summary.csv` produced by `setup/prepare_dataset.py` is **field-for-field identical**
  to the file actually used in the paper after parsing (500 rows × 4 fields, 71 multi-file instances)

---

## The corpus and that 500 vs 499

`instance_summary.csv` has 500 instances, while the paper reports **499 episodes**. The difference is
`django__django-15278`: the dataset only distributes 499 repository snapshots and this one is missing. Both `setup/check_env.py` and
`scripts/01_locate.sh` report such instances explicitly rather than dropping them silently.

The corpus is dominated by `django` (46%), so everywhere a predictor is fitted we use
**leave-one-repository-out**, with conformal sets calibrated Mondrian-style stratified by repository.

---

## Important limitations (paper §5.8 / Appendix E.3)

Before reproducing, know that these conclusions **are themselves negative results**, not a sign the code was run wrong:

1. **Weak step-level reference**: 0 of 190 effects pass BH-FDR. The paper's strongest attribution conclusion is a mechanistic claim about **how**
   the judge's verdict moves, not about **whether** the step it picks is correct.
2. **Narrow task scope**: file localization only -- because it is the minimal realistic environment that is replayable, prefix-reconstructible, and outcome-verifiable.
   The process structure of code editing or open-ended tool use may differ.
3. **The three-way decomposition is not validated on real traces**: it is validated only on a synthetic generative process (D.7); on real traces only U_policy is directly estimated.
4. **The graph channel covers only 46.2%**: only that fraction of agent operation objects resolve cleanly to dependency-graph nodes.
5. **Graph construction depends on external tools**: `pilot/runs/graphs/` is produced by two upstream tools not shipped with this package,
   see the "external dependencies of the graph channel" section of [REPRODUCE.md](REPRODUCE.md).

---

## Citation and license

The paper is under anonymous review at ICLR 2027; citation information will be added after acceptance.
The code is for academic reproduction only; the analyzed repositories come from SWE-bench-Verified and each follows its own original license.
All agent execution happens in a read-only sandbox and does not modify any repository under test.
