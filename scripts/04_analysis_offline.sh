#!/usr/bin/env bash
# Stage 4: all offline analyses -- zero API, zero GPU, seconds to minutes of CPU
#
# This stage only reads the recorded artifacts of stages 1-3; it calls no model. Every number in the
# paper is written into some JSON by one of the scripts here. Grouped by research question, matching
# paper §5.3-§5.7 one to one.
# For the number-by-number mapping to scripts see docs/PAPER_MAP.md.
#
# Usage:
#   bash scripts/04_analysis_offline.sh          # run everything
#   bash scripts/04_analysis_offline.sh rq1      # run only one group: rq1|rq2|rq3|rq4|rq5|report|selftest
set -euo pipefail
source "$(dirname "$0")/common.sh"

GROUP="${1:-all}"
LOCATE="$RUNS/full_locate"
ATTR="$RUNS/full_attr"
V4P="$RUNS/v4prime"

log "Stage 4: offline analysis (group=$GROUP)"
banner

FAILED=(); SKIPPED=()

# Run one analysis script; on missing input, skip rather than tear down the whole stage
run() {
  local label="$1"; shift
  printf '\n  \033[1m→ %s\033[0m\n' "$label"
  printf '    $ %s\n' "python3 $*"
  if python3 "$@"; then
    return 0
  else
    printf '    \033[31mFAILED\033[0m (usually a missing input artifact; see the error above)\n'
    FAILED+=("$label")
    return 0
  fi
}

need() {  # need <path> <label> -- register a skip if the input does not exist
  if [ ! -e "$1" ]; then
    printf '\n  \033[33mskipping %s\033[0m -- missing %s\n' "$2" "$1"
    SKIPPED+=("$2 (missing $1)")
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- self-check (no data needed)
if [ "$GROUP" = "all" ] || [ "$GROUP" = "selftest" ]; then
  log "UQ self-check (synthetic data, needs no recorded artifacts)"
  # Paper Appendix D.7: on a known generating process the three-way decomposition recovers parameters
  # with correlation 0.994, and U_model inflates 8.1x off support. This is the only positive validation
  # of the three-way decomposition.
  run "D.7 UQ three-way decomposition self-check" -m pilot.analyze.uq_selftest --out "$RUNS/uq_selftest.json"
fi

# ---------------------------------------------------------------- RQ1 next-action prediction
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq1" ]; then
  log "RQ1: what predicts the next action? (paper §5.3 / Table 1 / Table 10 / Table 11)"
  if need "$LOCATE" "RQ1"; then
    # Shared denominator of 509 decision points + first-visit/revisit two-regime decomposition → the denominator definition of Table 1
    run "V3g shared denominator and regime decomposition" -m pilot.analyze.v3g_regime \
      --runs-root "$LOCATE" --out "$RUNS/v3g_regime.json"
    # Execution provenance predictor family: uniform / frequency / recency / last-output → the main rows of Table 1
    run "V3j execution provenance graph prediction (core of Table 1)" -m pilot.analyze.v3j_trace_graph \
      --runs-root "$LOCATE" --out "$RUNS/v3j_trace_graph.json"
    # Distribution quality: top-k curve / MRR / ECE / NLL / effective candidate count exp(H)
    run "V3b prediction distribution quality" -m pilot.analyze.v3b_distribution_quality \
      --runs-root "$LOCATE" --out "$RUNS/v3b_distribution.json"
    # Tool channel entropy H_tool, reused by x4/x6 in RQ4
    run "tool channel Markov model" -m pilot.channels.tool_model \
      --runs-root "$LOCATE" --out "$RUNS/tool_model_official.json"
    run "X1 joint L1 distribution" -m pilot.analyze.x1_joint_l1 \
      --runs-root "$LOCATE" --tool-model "$RUNS/tool_model_official.json" \
      --prov-results "$RUNS/v3j_trace_graph.json" --out "$RUNS/x1_joint_l1.json"
    # Dependence between tool and object type I=0.159 nats (p=5e-4) ⇒ factorizing as P(tool)P(obj) is misspecified
    run "X1b tool/object conditional independence test (Table 11)" -m pilot.analyze.x1b_joint_independence \
      --runs-root "$LOCATE" --out "$RUNS/x1b_joint_independence.json"
    # Candidate set routed by predicted tool vs factorized shared candidate set → Table 10
    run "X1c tool-routed candidate set (Table 10)" -m pilot.analyze.x1c_tool_routed \
      --runs-root "$LOCATE" --out "$RUNS/x1c_tool_routed.json"
    # Query channel: is top-3=0.029 a real upper bound or a shlex parsing defect (26.6% of command segments)? → Appendix D.5
    run "X1d query channel and parsing defect (D.5)" -m pilot.analyze.x1d_query_channel \
      --runs-root "$LOCATE" --out "$RUNS/x1d_query_channel.json"
  fi
fi

# ---------------------------------------------------------------- RQ2 code dependency graph
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq2" ]; then
  log "RQ2: what is the code dependency graph actually good for? (paper §5.4 / Figure 7 / D.3)"
  if need "$LOCATE" "RQ2"; then
    # The graph has two roles: it does not describe step-to-step movement (one-hop lift 0.94x), but it does describe the working region (internal edge density 31x)
    # Requires the graph construction artifacts under pilot/runs/graphs/<instance_id>/; see the "graph channel" section of REPRODUCE.md
    if [ ! -d "$RUNS/graphs" ]; then
      printf '\n  \033[33mnote\033[0m RQ2 needs %s (graph construction artifacts, 71 instances).\n' "$RUNS/graphs"
      printf '    Graph construction depends on two external tools that are not shipped with the reproduction package; see "external dependencies of the graph channel" in REPRODUCE.md.\n'
      printf '    When they are missing, the script below degrades to reporting only the subset it can parse.\n'
      SKIPPED+=("RQ2 graph channel completeness (missing $RUNS/graphs)")
    fi
    run "V3i graph probe (Figure 7 / §5.4)" -m pilot.analyze.v3i_graph_probe \
      --runs-root "$LOCATE" --out "$RUNS/v3i_graph_probe.json"
  fi
fi

# ---------------------------------------------------------------- RQ3 reuse patterns
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq3" ]; then
  log "RQ3: which reuse patterns are real? (paper §5.5 / D.4)"
  if need "$LOCATE" "RQ3"; then
    # The reuse "rate" is a permutation invariant (zero variance, zero information); only the reuse "pattern" is informative:
    # consecutive-step repetition 1.83x (p~1e-141), visit count 1.70x, recency rank 1.74x;
    # tool self-repetition 1.02x (p=0.51, not significant) ⇒ the tool half of the folklore claim is false
    run "V3h reuse structure permutation test" -m pilot.analyze.v3h_reuse \
      --runs-root "$LOCATE" --out "$RUNS/v3h_reuse.json"
  fi
fi

# ---------------------------------------------------------------- RQ4 uncertainty hierarchy
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq4" ]; then
  log "RQ4: which layer does execution uncertainty live in? (paper §5.6 / Figure 8 / D.6 / D.7)"
  # Can zero-cost action entropy proxy for replay variance? → ρ=−0.088 (step level) / −0.219 (instance level) / −0.425 (positive control)
  if need "$V4P" "RQ4 step level"; then
    run "X4 structural entropy vs replay variance" -m pilot.analyze.x4_uexec_corr \
      --v4prime-dir "$V4P" --tool-model "$RUNS/tool_model_official.json" \
      --runs-root "$LOCATE" --out "$RUNS/x4_uexec_corr.json"
    # Does U^exec have non-mechanical structure: instance ICC=0.640 (permutation null 0.196), step position ρ=−0.131, tool type p=0.65
    run "X4b U^exec structural validity (Figure 8a)" -m pilot.analyze.x4b_uexec_validity \
      --v4prime "$V4P/attribution.jsonl" --runs-root "$LOCATE" \
      --out "$RUNS/x4b_uexec_validity.json"
    run "X4c instance-level re-check" -m pilot.analyze.x4c_instance_level \
      --v4prime "$V4P/attribution.jsonl" --runs-root "$LOCATE" \
      --out "$RUNS/x4c_instance_level.json"
    # The paper's core negative result: of 190 effects, 5.3% reach uncorrected p<0.05, and 0 survive BH-FDR(q=0.10)
    run "X5 step-level statistical power (core negative result of §5.6)" -m pilot.analyze.x5_step_power \
      --v4prime-dir "$V4P" --runs-root "$LOCATE" --out "$RUNS/x5_step_power.json"
    # The 5x reported by X6 and X6b's refutation of it: a threshold sweep shows the ratio decaying monotonically from 3.36 to 1.71 as α is relaxed
    run "X6 budget allocation (the result refuted by X6b)" -m pilot.analyze.x6_budget_alloc \
      --v4prime-dir "$V4P" --tool-model "$RUNS/tool_model_official.json" \
      --out "$RUNS/x6_budget_alloc.json"
    run "X6b artifacts of the binary criterion (D.6)" -m pilot.analyze.x6b_effect_criterion \
      --v4prime "$V4P/attribution.jsonl" --runs-root "$LOCATE" \
      --out "$RUNS/x6b_effect_criterion.json"
    # The superseded Bernoulli approximation U^exec=4v(1−v), kept in paper C.3 as a negative control
    run "V4 structural vs execution (negative control, C.3)" -m pilot.analyze.v4_structural_vs_exec \
      --attr "$V4P/attribution.jsonl" --locate "$LOCATE/results.jsonl" \
      --runs-root "$LOCATE" --tool-ent "$RUNS/tool_model_official.json" \
      --out "$RUNS/v4_structural_vs_exec.json"
  fi
  if need "$ATTR/attribution.jsonl" "RQ4 process uncertainty"; then
    # degenerate rate 22.9%, fragile successes, convergence point t*
    run "process uncertainty (degenerate rate)" -m pilot.analyze.process_uncertainty \
      --attr "$ATTR/attribution.jsonl" --out "$RUNS/process_uncertainty.json"
    # Three-state accounting: INESTIMABLE / NO_EFFECT / EFFECT
    run "estimability three-state accounting" -m pilot.analyze.estimability \
      --attr "$ATTR/attribution.jsonl" --locate "$LOCATE/results.jsonl" \
      --runs-root "$LOCATE" --out "$RUNS/estimability.json"
  fi
  # Outcome variable choice: intervention sensitivity of recall 2.505 > F1 2.165 > binary success 1.643 (Appendix C.6)
  if need "$LOCATE" "C.6 outcome variable screening"; then
    run "X8 outcome variable sensitivity (C.6)" -m pilot.analyze.x8_outcome_sensitivity \
      --runs-root "$LOCATE" --out "$RUNS/x8_outcome_sensitivity.json"
  fi
  # Locality of positivity: 42.9%/50% of probe points have action randomness, rank correlation with step position +0.072 (non-monotonic)
  if need "$RUNS/v5_scan" "C.4 entropy structure"; then
    run "C.4 entropy scan (Table 6 / Figure 5)" -m pilot.analyze.entropy_scan \
      --scan-dir "$RUNS/v5_scan"
  fi
fi

# ---------------------------------------------------------------- RQ5 attribution and collider bias
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq5" ]; then
  log "RQ5: what is observational attribution actually measuring? (paper §5.7 / Table 2 / Table 12 / Figure 4)"
  if need "$ATTR/attribution.jsonl" "RQ5"; then
    # D1 disagreement rate (retired by the paper, see D.12) and D3 positional displacement +0.537 (the core quantity of collider isolation)
    run "D1/D3 and collider isolation (Figure 4a)" -m pilot.analyze.attribution_compare \
      --attr "$ATTR/attribution.jsonl" --out "$RUNS/d1d3_report.json"
    # Is |Δ_t| concentrated on gold-provenance steps; the rate at which judge/causal top-1 falls in that class
    run "V3k trace causality and gold-provenance" -m pilot.analyze.v3k_trace_causal \
      --attr-dir "$ATTR" --runs-root "$LOCATE" --out "$RUNS/v3k_trace_causal.json"
    # Can zero-LLM structural rules reproduce the judge's choice? → Table 12; the best rule beats within-class guessing by only 1.7x
    run "X2 judge mechanism and zero-LLM rules (Table 12)" -m pilot.analyze.x2_judge_mechanism \
      --attr-dir "$ATTR" --runs-root "$LOCATE" --out "$RUNS/x2_judge_mechanism.json"
  fi
fi

# ---------------------------------------------------------------- summary report
if [ "$GROUP" = "all" ] || [ "$GROUP" = "report" ]; then
  log "summary report (the two commands recorded in paper Appendix E.1)"
  if need "$LOCATE/results.jsonl" "summary"; then
    # Localization table (Table 7) + LORO calibration and prediction for v_0 (Table 8 / D.8)
    run "full report (Table 7 / D.8)" -m pilot.analyze.full_report \
      --locate "$LOCATE/results.jsonl" --attr "$ATTR/attribution.jsonl" \
      --scan "$RUNS/v5_scan" "$RUNS/v5" --out "$RUNS/analysis.json"
  fi
  # Aggregate the individual JSONs into the single file used by the paper's macro definitions
  run "conclusion aggregation" -m pilot.analyze.x_synthesis --out "$RUNS/synthesis.json"
fi

# ---------------------------------------------------------------- closing
log "Stage 4 complete"
if [ ${#SKIPPED[@]} -gt 0 ]; then
  printf '\n\033[33mskipped %d item(s) (missing inputs):\033[0m\n' "${#SKIPPED[@]}"
  for s in "${SKIPPED[@]}"; do printf '  - %s\n' "$s"; done
fi
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '\n\033[31mfailed %d item(s):\033[0m\n' "${#FAILED[@]}"
  for f in "${FAILED[@]}"; do printf '  - %s\n' "$f"; done
  printf '\nA failure usually means an upstream artifact is missing or incomplete, not that the analysis logic is broken.\n'
fi
info "the artifact JSONs all live under $RUNS/. For which JSON field corresponds to each number in the paper, see docs/PAPER_MAP.md"
[ ${#FAILED[@]} -eq 0 ] || exit 1
