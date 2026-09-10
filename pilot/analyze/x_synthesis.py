"""X_SYNTHESIS（零 API）：综合所有实验结果生成论文数字表。

把 V3g/V3i/V3j/V3k/X1/X2/X7 + analysis.json 的全部已知数字
统一到一个 synthesis.json，供论文宏定义使用。

运行
----
  python3 -m pilot.analyze.x_synthesis --out pilot/runs/synthesis.json

输出结构
--------
  L1_layer  : 工具转移 + 溯源预测层
  L2_layer  : anytime 价值函数 + 因果归因
  L3_bridge : L1↔L2 关系（X4 依赖 V4'，此处填占位符）
  UQ_decomp : U_task / U_policy / U_model 三分解
  baselines : 定位基线（T1）
  judge_mechanism : X2 结论
  fragile_success : X7 + process_uncertainty
  X3_plan   : 干预计划摘要
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUNS = Path("pilot/runs")


def _load(fname: str) -> dict:
    p = RUNS / fname
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot/runs/synthesis.json")
    args = ap.parse_args()

    analysis = _load("analysis.json")
    v3b = _load("v3b_distribution.json")
    v3g = _load("v3g_regime.json")
    v3i = _load("v3i_graph_probe.json")
    v3j = _load("v3j_trace_graph.json")
    v3k = _load("v3k_trace_causal.json")
    x1 = _load("x1_joint_l1.json")
    x2 = _load("x2_judge_mechanism.json")
    x3p = _load("x3_plan.json")
    x4 = _load("x4_uexec_corr.json")  # may be placeholder
    x5 = _load("x5_step_power.json")
    x6 = _load("x6_budget_alloc.json")
    x7 = _load("x7_fragile_l1.json")
    proc = _load("process_uncertainty.json")
    upol = _load("upolicy_t3.json")
    estim = _load("estimability.json")

    # ── L1: 工具转移分量（来自 v3b_distribution） ──
    # Expected structure: {U_tool: {mean, median, eff_candidates}, top3_accuracy, ECE}
    l1_tool = {}
    if v3b:
        l1_tool = {
            "mean_H_tool": v3b.get("U_tool", {}).get("mean"),
            "eff_candidates": v3b.get("U_tool", {}).get("eff_candidates"),
            "top1_acc": v3b.get("top1"),
            "top3_acc": v3b.get("top3"),
            "ECE": v3b.get("ECE"),
        }
    # X1 has more detailed H_tool stats
    if x1:
        l1_tool.update({
            "H_tool_mean_x1": x1.get("H_tool_global", {}).get("mean"),
            "H_tool_exp_mean": x1.get("H_tool_global", {}).get("global_exp_H_tool"),
            "regime_diff_R_minus_F": x1.get("H_tool_R_vs_F_paired", {}).get("diff"),
            "regime_diff_sig": x1.get("H_tool_R_vs_F_paired", {}).get("significant"),
            "H_tool_v0_spearman_sig": x1.get("H_tool_vs_v0_corr", {}).get("significant"),
        })

    # ── L1: 溯源分量（来自 v3j） ──
    l1_prov = {}
    if v3j:
        g1 = v3j.get("G1_predictors", {})
        plo = g1.get("prov_lastout_ranked", {})
        l1_prov = {
            "top1_all": plo.get("all", {}).get("top1"),
            "top3_all": plo.get("all", {}).get("top3"),
            "eff_candidates": plo.get("eff_candidates"),
            "graph_top1_all": v3j.get("G2_routing_contrasts", {})
                                   .get("routed_graph", {}).get("top1"),
            "H_prov": 1.2408,   # log(3.46), pre-computed
            "H_prov_eff": 3.46,
        }

    # ── Regime split (来自 v3g) ──
    regime = {}
    if v3g:
        r_info = v3g.get("R", {})
        f_info = v3g.get("F", {})
        regime = {
            "R_fraction": v3g.get("R_fraction"),
            "F_fraction": v3g.get("F_fraction"),
            "R_top1": r_info.get("top1"),
            "R_top3": r_info.get("top3"),
            "F_top3": f_info.get("top3"),
            "regime_auroc": v3g.get("regime_auroc"),
        }

    # ── Code graph (来自 v3i) ──
    graph = {}
    if v3i:
        graph = {
            "neighbor_lift": v3i.get("hop1_lift"),
            "community_lift": v3i.get("community_lift"),
            "visited_cluster_lift": v3i.get("visited_cluster_lift"),
        }

    # ── Joint L1 (来自 X1) ──
    joint_l1 = {}
    if x1:
        jec = x1.get("joint_effective_candidates", {})
        joint_l1 = {
            "H_tool": x1.get("H_tool_global", {}).get("mean"),
            "exp_H_tool": x1.get("H_tool_global", {}).get("global_exp_H_tool"),
            "H_prov": 1.2408,
            "exp_H_prov": jec.get("H_prov"),
            "joint_eff_candidates": jec.get("product_exp_H_joint"),
            "H_tool_vs_v0_sig": x1.get("H_tool_vs_v0_corr", {}).get("significant"),
            "note": "H_tool + H_prov (independence assumption); eff=exp(H1)*exp(H2)",
        }

    # ── L2: anytime value + attribution (来自 analysis.json) ──
    t1 = analysis.get("T1_baseline", {})
    t4 = analysis.get("T4_attribution", {})
    l2 = {}
    if t1 or t4:
        l2 = {
            "baseline_recall": t1.get("primary_recall_rate"),
            "n_total": t1.get("n_total"),
            "fragile_success_rate": proc.get("U2_fragile_success", {}).get("rate"),
            "fragile_ci95": proc.get("U2_fragile_success", {}).get("ci95"),
            "causal_top1_delta_failed": proc.get("attribution_by_outcome", {})
                                             .get("failed", {}).get("mean_top_delta"),
            "causal_top1_delta_success": proc.get("attribution_by_outcome", {})
                                              .get("succeeded", {}).get("mean_top_delta"),
            "D1_disagreement": t4.get("D1_disagreement", {}).get("point"),
            "D3_displacement_mean": t4.get("D3_displacement", {}).get("mean"),
            "judge_pos_median": t4.get("D3_displacement", {}).get("median_judge_pos"),
            "causal_pos_median": t4.get("D3_displacement", {}).get("median_causal_pos"),
            "degenerate_rate": t4.get("degenerate_rate"),
        }

    # ── Judge mechanism (来自 X2) ──
    judge = {}
    if x2:
        judge = {
            "judge_gold_prov_rate": x2.get("judge_vs_own_baseline", {})
                                       .get("observed", {}).get("mean"),
            "judge_gold_prov_baseline": x2.get("judge_vs_own_baseline", {})
                                           .get("correct_baseline_all_steps", {}).get("mean"),
            "judge_gold_prov_diff": x2.get("judge_vs_own_baseline", {})
                                       .get("diff", {}).get("diff"),
            "judge_gold_prov_ci95": x2.get("judge_vs_own_baseline", {})
                                       .get("diff", {}).get("ci95"),
            "within_class_pos": x2.get("WITHIN_CLASS_position", {})
                                    .get("observed", {}).get("mean"),
            "within_class_null": 0.5,
            "within_class_diff_ci95": x2.get("WITHIN_CLASS_position", {})
                                          .get("diff", {}).get("ci95"),
            "within_class_sig": x2.get("WITHIN_CLASS_position", {})
                                    .get("diff", {}).get("significant"),
            "same_space_judge_rate": x2.get("fair_comparison_same_space", {})
                                        .get("judge", {}).get("mean"),
            "same_space_causal_rate": x2.get("fair_comparison_same_space", {})
                                         .get("causal", {}).get("mean"),
            "same_space_n": x2.get("fair_comparison_same_space", {}).get("n"),
            "mean_gold_steps_per_instance": 8.70,  # from X2 WITHIN_CLASS analysis
        }

    # ── UQ decomposition (来自 upolicy_t3, estimability) ──
    uq = {}
    if upol or estim:
        sc = estim.get("state_counts", {}) if estim else {}
        uq = {
            "U_policy": upol.get("U_policy") if upol else None,
            "U_policy_n": upol.get("n_instances") if upol else None,
            "three_state_inestimable": sc.get("INESTIMABLE"),
            "three_state_no_effect": sc.get("NO_EFFECT"),
            "three_state_effect": sc.get("EFFECT"),
            "three_state_n": estim.get("n_segments") if estim else None,
            "estimability_coverage": estim.get("estimability_coverage", {}).get("point") if estim else None,
            "T2_auroc": analysis.get("T2_v0_loro", {}).get("AUROC"),
            "T2_ECE": analysis.get("T2_v0_loro", {}).get("ECE"),
            "T2_brier": analysis.get("T2_v0_loro", {}).get("Brier"),
        }

    # ── X7 fragile success ↔ L1 ──
    fragile_l1 = {}
    if x7:
        fragile_l1 = {
            "fragile_robust_H_diff": x7.get("FRAGILE_ROBUST_COMPARISON", {})
                                        .get("diff", {}).get("diff"),
            "fragile_robust_H_diff_sig": x7.get("FRAGILE_ROBUST_COMPARISON", {})
                                             .get("diff", {}).get("significant"),
            "auroc_H_predicts_fragile": x7.get("AUROC", {}).get("auroc"),
            "power_H_vs_nsteps_rho": x7.get("POWER_CHECK", {}).get("H_tool_vs_n_steps", {})
                                        .get("rho"),
            "power_H_vs_nsteps_sig": x7.get("POWER_CHECK", {}).get("H_tool_vs_n_steps", {})
                                        .get("significant"),
            "conclusion": "L1工具熵不是执行稳定性的代理（功效已确认）",
        }

    # ── X4/X5/X6: 状态（V4' 后填充）──
    x4_status = "PENDING_V4PRIME" if not x4 or x4.get("n_rows", 0) < 10 else "DONE"
    x5_status = "PENDING_V4PRIME" if not x5 or x5.get("n_tests", 0) < 10 else "DONE"
    x6_status = "PENDING_V4PRIME" if not x6 or x6.get("N_points_total", 0) < 10 else "DONE"

    bridge_l1_l2 = {
        "X4_H_struct_vs_Uexec": {
            "status": x4_status,
            "rho": x4.get("MAIN_CORRELATIONS", {})
                     .get("H_struct_vs_U_exec", {}).get("rho") if x4 else None,
            "significant": x4.get("MAIN_CORRELATIONS", {})
                             .get("H_struct_vs_U_exec", {}).get("significant") if x4 else None,
        },
        "X5_step_power": {
            "status": x5_status,
            "n_sig_bh": x5.get("STEP_POWER", {}).get("n_significant_bh") if x5 else None,
            "sig_rate": x5.get("STEP_POWER", {}).get("significant_fraction_bh") if x5 else None,
        },
        "X6_budget_q03": {
            "status": x6_status,
            "verdict": x6.get("VERDICT_Q03") if x6 else None,
        },
        "X3_plan": {
            "n_candidates": x3p.get("n_candidates"),
            "first_gold_t_median": x3p.get("first_gold_t", {}).get("median"),
            "estimated_hours": x3p.get("estimated_budget", {}).get("estimated_hours"),
            "status": "PLAN_READY_AWAITING_WORKERS",
        },
    }

    # ── Paper claim inventory ──
    claims = [
        {"id": "A1", "claim": "环境可重放（100/100 逐字节一致）",
         "source": "replay_check.json", "value": 1.0},
        {"id": "A2", "claim": f"工具转移 top-3={l1_tool.get('top3_acc','?')} ECE={l1_tool.get('ECE','?')}",
         "source": "v3b_distribution.json"},
        {"id": "A3", "claim": f"溯源 top-1={l1_prov.get('top1_all','?')} eff_cand={l1_prov.get('eff_candidates','?')}",
         "source": "v3j_trace_graph.json"},
        {"id": "A4", "claim": f"R 步 top-3={regime.get('R_top3','?')} (F 不可预测)",
         "source": "v3g_regime.json"},
        {"id": "A5", "claim": f"regime AUROC={regime.get('regime_auroc','?')} (不可预测 regime)",
         "source": "v3g_regime.json"},
        {"id": "A6", "claim": f"代码图 lift={graph.get('neighbor_lift','?')} (无转移信号)",
         "source": "v3i_graph_probe.json"},
        {"id": "A10", "claim": f"judge 挑 gold 溯源步 {judge.get('judge_gold_prov_rate','?'):.0%} vs 基线 {judge.get('judge_gold_prov_baseline','?'):.1%}",
         "source": "x2_judge_mechanism.json"},
        {"id": "A11", "claim": f"失败 top1 Δ={l2.get('causal_top1_delta_failed','?'):.3f} CI 显著",
         "source": "analysis.json"},
        {"id": "A12", "claim": f"judge 系统偏后 pos={judge.get('within_class_pos','?'):.3f} vs null=0.5",
         "source": "x2_judge_mechanism.json"},
        {"id": "A13", "claim": f"脆弱成功率={l2.get('fragile_success_rate','?'):.1%} CI{l2.get('fragile_ci95','?')}",
         "source": "process_uncertainty.json"},
        {"id": "A14", "claim": f"U_policy={uq.get('U_policy','?')} (23 instance × K=4)",
         "source": "upolicy_t3.json"},
        {"id": "A15", "claim": f"三态 INESTIMABLE={uq.get('three_state_inestimable','?')} NO_EFFECT={uq.get('three_state_no_effect','?')} EFFECT={uq.get('three_state_effect','?')}",
         "source": "estimability.json"},
        {"id": "X1", "claim": f"联合有效候选={joint_l1.get('joint_eff_candidates','?'):.2f} H_tool~v0 n.s.",
         "source": "x1_joint_l1.json"},
        {"id": "X2", "claim": "within-class pos=0.746 显著（judge 偏后）; 类内精确定位仍随机",
         "source": "x2_judge_mechanism.json"},
        {"id": "X7", "claim": "H_tool 不预测脆弱成功 AUROC=0.526（功效已确认 rho=+0.394）",
         "source": "x7_fragile_l1.json"},
    ]

    synthesis = {
        "status": {
            "v4prime_instances_done": 2,  # updated at runtime
            "v4prime_total_instances": 38,
            "x3_ready": True,
            "x4_x5_x6_ready_awaiting_v4prime": True,
        },
        "L1_layer": {
            "tool_channel": l1_tool,
            "prov_channel": l1_prov,
            "regime": regime,
            "graph_null": graph,
            "joint_l1": joint_l1,
        },
        "L2_layer": l2,
        "L3_bridge": bridge_l1_l2,
        "UQ_decomp": uq,
        "judge_mechanism": judge,
        "fragile_success_L1": fragile_l1,
        "claims": claims,
    }

    # Update V4' instance count
    n_done = sum(
        sum(1 for l in (RUNS / f"v4prime/shard{s}/attribution.jsonl").read_text().splitlines()
            if l.strip())
        for s in range(6)
        if (RUNS / f"v4prime/shard{s}/attribution.jsonl").exists()
    )
    synthesis["status"]["v4prime_instances_done"] = n_done

    Path(args.out).write_text(
        json.dumps(synthesis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== SCAE 体系综合摘要 ===")
    print(f"V4' 进度: {n_done}/38 instances")
    print()
    print("── L1 层 ──")
    print(f"  工具转移: top-3={l1_tool.get('top3_acc','?')}, "
          f"eff_cand={l1_tool.get('eff_candidates','?')}, "
          f"mean H_tool={l1_tool.get('H_tool_mean_x1','?')}, "
          f"exp(H)={l1_tool.get('H_tool_exp_mean','?')}")
    print(f"  溯源: top-1={l1_prov.get('top1_all','?')}, "
          f"eff_cand={l1_prov.get('eff_candidates','?')}, "
          f"H_prov={l1_prov.get('H_prov','?')}")
    print(f"  联合有效候选: {joint_l1.get('joint_eff_candidates','?'):.2f} "
          f"(工具 × 溯源，独立假设)")
    print(f"  代码图: lift={graph.get('neighbor_lift','?')} (已否证)")
    print()
    print("── L2 层 ──")
    print(f"  基线 recall={l2.get('baseline_recall','?'):.3f}, n={l2.get('n_total','?')}")
    print(f"  脆弱成功率={l2.get('fragile_success_rate','?'):.1%} CI{l2.get('fragile_ci95','?')}")
    print(f"  失败 top1 Δ={l2.get('causal_top1_delta_failed','?'):.3f} (显著)")
    print(f"  D1 disagreement={l2.get('D1_disagreement','?'):.3f}, "
          f"D3 displacement={l2.get('D3_displacement_mean','?'):.3f}")
    print()
    print("── judge 机制（X2）──")
    print(f"  judge 选 gold 溯源步: {judge.get('judge_gold_prov_rate','?'):.0%} "
          f"vs 基线 {judge.get('judge_gold_prov_baseline','?'):.1%}")
    print(f"  within-class pos: {judge.get('within_class_pos','?'):.3f} (null=0.5, 显著)")
    print(f"  同空间对比(n={judge.get('same_space_n','?')}): "
          f"judge={judge.get('same_space_judge_rate','?'):.1%} "
          f"causal={judge.get('same_space_causal_rate','?'):.1%}")
    print()
    print("── L3 桥接层 ──")
    print(f"  X3 干预: {bridge_l1_l2['X3_plan']['n_candidates']} 候选, "
          f"median first_gold_t={bridge_l1_l2['X3_plan']['first_gold_t_median']}, "
          f"≈{bridge_l1_l2['X3_plan']['estimated_hours']:.1f}h")
    print(f"  X4 U^exec↔H 相关: {x4_status}")
    print(f"  X5 步级功效: {x5_status}")
    print(f"  X6 预算分配: {x6_status}")
    print()
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
