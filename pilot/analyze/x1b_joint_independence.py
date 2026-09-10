"""X1b：补完 X1 —— 联合分布的**实测**评估与**条件独立假设的检验**。零 API。

X1 缺了什么
-----------
X1 报「联合有效候选数 = exp(H_tool) × exp(H_prov) = 2.757 × 3.46 = 9.54」。
但这个乘法依赖 **P(tool,obj) = P(tool)·P(obj)** 这个条件独立假设，
而 X1 自己注明"独立性尚未检验"。若不独立，9.54 这个数是错的
（不独立时联合熵 < 边际熵之和，故 9.54 是**上界**而非估计）。

方案 X1 要求的是联合分布的 **实测 top-k / ECE / exp(H)**，而非边际相乘。本脚本补齐：

  J1 实测联合：候选 = (tool, obj) 笛卡尔积，P = P(tool)·P(obj_prov)，
     共同分母、闭式期望破并列，给 top-k / ECE / 有效候选
  J2 独立性检验：I(tool_{t+1}; objcat_{t+1}) vs **episode 内置换零假设**
     objcat 用粗分类 {在最近输出的路径里 / 在已访问集合里 / 其他}，
     因为对象空间太大无法直接估互信息
  J3 若不独立：拟合 tool 条件化的对象模型（每个工具类别一套溯源加权），
     用 NLL 与联合 top-k 对比因子化模型，量化误配代价
  J4 联合 ECE 分层：按 (工具类型 × 溯源覆盖 × 步位) 找"何时可信"

判据（预注册）
--------------
| J2 结果 | 走向 |
|---|---|
| 互信息显著 > 零假设 | 因子化误配 → 9.54 只能作上界报告；改用 J1 实测值与 J3 条件模型 |
| 不显著 | 独立假设可接受 → 9.54 可作估计，但仍以 J1 实测为准 |
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.analyze.v3g_regime import rank_ge, exp_hit, eff_cand, ece, KS  # noqa: E402
from pilot.analyze.v3_nextstep_dual import fit_tool_trans, p_tool        # noqa: E402
from pilot.analyze.v3j_trace_graph import (                              # noqa: E402
    load_episodes_with_output, mentioned_nodes, boot_ci, paired_ci,
)

SEED = 20260814
B_PERM = 2000
OBJCATS = ("in_last_output", "in_visited", "other")


def objcat(truth: str, last_out: list[str], seen: set[str]) -> str:
    if truth in last_out:
        return "in_last_output"
    if truth in seen:
        return "in_visited"
    return "other"


def mutual_info(pairs: list[tuple[str, str]]) -> float:
    """I(X;Y) in nats。"""
    if not pairs:
        return 0.0
    n = len(pairs)
    pj, px, py = Counter(pairs), Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    return sum((c / n) * math.log((c / n) / ((px[a] / n) * (py[b] / n)))
               for (a, b), c in pj.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x1b_joint_independence.json")
    args = ap.parse_args()

    eps = load_episodes_with_output(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"载入 {len(eps)} episode，工具 {len(tool_vocab)} 类")

    # J1 实测联合
    joint_hits = {k: [] for k in KS}
    joint_cov, joint_eff, joint_cal = [], [], []
    # 因子化 vs 条件化的 NLL
    nll_fact, nll_cond = [], []
    factored_hits = {k: [] for k in KS}       # 仅对象分量（对照）
    # J2 独立性
    mi_pairs_per_ep: list[list[tuple[str, str]]] = []
    # J4 分层
    strat: dict[str, list[tuple[float, int]]] = defaultdict(list)
    strat_top3: dict[str, list[float]] = defaultdict(list)

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            # 条件化模型：每个"当前工具"类别一套溯源位置权重（从 train 估）
            cond_w: dict[str, Counter] = defaultdict(Counter)
            for x in train:
                sq = x["seq"]
                o_prev: list[str] = []
                for t2 in range(len(sq) - 1):
                    mo = mentioned_nodes(x["graph"], sq[t2]["out"])
                    if mo:
                        o_prev = mo
                    tr = sq[t2 + 1]["node"]
                    if tr and o_prev:
                        # 真值在最近输出里的位置桶（0,1,2,3+）
                        if tr in o_prev:
                            b = min(3, o_prev.index(tr))
                            cond_w[sq[t2 + 1]["tool"]][b] += 1
            g, seq = held["graph"], held["seq"]
            nodes = [s["node"] for s in seq]
            outs = [mentioned_nodes(g, s["out"]) for s in seq]
            seen: set[str] = set()
            last_out: list[str] = []
            ep_pairs: list[tuple[str, str]] = []
            for t in range(len(seq) - 1):
                if nodes[t]:
                    seen.add(nodes[t])
                if outs[t]:
                    last_out = outs[t]
                truth = nodes[t + 1]
                if not truth:
                    continue
                nxt_tool = seq[t + 1]["tool"]
                ep_pairs.append((nxt_tool, objcat(truth, last_out, seen)))

                # 工具分量
                pt = {v: p_tool(tm, seq[t]["tool"], v, tool_vocab) for v in tool_vocab}
                zt = sum(pt.values())
                pt = {k: v / zt for k, v in pt.items()}
                # 对象分量（prov_lastout_ranked：最近一条输出按位置加权）
                po_raw = {x: 1.0 / (i + 1) for i, x in enumerate(last_out)}
                zo = sum(po_raw.values())
                po = {k: v / zo for k, v in po_raw.items()} if zo else {}

                # --- J1 联合 ---
                if po:
                    jd = {(a, b): pa * pb
                          for a, pa in pt.items() for b, pb in po.items()}
                    rs = rank_ge(jd, (nxt_tool, truth))
                    joint_cov.append(1.0 if rs else 0.0)
                    for k in KS:
                        joint_hits[k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        joint_eff.append(eff_cand(jd))
                        pt_true = jd[(nxt_tool, truth)]
                        joint_cal.append((pt_true, 1))
                        for v in sorted((v for kk, v in jd.items()
                                         if kk != (nxt_tool, truth)),
                                        reverse=True)[:20]:
                            joint_cal.append((v, 0))
                    # 对象分量单独（对照）
                    rso = rank_ge(po, truth)
                    for k in KS:
                        factored_hits[k].append(
                            exp_hit(rso[0], rso[1], k) if rso else 0.0)
                    # --- J3 NLL：因子化 vs 条件化 ---
                    if truth in po:
                        nll_fact.append(-math.log(max(po[truth], 1e-12)))
                        row = cond_w.get(nxt_tool)
                        if row and sum(row.values()) >= 5:
                            tot = sum(row.values()) + 4 * 0.5
                            w2 = {x: (row.get(min(3, i), 0) + 0.5) / tot
                                  for i, x in enumerate(last_out)}
                            z2 = sum(w2.values())
                            pc = {k: v / z2 for k, v in w2.items()} if z2 else {}
                            nll_cond.append(-math.log(max(pc.get(truth, 1e-12), 1e-12)))
                        else:
                            nll_cond.append(-math.log(max(po[truth], 1e-12)))
                    # --- J4 分层 ---
                    covered = truth in po
                    pos = "early" if t < len(seq) / 3 else (
                        "mid" if t < 2 * len(seq) / 3 else "late")
                    # ⚠️ 分层 ECE 必须同时收正例与负例，否则算出的只是
                    # 1−mean(p_true)，不是校准误差（本脚本首版就犯了这个错）。
                    negs = sorted((v for kk, v in jd.items()
                                   if kk != (nxt_tool, truth)), reverse=True)[:20]
                    # ⚠️ prov_cov 条件在**真值**上（truth in po），预测时不可知，
                    # 故它只是「给定覆盖的条件精度」，**不能当可操作规则**。
                    # 下面三个键才是预测时可观测的，可用于真正的"何时可信"规则。
                    nlast = len(last_out)
                    nb = ("nlast=0" if nlast == 0 else "nlast=1-3" if nlast <= 3
                          else "nlast=4-10" if nlast <= 10 else "nlast=11+")
                    ec = eff_cand(jd)
                    eb = ("eff<=5" if ec <= 5 else "eff5-15" if ec <= 15
                          else "eff>15")
                    for key in (f"tool={seq[t]['tool']}", f"pos={pos}",
                                f"OBS_{nb}", f"OBS_{eb}",
                                f"POSTHOC_prov_cov={covered}"):
                        if rs:
                            strat[key].append((jd[(nxt_tool, truth)], 1))
                            for v in negs:
                                strat[key].append((v, 0))
                        strat_top3[key].append(
                            exp_hit(rs[0], rs[1], 3) if rs else 0.0)
            if ep_pairs:
                mi_pairs_per_ep.append(ep_pairs)

    # ---- J2 独立性：episode 内置换 tool 标签 ----
    rng = random.Random(SEED)
    obs_mi, null_mi, per_ep_p = [], [], []
    for pairs in mi_pairs_per_ep:
        if len(pairs) < 4:
            continue
        o = mutual_info(pairs)
        tools = [a for a, _ in pairs]
        cats = [b for _, b in pairs]
        nulls = []
        for _ in range(200):
            rng.shuffle(tools)
            nulls.append(mutual_info(list(zip(tools, cats))))
        obs_mi.append(o)
        null_mi.append(statistics.mean(nulls))
        per_ep_p.append((sum(1 for v in nulls if v >= o) + 1) / (len(nulls) + 1))
    # 全局池化互信息（更有功效）
    allp = [p for ps in mi_pairs_per_ep for p in ps]
    g_obs = mutual_info(allp)
    tools_all = [a for a, _ in allp]
    cats_all = [b for _, b in allp]
    g_nulls = []
    for _ in range(B_PERM):
        rng.shuffle(tools_all)
        g_nulls.append(mutual_info(list(zip(tools_all, cats_all))))
    g_nulls.sort()
    g_p = (sum(1 for v in g_nulls if v >= g_obs) + 1) / (len(g_nulls) + 1)

    n_pts = len(joint_cov)
    res = {
        "n_episodes": len(eps), "n_decision_points": n_pts,
        "J1_measured_joint": {
            "coverage": boot_ci(joint_cov),
            "topk_uncond": {f"top{k}": boot_ci(joint_hits[k]) for k in KS},
            "eff_candidates_median": (round(statistics.median(joint_eff), 2)
                                      if joint_eff else None),
            "ECE_cond": round(ece(joint_cal), 4),
            "object_component_only_topk": {
                f"top{k}": round(statistics.mean(factored_hits[k]), 4) for k in KS},
        },
        "J2_independence_test": {
            "note": "objcat={in_last_output,in_visited,other}；零假设=episode 内置换 tool 标签",
            "pooled_MI_observed_nats": round(g_obs, 5),
            "pooled_MI_null_mean": round(statistics.mean(g_nulls), 5),
            "pooled_MI_null_p975": round(g_nulls[int(0.975 * len(g_nulls))], 5),
            "pooled_perm_p": g_p,
            "per_episode_MI_obs": boot_ci(obs_mi),
            "per_episode_MI_null": boot_ci(null_mi),
            "per_episode_frac_p_below_05": (
                round(sum(1 for p in per_ep_p if p < 0.05) / len(per_ep_p), 4)
                if per_ep_p else None),
            "verdict": None,
        },
        "J3_factored_vs_conditional_NLL": {
            "nll_factored": boot_ci(nll_fact),
            "nll_tool_conditional": boot_ci(nll_cond),
            "diff_factored_minus_conditional": paired_ci(nll_fact, nll_cond),
            "note": "差值 >0 且显著 ⇒ 条件化更好 ⇒ 因子化误配",
        },
        "J4_stratified_top3_and_ECE": {
            k: {"top3": boot_ci(v)["mean"], "n": len(v),
                "ECE": round(ece(strat.get(k, [])), 4),
                "n_calib_pairs": len(strat.get(k, []))}
            for k, v in sorted(strat_top3.items())},
        "independence_upper_bound_note": (
            "X1 的 exp(H_tool)×exp(H_prov)=9.54 在不独立时是**上界**："
            "互信息 I>0 ⇒ 联合熵 = H_tool+H_prov−I < 边际之和"),
    }
    mi_excess = g_obs - statistics.mean(g_nulls)
    res["J2_independence_test"]["verdict"] = (
        f"独立假设**不成立**（池化互信息超出零假设 {mi_excess:.5f} nat，"
        f"perm p={g_p:.4g}）⇒ 9.54 只能作上界"
        if g_p < 0.05 else
        f"独立假设可接受（perm p={g_p:.4g}）⇒ 9.54 可作估计，仍以 J1 实测为准")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    j1 = res["J1_measured_joint"]
    print(f"\n=== J1 实测联合分布（n={n_pts}，共同分母）===")
    print(f"  覆盖 {j1['coverage']['mean']:.4f} CI{j1['coverage']['ci95']}")
    for k in KS:
        b = j1["topk_uncond"][f"top{k}"]
        print(f"  联合 top-{k:<2} {b['mean']:.4f} CI{b['ci95']}")
    print(f"  联合有效候选（实测中位）**{j1['eff_candidates_median']}**"
          f"   ECE {j1['ECE_cond']}")
    print(f"  对象分量单独 top-k: " +
          "  ".join(f"{k}={v}" for k, v in j1["object_component_only_topk"].items()))

    j2 = res["J2_independence_test"]
    print(f"\n=== J2 条件独立性检验 ===")
    print(f"  池化互信息 观测 {j2['pooled_MI_observed_nats']} nat   "
          f"零假设均值 {j2['pooled_MI_null_mean']}   "
          f"零假设 97.5% 分位 {j2['pooled_MI_null_p975']}")
    print(f"  置换 p = {j2['pooled_perm_p']:.4g}   "
          f"单 episode 显著率 {j2['per_episode_frac_p_below_05']}")
    print(f"  裁决：{j2['verdict']}")

    j3 = res["J3_factored_vs_conditional_NLL"]
    if "note" not in j3["nll_factored"]:
        d = j3["diff_factored_minus_conditional"]
        print(f"\n=== J3 因子化 vs 工具条件化（NLL，越小越好）===")
        print(f"  因子化 {j3['nll_factored']['mean']:.4f}   "
              f"工具条件化 {j3['nll_tool_conditional']['mean']:.4f}")
        if "note" not in d:
            print(f"  差值(因子化−条件化) {d['diff']:+.4f} "
                  f"CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
                  f"{'显著（条件化更好）' if d['significant'] and d['diff'] > 0 else '不显著' if not d['significant'] else '显著（因子化更好）'}")

    print(f"\n=== J4 分层 top-3 / ECE（找'何时可信'）===")
    print(f"  注：tool=X 指**当前步**的工具（做完 X 之后下一步好不好预测）；")
    print(f"      OBS_* 为预测时**可观测**的分层，可作可操作规则；")
    print(f"      POSTHOC_* 条件在真值上，**不可作规则**，仅为条件精度参考。")
    print(f"{'分层':<26}{'n':>7}{'top3':>9}{'ECE':>9}{'校准对':>8}")
    for k, v in res["J4_stratified_top3_and_ECE"].items():
        print(f"{k:<26}{v['n']:>7}{v['top3']:>9.4f}{v['ECE']:>9.4f}"
              f"{v['n_calib_pairs']:>8}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
