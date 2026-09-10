"""V3h：「agent 倾向复用已用过的工具/文件」这一假设的严格检验。零 API。

为什么需要零假设对照
--------------------
朴素读法是看复用率（P(next ∈ 之前出现过的)）。但该量是**置换不变量**：
序列位置 2..n 中"首次出现"的个数恒为 #distinct−1，故

    复用率 = (n − #distinct) / (n − 1)

完全由"多少步、多少个不同对象"决定，把序列任意打乱都不变。
所以 33.0% 的重访率**不构成任何"倾向"的证据** —— 它只是
「24 步里只碰了 14 个不同对象」的算术后果。本脚本把这一点作为
零方差检查 (O0) 显式验证，避免把算术当发现。

真正可检验的是复用的**结构**，即在给定成分下时序排列是否比随机更"聚集"：

  T1/O1  紧邻重复      P(x_{t+1} = x_t)                    置换检验
  T2/O2  序列依赖      互信息 I(x_t; x_{t+1})              置换检验
  O3     突发性        同一对象相邻两次出现的间隔          置换检验
  O4     高频偏好      重访目标的历史访问次数 vs 均匀       闭式期望
  O5     近期偏好      重访目标的新近度排名 vs 均匀         闭式期望

零假设：**episode 内置换**（保留该 episode 的成分与长度，只打乱顺序）。
这正好隔离"倾向"（时序结构）与"词表小"（成分）两种解释。
O4/O5 用闭式期望而非采样（踩坑 #15）。
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

from pilot.analyze.v3_nextstep_dual import load_episodes    # noqa: E402

B = 2000
SEED = 20260814


# ---------------------------------------------------------------- 统计量

def self_repeat(seq: list) -> tuple[float, int]:
    """紧邻重复率 P(x_{t+1} = x_t)。"""
    n = sum(1 for a, b in zip(seq, seq[1:]))
    if not n:
        return float("nan"), 0
    return sum(1 for a, b in zip(seq, seq[1:]) if a == b) / n, n


def reuse_rate(seq: list) -> tuple[float, int]:
    """复用率 P(x_{t+1} ∈ {x_1..x_t})。**置换不变**，仅用于 O0 检查。"""
    seen, hit, n = {seq[0]}, 0, 0
    for x in seq[1:]:
        n += 1
        if x in seen:
            hit += 1
        seen.add(x)
    return (hit / n if n else float("nan")), n


def mutual_info(seq: list) -> float:
    """I(x_t; x_{t+1})，nat。度量超出边际分布的序列依赖。"""
    pairs = list(zip(seq, seq[1:]))
    if not pairs:
        return 0.0
    n = len(pairs)
    pj = Counter(pairs)
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    mi = 0.0
    for (a, b), c in pj.items():
        mi += (c / n) * math.log((c / n) / ((pa[a] / n) * (pb[b] / n)))
    return mi


def mean_gap(seq: list) -> float | None:
    """同一对象相邻两次出现的平均间隔。越小越"突发/聚集"。"""
    pos: dict = defaultdict(list)
    for i, x in enumerate(seq):
        pos[x].append(i)
    gaps = [b - a for v in pos.values() for a, b in zip(v, v[1:])]
    return statistics.mean(gaps) if gaps else None


def perm_test(seq: list, stat_fn, n_perm: int = B, seed: int = SEED,
              alternative: str = "greater") -> dict:
    """episode 内置换检验。返回观测值、零分布均值、p 值。"""
    obs = stat_fn(seq)
    if obs is None or (isinstance(obs, float) and math.isnan(obs)):
        return {"obs": None}
    rng = random.Random(seed)
    work = list(seq)
    null = []
    for _ in range(n_perm):
        rng.shuffle(work)
        v = stat_fn(work)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            null.append(v)
    if not null:
        return {"obs": obs}
    if alternative == "greater":
        ge = sum(1 for v in null if v >= obs)
    else:
        ge = sum(1 for v in null if v <= obs)
    return {"obs": obs, "null_mean": statistics.mean(null),
            "null_sd": (statistics.stdev(null) if len(null) > 1 else 0.0),
            "p": (ge + 1) / (len(null) + 1)}


# ---------------------------------------------------------------- 闭式偏好

def freq_recency_bias(seq: list) -> dict:
    """重访事件上的高频/近期偏好，用**闭式期望**做零假设。

    每个重访事件（x_{t+1} ∈ seen_t）：
      高频：目标的历史访问次数 c*  vs  均匀取一个 seen 对象的期望次数 E[c]
      近期：目标的新近度排名 r*（1=最近访问过）vs 均匀期望 (m+1)/2
    """
    obs_c, exp_c, obs_r, exp_r, n = [], [], [], [], 0
    counts: Counter = Counter()
    order: list = []            # 按最近访问排序（前=最近）
    for i, x in enumerate(seq):
        if i > 0 and counts and x in counts:
            n += 1
            m = len(counts)
            obs_c.append(counts[x])
            exp_c.append(sum(counts.values()) / m)          # 均匀取一个 seen 的期望次数
            obs_r.append(order.index(x) + 1)
            exp_r.append((m + 1) / 2)                        # 均匀排名期望
        counts[x] += 1
        if x in order:
            order.remove(x)
        order.insert(0, x)
    if not n:
        return {"n": 0}
    return {
        "n_revisit_events": n,
        "freq_obs": round(statistics.mean(obs_c), 4),
        "freq_null": round(statistics.mean(exp_c), 4),
        "freq_lift": round(statistics.mean(obs_c) / statistics.mean(exp_c), 4),
        "recency_obs_rank": round(statistics.mean(obs_r), 4),
        "recency_null_rank": round(statistics.mean(exp_r), 4),
        "recency_lift": round(statistics.mean(exp_r) / statistics.mean(obs_r), 4),
    }


# ---------------------------------------------------------------- 汇总

def agg(per_ep: list[dict], key: str) -> dict:
    """跨 episode 汇总：对每 episode 的统计量做 episode 级 bootstrap。"""
    vals = [e[key] for e in per_ep if e.get(key) is not None]
    if not vals:
        return {"note": "无样本"}
    rng = random.Random(SEED)
    pts = []
    for _ in range(B):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        pts.append(statistics.mean(s))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * B)], 4), round(pts[int(0.975 * B)], 4)],
            "n_episodes": len(vals)}


def fisher_combine(ps: list[float]) -> dict:
    """把每 episode 的置换 p 值合并（Fisher's method）。"""
    ps = [p for p in ps if p is not None and 0 < p <= 1]
    if not ps:
        return {"note": "无 p 值"}
    x2 = -2 * sum(math.log(p) for p in ps)
    df = 2 * len(ps)
    # 卡方生存函数（df 偶数的闭式）：P = e^{-t}·Σ_{i<k} t^i/i!
    # 直接算会溢出（k 达数百），故在**对数空间**用 log-sum-exp。
    k, t = df // 2, x2 / 2
    terms = [i * math.log(t) - math.lgamma(i + 1) for i in range(k)] if t > 0 else [0.0]
    mx = max(terms)
    log_p = -t + mx + math.log(sum(math.exp(v - mx) for v in terms))
    return {"n_episodes": len(ps), "chi2": round(x2, 2), "df": df,
            "log10_p_combined": round(log_p / math.log(10), 2),
            "p_combined": (math.exp(log_p) if log_p > -700 else 0.0),
            "median_p": round(statistics.median(ps), 4),
            "frac_p_below_0.05": round(sum(1 for p in ps if p < 0.05) / len(ps), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3h_reuse.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root), graph_only=False)
    print(f"载入 {len(eps)} episode（不限已构图）")

    rows = []
    for e in eps:
        tools = [s["tool"] for s in e["seq"]]
        objs = [s["obj"] for s in e["seq"] if s["obj"]]
        if len(tools) < 4:
            continue
        r: dict = {"id": e["id"], "n_steps": len(tools),
                   "n_tools_distinct": len(set(tools)),
                   "n_objs": len(objs), "n_objs_distinct": len(set(objs))}
        # --- 工具 ---
        t1 = perm_test(tools, lambda s: self_repeat(s)[0])
        t2 = perm_test(tools, mutual_info)
        r["tool_selfrep"] = t1.get("obs")
        r["tool_selfrep_null"] = t1.get("null_mean")
        r["tool_selfrep_p"] = t1.get("p")
        r["tool_mi"] = t2.get("obs")
        r["tool_mi_null"] = t2.get("null_mean")
        r["tool_mi_p"] = t2.get("p")
        r["tool_reuse"] = reuse_rate(tools)[0]
        r["tool_reuse_perm"] = perm_test(tools, lambda s: reuse_rate(s)[0]).get("null_sd")
        # --- 对象 ---
        if len(objs) >= 4:
            o1 = perm_test(objs, lambda s: self_repeat(s)[0])
            o2 = perm_test(objs, mutual_info)
            o3 = perm_test(objs, mean_gap, alternative="less")
            r["obj_selfrep"] = o1.get("obs")
            r["obj_selfrep_null"] = o1.get("null_mean")
            r["obj_selfrep_p"] = o1.get("p")
            r["obj_mi"] = o2.get("obs")
            r["obj_mi_null"] = o2.get("null_mean")
            r["obj_mi_p"] = o2.get("p")
            r["obj_gap"] = o3.get("obs")
            r["obj_gap_null"] = o3.get("null_mean")
            r["obj_gap_p"] = o3.get("p")
            r["obj_reuse"] = reuse_rate(objs)[0]
            r["obj_reuse_perm"] = perm_test(objs, lambda s: reuse_rate(s)[0]).get("null_sd")
            r.update({f"bias_{k}": v for k, v in freq_recency_bias(objs).items()})
        rows.append(r)

    res: dict = {
        "n_episodes": len(rows),
        "note": "零假设=episode 内置换（保留成分与长度，打乱顺序），B=2000",
        "O0_reuse_is_permutation_invariant": {
            "tool_reuse_null_sd_max": max((r.get("tool_reuse_perm") or 0)
                                          for r in rows),
            "obj_reuse_null_sd_max": max((r.get("obj_reuse_perm") or 0)
                                         for r in rows),
            "tool_reuse_mean": agg(rows, "tool_reuse"),
            "obj_reuse_mean": agg(rows, "obj_reuse"),
            "verdict": None,      # 下面填
        },
        "T1_tool_self_repeat": {"obs": agg(rows, "tool_selfrep"),
                                "null": agg(rows, "tool_selfrep_null"),
                                "perm": fisher_combine([r.get("tool_selfrep_p")
                                                        for r in rows])},
        "T2_tool_mutual_info": {"obs": agg(rows, "tool_mi"),
                                "null": agg(rows, "tool_mi_null"),
                                "perm": fisher_combine([r.get("tool_mi_p")
                                                        for r in rows])},
        "O1_obj_self_repeat": {"obs": agg(rows, "obj_selfrep"),
                               "null": agg(rows, "obj_selfrep_null"),
                               "perm": fisher_combine([r.get("obj_selfrep_p")
                                                       for r in rows])},
        "O2_obj_mutual_info": {"obs": agg(rows, "obj_mi"),
                               "null": agg(rows, "obj_mi_null"),
                               "perm": fisher_combine([r.get("obj_mi_p")
                                                       for r in rows])},
        "O3_obj_burstiness_gap": {"obs": agg(rows, "obj_gap"),
                                  "null": agg(rows, "obj_gap_null"),
                                  "perm": fisher_combine([r.get("obj_gap_p")
                                                          for r in rows])},
        "O4_frequency_bias": {"obs": agg(rows, "bias_freq_obs"),
                              "null": agg(rows, "bias_freq_null"),
                              "lift": agg(rows, "bias_freq_lift")},
        "O5_recency_bias": {"obs_rank": agg(rows, "bias_recency_obs_rank"),
                            "null_rank": agg(rows, "bias_recency_null_rank"),
                            "lift": agg(rows, "bias_recency_lift")},
        "per_episode": rows,
    }
    sd_max = max(res["O0_reuse_is_permutation_invariant"]["tool_reuse_null_sd_max"],
                 res["O0_reuse_is_permutation_invariant"]["obj_reuse_null_sd_max"])
    res["O0_reuse_is_permutation_invariant"]["verdict"] = (
        f"置换零分布标准差最大 {sd_max:.2e} ⇒ 复用率确为置换不变量，"
        f"不能作为『倾向』的证据" if sd_max < 1e-9 else
        f"⚠️ 标准差 {sd_max:.2e} 非零，与理论不符，需检查实现")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---------------------------------------------------------- 打印
    o0 = res["O0_reuse_is_permutation_invariant"]
    print(f"\n=== O0 复用率是置换不变量（先排除算术假象）===")
    print(f"  工具复用率 {o0['tool_reuse_mean']['mean']}  "
          f"CI{o0['tool_reuse_mean']['ci95']}")
    print(f"  对象复用率 {o0['obj_reuse_mean']['mean']}  "
          f"CI{o0['obj_reuse_mean']['ci95']}")
    print(f"  {o0['verdict']}")

    print(f"\n=== 时序结构的置换检验（这些才是真『倾向』）===")
    print(f"{'统计量':<22}{'观测':>9}{'零假设':>9}{'比值':>8}   合并 p")
    for key, label, inv in (
        ("T1_tool_self_repeat", "工具 紧邻重复", False),
        ("T2_tool_mutual_info", "工具 互信息(nat)", False),
        ("O1_obj_self_repeat", "对象 紧邻重复", False),
        ("O2_obj_mutual_info", "对象 互信息(nat)", False),
        ("O3_obj_burstiness_gap", "对象 平均间隔↓", True),
    ):
        d = res[key]
        o, nl = d["obs"].get("mean"), d["null"].get("mean")
        if o is None or nl is None:
            continue
        ratio = (nl / o if inv and o else (o / nl if nl else float("nan")))
        lp = d["perm"].get("log10_p_combined")
        ps = (f"1e{lp:.0f}" if lp is not None and lp < -3
              else ("%.3g" % d["perm"]["p_combined"]) if lp is not None else "—")
        frac = d["perm"].get("frac_p_below_0.05")
        print(f"{label:<22}{o:>9.4f}{nl:>9.4f}{ratio:>8.2f}   {ps}"
              f"   单ep显著率 {frac}   (n={d['obs']['n_episodes']} ep)")
        print(f"{'':<22}观测 CI{d['obs']['ci95']}  零假设 CI{d['null']['ci95']}")

    print(f"\n=== 重访目标的选择偏好（闭式期望零假设）===")
    o4, o5 = res["O4_frequency_bias"], res["O5_recency_bias"]
    print(f"  高频偏好：目标历史访问次数 {o4['obs']['mean']} vs 均匀期望 "
          f"{o4['null']['mean']}   lift {o4['lift']['mean']} CI{o4['lift']['ci95']}")
    print(f"  近期偏好：目标新近度排名 {o5['obs_rank']['mean']} vs 均匀期望 "
          f"{o5['null_rank']['mean']}   lift {o5['lift']['mean']} CI{o5['lift']['ci95']}")

    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
