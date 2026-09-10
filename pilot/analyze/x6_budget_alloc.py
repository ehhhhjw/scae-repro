"""X6（零 API，V4′ 后）：用 H_t 分配重放预算 —— 桥接层实用产出。

命题
----
按 H_t 降序只重放前 q 比例的点，可保留大部分 EFFECT 点的召回；
优于随机分配（闭式期望）和均匀分配。

被估量
------
给定预算 q ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}，
EFFECT 点召回率（H_t 排序 top-q）；
对照：随机分配（q 比例随机采样，闭式期望=q），均匀等距采样。

三档判据（RESEARCH_PLAN_v5 X6）
-------------------------------
| q=0.3 保留 ≥70% EFFECT | 实用结论：归因成本可降 3 倍                 |
| 与随机无异              | 如实报告；说明必须全扫描                      |

"EFFECT" 定义
-------------
V4′ 中一个估值点被标记为 EFFECT 当且仅当：
  ys_t 与 ys_prev 的 bootstrap t-test p < 0.05（m=8 原始值，双尾）

这与 X5 的 BH-FDR 门槛一致，但 X6 用更宽松的 p=0.05 单重（不做全局 FDR），
理由是：X6 测算法优劣，不用正式假设检验，只需标记"此点有信号"。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 20260814
BUDGET_QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EFFECT_P_THRESHOLD = 0.05          # 单重，X6 内部使用
BH_Q = 0.10                        # 与 X5 一致的 BH-FDR 门槛（也报告）
BOOTSTRAP_B = 4000


def boot_pvalue_twosided(a: list[float], b_: list[float],
                         b: int = BOOTSTRAP_B, seed: int = SEED) -> float:
    """Bootstrap permutation p-value for H0: mean(a) - mean(b) = 0."""
    if len(a) < 2 or len(b_) < 2:
        return 1.0
    obs = statistics.mean(a) - statistics.mean(b_)
    combined = a + b_
    na = len(a)
    rng = random.Random(seed)
    count = 0
    for _ in range(b):
        rng.shuffle(combined)
        null = statistics.mean(combined[:na]) - statistics.mean(combined[na:])
        if abs(null) >= abs(obs):
            count += 1
    return count / b


def bh_threshold(pvalues: list[float], q: float = BH_Q) -> float:
    """Benjamini-Hochberg: return the largest p that passes at FDR q.
    Returns 0.0 if nothing passes."""
    if not pvalues:
        return 0.0
    n = len(pvalues)
    sorted_p = sorted(pvalues)
    threshold = 0.0
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) * q / n:
            threshold = p
    return threshold


def rankdata(v: list[float]) -> list[float]:
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def compute_h_strategy_recall(
    points: list[dict],       # [{"h_tool": float|None, "is_effect": bool, "pos": float}, ...]
    q: float,
    strategy: str,            # "h_tool", "uniform", "random_k"
    rng: random.Random | None = None,
) -> float:
    """Given a budget q ∈ (0,1], compute fraction of EFFECT points captured.

    strategy choices:
      "h_tool"  : pick top-q by H_tool (descending); ties broken by position
      "uniform" : pick evenly spaced q fraction by position
      "random_k": random subset of size ceil(n*q)
    Returns EFFECT recall = |{selected & EFFECT}| / |{EFFECT}|
    """
    n = len(points)
    k = max(1, math.ceil(n * q))
    effect_total = sum(1 for p in points if p["is_effect"])
    if effect_total == 0:
        return float("nan")

    if strategy == "h_tool":
        # sort by H_tool desc (use pos as tiebreak asc)
        has_h = [p for p in points if p.get("h_tool") is not None]
        no_h = [p for p in points if p.get("h_tool") is None]
        has_h_sorted = sorted(has_h, key=lambda p: (-p["h_tool"], p["pos"]))
        # fill to k: first has_h (descending H), then no_h
        selected_pool = has_h_sorted + no_h
        selected = selected_pool[:k]
    elif strategy == "uniform":
        # evenly spaced indices
        pts_by_pos = sorted(range(n), key=lambda i: points[i]["pos"])
        step = n / k
        selected_idx = {pts_by_pos[min(n - 1, int(i * step))] for i in range(k)}
        selected = [points[i] for i in range(n) if i in selected_idx]
    elif strategy == "random_k":
        assert rng is not None
        idx = list(range(n))
        rng.shuffle(idx)
        selected = [points[i] for i in idx[:k]]
    else:
        raise ValueError(strategy)

    effect_captured = sum(1 for p in selected if p["is_effect"])
    return effect_captured / effect_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime-dir", default="pilot/runs/v4prime",
                    help="V4′ 分片目录（含 shard*/attribution.jsonl）")
    ap.add_argument("--tool-model", default="pilot/runs/tool_model_official.json")
    ap.add_argument("--out", default="pilot/runs/x6_budget_alloc.json")
    ap.add_argument("--random-draws", type=int, default=200,
                    help="随机策略的蒙特卡洛次数（取均值）")
    args = ap.parse_args()

    # ---- 1. 载入工具熵 ----
    tm = json.loads(Path(args.tool_model).read_text())
    h_tool_idx: dict[tuple[str, int], float] = {
        (row["instance_id"], row["t"]): row["H_tool"]
        for row in tm.get("entropy_rows", [])
    }
    print(f"H_tool 步: {len(h_tool_idx)}")

    # ---- 2. 载入 V4′ attribution ----
    recs: list[dict] = []
    v4p = Path(args.v4prime_dir)
    for f in sorted(v4p.glob("shard*/attribution.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"V4′ 归因记录: {len(recs)}")

    if not recs:
        print("⚠️  V4′ 归因尚无数据。请等 V4′ 完成后重跑本脚本。")
        return 1

    # ---- 3. 逐 instance 构建估值点列表，标注 EFFECT ----
    all_instance_points: list[list[dict]] = []  # outer=instance, inner=segment
    all_pvalues: list[float] = []               # 全局 p-values（供 BH-FDR）
    global_pts: list[dict] = []                 # 所有点（供全局统计）

    for r in recs:
        iid = r["instance_id"]
        segs = r.get("causal_segments") or []
        if not segs:
            continue
        inst_pts: list[dict] = []
        for seg in segs:
            t = seg.get("t")
            ys_t = seg.get("ys_t") or []
            ys_prev = seg.get("ys_prev") or []
            v_t = seg.get("v_t", 0.0)
            pos = seg.get("pos", 0.0)
            if not ys_t or not ys_prev or t is None:
                continue
            if len(ys_t) < 2 or len(ys_prev) < 2:
                continue
            pval = boot_pvalue_twosided(list(ys_t), list(ys_prev))
            all_pvalues.append(pval)
            h_t = h_tool_idx.get((iid, t))
            pt = {
                "iid": iid, "t": t, "pos": pos,
                "h_tool": h_t,
                "v_t": v_t,
                "delta": seg.get("delta", 0.0),
                "pval": pval,
                "is_effect": pval < EFFECT_P_THRESHOLD,
            }
            inst_pts.append(pt)
            global_pts.append(pt)
        if inst_pts:
            all_instance_points.append(inst_pts)

    total_pts = len(global_pts)
    effect_pts = sum(1 for p in global_pts if p["is_effect"])
    print(f"估值点总数: {total_pts}, EFFECT (p<{EFFECT_P_THRESHOLD}): "
          f"{effect_pts} ({effect_pts/max(1,total_pts)*100:.1f}%)")

    # BH-FDR at q=0.10
    bh_thresh = bh_threshold(all_pvalues, q=BH_Q)
    effect_bh = sum(1 for p in all_pvalues if p <= bh_thresh)
    print(f"BH-FDR q={BH_Q} 阈值: {bh_thresh:.4f}, 显著点: {effect_bh}")

    if total_pts < 10:
        print("⚠️  估值点不足 10，功效分析意义有限。")

    # ---- 4. 分策略计算召回 ----
    rng = random.Random(SEED)

    # 4a. 全局（所有点打平）
    global_recall: dict[str, dict] = {}
    for q in BUDGET_QS:
        h_recall = compute_h_strategy_recall(global_pts, q, "h_tool")
        u_recall = compute_h_strategy_recall(global_pts, q, "uniform")
        # random: average over draws
        rand_recalls = [compute_h_strategy_recall(global_pts, q, "random_k", rng)
                        for _ in range(args.random_draws)]
        rand_valid = [x for x in rand_recalls if not math.isnan(x)]
        rand_mean = statistics.mean(rand_valid) if rand_valid else float("nan")
        global_recall[f"q={q:.1f}"] = {
            "h_tool_recall": round(h_recall, 4) if not math.isnan(h_recall) else None,
            "uniform_recall": round(u_recall, 4) if not math.isnan(u_recall) else None,
            "random_recall_mean": round(rand_mean, 4) if not math.isnan(rand_mean) else None,
            "random_expected": round(q, 4),   # 理论期望
            "h_over_random": round(h_recall - q, 4) if not math.isnan(h_recall) else None,
        }

    # 4b. Per-instance 平均召回（避免 pooling 偏差）
    per_inst_h_recalls: dict[float, list[float]] = {q: [] for q in BUDGET_QS}
    per_inst_u_recalls: dict[float, list[float]] = {q: [] for q in BUDGET_QS}
    for inst_pts in all_instance_points:
        for q in BUDGET_QS:
            h_r = compute_h_strategy_recall(inst_pts, q, "h_tool")
            u_r = compute_h_strategy_recall(inst_pts, q, "uniform")
            if not math.isnan(h_r):
                per_inst_h_recalls[q].append(h_r)
            if not math.isnan(u_r):
                per_inst_u_recalls[q].append(u_r)

    per_instance_recall: dict[str, dict] = {}
    for q in BUDGET_QS:
        h_vals = per_inst_h_recalls[q]
        u_vals = per_inst_u_recalls[q]
        per_instance_recall[f"q={q:.1f}"] = {
            "h_tool_mean": round(statistics.mean(h_vals), 4) if h_vals else None,
            "uniform_mean": round(statistics.mean(u_vals), 4) if u_vals else None,
            "h_n_instances": len(h_vals),
            "random_expected": round(q, 4),
        }

    # ---- 5. H_tool vs EFFECT 频率（分层：高/低 H_tool 的 EFFECT 率）----
    pts_with_h = [p for p in global_pts if p.get("h_tool") is not None]
    h_effect_split: dict = {}
    if pts_with_h:
        pts_by_h = sorted(pts_with_h, key=lambda p: p["h_tool"])
        n_h = len(pts_by_h)
        low_h = pts_by_h[: n_h // 2]
        high_h = pts_by_h[n_h // 2:]
        h_effect_split = {
            "low_H_effect_rate": round(sum(p["is_effect"] for p in low_h) / len(low_h), 4),
            "high_H_effect_rate": round(sum(p["is_effect"] for p in high_h) / len(high_h), 4),
            "low_H_n": len(low_h),
            "high_H_n": len(high_h),
            "note": "高 H_tool 步是否更常是 EFFECT（支持'高熵步更值得重放'假说）",
        }

    # ---- 6. 与 X4/X5 联合解读摘要 ----
    # Find q=0.3 result for quick verdict
    q3 = global_recall.get("q=0.3", {})
    h_r30 = q3.get("h_tool_recall")
    verdict = (
        "PRACTICAL: H_tool budget allocation retains ≥70% EFFECT at q=0.3"
        if h_r30 is not None and h_r30 >= 0.70
        else "NO_ADVANTAGE: H_tool allocation not better than random at q=0.3"
        if h_r30 is not None
        else "INSUFFICIENT_DATA"
    )

    res = {
        "N_instances": len(all_instance_points),
        "N_points_total": total_pts,
        "N_effect_p005": effect_pts,
        "effect_rate_p005": round(effect_pts / max(1, total_pts), 4),
        "N_effect_bh": effect_bh,
        "bh_threshold": round(bh_thresh, 4),
        "GLOBAL_RECALL_BY_BUDGET": global_recall,
        "PER_INSTANCE_RECALL_BY_BUDGET": per_instance_recall,
        "H_EFFECT_STRATIFICATION": h_effect_split,
        "VERDICT_Q03": verdict,
        "params": {
            "EFFECT_P_THRESHOLD": EFFECT_P_THRESHOLD,
            "BH_Q": BH_Q,
            "BUDGET_QS": BUDGET_QS,
            "RANDOM_DRAWS": args.random_draws,
        },
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---- 打印摘要 ----
    print(f"\n=== X6：H_t 预算分配效果 ===")
    print(f"EFFECT 点 (p<0.05): {effect_pts}/{total_pts} = {effect_pts/max(1,total_pts)*100:.1f}%")
    print(f"BH-FDR q=0.10 显著点: {effect_bh}")
    if h_effect_split:
        print(f"低 H_tool EFFECT率={h_effect_split['low_H_effect_rate']:.3f}, "
              f"高 H_tool EFFECT率={h_effect_split['high_H_effect_rate']:.3f}")
    print()
    print(f"{'q':>6}  {'H_tool':>8}  {'uniform':>8}  {'random(expect)':>14}  {'H-random':>9}")
    for q in BUDGET_QS:
        key = f"q={q:.1f}"
        g = global_recall[key]
        h_r = g['h_tool_recall'] if g['h_tool_recall'] is not None else float('nan')
        u_r = g['uniform_recall'] if g['uniform_recall'] is not None else float('nan')
        print(f"  {q:.1f}    {h_r:8.3f}  {u_r:8.3f}  {q:14.3f}  {h_r-q:+9.3f}")
    print(f"\n判据（q=0.3）: {verdict}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
