"""V3f：三层结构先验的消融 —— 排除"全靠历史先验"这一替代解释。零 API。

必须先做的理由
--------------
V3b 得到正面结果（对象分布 top-3=0.513 超基线、ECE=0.0202），但候选打分含三层：
    L1 一跳邻居 (W=0.62) / L2 同社区 (W=0.30) / L3 已访问历史 (W=0.08 + 自重复 0.04)
若 only-L3（仅历史先验）就能达到同样的 top-3，则**代码图没有增量贡献**，
V3b 的正面结论必须重新表述为"自重复先验有效"而非"代码结构有效"。

这是最需要排除的替代解释，优先级高于其他 V3 子实验。

消融臂
------
  full     L1+L2+L3          基线
  no_L1    去一跳邻居         依赖图边是否有用
  no_L2    去同社区           Leiden 聚类是否有用
  no_L3    去历史先验         自重复是否主导
  only_L3  仅历史             **关键臂**：图结构是否有增量贡献
  only_L1  仅一跳邻居         图边单独的能力
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import W_L1, W_L2, W_L3               # noqa: E402
from pilot.analyze.v3_nextstep_dual import load_episodes               # noqa: E402

KS = (1, 2, 3, 5, 10)
ARMS = ("full", "no_L1", "no_L2", "no_L3", "only_L3", "only_L1")


def candidates(g, cur_node, seen, arm: str, topk_seen: int = 30) -> dict[str, float]:
    """按消融臂构造候选与权重。"""
    use_l1 = arm in ("full", "no_L2", "no_L3", "only_L1")
    use_l2 = arm in ("full", "no_L1", "no_L3")
    use_l3 = arm in ("full", "no_L1", "no_L2", "only_L3")
    c: dict[str, float] = {}
    if cur_node and not cur_node.startswith("dir:"):
        L1 = g.neighbors(cur_node)
        L2 = g.same_community(cur_node) - L1
        if use_l1:
            for x in L1:
                c[x] = c.get(x, 0.0) + W_L1 / max(1, len(L1))
        if use_l2:
            for x in L2:
                c[x] = c.get(x, 0.0) + W_L2 / max(1, len(L2))
    if use_l3:
        hist = list(seen)[-topk_seen:]
        for x in hist:
            c[x] = c.get(x, 0.0) + W_L3 / max(1, len(hist))
        if cur_node:
            c[cur_node] = c.get(cur_node, 0.0) + W_L3 / 2
    return c


def ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> float:
    if not pairs:
        return 0.0
    bins: list[list] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(n_bins - 1, int(p * n_bins))].append((p, y))
    e, n = 0.0, len(pairs)
    for b in bins:
        if b:
            e += len(b) / n * abs(statistics.mean(x for x, _ in b)
                                  - statistics.mean(y for _, y in b))
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3f_ablation.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    print(f"载入 {len(eps)} episode")

    stat = {a: {"hits": {k: 0 for k in KS}, "tot": 0, "cov_num": 0, "cov_den": 0,
                "ranks": [], "calib": [], "eff": [], "nll": 0.0} for a in ARMS}

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            g, seq = held["graph"], held["seq"]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    seen.add(cur["node"])
                if not nxt["node"]:
                    continue
                for arm in ARMS:
                    s = stat[arm]
                    cands = candidates(g, cur["node"], seen, arm)
                    s["cov_den"] += 1
                    if not cands or nxt["node"] not in cands:
                        continue
                    s["cov_num"] += 1
                    z = sum(cands.values())
                    p = {k: v / z for k, v in cands.items()}
                    ranked = sorted(p, key=lambda k: -p[k])
                    pos = ranked.index(nxt["node"]) + 1
                    s["tot"] += 1
                    s["ranks"].append(pos)
                    for k in KS:
                        if pos <= k:
                            s["hits"][k] += 1
                    s["nll"] += -math.log(max(p[nxt["node"]], 1e-12))
                    s["eff"].append(math.exp(-sum(v * math.log(v)
                                                  for v in p.values() if v > 0)))
                    s["calib"].append((p[nxt["node"]], 1))
                    for cc in ranked[:20]:
                        if cc != nxt["node"]:
                            s["calib"].append((p[cc], 0))

    res = {"arms": {}}
    for a in ARMS:
        s = stat[a]
        if not s["tot"]:
            res["arms"][a] = {"note": "无有效点"}
            continue
        res["arms"][a] = {
            "n": s["tot"],
            "coverage": round(s["cov_num"] / s["cov_den"], 4) if s["cov_den"] else None,
            "topk": {f"top{k}": round(s["hits"][k] / s["tot"], 4) for k in KS},
            "MRR": round(statistics.mean(1 / r for r in s["ranks"]), 4),
            "median_rank": statistics.median(s["ranks"]),
            "ECE": round(ece(s["calib"]), 4),
            "eff_candidates": round(statistics.median(s["eff"]), 2),
            "perplexity": round(math.exp(s["nll"] / s["tot"]), 2),
        }

    full = res["arms"].get("full", {})
    only3 = res["arms"].get("only_L3", {})
    if full.get("topk") and only3.get("topk"):
        d3 = full["topk"]["top3"] - only3["topk"]["top3"]
        res["key_contrast"] = {
            "full_top3": full["topk"]["top3"],
            "only_L3_top3": only3["topk"]["top3"],
            "graph_increment": round(d3, 4),
            "verdict": ("代码图有增量贡献：full 显著高于 only_L3"
                        if d3 > 0.03 else
                        "⚠️ 代码图无增量贡献：正面结果实际来自历史先验，"
                        "V3b 结论须重新表述"),
        }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V3f 三层先验消融 ===")
    print(f"{'臂':<10}{'n':>6}{'覆盖':>8}{'top1':>8}{'top3':>8}{'top10':>8}"
          f"{'MRR':>8}{'ECE':>8}{'有效候选':>9}")
    for a in ARMS:
        r = res["arms"][a]
        if "note" in r:
            print(f"{a:<10} {r['note']}")
            continue
        print(f"{a:<10}{r['n']:>6}{r['coverage']:>8}{r['topk']['top1']:>8}"
              f"{r['topk']['top3']:>8}{r['topk']['top10']:>8}"
              f"{r['MRR']:>8}{r['ECE']:>8}{r['eff_candidates']:>9}")
    if "key_contrast" in res:
        kc = res["key_contrast"]
        print(f"\n关键对比 full({kc['full_top3']}) - only_L3({kc['only_L3_top3']}) "
              f"= {kc['graph_increment']:+.4f}")
        print(f"裁决: {kc['verdict']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
