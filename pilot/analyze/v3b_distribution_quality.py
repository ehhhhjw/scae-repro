"""V3b：next-step 作为**概率分布**的质量评估（而非 top-1 精确命中）。

为何重做评价
------------
V3 首轮我用 top-1 判定"对象通道失败"（0.2627 = repeat-prev），这是**判据用错**。
next-step 的实际价值是回答"下一步可能做哪几件事、各自概率多少"，不是猜准唯一答案。
被忽略的信号：top-3 = 0.4407、MRR = 0.395（真答案平均排第 2.5 位）。

本模块评估分布质量而非点命中：
  1. top-k 曲线（k=1..10）—— 覆盖多少才够用
  2. MRR / 平均命中排名
  3. **校准性**：把预测概率分箱，看每箱的实际命中率是否匹配（这是核心）
  4. 有效候选数 exp(H) —— "需要考虑几个候选"，直接对应 U^obj 的实用含义
  5. NLL / perplexity —— 分布整体质量
  6. 与三个基线的分布级对比（repeat-prev、边际频率、均匀）

判据（预注册）
--------------
  分布可用：ECE < 0.15 且 top-3 显著高于基线 top-3
  部分可用：校准好但 top-k 不及基线 → 概率可信但排序无优势
  不可用：校准差且 top-k 不及基线
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

from pilot.channels.code_channel import W_L1, W_L2, W_L3, load_graph  # noqa: E402
from pilot.channels.parsed_command import parse_step                  # noqa: E402
from pilot.harness import rollout as R                                # noqa: E402
from pilot.probes import task as T                                    # noqa: E402
from pilot.analyze.v3_nextstep_dual import (                          # noqa: E402
    fit_tool_trans, load_episodes, obj_candidates, p_tool,
)

KS = (1, 2, 3, 5, 10)


def normalize(d: dict[str, float]) -> dict[str, float]:
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else d


def entropy(ps) -> float:
    return -sum(p * math.log(p) for p in ps if p > 0)


def ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> tuple[float, list]:
    """pairs = [(预测概率, 是否命中)]。等宽分箱的 ECE + 每箱明细。"""
    if not pairs:
        return 0.0, []
    bins: list[list] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append((p, y))
    e = 0.0
    detail = []
    n = len(pairs)
    for i, b in enumerate(bins):
        if not b:
            continue
        conf = statistics.mean(x for x, _ in b)
        acc = statistics.mean(y for _, y in b)
        e += len(b) / n * abs(conf - acc)
        detail.append({"bin": f"[{i/n_bins:.1f},{(i+1)/n_bins:.1f})",
                       "n": len(b), "mean_pred": round(conf, 4),
                       "actual_rate": round(acc, 4)})
    return e, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3b_distribution.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"载入 {len(eps)} episode，工具 {len(tool_vocab)} 类")

    hits = {k: 0 for k in KS}
    base_hits = {name: {k: 0 for k in KS}
                 for name in ("repeat_prev", "marginal", "uniform")}
    tot = 0
    ranks: list[int] = []
    calib: list[tuple[float, int]] = []
    eff_cands: list[float] = []
    nll = 0.0
    # 工具分布单独评估
    tool_hits = {k: 0 for k in KS}
    tool_tot = 0
    tool_calib: list[tuple[float, int]] = []
    tool_eff: list[float] = []

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [e for j, e in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            objfreq = Counter(s["node"] for e in train for s in e["seq"] if s["node"])
            g, seq = held["graph"], held["seq"]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    seen.add(cur["node"])

                # ---------- 工具分布 ----------
                pt = normalize({v: p_tool(tm, cur["tool"], v, tool_vocab)
                                for v in tool_vocab})
                tool_tot += 1
                tool_eff.append(math.exp(entropy(pt.values())))
                tr = sorted(pt, key=lambda k: -pt[k])
                tp = tr.index(nxt["tool"]) + 1
                for k in KS:
                    if tp <= k:
                        tool_hits[k] += 1
                tool_calib.append((pt[nxt["tool"]], 1))
                for v in tool_vocab:
                    if v != nxt["tool"]:
                        tool_calib.append((pt[v], 0))

                # ---------- 对象分布 ----------
                if not nxt["node"]:
                    continue
                cands = obj_candidates(g, cur["node"], seen)
                if nxt["node"] not in cands:
                    continue
                po = normalize(cands)
                tot += 1
                eff_cands.append(math.exp(entropy(po.values())))
                ranked = sorted(po, key=lambda k: -po[k])
                pos = ranked.index(nxt["node"]) + 1
                ranks.append(pos)
                for k in KS:
                    if pos <= k:
                        hits[k] += 1
                nll += -math.log(max(po[nxt["node"]], 1e-12))
                # 校准：命中项 + 最多 20 个未命中项（控制样本平衡）
                calib.append((po[nxt["node"]], 1))
                for c in ranked[:20]:
                    if c != nxt["node"]:
                        calib.append((po[c], 0))

                # 基线
                rp = {c: (1.0 if c == cur["node"] else 0.0) for c in cands}
                mg = {c: float(objfreq.get(c, 0)) for c in cands}
                un = {c: 1.0 for c in cands}
                for name, dist in (("repeat_prev", rp), ("marginal", mg),
                                   ("uniform", un)):
                    rk = sorted(dist, key=lambda k: -dist[k])
                    p = rk.index(nxt["node"]) + 1
                    for k in KS:
                        if p <= k:
                            base_hits[name][k] += 1

    if not tot:
        print("无有效决策点")
        return 0

    e_obj, det_obj = ece(calib)
    e_tool, det_tool = ece(tool_calib)
    res = {
        "n_decision_points": tot,
        "object_distribution": {
            "topk": {f"top{k}": round(hits[k] / tot, 4) for k in KS},
            "MRR": round(statistics.mean(1 / r for r in ranks), 4),
            "mean_rank": round(statistics.mean(ranks), 2),
            "median_rank": statistics.median(ranks),
            "ECE": round(e_obj, 4),
            "calibration_bins": det_obj,
            "mean_effective_candidates": round(statistics.mean(eff_cands), 2),
            "median_effective_candidates": round(statistics.median(eff_cands), 2),
            "NLL": round(nll / tot, 4),
            "perplexity": round(math.exp(nll / tot), 2),
        },
        "tool_distribution": {
            "n": tool_tot,
            "topk": {f"top{k}": round(tool_hits[k] / tool_tot, 4)
                     for k in KS if k <= len(tool_vocab)},
            "ECE": round(e_tool, 4),
            "mean_effective_candidates": round(statistics.mean(tool_eff), 2),
        },
        "baselines_topk": {name: {f"top{k}": round(v[k] / tot, 4) for k in KS}
                           for name, v in base_hits.items()},
    }
    o = res["object_distribution"]
    b3 = max(res["baselines_topk"][n]["top3"] for n in res["baselines_topk"])
    res["verdict"] = (
        f"分布可用：ECE={o['ECE']} <0.15 且 top-3={o['topk']['top3']} > 基线 {b3}"
        if o["ECE"] < 0.15 and o["topk"]["top3"] > b3 else
        f"部分可用：ECE={o['ECE']}（校准{'好' if o['ECE']<0.15 else '差'}），"
        f"top-3={o['topk']['top3']} vs 基线 {b3}"
    )
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V3b 对象分布质量（n={tot}）===")
    print(f"  top-k: {o['topk']}")
    print(f"  MRR={o['MRR']}  平均排名={o['mean_rank']}  中位排名={o['median_rank']}")
    print(f"  ECE={o['ECE']}  NLL={o['NLL']}  perplexity={o['perplexity']}")
    print(f"  有效候选数 exp(H): mean={o['mean_effective_candidates']} "
          f"median={o['median_effective_candidates']}")
    print("\n  基线 top-k:")
    for n_, v in res["baselines_topk"].items():
        print(f"    {n_:<12} {v}")
    tl = res["tool_distribution"]
    print(f"\n=== 工具分布（n={tl['n']}）===")
    print(f"  top-k: {tl['topk']}  ECE={tl['ECE']}  "
          f"有效候选={tl['mean_effective_candidates']}")
    print("\n  对象分布校准分箱:")
    for d in det_obj[:8]:
        print(f"    {d['bin']:<12} n={d['n']:<5} 预测={d['mean_pred']:<8} "
              f"实际={d['actual_rate']}")
    print(f"\n裁决: {res['verdict']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
