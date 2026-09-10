"""V3：next-step 预测重做 —— 双通道分解（工具 × 对象）。零 API。

首轮 next-step 为何失败
----------------------
三种设置全部输给 repeat-prev（0.354/0.307/0.220 vs 0.371/0.294/0.236），
候选覆盖率仅 48%。根因：候选集是**字符串路径集合**，路径跨 repo 不存在，
且未建模"步与步之间为什么这样衔接"。

本次改动
--------
按双通道分解 P(u_{t+1}) = P(tool_{t+1}|·) · P(obj_{t+1}|obj_t, G)：
  工具分量  用 V1' 的官方 6 类条件转移模型（已验证 top-1 56.6%）
  对象分量  候选集改为**代码图结构邻域**：
             L1 一跳邻居 / L2 同社区 / L3 其余
           打分用结构先验（W_L1=0.62 等，由 E3 实测相邻步图距离定出）

评估
----
同 repo 内 leave-one-episode-out（next-step 的正确设置：实际用途是在当前
仓库里预测下一步，不需要跨仓库迁移）。指标 top-1/top-3/MRR + 候选覆盖率。
基线：repeat-prev（曾击败我们）、边际频率、随机。
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


def load_episodes(runs_root: Path, graph_only: bool = True) -> list[dict]:
    """载入 (tool, obj) 序列。graph_only=True 时只保留已构图的仓库。"""
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        g = load_graph(iid)
        if graph_only and not g.ok:
            continue
        try:
            steps = R.steps(R.load(f))
        except (OSError, ValueError):
            continue
        if len(steps) < 3:
            continue
        seq = []
        for s in steps:
            p = parse_step(s)
            node = g.resolve(p.path or p.obj) if g.ok else None
            seq.append({"tool": p.kind, "obj": p.obj, "node": node})
        eps.append({"id": iid, "project": inst.project, "seq": seq, "graph": g})
    return eps


def obj_candidates(g, cur_node: str | None, seen: set[str],
                   topk_seen: int = 30) -> dict[str, float]:
    """对象候选集 + 结构先验权重。

    L1 一跳邻居 / L2 同社区 / L3 已访问过的（历史先验）
    L3 用"已访问"而非"全仓库"，因为全仓库候选会把 top-1 稀释到无意义。
    """
    cands: dict[str, float] = {}
    if cur_node and not cur_node.startswith("dir:"):
        L1 = g.neighbors(cur_node)
        L2 = g.same_community(cur_node) - L1
        for x in L1:
            cands[x] = cands.get(x, 0.0) + W_L1 / max(1, len(L1))
        for x in L2:
            cands[x] = cands.get(x, 0.0) + W_L2 / max(1, len(L2))
    hist = list(seen)[-topk_seen:]
    for x in hist:
        cands[x] = cands.get(x, 0.0) + W_L3 / max(1, len(hist))
    if cur_node:
        cands[cur_node] = cands.get(cur_node, 0.0) + W_L3 / 2   # 自重复先验
    return cands


def fit_tool_trans(eps: list[dict], alpha: float = 0.5) -> dict:
    tr: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        seq = e["seq"]
        for a, b in zip(seq, seq[1:]):
            tr[a["tool"]][b["tool"]] += 1
    return {"trans": tr, "alpha": alpha}


def p_tool(m: dict, cur: str, nxt: str, vocab: list[str]) -> float:
    a = m["alpha"]
    row = m["trans"].get(cur)
    if not row or sum(row.values()) == 0:
        return 1.0 / len(vocab)
    return (row.get(nxt, 0) + a) / (sum(row.values()) + a * len(vocab))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3_nextstep_dual.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"载入 {len(eps)} episode（已构图仓库），"
          f"{sum(len(e['seq'])-1 for e in eps)} 个步转移，工具 {len(tool_vocab)} 类")

    tot = h1 = h3 = 0
    rr = 0.0
    cov_num = cov_den = 0
    base = {"repeat_prev": 0, "marginal": 0}
    tool_h1 = obj_h1 = tool_den = 0

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [e for j, e in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            objfreq = Counter(s["node"] for e in train for s in e["seq"] if s["node"])
            g = held["graph"]
            seq = held["seq"]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    seen.add(cur["node"])
                # --- 工具分量 ---
                pt = {v: p_tool(tm, cur["tool"], v, tool_vocab) for v in tool_vocab}
                tool_den += 1
                if max(pt, key=pt.get) == nxt["tool"]:
                    tool_h1 += 1
                # --- 对象分量 ---
                if not nxt["node"]:
                    continue
                cands = obj_candidates(g, cur["node"], seen)
                cov_den += 1
                if nxt["node"] not in cands:
                    continue
                cov_num += 1
                # 联合打分：P(tool) * P(obj)
                joint = {}
                for node, w in cands.items():
                    joint[node] = w * pt.get(nxt["tool"], 1e-6)
                ranked = sorted(joint, key=lambda k: -joint[k])
                pos = ranked.index(nxt["node"]) + 1
                tot += 1
                if pos == 1:
                    h1 += 1
                    obj_h1 += 1
                if pos <= 3:
                    h3 += 1
                rr += 1.0 / pos
                if cur["node"] == nxt["node"]:
                    base["repeat_prev"] += 1
                if objfreq and max(cands, key=lambda c: objfreq.get(c, 0)) == nxt["node"]:
                    base["marginal"] += 1

    if not tot:
        print("无有效决策点")
        return 0
    res = {
        "setting": "within-repo LOO, dual-channel (tool x object-graph)",
        "n_episodes": len(eps), "n_decision_points": tot,
        "candidate_coverage": round(cov_num / cov_den, 4) if cov_den else None,
        "top1": round(h1 / tot, 4), "top3": round(h3 / tot, 4),
        "MRR": round(rr / tot, 4),
        "baselines_top1": {k: round(v / tot, 4) for k, v in base.items()},
        "tool_component_top1": round(tool_h1 / max(1, tool_den), 4),
        "tool_component_n": tool_den,
        "prior_result": {"note": "首轮三设置", "coverage": [0.4815, 0.6074, 0.9774],
                         "top1": [0.354, 0.307, 0.220],
                         "repeat_prev": [0.371, 0.294, 0.236]},
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n=== V3 双通道 next-step（{res['setting']}）===")
    print(f"  决策点 {tot}   候选覆盖率 {res['candidate_coverage']}")
    print(f"  top-1 {res['top1']}   top-3 {res['top3']}   MRR {res['MRR']}")
    print(f"  基线 top-1: {res['baselines_top1']}")
    print(f"  工具分量单独 top-1: {res['tool_component_top1']}")
    print(f"\n  对比首轮: 覆盖率 48.2/60.7/97.7% → {res['candidate_coverage']:.1%}")
    print(f"            top-1 0.354/0.307/0.220 → {res['top1']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
