"""V3j：把 trace 本身当图 —— 执行溯源图（provenance graph）上的预测与归因。零 API。

动机
----
前面所有实验都把 trace 当**序列**（tool_t → tool_{t+1}），把结构信息寄托在
**代码依赖图**上。但 trace 本身就是一张图：第 t 步的**输出**是第 t' 步的
**输入依据**。V3i 已证代码图不刻画步间转移（相邻步一跳率 lift 0.94），
而一个更贴近机制的假设从未被检验：

    agent 不是走到"代码上的邻居"，而是走到**它刚在某个工具输出里看到的东西**。

grep 返回 `astropy/io/ascii/rst.py:35:class RST(...)`，下一步就去读这个文件。
这是一条**溯源边** s_t → s_{t'}，与代码依赖无关。这正好能覆盖 V3g 判定
"不可预测"的 67% 首访步 —— 因为首访对象虽未被**访问**过，却很可能已被**提及**过。

溯源图定义
----------
节点 = 步 s_1..s_n（+ 终止答案节点 A）
边：
  P 溯源   s_t → s_{t'}  若 s_{t'} 的操作对象**字面出现在** s_t 的输出里 (t<t')
  S 顺序   s_t → s_{t+1}
  A 答案   s_t → A       若 s_t 的对象出现在最终答案里

本脚本做的实验
--------------
  G0 溯源覆盖率：首访对象中有多大比例**先被提及过**？（决定假设的上限）
  G1 溯源预测器：uniform / recency / freq 三种加权，在**与 V3g 逐点一致**的
     点集上评估，并与 marginal、代码图多源邻近（V3i 的 C1）配对对比
  G2 路由修正：pi·历史 + (1−pi)·溯源(未访问)，对比 V3g 的 pi·历史 + (1−pi)·代码图
  G3 溯源图的结构量：入度/出度/是否答案祖先，作为逐步 UQ 候选量
     （与 U^exec 的相关待 V4′ 完成后接入 V4′ 的真实重放方差）

判据
----
若 G0 覆盖率高且 G1 在 F regime 上显著超过 marginal 与 C1，
则「首访步不可预测」这一负面结论应修正为「**用错了结构**」——
正确的结构不是代码图而是 agent 自身的观测溯源。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import load_graph                  # noqa: E402
from pilot.channels.parsed_command import parse_step                # noqa: E402
from pilot.harness import rollout as R                              # noqa: E402
from pilot.probes import task as T                                  # noqa: E402
from pilot.analyze.v3g_regime import (                              # noqa: E402
    p_hist, p_graph, rank_ge, exp_hit, exp_recip, eff_cand, ece, Acc, KS,
)
from pilot.analyze.v3i_graph_probe import (                         # noqa: E402
    multi_source_dist, seed_adjacency,
)

SEED = 20260814
# 路径样式：含扩展名的相对路径，或 grep 的 `path:line:` 前缀
PATH_RE = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:py|pyx|pyi|rst|txt|cfg|toml|ini|md|json|ya?ml|c|h|cpp|hpp|js|html))"
)


def load_episodes_with_output(runs_root: Path) -> list[dict]:
    """与 v3_nextstep_dual.load_episodes 同构，但**保留每步的输出文本**。"""
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        g = load_graph(iid)
        if not g.ok:
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
            seq.append({"tool": p.kind, "obj": p.obj,
                        "node": g.resolve(p.path or p.obj),
                        "out": s.output_text or ""})
        eps.append({"id": iid, "project": inst.project, "seq": seq,
                    "graph": g, "gold": inst.gold})
    return eps


def mentioned_nodes(g, text: str, cap: int = 4000) -> list[str]:
    """从一段工具输出里抽出可解析为图节点的路径（去重，保序）。"""
    if not text:
        return []
    out, seen = [], set()
    for m in PATH_RE.finditer(text[:200_000]):
        raw = m.group(1)
        n = g.resolve(raw)
        if n and not n.startswith("dir:") and n not in seen:
            seen.add(n)
            out.append(n)
            if len(out) >= cap:
                break
    return out


def p_prov(mentions: dict[str, list[int]], t: int, mode: str,
           gamma: float = 0.8, exclude: set[str] | None = None) -> dict[str, float]:
    """溯源先验。mentions[node] = 被提及的步号列表。

    mode: uniform / recency（按最近一次提及衰减）/ freq（按提及次数）
    """
    w: dict[str, float] = {}
    for node, occ in mentions.items():
        if exclude and node in exclude:
            continue
        if mode == "uniform":
            w[node] = 1.0
        elif mode == "recency":
            w[node] = gamma ** max(0, t - max(occ))
        elif mode == "freq":
            w[node] = float(len(occ))
        else:
            raise ValueError(mode)
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def boot_ci(vals: list[float], b: int = 2000, seed: int = SEED) -> dict:
    if not vals:
        return {"note": "无样本"}
    rng = random.Random(seed)
    pts = []
    for _ in range(b):
        pts.append(sum(vals[rng.randrange(len(vals))]
                       for _ in range(len(vals))) / len(vals))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * b)], 4), round(pts[int(0.975 * b)], 4)],
            "n": len(vals)}


def paired_ci(a: list[float], b: list[float], bnum: int = 2000,
              seed: int = SEED) -> dict:
    if len(a) != len(b) or not a:
        return {"note": f"点集不匹配 {len(a)} vs {len(b)}"}
    rng = random.Random(seed)
    pts = []
    for _ in range(bnum):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        pts.append(sum(a[i] - b[i] for i in idx) / len(idx))
    pts.sort()
    lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
    return {"diff": round(statistics.mean(a) - statistics.mean(b), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3j_trace_graph.json")
    args = ap.parse_args()

    eps = load_episodes_with_output(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    print(f"载入 {len(eps)} episode（已构图，含输出文本）")

    PRED = ["prov_uniform", "prov_recency", "prov_freq",
            "prov_lastout", "prov_lastout_ranked",
            "hist_freq", "marginal_repo", "graph_C1",
            "routed_prov", "routed_graph"]
    # 逐点期望命中，按 regime 分桶；F 桶用于关键对比
    hits = {p: {r: {k: [] for k in KS} for r in ("all", "R", "F")} for p in PRED}
    cov = {p: {r: [] for r in ("all", "R", "F")} for p in PRED}
    effc = {p: [] for p in PRED}

    g0_mentioned_before, g0_first_visit = [], 0
    g0_by_regime = {"R": [], "F": []}
    n_R = n_F = 0
    # G3 结构量
    g3_rows = []

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            objfreq = Counter(s["node"] for x in train for s in x["seq"] if s["node"])
            g, seq = held["graph"], held["seq"]
            nodes = [s["node"] for s in seq]
            # 预抽每步输出里提到的节点
            outs = [mentioned_nodes(g, s["out"]) for s in seq]

            mentions: dict[str, list[int]] = defaultdict(list)
            visits: list[str] = []
            seen: set[str] = set()
            prev_rev = False
            # regime 转移率（与 V3g 同法，从 train 估）
            rc: dict[bool, Counter] = {True: Counter(), False: Counter()}
            for x in train:
                sq, sn = x["seq"], set()
                pr = False
                for t2 in range(len(sq) - 1):
                    if sq[t2]["node"]:
                        sn.add(sq[t2]["node"])
                    if not sq[t2 + 1]["node"]:
                        continue
                    isr = sq[t2 + 1]["node"] in sn
                    rc[pr][isr] += 1
                    pr = isr

            def pi_hat(pr: bool) -> float:
                row = rc[pr]
                tot = sum(row.values())
                return (row[True] + 0.5) / (tot + 1.0) if tot else 0.5

            for t in range(len(seq) - 1):
                if nodes[t]:
                    visits.append(nodes[t])
                    seen.add(nodes[t])
                # 第 t 步的输出在预测第 t+1 步时**已可见**
                for nd in outs[t]:
                    mentions[nd].append(t)
                truth = nodes[t + 1]
                if not truth:
                    continue
                is_rev = truth in seen
                reg = "R" if is_rev else "F"
                if is_rev:
                    n_R += 1
                else:
                    n_F += 1
                    g0_first_visit += 1
                    g0_mentioned_before.append(1.0 if truth in mentions else 0.0)
                g0_by_regime[reg].append(1.0 if truth in mentions else 0.0)

                pi = pi_hat(prev_rev)
                fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
                # 代码图 C1 多源邻近（V3i 的最强形式），排除已访问
                if fseeds:
                    dd = multi_source_dist(g, fseeds)
                    aa = seed_adjacency(g, fseeds)
                    mx = max(dd.values()) or 1
                    c1 = {x: (mx - dd[x] + aa.get(x, 0) / 100.0)
                          for x in dd if x not in seen}
                    z1 = sum(c1.values())
                    c1 = {k: v / z1 for k, v in c1.items()} if z1 else {}
                else:
                    c1 = {}
                prov_uns = p_prov(mentions, t, "freq", exclude=seen)
                ph = p_hist(visits, nodes[t], "freq")
                # 更锋利的机制检验：只用**最近一次有提及的输出**。
                # 若它已达到 prov_recency 的水平，则机制就是
                # "读刚才那条输出里列出的文件"，而非"整段观测史的加权"。
                last: list[str] = []
                for tt in range(t, -1, -1):
                    if outs[tt]:
                        last = outs[tt]
                        break
                lo_u = _norm({x: 1.0 for x in last})
                lo_r = _norm({x: 1.0 / (i + 1) for i, x in enumerate(last)})
                dists = {
                    "prov_uniform": p_prov(mentions, t, "uniform"),
                    "prov_recency": p_prov(mentions, t, "recency"),
                    "prov_freq":    p_prov(mentions, t, "freq"),
                    "prov_lastout": lo_u,
                    "prov_lastout_ranked": lo_r,
                    "hist_freq":    ph,
                    "marginal_repo": _norm({k: float(v) for k, v in objfreq.items()}),
                    "graph_C1":     c1,
                    "routed_prov":  _mix(ph, prov_uns, pi),
                    "routed_graph": _mix(ph, c1, pi),
                }
                for nm, d in dists.items():
                    rs = rank_ge(d, truth)
                    for bucket in ("all", reg):
                        cov[nm][bucket].append(1.0 if rs else 0.0)
                        for k in KS:
                            hits[nm][bucket][k].append(
                                exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        effc[nm].append(eff_cand(d))
                prev_rev = is_rev

            # ---- G3 溯源图结构量 ----
            n_edges = sum(1 for t in range(len(seq))
                          for nd in outs[t] if nd in nodes[t + 1:t + 2])
            g3_rows.append({
                "id": held["id"], "n_steps": len(seq),
                "n_mentioned_total": len({n for o in outs for n in o}),
                "n_visited": len({n for n in nodes if n}),
                "prov_immediate_edges": n_edges,
                "mention_to_visit_rate": (
                    len({n for o in outs for n in o} & {n for n in nodes if n})
                    / max(1, len({n for o in outs for n in o}))),
            })

    res: dict = {
        "n_episodes": len(eps),
        "regime_split": {"n_revisit": n_R, "n_first_visit": n_F},
        "G0_provenance_coverage": {
            "first_visit_target_was_mentioned_before": boot_ci(g0_mentioned_before),
            "by_regime": {r: boot_ci(v) for r, v in g0_by_regime.items()},
            "note": "首访对象中『此前已被某步输出提及』的比例 —— 溯源假设的上限",
        },
        "G1_predictors": {
            nm: {
                "all": {"coverage": boot_ci(cov[nm]["all"])["mean"],
                        "topk": {f"top{k}": round(statistics.mean(hits[nm]["all"][k]), 4)
                                 for k in KS}},
                "R": {"coverage": boot_ci(cov[nm]["R"])["mean"],
                      "topk": {f"top{k}": round(statistics.mean(hits[nm]["R"][k]), 4)
                               for k in KS}},
                "F": {"coverage": boot_ci(cov[nm]["F"])["mean"],
                      "topk": {f"top{k}": boot_ci(hits[nm]["F"][k]) for k in KS}},
                "eff_candidates": (round(statistics.median(effc[nm]), 2)
                                   if effc[nm] else None),
            } for nm in PRED},
        "G1_contrasts_on_first_visit": {
            "prov_freq_vs_marginal_top10": paired_ci(hits["prov_freq"]["F"][10],
                                                     hits["marginal_repo"]["F"][10]),
            "prov_freq_vs_marginal_top3": paired_ci(hits["prov_freq"]["F"][3],
                                                    hits["marginal_repo"]["F"][3]),
            "prov_freq_vs_graphC1_top10": paired_ci(hits["prov_freq"]["F"][10],
                                                    hits["graph_C1"]["F"][10]),
            "prov_freq_vs_graphC1_top3": paired_ci(hits["prov_freq"]["F"][3],
                                                   hits["graph_C1"]["F"][3]),
            "prov_recency_vs_prov_freq_top3": paired_ci(hits["prov_recency"]["F"][3],
                                                        hits["prov_freq"]["F"][3]),
            # 机制辨析：整段观测史 vs 只看最近一条输出
            "prov_recency_vs_lastout_top3": paired_ci(hits["prov_recency"]["F"][3],
                                                      hits["prov_lastout"]["F"][3]),
            "prov_recency_vs_lastout_top10": paired_ci(hits["prov_recency"]["F"][10],
                                                       hits["prov_lastout"]["F"][10]),
            "lastout_ranked_vs_lastout_top3": paired_ci(
                hits["prov_lastout_ranked"]["F"][3], hits["prov_lastout"]["F"][3]),
        },
        "G2_routing_contrasts_all": {
            "routed_prov_vs_routed_graph_top10": paired_ci(
                hits["routed_prov"]["all"][10], hits["routed_graph"]["all"][10]),
            "routed_prov_vs_routed_graph_top3": paired_ci(
                hits["routed_prov"]["all"][3], hits["routed_graph"]["all"][3]),
            "routed_prov_vs_hist_top10": paired_ci(
                hits["routed_prov"]["all"][10], hits["hist_freq"]["all"][10]),
        },
        "G3_trace_graph_structure": {
            "per_episode_mean": {
                k: round(statistics.mean([r[k] for r in g3_rows]), 4)
                for k in ("n_steps", "n_mentioned_total", "n_visited",
                          "mention_to_visit_rate")},
            "note": "n_mentioned_total 远大于 n_visited 说明 agent 看到的远多于它去的；"
                    "mention_to_visit_rate 是『被提及者最终被访问』的比例",
        },
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ------------------------------------------------------------- 打印
    g0 = res["G0_provenance_coverage"]
    print(f"\n=== G0 溯源覆盖率（假设的上限）===")
    b = g0["first_visit_target_was_mentioned_before"]
    print(f"  首访对象『此前已被输出提及』：{b['mean']:.4f} CI{b['ci95']} (n={b['n']})")
    for r in ("R", "F"):
        v = g0["by_regime"][r]
        print(f"    regime {r}: {v['mean']:.4f} CI{v['ci95']} (n={v['n']})")

    print(f"\n=== G1 预测器：全部点（分母={n_R + n_F}）===")
    print(f"{'预测器':<16}{'覆盖':>8}{'top1':>8}{'top3':>8}{'top10':>8}{'有效候选':>10}")
    for nm in PRED:
        d = res["G1_predictors"][nm]["all"]
        print(f"{nm:<16}{d['coverage']:>8.4f}{d['topk']['top1']:>8.4f}"
              f"{d['topk']['top3']:>8.4f}{d['topk']['top10']:>8.4f}"
              f"{str(res['G1_predictors'][nm]['eff_candidates']):>10}")

    print(f"\n=== G1 首访步 F（n={n_F}）—— 关键战场 ===")
    print(f"{'预测器':<16}{'覆盖':>8}{'top3':>18}{'top10':>18}")
    for nm in PRED:
        d = res["G1_predictors"][nm]["F"]
        t3, t10 = d["topk"]["top3"], d["topk"]["top10"]
        if "note" in t3:
            continue
        print(f"{nm:<16}{d['coverage']:>8.4f}"
              f"{t3['mean']:>10.4f} CI{t3['ci95']}"
              f"{t10['mean']:>8.4f} CI{t10['ci95']}")

    print(f"\n=== 首访步上的配对对比 ===")
    for k, v in res["G1_contrasts_on_first_visit"].items():
        if "note" in v:
            print(f"  {k:<34} {v['note']}")
            continue
        print(f"  {k:<34} {v['diff']:+.4f}  CI[{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]  {'显著' if v['significant'] else '不显著'}")

    print(f"\n=== G2 路由：溯源 vs 代码图 ===")
    for k, v in res["G2_routing_contrasts_all"].items():
        if "note" in v:
            continue
        print(f"  {k:<38} {v['diff']:+.4f}  CI[{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]  {'显著' if v['significant'] else '不显著'}")

    print(f"\n=== G3 溯源图规模 ===")
    for k, v in res["G3_trace_graph_structure"]["per_episode_mean"].items():
        print(f"  {k:<24} {v}")

    print(f"\n→ {args.out}")
    return 0


def _norm(w: dict[str, float]) -> dict[str, float]:
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def _mix(a: dict[str, float], b: dict[str, float], pi: float) -> dict[str, float]:
    w: dict[str, float] = {}
    for k, v in a.items():
        w[k] = w.get(k, 0.0) + pi * v
    for k, v in b.items():
        w[k] = w.get(k, 0.0) + (1 - pi) * v
    return _norm(w)


if __name__ == "__main__":
    raise SystemExit(main())
