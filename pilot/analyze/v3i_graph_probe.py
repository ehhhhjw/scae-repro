"""V3i：代码图的多角度论证。零 API。

为什么需要多角度
----------------
V3g 已证代码图在 next-object 上无用（首访 top-10 0.077，显著不如仓库词频 0.163）。
但单一失败不足以断言"代码图无价值"，因为至少有五种替代解释：

  (a) 测量问题：agent 的对象根本没解析到图节点 → 图从未有机会
  (b) 用法问题：只用了「当前节点的一跳邻居」，没用「整个已访问集合」做多源邻近
  (c) 粒度问题：节点级不行，社区级可能行
  (d) 目标问题：图不预测「下一步」，但可能预测「哪些文件是 gold」
  (e) 描述 vs 预测：图可能刻画了轨迹（相邻步比随机近），只是不足以排序

本脚本逐一检验，且每项都配零假设。若这些角度全部失败，"代码图对刻画
agent 注意力移动无效"才是可断言的结论；若某项成功，则该项就是图的正确用途。

角度与判据
----------
A 测量有效性     A1 对象→图节点解析率   A2 gold 是否在图内、离已访问多远
B 描述性关联     B1 相邻步图距离 vs 置换零假设（**分离距离 0 的自我重复**）
                 B2 相邻步同社区率 vs 置换零假设
C 加强版预测器   C1 多源邻近（整个已访问集合为种子，全仓库排序）
                 C2 k 跳扫描（1/2/3 跳的覆盖–精度权衡）
                 C3 纯度数中心性基线（不用轨迹信息）—— 若与 C1 相当，
                    则「图与轨迹的关系」无增量，图只是在推荐热门文件
D 换目标         D1 图邻近能否预测 gold 文件（可能的正面用途）
                 D2 已访问集合是否比随机同规模集合更聚集（描述性主张）

关键陷阱
--------
W_L1=0.62 的来源是「548 个相邻步对中 62% 在一跳内」。若该统计把
**同一文件连续两步（距离 0）**也计入"一跳内"，则该权重是自我重复的假象
（V3h 已证对象紧邻重复 1.83 倍于随机）。B1 显式分离 d=0 与 d=1。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import load_graph          # noqa: E402
from pilot.analyze.v3_nextstep_dual import load_episodes    # noqa: E402
from pilot.probes import task as T                          # noqa: E402

B_PERM = 1000
SEED = 20260814
KS = (1, 3, 5, 10, 20, 50)


# ---------------------------------------------------------------- 工具

def multi_source_dist(g, seeds: set[str], max_hops: int = 4) -> dict[str, int]:
    """从 seeds 出发的多源 BFS，返回每个可达节点的最短跳数。"""
    dist: dict[str, int] = {s: 0 for s in seeds if s in g.edges or s in g.files}
    dq = deque(dist)
    while dq:
        x = dq.popleft()
        d = dist[x]
        if d >= max_hops:
            continue
        for y in g.edges.get(x, ()):
            if y not in dist:
                dist[y] = d + 1
                dq.append(y)
    return dist


def seed_adjacency(g, seeds: set[str]) -> Counter:
    """每个节点与多少个种子直接相邻（多源邻近的 tie-break 信号）。"""
    c: Counter = Counter()
    for s in seeds:
        for y in g.edges.get(s, ()):
            c[y] += 1
    return c


def exp_hit(g_: int, e_: int, k: int) -> float:
    """闭式期望 P(rank ≤ k)，随机破并列。"""
    return min(1.0, max(0.0, (k - g_) / e_))


def rank_stats(scores: dict[str, float], truth: str) -> tuple[int, int] | None:
    if truth not in scores:
        return None
    p = scores[truth]
    return (sum(1 for v in scores.values() if v > p),
            sum(1 for v in scores.values() if v == p))


def _best_rank(cand: dict[str, float], targets: set[str], hits: dict,
               rankfrac: list[float], n_pool: int) -> None:
    """把 targets 里排名最好的那个折算成 top-k 命中与排名分位，累加进 hits。"""
    best = None
    for gp in targets:
        rs = rank_stats(cand, gp)
        if rs:
            r = rs[0] + (rs[1] + 1) / 2          # 期望排名（随机破并列）
            best = r if best is None else min(best, r)
    if best is None:
        return
    for k in KS:
        hits[k].append(1.0 if best <= k else 0.0)
    rankfrac.append(best / max(1, n_pool))


def boot_ci(vals: list[float], b: int = 2000, seed: int = SEED) -> dict:
    if not vals:
        return {"note": "无样本"}
    rng = random.Random(seed)
    pts = []
    for _ in range(b):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        pts.append(statistics.mean(s))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * b)], 4), round(pts[int(0.975 * b)], 4)],
            "n": len(vals)}


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3i_graph_probe.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    eps = load_episodes(Path(args.runs_root))
    print(f"载入 {len(eps)} episode（已构图）")

    # ---- A1 解析率 ----
    a1 = {"n_steps": 0, "n_with_obj": 0, "resolved": 0, "dir_node": 0,
          "file_node": 0, "unresolved": 0}
    # ---- B ----
    b1_obs_d0, b1_obs_d1, b1_obs_le3 = [], [], []
    b1_null_d0, b1_null_d1, b1_null_le3 = [], [], []
    b1_obs_d1_excl0, b1_null_d1_excl0 = [], []
    b2_obs, b2_null = [], []
    # ---- C/D 累加 ----
    C1 = {k: [] for k in KS}
    C3 = {k: [] for k in KS}
    C1_cov, C3_cov = [], []
    C2 = {h: {"cov": [], "size": []} for h in (1, 2, 3)}
    D1 = {k: [] for k in KS}
    D1_uns = {k: [] for k in KS}
    D1_pre = {k: [] for k in KS}
    D1_rankfrac, D1_gold_in_graph, D1_gold_dist = [], [], []
    D1_uns_rankfrac, D1_pre_rankfrac, D1_uns_n = [], [], []
    # E 同点集配对：四个预测器在**完全相同**的首访点上对打
    E_names = ("graph_1hop_current", "C1_multisource", "C3_degree", "marginal_repo")
    E = {nm: {k: [] for k in KS} for nm in E_names}
    E_cov = {nm: [] for nm in E_names}
    D2_obs, D2_null = [], []
    n_first_visit = 0

    rng = random.Random(SEED)

    for e in eps:
        g, seq, iid = e["graph"], e["seq"], e["id"]
        inst = insts.get(iid)
        nodes = [s["node"] for s in seq]
        # A1
        for s in seq:
            a1["n_steps"] += 1
            if s["obj"]:
                a1["n_with_obj"] += 1
            if s["node"]:
                a1["resolved"] += 1
                if s["node"].startswith("dir:"):
                    a1["dir_node"] += 1
                else:
                    a1["file_node"] += 1
            elif s["obj"]:
                a1["unresolved"] += 1

        # 只保留可用于图分析的真实文件节点序列
        fseq = [n for n in nodes if n and not n.startswith("dir:") and n in g.files]
        if len(fseq) >= 3:
            # ---- B1 相邻步图距离（分离 d=0）----
            def dist_profile(s: list[str]) -> tuple[float, float, float, float]:
                d0 = d1 = le3 = tot = 0
                d1x = tot_x = 0                     # 排除 d=0 后的一跳率
                for a, b in zip(s, s[1:]):
                    dd = g.graph_distance(a, b, max_hops=3)
                    tot += 1
                    if dd == 0:
                        d0 += 1
                    else:
                        tot_x += 1
                        if dd == 1:
                            d1x += 1
                    if dd == 1:
                        d1 += 1
                    if dd is not None and dd <= 3:
                        le3 += 1
                return (d0 / tot, d1 / tot, le3 / tot,
                        (d1x / tot_x if tot_x else float("nan")))

            o = dist_profile(fseq)
            b1_obs_d0.append(o[0]); b1_obs_d1.append(o[1]); b1_obs_le3.append(o[2])
            if not math.isnan(o[3]):
                b1_obs_d1_excl0.append(o[3])
            work = list(fseq)
            n0 = n1 = n3 = 0.0
            n1x, cnt_x = 0.0, 0
            NP = 60          # 每 episode 置换次数（图距离计算较贵）
            for _ in range(NP):
                rng.shuffle(work)
                v = dist_profile(work)
                n0 += v[0]; n1 += v[1]; n3 += v[2]
                if not math.isnan(v[3]):
                    n1x += v[3]; cnt_x += 1
            b1_null_d0.append(n0 / NP); b1_null_d1.append(n1 / NP)
            b1_null_le3.append(n3 / NP)
            if cnt_x:
                b1_null_d1_excl0.append(n1x / cnt_x)

            # ---- B2 同社区率（排除同一节点）----
            def comm_rate(s: list[str]) -> float | None:
                hit = tot = 0
                for a, b in zip(s, s[1:]):
                    if a == b:
                        continue
                    ca, cb = g.community.get(a), g.community.get(b)
                    if ca is None or cb is None:
                        continue
                    tot += 1
                    if ca == cb:
                        hit += 1
                return hit / tot if tot else None

            co = comm_rate(fseq)
            if co is not None:
                b2_obs.append(co)
                acc, cnt = 0.0, 0
                for _ in range(200):
                    rng.shuffle(work)
                    v = comm_rate(work)
                    if v is not None:
                        acc += v; cnt += 1
                if cnt:
                    b2_null.append(acc / cnt)

        # ---- C1/C3 加强版预测器（只在首访步评估）----
        seen: set[str] = set()
        for t in range(len(seq) - 1):
            if nodes[t]:
                seen.add(nodes[t])
            truth = nodes[t + 1]
            if not truth or truth in seen:
                continue                      # 只看首访步
            n_first_visit += 1
            fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
            if not fseeds:
                continue
            dist = multi_source_dist(g, fseeds)
            adj = seed_adjacency(g, fseeds)
            # C1 多源邻近：先按最短跳数，再按"与多少种子相邻"
            cand = {x: (-dist[x] * 1000 + adj.get(x, 0))
                    for x in dist if x not in seen}
            C1_cov.append(1.0 if truth in cand else 0.0)
            rs = rank_stats(cand, truth)
            for k in KS:
                C1[k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
            # C3 纯度数中心性（不用轨迹）
            deg = {x: float(len(g.edges.get(x, ()))) for x in g.files if x not in seen}
            C3_cov.append(1.0 if truth in deg else 0.0)
            rs3 = rank_stats(deg, truth)
            for k in KS:
                C3[k].append(exp_hit(rs3[0], rs3[1], k) if rs3 else 0.0)
            # C2 k 跳覆盖–规模
            for h in (1, 2, 3):
                reach = {x for x, d in dist.items() if 0 < d <= h and x not in seen}
                C2[h]["cov"].append(1.0 if truth in reach else 0.0)
                C2[h]["size"].append(len(reach))

        # ---- D1 图邻近能否预测 gold ----
        # ⚠️ 泄漏控制：agent 成功率 94%，gold 多半**已被访问**（距离 0）。
        # 若把已访问文件留在候选里，"命中 gold"只是在复述 agent 自己的历史。
        # 故三档并列上报：
        #   full   ：全仓库排序（含已访问）—— 有泄漏，仅作对照
        #   unseen ：排除已访问 —— 真正问"图能否指出还没看的 gold"
        #   prefix ：只用前 30% 轨迹作种子，且排除该前缀已访问 —— 最严格
        if inst and inst.gold:
            gold_in = {p for p in inst.gold if p in g.files}
            D1_gold_in_graph.append(len(gold_in) / len(inst.gold))
            fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
            if gold_in and fseeds:
                dist = multi_source_dist(g, fseeds)
                adj = seed_adjacency(g, fseeds)
                cand_full = {x: (-dist.get(x, 99) * 1000 + adj.get(x, 0))
                             for x in g.files}
                _best_rank(cand_full, gold_in, D1, D1_rankfrac, len(g.files))
                # unseen：候选与 gold 均排除已访问
                gold_unseen = gold_in - fseeds
                cand_uns = {x: v for x, v in cand_full.items() if x not in fseeds}
                if gold_unseen:
                    _best_rank(cand_uns, gold_unseen, D1_uns, D1_uns_rankfrac,
                               len(cand_uns))
                    D1_uns_n.append(1.0)
                else:
                    D1_uns_n.append(0.0)
                for gp in gold_in:
                    d = dist.get(gp)
                    D1_gold_dist.append(d if d is not None else 99)
                # prefix：只用前 30% 步的已访问集合
                cut = max(1, int(0.3 * len(seq)))
                pre = {n for n in nodes[:cut]
                       if n and not n.startswith("dir:") and n in g.files}
                gold_pre = gold_in - pre
                if pre and gold_pre:
                    dp = multi_source_dist(g, pre)
                    ap = seed_adjacency(g, pre)
                    cp = {x: (-dp.get(x, 99) * 1000 + ap.get(x, 0))
                          for x in g.files if x not in pre}
                    _best_rank(cp, gold_pre, D1_pre, D1_pre_rankfrac, len(cp))

        # ---- D2 已访问集合是否比随机集合更聚集 ----
        fs = {x for x in seen if not x.startswith("dir:") and x in g.files}
        if 3 <= len(fs) <= 60 and len(g.files) > len(fs) * 3:
            def clustering(S: set[str]) -> float:
                """集合内部边密度：内部边数 / 可能对数。"""
                S = set(S)
                inner = sum(1 for x in S for y in g.edges.get(x, ()) if y in S) / 2
                m = len(S) * (len(S) - 1) / 2
                return inner / m if m else 0.0
            D2_obs.append(clustering(fs))
            pool = list(g.files)
            acc = 0.0
            for _ in range(200):
                acc += clustering(set(rng.sample(pool, len(fs))))
            D2_null.append(acc / 200)

    # ------------------------------------------------ E 同点集配对对打
    # V3g 的 marginal 是 within-repo LOO 拟合的，故这里也用同样的 LOO 与
    # `len(group)>=6` 过滤，保证 E 的点集与 V3g 的 F regime **逐点一致**，
    # 否则 C1(n=381) 与 marginal(n=341) 不可比。
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            objfreq = Counter(s["node"] for x in train for s in x["seq"] if s["node"])
            g, seq = held["graph"], held["seq"]
            nodes = [s["node"] for s in seq]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                if nodes[t]:
                    seen.add(nodes[t])
                truth = nodes[t + 1]
                if not truth or truth in seen:
                    continue
                cur = nodes[t]
                fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
                # 1) V3g 用法：当前节点一跳+同社区，排除已访问
                s1: dict[str, float] = {}
                if cur and not cur.startswith("dir:"):
                    L1 = g.neighbors(cur) - seen
                    L2 = (g.same_community(cur) - g.neighbors(cur)) - seen
                    for x in L1:
                        s1[x] = 2.0
                    for x in L2:
                        s1[x] = 1.0
                # 2) 多源邻近
                s2: dict[str, float] = {}
                if fseeds:
                    dd = multi_source_dist(g, fseeds)
                    aa = seed_adjacency(g, fseeds)
                    s2 = {x: -dd[x] * 1000 + aa.get(x, 0)
                          for x in dd if x not in seen}
                # 3) 度数中心性
                s3 = {x: float(len(g.edges.get(x, ()))) for x in g.files
                      if x not in seen}
                # 4) 仓库词频（LOO）
                s4 = {x: float(c) for x, c in objfreq.items() if x not in seen}
                for nm, sc in (("graph_1hop_current", s1), ("C1_multisource", s2),
                               ("C3_degree", s3), ("marginal_repo", s4)):
                    rs = rank_stats(sc, truth)
                    E_cov[nm].append(1.0 if rs else 0.0)
                    for k in KS:
                        E[nm][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)

    def paired(a: str, b: str, k: int, bnum: int = 2000) -> dict:
        va, vb = E[a][k], E[b][k]
        if len(va) != len(vb) or not va:
            return {"note": "点集不匹配"}
        r = random.Random(SEED)
        pts = []
        for _ in range(bnum):
            idx = [r.randrange(len(va)) for _ in range(len(va))]
            pts.append(sum(va[i] - vb[i] for i in idx) / len(idx))
        pts.sort()
        lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
        return {"diff": round(statistics.mean(va) - statistics.mean(vb), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0 or hi < 0)}

    # ---------------------------------------------------------- 汇总
    res: dict = {
        "n_episodes": len(eps),
        "A1_resolution": {
            **a1,
            "resolve_rate_of_steps_with_obj": round(
                a1["resolved"] / max(1, a1["n_with_obj"]), 4),
            "file_node_share_of_resolved": round(
                a1["file_node"] / max(1, a1["resolved"]), 4),
            "dir_node_share_of_resolved": round(
                a1["dir_node"] / max(1, a1["resolved"]), 4),
        },
        "B1_adjacent_step_distance": {
            "d0_selfrepeat": {"obs": boot_ci(b1_obs_d0), "null": boot_ci(b1_null_d0)},
            "d1_onehop_incl_d0_denominator": {
                "obs": boot_ci(b1_obs_d1), "null": boot_ci(b1_null_d1)},
            "d1_onehop_EXCLUDING_selfrepeat": {
                "obs": boot_ci(b1_obs_d1_excl0), "null": boot_ci(b1_null_d1_excl0)},
            "within_3hops": {"obs": boot_ci(b1_obs_le3), "null": boot_ci(b1_null_le3)},
        },
        "B2_same_community": {"obs": boot_ci(b2_obs), "null": boot_ci(b2_null)},
        "C1_multisource_proximity": {
            "coverage": boot_ci(C1_cov),
            "topk": {f"top{k}": boot_ci(C1[k]) for k in KS},
            "n_first_visit_points": len(C1_cov),
        },
        "C3_degree_centrality_baseline": {
            "coverage": boot_ci(C3_cov),
            "topk": {f"top{k}": boot_ci(C3[k]) for k in KS},
        },
        "C2_khop_tradeoff": {
            f"{h}hop": {"coverage": boot_ci(C2[h]["cov"]),
                        "median_candidate_set": (statistics.median(C2[h]["size"])
                                                 if C2[h]["size"] else None)}
            for h in (1, 2, 3)},
        "D1_gold_prediction": {
            "gold_in_graph_share": boot_ci(D1_gold_in_graph),
            "leakage_warning": "full 档含已访问文件，gold 多半已被访问 ⇒ 有泄漏，"
                               "只作对照；结论看 unseen 与 prefix 两档",
            "full_LEAKY": {
                "topk_any_gold": {f"top{k}": boot_ci(D1[k]) for k in KS},
                "best_rank_frac": boot_ci(D1_rankfrac)},
            "unseen": {
                "topk_any_gold": {f"top{k}": boot_ci(D1_uns[k]) for k in KS},
                "best_rank_frac": boot_ci(D1_uns_rankfrac),
                "share_episodes_with_unvisited_gold": boot_ci(D1_uns_n)},
            "prefix30pct": {
                "topk_any_gold": {f"top{k}": boot_ci(D1_pre[k]) for k in KS},
                "best_rank_frac": boot_ci(D1_pre_rankfrac)},
            "gold_hop_distance_from_visited": (
                dict(sorted(Counter(D1_gold_dist).items())) if D1_gold_dist else None),
        },
        "E_paired_headtohead_first_visit": {
            "n_points": len(E_cov["C1_multisource"]),
            "note": "与 V3g 的 F regime 逐点一致（within-repo LOO，group>=6）",
            "coverage": {nm: boot_ci(E_cov[nm])["mean"] for nm in E_names},
            "topk": {nm: {f"top{k}": round(statistics.mean(E[nm][k]), 4)
                          for k in KS} for nm in E_names},
            "contrasts": {
                "C1_vs_graph1hop_top10": paired("C1_multisource",
                                                "graph_1hop_current", 10),
                "C1_vs_marginal_top10": paired("C1_multisource",
                                               "marginal_repo", 10),
                "C1_vs_marginal_top3": paired("C1_multisource",
                                              "marginal_repo", 3),
                "C1_vs_C3degree_top10": paired("C1_multisource", "C3_degree", 10),
                "C1_vs_marginal_top50": paired("C1_multisource",
                                               "marginal_repo", 50),
            },
        },
        "D2_visited_set_clustering": {
            "obs": boot_ci(D2_obs), "null_random_sets": boot_ci(D2_null),
            "lift": (round(statistics.mean(D2_obs) / statistics.mean(D2_null), 3)
                     if D2_obs and D2_null and statistics.mean(D2_null) > 0 else None),
        },
        "n_first_visit_total": n_first_visit,
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---------------------------------------------------------- 打印
    a = res["A1_resolution"]
    print(f"\n=== A1 测量有效性：对象→图节点解析 ===")
    print(f"  有对象的步 {a['n_with_obj']}/{a['n_steps']}，"
          f"解析成功率 {a['resolve_rate_of_steps_with_obj']:.1%}")
    print(f"  解析结果中：真实文件节点 {a['file_node_share_of_resolved']:.1%}，"
          f"目录节点 {a['dir_node_share_of_resolved']:.1%}")

    print(f"\n=== B1 相邻步图距离 vs 置换零假设 ===")
    for key, label in (("d0_selfrepeat", "距离 0（同一文件）"),
                       ("d1_onehop_incl_d0_denominator", "距离 1（全体为分母）"),
                       ("d1_onehop_EXCLUDING_selfrepeat", "距离 1（**排除距离 0**）"),
                       ("within_3hops", "3 跳内")):
        d = res["B1_adjacent_step_distance"][key]
        if "note" in d["obs"]:
            continue
        o, n = d["obs"], d["null"]
        lift = o["mean"] / n["mean"] if n["mean"] else float("nan")
        print(f"  {label:<24} 观测 {o['mean']:.4f} CI{o['ci95']}   "
              f"零假设 {n['mean']:.4f} CI{n['ci95']}   lift {lift:.2f}")

    d = res["B2_same_community"]
    if "note" not in d["obs"]:
        lift = d["obs"]["mean"] / d["null"]["mean"] if d["null"]["mean"] else float("nan")
        print(f"\n=== B2 相邻步同社区率（已排除同一节点）===")
        print(f"  观测 {d['obs']['mean']:.4f} CI{d['obs']['ci95']}   "
              f"零假设 {d['null']['mean']:.4f} CI{d['null']['ci95']}   lift {lift:.2f}")

    c1, c3 = res["C1_multisource_proximity"], res["C3_degree_centrality_baseline"]
    print(f"\n=== C1/C3 加强版预测器（仅首访步，n={c1['n_first_visit_points']}）===")
    print(f"{'':<26}" + "".join(f"{'top'+str(k):>9}" for k in KS))
    print(f"{'C1 多源邻近(用轨迹)':<26}" +
          "".join(f"{c1['topk']['top'+str(k)]['mean']:>9.4f}" for k in KS))
    print(f"{'C3 度数中心性(不用轨迹)':<24}" +
          "".join(f"{c3['topk']['top'+str(k)]['mean']:>9.4f}" for k in KS))
    print(f"  覆盖率：C1 {c1['coverage']['mean']:.4f}  C3 {c3['coverage']['mean']:.4f}")

    print(f"\n=== C2 k 跳覆盖–候选集规模权衡 ===")
    for h in (1, 2, 3):
        d = res["C2_khop_tradeoff"][f"{h}hop"]
        print(f"  {h} 跳：覆盖 {d['coverage']['mean']:.4f} "
              f"CI{d['coverage']['ci95']}   候选集中位规模 {d['median_candidate_set']}")

    ee = res["E_paired_headtohead_first_visit"]
    print(f"\n=== E 同点集配对对打（首访步，n={ee['n_points']}，与 V3g 的 F 一致）===")
    print(f"{'预测器':<22}{'覆盖':>8}" + "".join(f"{'top'+str(k):>9}" for k in KS))
    for nm in E_names:
        print(f"{nm:<22}{ee['coverage'][nm]:>8.4f}" +
              "".join(f"{ee['topk'][nm]['top'+str(k)]:>9.4f}" for k in KS))
    print("  配对对比（差值 CI 不含 0 才显著）：")
    for kk, vv in ee["contrasts"].items():
        if "note" in vv:
            print(f"    {kk:<28} {vv['note']}")
            continue
        print(f"    {kk:<28} {vv['diff']:+.4f}  "
              f"CI[{vv['ci95'][0]:+.4f}, {vv['ci95'][1]:+.4f}]  "
              f"{'显著' if vv['significant'] else '不显著'}")

    d1 = res["D1_gold_prediction"]
    print(f"\n=== D1 换目标：图邻近能否预测 gold 文件 ===")
    print(f"  gold 在图内的比例 {d1['gold_in_graph_share']['mean']:.4f} "
          f"CI{d1['gold_in_graph_share']['ci95']}")
    print(f"  ⚠️ {d1['leakage_warning']}")
    print(f"  gold 距已访问集合的跳数分布 {d1['gold_hop_distance_from_visited']}"
          f"  ← 距离 0 即「已被访问」")
    for tag, label in (("full_LEAKY", "full（含已访问，泄漏）"),
                       ("unseen", "unseen（排除已访问）"),
                       ("prefix30pct", "prefix30%（仅前 30% 轨迹）")):
        blk = d1[tag]
        tk = blk["topk_any_gold"]
        if "note" in tk[f"top{KS[0]}"]:
            print(f"  {label:<26} 无样本")
            continue
        print(f"  {label:<26}" +
              "  ".join(f"top{k} {tk['top'+str(k)]['mean']:.3f}" for k in KS))
        rf = blk["best_rank_frac"]
        if "note" not in rf:
            print(f"  {'':<26}最佳 gold 排名分位 {rf['mean']:.4f} "
                  f"CI{rf['ci95']}（随机=0.5，n={rf['n']}）")

    d2 = res["D2_visited_set_clustering"]
    print(f"\n=== D2 已访问集合的图聚集性 ===")
    if "note" not in d2["obs"]:
        print(f"  观测内部边密度 {d2['obs']['mean']:.5f} CI{d2['obs']['ci95']}")
        print(f"  随机同规模集合 {d2['null_random_sets']['mean']:.5f} "
              f"CI{d2['null_random_sets']['ci95']}   lift {d2['lift']}")

    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
