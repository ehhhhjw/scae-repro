"""V3g：共同分母评估 + 双 regime 分解 + L3 加权消融 + 路由混合。零 API。

为什么必须重做 V3f
------------------
V3f 各臂的 top-k 是**条件于"真值落在本臂候选集内"**算的（full n=236 / only_L3
n=168 / no_L3 n=90），而 full 的候选集 ⊇ only_L3 的候选集，故 only_L3 的 168 点
是 full 的 236 点的子集，且恰是"重访"这类简单点。用各自条件后的 top-k 互比
不是同基准比较。换成共同分母（全部 509 个决策点，未覆盖记为 miss）后：

    臂        条件top3  无条件top3   条件top10  无条件top10
    full       0.491     0.228       0.733      0.340
    only_L3    0.744     0.246       0.994      0.328
    no_L1      0.670     0.259       0.858      0.332

→ V3f 的"only_L3 全面优于 full"与"图结构越少越好的单调关系"**均不成立**。
   唯一存活的结论是 no_L3 崩塌（历史先验必要）。

本实验回答的真问题
------------------
only_L3 结构上**只能预测重访**（候选集就是已访问集合），首次访问步它覆盖率恒 0。
所以两类步应分开建模，而不是加性混在一个打分里：

  regime R (revisit)     next ∈ seen      —— 历史先验适用
  regime F (first-visit) next ∉ seen      —— 只有代码图能触达

据此四问：
  Q1 两 regime 各占多少？（决定问题的形状）
  Q2 历史先验在 R 上多好？代码图在 F 上多好？（各自的适用域）
  Q3 regime 本身可预测吗？（路由是否可实现）
  Q4 路由混合是否优于两个单一模型？（正确的模型形式）

顺带修两个 bug 与补一个消融
---------------------------
  bug1  `list(seen)[-30:]`：seen 是 set，顺序无保证 → "最近 30 个"实为"任意 30 个"。
        改为按访问顺序的 list。
  bug2  ECE 只在覆盖点上算，未覆盖点（模型给真值 0 概率）被排除 → 高估校准。
        并列上报 ECE_cond（可比 V3b）与 ECE_uncond（含 miss）。
  消融  既然 L3 是唯一有效成分，L3 内部加权才是收益所在：
        uniform（现状） / recency（指数衰减） / freq（访问次数）。
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

from pilot.channels.code_channel import W_L1, W_L2, W_L3, load_graph   # noqa: E402
from pilot.analyze.v3_nextstep_dual import (                            # noqa: E402
    load_episodes, fit_tool_trans, p_tool,
)

KS = (1, 2, 3, 5, 10)
MISS = math.inf
TOPK_SEEN = 30
GAMMA = 0.8          # recency 衰减因子，见 --scan-gamma
NEG_CAP = 20         # ECE_cond 的负例上限（与 V3b 一致）


# ---------------------------------------------------------------- 候选构造

def p_hist(visits: list[str], cur: str | None, mode: str,
           topk: int = TOPK_SEEN, gamma: float = GAMMA) -> dict[str, float]:
    """历史先验（L3）。visits 按访问顺序，可含重复。

    mode: uniform（V3f 现状，均匀）/ recency（指数衰减）/ freq（访问次数）
    """
    if not visits:
        return {}
    w: dict[str, float] = {}
    if mode == "uniform":
        # 修 bug1：按访问顺序取最近 topk 个**去重后**的对象
        uniq: list[str] = []
        for x in reversed(visits):
            if x not in uniq:
                uniq.append(x)
            if len(uniq) >= topk:
                break
        for x in uniq:
            w[x] = 1.0 / len(uniq)
    elif mode == "recency":
        n = len(visits)
        for i, x in enumerate(visits):
            w[x] = w.get(x, 0.0) + gamma ** (n - 1 - i)
    elif mode == "freq":
        for x, c in Counter(visits).items():
            w[x] = float(c)
    else:
        raise ValueError(mode)
    if cur:                                   # 自重复先验
        w[cur] = w.get(cur, 0.0) + 0.5 * max(w.values())
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def p_graph(g, cur_node: str | None, exclude: set[str] | None = None) -> dict[str, float]:
    """代码图先验（L1+L2），exclude 用于只保留未访问对象（regime F 用）。"""
    if not cur_node or cur_node.startswith("dir:"):
        return {}
    L1 = g.neighbors(cur_node)
    L2 = g.same_community(cur_node) - L1
    if exclude:
        L1, L2 = L1 - exclude, L2 - exclude
    w: dict[str, float] = {}
    for x in L1:
        w[x] = w.get(x, 0.0) + W_L1 / max(1, len(L1))
    for x in L2:
        w[x] = w.get(x, 0.0) + W_L2 / max(1, len(L2))
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def p_additive_full(g, visits, cur, seen) -> dict[str, float]:
    """V3f 的 full 臂：三层加性混合（作为对照）。"""
    w: dict[str, float] = {}
    if cur and not cur.startswith("dir:"):
        L1 = g.neighbors(cur)
        L2 = g.same_community(cur) - L1
        for x in L1:
            w[x] = w.get(x, 0.0) + W_L1 / max(1, len(L1))
        for x in L2:
            w[x] = w.get(x, 0.0) + W_L2 / max(1, len(L2))
    for x, p in p_hist(visits, cur, "uniform").items():
        w[x] = w.get(x, 0.0) + W_L3 * p
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def p_routed(g, visits, cur, seen, pi: float, hist_mode: str) -> dict[str, float]:
    """路由混合：pi·P_hist(已访问) + (1−pi)·P_graph(未访问)。

    与 additive_full 的区别是**两个分量的支撑集不重叠**且权重由 regime 概率给出，
    而非固定常数 W。这样 regime 预测对了就自动把质量放到正确的半边。
    """
    ph = p_hist(visits, cur, hist_mode)
    pg = p_graph(g, cur, exclude=seen)
    w: dict[str, float] = {}
    for x, p in ph.items():
        w[x] = w.get(x, 0.0) + pi * p
    for x, p in pg.items():
        w[x] = w.get(x, 0.0) + (1.0 - pi) * p
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


# ---------------------------------------------------------------- 指标

def rank_ge(dist: dict[str, float], truth: str) -> tuple[int, int] | None:
    """返回 (g, e)：g=严格大于真值概率的候选数，e=与真值并列的候选数(含真值)。

    未覆盖返回 None。**不做任意破并列** —— 均匀分布下所有候选并列，
    用字典序破并列会凭字母顺序造出假信号（踩坑 #15 的同类错误）。
    调用方用闭式期望把 (g, e) 折算成 top-k 命中概率与 E[1/rank]。
    """
    if truth not in dist:
        return None
    p = dist[truth]
    g = sum(1 for v in dist.values() if v > p)
    e = sum(1 for v in dist.values() if v == p)
    return g, e


def exp_hit(g: int, e: int, k: int) -> float:
    """随机破并列下 P(rank ≤ k)：rank = g + U，U 均匀于 {1..e}。"""
    return min(1.0, max(0.0, (k - g) / e))


def exp_recip(g: int, e: int) -> float:
    """E[1/rank] 的闭式：(1/e)·Σ_{j=g+1}^{g+e} 1/j。"""
    return sum(1.0 / j for j in range(g + 1, g + e + 1)) / e


def eff_cand(dist: dict[str, float]) -> float:
    if not dist:
        return 0.0
    return math.exp(-sum(v * math.log(v) for v in dist.values() if v > 0))


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


class Acc:
    """一个预测器的累加器。分母是**全部决策点**，miss 计入分母。

    同时保留 per-point 记录（带 episode id），供 episode 级 bootstrap。
    """

    def __init__(self) -> None:
        self.n = 0
        self.hits = {k: 0.0 for k in KS}          # 期望命中数（闭式，非整数）
        self.rr = 0.0
        self.cov = 0
        self.eff: list[float] = []
        self.nll: list[float] = []
        self.cal_cond: list[tuple[float, int]] = []
        self.cal_unc: list[tuple[float, int]] = []
        # (episode_id, {k: 期望命中概率}) —— miss 点各 k 记 0.0
        self.recs: list[tuple[str, dict[int, float]]] = []

    def add(self, dist: dict[str, float], truth: str, ep: str = "") -> None:
        self.n += 1
        ge = rank_ge(dist, truth)
        if ge is None:
            self.recs.append((ep, {k: 0.0 for k in KS}))
            self.cal_unc.append((0.0, 1))
            return
        g, e = ge
        self.cov += 1
        self.rr += exp_recip(g, e)
        h = {k: exp_hit(g, e, k) for k in KS}
        self.recs.append((ep, h))
        for k in KS:
            self.hits[k] += h[k]
        self.eff.append(eff_cand(dist))
        p_true = dist[truth]
        self.nll.append(-math.log(max(p_true, 1e-12)))
        self.cal_cond.append((p_true, 1))
        self.cal_unc.append((p_true, 1))
        negs = sorted((v for k, v in dist.items() if k != truth), reverse=True)[:NEG_CAP]
        for v in negs:
            self.cal_cond.append((v, 0))
            self.cal_unc.append((v, 0))

    def report(self) -> dict:
        if not self.n:
            return {"note": "无有效点"}
        return {
            "n_points": self.n,
            "coverage": round(self.cov / self.n, 4),
            "topk_uncond": {f"top{k}": round(self.hits[k] / self.n, 4) for k in KS},
            "topk_cond": ({f"top{k}": round(self.hits[k] / self.cov, 4) for k in KS}
                          if self.cov else None),
            "MRR_uncond": round(self.rr / self.n, 4),
            "ECE_cond": round(ece(self.cal_cond), 4),
            "ECE_uncond": round(ece(self.cal_unc), 4),
            "eff_candidates": (round(statistics.median(self.eff), 2)
                               if self.eff else None),
            "NLL_cond": (round(statistics.mean(self.nll), 4) if self.nll else None),
            "perplexity_cond": (round(math.exp(statistics.mean(self.nll)), 2)
                                if self.nll else None),
        }


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3g_regime.json")
    ap.add_argument("--scan-gamma", action="store_true",
                    help="扫 recency 衰减因子 gamma")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"载入 {len(eps)} episode，工具 {len(tool_vocab)} 类")

    PREDICTORS = [
        "uniform_seen", "hist_uniform", "hist_recency", "hist_freq",
        "graph_L1L2", "graph_unseen",
        "additive_full", "routed_oracle", "routed_pred",
        "repeat_prev", "marginal",
    ]
    # 全局 + 分 regime 三套累加器
    acc = {p: {"all": Acc(), "R": Acc(), "F": Acc()} for p in PREDICTORS}

    n_R = n_F = 0
    max_seen = 0
    regime_pred: list[tuple[float, int]] = []       # (π̂, 是否 revisit) 供 AUROC
    joint_acc, joint_n = {k: 0.0 for k in KS}, 0

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [e for j, e in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            objfreq = Counter(s["node"] for e in train for s in e["seq"] if s["node"])

            # --- 从训练集估 regime 转移（Q3：regime 可预测吗）---
            rc: dict[bool, Counter] = {True: Counter(), False: Counter()}
            glob = Counter()
            for e in train:
                sq, sn = e["seq"], set()
                prev_rev = False
                for t in range(len(sq) - 1):
                    if sq[t]["node"]:
                        sn.add(sq[t]["node"])
                    if not sq[t + 1]["node"]:
                        continue
                    is_rev = sq[t + 1]["node"] in sn
                    rc[prev_rev][is_rev] += 1
                    glob[is_rev] += 1
                    prev_rev = is_rev

            def pi_hat(prev_rev: bool) -> float:
                row = rc[prev_rev]
                tot = sum(row.values())
                if tot < 5:                      # 退回全局率
                    tot, row = sum(glob.values()), glob
                return (row[True] + 0.5) / (tot + 1.0) if tot else 0.5

            g, seq = held["graph"], held["seq"]
            visits: list[str] = []
            seen: set[str] = set()
            prev_rev = False
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    visits.append(cur["node"])
                    seen.add(cur["node"])
                max_seen = max(max_seen, len(seen))
                if not nxt["node"]:
                    continue
                truth = nxt["node"]
                is_rev = truth in seen
                reg = "R" if is_rev else "F"
                if is_rev:
                    n_R += 1
                else:
                    n_F += 1
                pi = pi_hat(prev_rev)
                regime_pred.append((pi, int(is_rev)))

                dists = {
                    # 纯均匀于已访问集合：隔离"候选集小"与"排序有效"两种贡献
                    "uniform_seen":  _norm({x: 1.0 for x in seen}),
                    "hist_uniform":  p_hist(visits, cur["node"], "uniform"),
                    "hist_recency":  p_hist(visits, cur["node"], "recency"),
                    "hist_freq":     p_hist(visits, cur["node"], "freq"),
                    "graph_L1L2":    p_graph(g, cur["node"]),
                    "graph_unseen":  p_graph(g, cur["node"], exclude=seen),
                    "additive_full": p_additive_full(g, visits, cur["node"], seen),
                    # oracle 路由：用真 regime 当 π（上界，说明路由的潜力）
                    "routed_oracle": p_routed(g, visits, cur["node"], seen,
                                              1.0 if is_rev else 0.0, "recency"),
                    # 可实现路由：用训练集估的 π̂
                    "routed_pred":   p_routed(g, visits, cur["node"], seen,
                                              pi, "recency"),
                    "repeat_prev":   ({cur["node"]: 1.0} if cur["node"] else {}),
                    "marginal":      _norm({k: v for k, v in objfreq.items()}),
                }
                for name, d in dists.items():
                    acc[name]["all"].add(d, truth, held["id"])
                    acc[name][reg].add(d, truth, held["id"])

                # --- 联合动作 (tool, obj) ---
                pt = {v: p_tool(tm, cur["tool"], v, tool_vocab) for v in tool_vocab}
                zt = sum(pt.values())
                pt = {k: v / zt for k, v in pt.items()}
                po = dists["routed_pred"]
                if po:
                    joint = {(tk, ok): tv * ov
                             for tk, tv in pt.items() for ok, ov in po.items()}
                    jge = rank_ge(joint, (nxt["tool"], truth))
                    joint_n += 1
                    if jge is not None:
                        for k in KS:
                            joint_acc[k] += exp_hit(jge[0], jge[1], k)
                prev_rev = is_rev

    # ---------------------------------------------------------- 汇总
    res: dict = {
        "setting": "within-repo LOO；分母=全部决策点（未覆盖记 miss，不排除）",
        "n_episodes": len(eps),
        "regime_split": {
            "n_revisit": n_R, "n_first_visit": n_F,
            "revisit_rate": round(n_R / max(1, n_R + n_F), 4),
        },
        "max_distinct_objects_in_episode": max_seen,
        "predictors": {p: {r: acc[p][r].report() for r in ("all", "R", "F")}
                       for p in PREDICTORS},
        "regime_predictability": _auroc(regime_pred),
        "bootstrap_ci": _bootstrap_block(acc),
        "joint_action": ({"n": joint_n,
                          "topk_uncond": {f"top{k}": round(joint_acc[k] / joint_n, 4)
                                          for k in KS}} if joint_n else None),
    }

    if args.scan_gamma:
        res["gamma_scan"] = _scan_gamma(by_repo)

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---------------------------------------------------------- 打印
    rs = res["regime_split"]
    print(f"\n=== Q1 regime 构成 ===")
    print(f"  重访 R {rs['n_revisit']}  首访 F {rs['n_first_visit']}  "
          f"重访率 {rs['revisit_rate']:.1%}")
    print(f"  单 episode 最多访问 {max_seen} 个不同对象"
          f"（>{TOPK_SEEN} 则 uniform 截断生效）")

    print(f"\n=== Q2 各预测器（无条件，分母={n_R + n_F}）===")
    hdr = f"{'预测器':<16}{'覆盖':>7}{'top1':>8}{'top3':>8}{'top10':>8}{'MRR':>8}"
    print(hdr + f"{'ECEc':>8}{'ECEu':>8}{'有效候选':>9}")
    for p in PREDICTORS:
        r = acc[p]["all"].report()
        if "note" in r:
            continue
        print(f"{p:<16}{r['coverage']:>7.3f}{r['topk_uncond']['top1']:>8.3f}"
              f"{r['topk_uncond']['top3']:>8.3f}{r['topk_uncond']['top10']:>8.3f}"
              f"{r['MRR_uncond']:>8.3f}{r['ECE_cond']:>8.3f}{r['ECE_uncond']:>8.3f}"
              f"{str(r['eff_candidates']):>9}")

    print(f"\n--- 分 regime：R=重访(n={n_R}) / F=首访(n={n_F}) ---")
    print(f"{'预测器':<16}{'R:cov':>8}{'R:top3':>8}{'R:top10':>9}"
          f"{'F:cov':>8}{'F:top3':>8}{'F:top10':>9}")
    for p in PREDICTORS:
        rr_, rf = acc[p]["R"].report(), acc[p]["F"].report()
        if "note" in rr_ or "note" in rf:
            continue
        print(f"{p:<16}{rr_['coverage']:>8.3f}{rr_['topk_uncond']['top3']:>8.3f}"
              f"{rr_['topk_uncond']['top10']:>9.3f}"
              f"{rf['coverage']:>8.3f}{rf['topk_uncond']['top3']:>8.3f}"
              f"{rf['topk_uncond']['top10']:>9.3f}")

    ra = res["regime_predictability"]
    print(f"\n=== Q3 regime 可预测性 ===")
    print(f"  AUROC {ra['auroc']}  (n={ra['n']}，基线率 {ra['base_rate']})")

    print(f"\n=== Q4 路由 vs 单一模型（无条件 top-3 / top-10）===")
    for p in ("hist_recency", "graph_unseen", "additive_full",
              "routed_pred", "routed_oracle"):
        r = acc[p]["all"].report()
        print(f"  {p:<15} top3 {r['topk_uncond']['top3']:.4f}   "
              f"top10 {r['topk_uncond']['top10']:.4f}   "
              f"覆盖 {r['coverage']:.4f}   有效候选 {r['eff_candidates']}")

    if res["joint_action"]:
        j = res["joint_action"]
        print(f"\n=== 联合动作 (tool, obj)，n={j['n']} ===")
        print("  " + "  ".join(f"{k} {v}" for k, v in j["topk_uncond"].items()))

    bc = res["bootstrap_ci"]
    print(f"\n=== 关键量 95% CI（episode 级 bootstrap）===")
    for key in ("hist_freq_R_top1", "hist_freq_R_top3", "graph_F_top3",
                "graph_F_top10", "hist_freq_all_top10", "routed_pred_all_top10"):
        v = bc[key]
        if v.get("point") is None:
            continue
        print(f"  {key:<26} {v['point']:.4f}  CI[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
              f"  (n={v['n_points']}, {v['n_episodes']} episode)")
    print(f"\n=== 配对对比（差值 CI 不含 0 才算显著）===")
    for key, v in bc["contrasts"].items():
        if "note" in v:
            print(f"  {key:<40} {v['note']}")
            continue
        mark = "显著" if v["significant"] else "不显著"
        print(f"  {key:<40} {v['diff']:+.4f}  "
              f"CI[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {mark}")

    if "gamma_scan" in res:
        print(f"\n=== recency gamma 扫描 ===")
        for row in res["gamma_scan"]:
            print(f"  gamma={row['gamma']:.2f}  top3 {row['top3']:.4f}  "
                  f"top10 {row['top10']:.4f}  MRR {row['MRR']:.4f}")

    print(f"\n→ {args.out}")
    return 0


def _norm(w: dict[str, float]) -> dict[str, float]:
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def _auroc(pairs: list[tuple[float, int]]) -> dict:
    """Mann-Whitney U 形式的 AUROC，含平局 0.5 计。"""
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return {"auroc": None, "n": len(pairs), "base_rate": None,
                "note": "单一类别，无法算 AUROC"}
    s = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return {"auroc": round(s / (len(pos) * len(neg)), 4), "n": len(pairs),
            "base_rate": round(len(pos) / len(pairs), 4),
            "n_pos": len(pos), "n_neg": len(neg)}


def _topk_from_recs(recs: list[tuple[str, dict[int, float]]], k: int) -> float:
    """闭式期望 top-k：对每点的期望命中概率取均值（无蒙特卡洛噪声）。"""
    if not recs:
        return float("nan")
    return sum(h[k] for _, h in recs) / len(recs)


def _boot_ci(recs: list[tuple[str, dict[int, float]]], k: int, B: int = 2000,
             seed: int = 20260814) -> dict:
    """episode 级 bootstrap（整段重采样，因同 episode 内步骤强相关）。"""
    import random
    if not recs:
        return {"point": None, "note": "无样本"}
    by_ep: dict[str, list[float]] = defaultdict(list)
    for ep, h in recs:
        by_ep[ep].append(h[k])
    eps = list(by_ep)
    rng = random.Random(seed)
    pts = []
    for _ in range(B):
        pool: list[float] = []
        for _ in range(len(eps)):
            pool.extend(by_ep[eps[rng.randrange(len(eps))]])
        if pool:
            pts.append(sum(pool) / len(pool))
    pts.sort()
    return {
        "point": round(_topk_from_recs(recs, k), 4),
        "ci95": [round(pts[int(0.025 * len(pts))], 4),
                 round(pts[int(0.975 * len(pts))], 4)],
        "n_episodes": len(eps), "n_points": len(recs),
    }


def _paired_diff_ci(a: list[tuple[str, dict[int, float]]],
                    b: list[tuple[str, dict[int, float]]],
                    k: int, B: int = 2000, seed: int = 20260814) -> dict:
    """配对差 (a−b) 的 episode 级 bootstrap CI；两者逐点对齐。"""
    import random
    if len(a) != len(b):
        return {"note": f"长度不匹配 {len(a)} vs {len(b)}"}
    by_ep: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (ep, ha), (_, hb) in zip(a, b):
        by_ep[ep].append((ha[k], hb[k]))
    eps = list(by_ep)
    rng = random.Random(seed)
    pts = []
    for _ in range(B):
        pool: list[tuple[float, float]] = []
        for _ in range(len(eps)):
            pool.extend(by_ep[eps[rng.randrange(len(eps))]])
        if pool:
            pts.append((sum(x for x, _ in pool) - sum(y for _, y in pool)) / len(pool))
    pts.sort()
    obs = _topk_from_recs(a, k) - _topk_from_recs(b, k)
    lo, hi = pts[int(0.025 * len(pts))], pts[int(0.975 * len(pts))]
    return {"diff": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def _bootstrap_block(acc: dict) -> dict:
    """论文要用的几个关键量的 CI 与配对对比。"""
    out: dict = {"note": "episode 级 bootstrap B=2000，seed 固定"}
    out["hist_freq_R_top3"] = _boot_ci(acc["hist_freq"]["R"].recs, 3)
    out["hist_freq_R_top1"] = _boot_ci(acc["hist_freq"]["R"].recs, 1)
    out["graph_F_top3"] = _boot_ci(acc["graph_unseen"]["F"].recs, 3)
    out["graph_F_top10"] = _boot_ci(acc["graph_unseen"]["F"].recs, 10)
    out["routed_pred_all_top10"] = _boot_ci(acc["routed_pred"]["all"].recs, 10)
    out["hist_freq_all_top10"] = _boot_ci(acc["hist_freq"]["all"].recs, 10)
    out["contrasts"] = {
        # 论文主张 1：图在首访上是否优于"什么都不做"——与其自身 top-10 对比无意义，
        # 故直接看 CI 上界是否触及可用水平（见 graph_F_top3/top10）
        "hist_freq_vs_uniform_seen_R_top3": _paired_diff_ci(
            acc["hist_freq"]["R"].recs, acc["uniform_seen"]["R"].recs, 3),
        "hist_freq_vs_repeat_prev_all_top1": _paired_diff_ci(
            acc["hist_freq"]["all"].recs, acc["repeat_prev"]["all"].recs, 1),
        "routed_pred_vs_hist_freq_all_top10": _paired_diff_ci(
            acc["routed_pred"]["all"].recs, acc["hist_freq"]["all"].recs, 10),
        "routed_pred_vs_additive_full_all_top10": _paired_diff_ci(
            acc["routed_pred"]["all"].recs, acc["additive_full"]["all"].recs, 10),
        "additive_full_vs_hist_freq_R_top3": _paired_diff_ci(
            acc["additive_full"]["R"].recs, acc["hist_freq"]["R"].recs, 3),
        # 最锋利的否证：完全不用代码图的仓库词频基线，在代码图的**专属域**(首访)上
        # 是否反而更好。若显著为正，则代码图连自己唯一能触达的地方都不如平凡基线。
        "marginal_vs_graph_F_top10": _paired_diff_ci(
            acc["marginal"]["F"].recs, acc["graph_unseen"]["F"].recs, 10),
        "marginal_vs_graph_F_top3": _paired_diff_ci(
            acc["marginal"]["F"].recs, acc["graph_unseen"]["F"].recs, 3),
    }
    return out


def _scan_gamma(by_repo) -> list[dict]:
    out = []
    for gm in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        a = Acc()
        for proj, group in by_repo.items():
            if len(group) < 6:
                continue
            for held in group:
                seq = held["seq"]
                visits: list[str] = []
                seen: set[str] = set()
                for t in range(len(seq) - 1):
                    cur, nxt = seq[t], seq[t + 1]
                    if cur["node"]:
                        visits.append(cur["node"])
                        seen.add(cur["node"])
                    if not nxt["node"]:
                        continue
                    a.add(p_hist(visits, cur["node"], "recency", gamma=gm),
                          nxt["node"])
        r = a.report()
        out.append({"gamma": gm, "top3": r["topk_uncond"]["top3"],
                    "top10": r["topk_uncond"]["top10"], "MRR": r["MRR_uncond"]})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
