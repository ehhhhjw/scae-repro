"""V3g: common-denominator evaluation + dual-regime decomposition + L3 weighted ablation + routed mixture. Zero API.

Why V3f must be redone
----------------------
V3f computed each arm's top-k **conditional on "the truth falling inside that arm's candidate
set"** (full n=236 / only_L3 n=168 / no_L3 n=90), but full's candidate set ⊇ only_L3's, so
only_L3's 168 points are a subset of full's 236 points, and are exactly the easy "revisit"
points. Comparing arms via their individually conditioned top-k is not a same-baseline
comparison. Switching to a common denominator (all 509 decision points, uncovered = miss):

    arm      cond_top3 uncond_top3 cond_top10 uncond_top10
    full       0.491     0.228       0.733      0.340
    only_L3    0.744     0.246       0.994      0.328
    no_L1      0.670     0.259       0.858      0.332

→ V3f's "only_L3 dominates full" and "the monotone relation that less graph structure is
   better" **both fail**. The only surviving conclusion is that no_L3 collapses (the
   history prior is necessary).

The real questions this experiment answers
------------------------------------------
only_L3 can structurally **only predict revisits** (its candidate set is exactly the visited
set); on first-visit steps its coverage is identically 0. So the two kinds of steps should be
modeled separately rather than additively mixed into a single score:

  regime R (revisit)     next ∈ seen      -- the history prior applies
  regime F (first-visit) next ∉ seen      -- only the code graph can reach it

Hence four questions:
  Q1 What fraction is each regime? (fixes the shape of the problem)
  Q2 How good is the history prior on R? the code graph on F? (their domains of validity)
  Q3 Is the regime itself predictable? (whether routing is realizable)
  Q4 Does a routed mixture beat either single model? (the right model form)

Also fixes two bugs and adds one ablation
-----------------------------------------
  bug1  `list(seen)[-30:]`: seen is a set with no order guarantee → "the most recent 30"
        was really "an arbitrary 30". Changed to a visit-ordered list.
  bug2  ECE was computed only on covered points; uncovered points (where the model gives the
        truth probability 0) were excluded → calibration overstated.
        Report both ECE_cond (comparable to V3b) and ECE_uncond (includes miss).
  ablation  Since L3 is the only effective component, in-L3 weighting is where the gain is:
        uniform (status quo) / recency (exponential decay) / freq (visit count).
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
GAMMA = 0.8          # recency decay factor, see --scan-gamma
NEG_CAP = 20         # cap on negatives for ECE_cond (matches V3b)


# ---------------------------------------------------------------- candidates

def p_hist(visits: list[str], cur: str | None, mode: str,
           topk: int = TOPK_SEEN, gamma: float = GAMMA) -> dict[str, float]:
    """History prior (L3). visits is in visit order and may contain duplicates.

    mode: uniform (V3f status quo, flat) / recency (exponential decay) / freq (visit count)
    """
    if not visits:
        return {}
    w: dict[str, float] = {}
    if mode == "uniform":
        # bug1 fix: take the most recent topk **deduplicated** objects in visit order
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
    if cur:                                   # self-repeat prior
        w[cur] = w.get(cur, 0.0) + 0.5 * max(w.values())
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def p_graph(g, cur_node: str | None, exclude: set[str] | None = None) -> dict[str, float]:
    """Code-graph prior (L1+L2); exclude keeps only unvisited objects (used by regime F)."""
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
    """V3f's full arm: additive mixture of the three layers (as a control)."""
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
    """Routed mixture: pi·P_hist(visited) + (1−pi)·P_graph(unvisited).

    Unlike additive_full, **the two components have disjoint support** and the weights come
    from the regime probability rather than fixed constants W. That way, whenever the regime
    is predicted correctly, the mass automatically lands on the right half.
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


# ---------------------------------------------------------------- metrics

def rank_ge(dist: dict[str, float], truth: str) -> tuple[int, int] | None:
    """Return (g, e): g = candidates strictly above the truth's probability, e = candidates tied with the truth (incl. it).

    Returns None when uncovered. **No arbitrary tie-breaking** -- all candidates tie under a
    flat distribution, and lexicographic tie-breaking fabricates signal out of alphabetical
    order (the same class of error as pitfall #15).
    The caller converts (g, e) into top-k hit probability and E[1/rank] in closed form.
    """
    if truth not in dist:
        return None
    p = dist[truth]
    g = sum(1 for v in dist.values() if v > p)
    e = sum(1 for v in dist.values() if v == p)
    return g, e


def exp_hit(g: int, e: int, k: int) -> float:
    """P(rank ≤ k) under random tie-breaking: rank = g + U, U uniform on {1..e}."""
    return min(1.0, max(0.0, (k - g) / e))


def exp_recip(g: int, e: int) -> float:
    """Closed form for E[1/rank]: (1/e)·Σ_{j=g+1}^{g+e} 1/j."""
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
    """Accumulator for one predictor. The denominator is **all decision points**; misses count.

    Also keeps per-point records (with episode id) for episode-level bootstrap.
    """

    def __init__(self) -> None:
        self.n = 0
        self.hits = {k: 0.0 for k in KS}          # expected hits (closed form, non-integer)
        self.rr = 0.0
        self.cov = 0
        self.eff: list[float] = []
        self.nll: list[float] = []
        self.cal_cond: list[tuple[float, int]] = []
        self.cal_unc: list[tuple[float, int]] = []
        # (episode_id, {k: expected hit probability}) -- miss points record 0.0 for every k
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
            return {"note": "no valid points"}
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


# ---------------------------------------------------------------- main flow

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3g_regime.json")
    ap.add_argument("--scan-gamma", action="store_true",
                    help="scan the recency decay factor gamma")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"loaded {len(eps)} episodes, {len(tool_vocab)} tool classes")

    PREDICTORS = [
        "uniform_seen", "hist_uniform", "hist_recency", "hist_freq",
        "graph_L1L2", "graph_unseen",
        "additive_full", "routed_oracle", "routed_pred",
        "repeat_prev", "marginal",
    ]
    # three accumulator sets: global + per regime
    acc = {p: {"all": Acc(), "R": Acc(), "F": Acc()} for p in PREDICTORS}

    n_R = n_F = 0
    max_seen = 0
    regime_pred: list[tuple[float, int]] = []       # (π̂, is_revisit) for AUROC
    joint_acc, joint_n = {k: 0.0 for k in KS}, 0

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [e for j, e in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            objfreq = Counter(s["node"] for e in train for s in e["seq"] if s["node"])

            # --- estimate regime transitions from the training set (Q3: is regime predictable) ---
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
                if tot < 5:                      # fall back to the global rate
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
                    # flat over the visited set: separates "small candidate set" from "useful ranking"
                    "uniform_seen":  _norm({x: 1.0 for x in seen}),
                    "hist_uniform":  p_hist(visits, cur["node"], "uniform"),
                    "hist_recency":  p_hist(visits, cur["node"], "recency"),
                    "hist_freq":     p_hist(visits, cur["node"], "freq"),
                    "graph_L1L2":    p_graph(g, cur["node"]),
                    "graph_unseen":  p_graph(g, cur["node"], exclude=seen),
                    "additive_full": p_additive_full(g, visits, cur["node"], seen),
                    # oracle routing: use the true regime as π (upper bound, shows routing's potential)
                    "routed_oracle": p_routed(g, visits, cur["node"], seen,
                                              1.0 if is_rev else 0.0, "recency"),
                    # realizable routing: use the π̂ estimated from the training set
                    "routed_pred":   p_routed(g, visits, cur["node"], seen,
                                              pi, "recency"),
                    "repeat_prev":   ({cur["node"]: 1.0} if cur["node"] else {}),
                    "marginal":      _norm({k: v for k, v in objfreq.items()}),
                }
                for name, d in dists.items():
                    acc[name]["all"].add(d, truth, held["id"])
                    acc[name][reg].add(d, truth, held["id"])

                # --- joint action (tool, obj) ---
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

    # ---------------------------------------------------------- summary
    res: dict = {
        "setting": "within-repo LOO; denominator = all decision points (uncovered = miss, not excluded)",
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

    # ---------------------------------------------------------- printing
    rs = res["regime_split"]
    print(f"\n=== Q1 regime composition ===")
    print(f"  revisit R {rs['n_revisit']}  first-visit F {rs['n_first_visit']}  "
          f"revisit rate {rs['revisit_rate']:.1%}")
    print(f"  max distinct objects visited in one episode: {max_seen}"
          f" (>{TOPK_SEEN} means the uniform truncation takes effect)")

    print(f"\n=== Q2 per predictor (unconditional, denominator={n_R + n_F}) ===")
    hdr = f"{'predictor':<16}{'cov':>7}{'top1':>8}{'top3':>8}{'top10':>8}{'MRR':>8}"
    print(hdr + f"{'ECEc':>8}{'ECEu':>8}{'eff_cand':>9}")
    for p in PREDICTORS:
        r = acc[p]["all"].report()
        if "note" in r:
            continue
        print(f"{p:<16}{r['coverage']:>7.3f}{r['topk_uncond']['top1']:>8.3f}"
              f"{r['topk_uncond']['top3']:>8.3f}{r['topk_uncond']['top10']:>8.3f}"
              f"{r['MRR_uncond']:>8.3f}{r['ECE_cond']:>8.3f}{r['ECE_uncond']:>8.3f}"
              f"{str(r['eff_candidates']):>9}")

    print(f"\n--- by regime: R=revisit(n={n_R}) / F=first-visit(n={n_F}) ---")
    print(f"{'predictor':<16}{'R:cov':>8}{'R:top3':>8}{'R:top10':>9}"
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
    print(f"\n=== Q3 regime predictability ===")
    print(f"  AUROC {ra['auroc']}  (n={ra['n']}, base rate {ra['base_rate']})")

    print(f"\n=== Q4 routing vs single models (unconditional top-3 / top-10) ===")
    for p in ("hist_recency", "graph_unseen", "additive_full",
              "routed_pred", "routed_oracle"):
        r = acc[p]["all"].report()
        print(f"  {p:<15} top3 {r['topk_uncond']['top3']:.4f}   "
              f"top10 {r['topk_uncond']['top10']:.4f}   "
              f"cov {r['coverage']:.4f}   eff_cand {r['eff_candidates']}")

    if res["joint_action"]:
        j = res["joint_action"]
        print(f"\n=== joint action (tool, obj), n={j['n']} ===")
        print("  " + "  ".join(f"{k} {v}" for k, v in j["topk_uncond"].items()))

    bc = res["bootstrap_ci"]
    print(f"\n=== key quantities 95% CI (episode-level bootstrap) ===")
    for key in ("hist_freq_R_top1", "hist_freq_R_top3", "graph_F_top3",
                "graph_F_top10", "hist_freq_all_top10", "routed_pred_all_top10"):
        v = bc[key]
        if v.get("point") is None:
            continue
        print(f"  {key:<26} {v['point']:.4f}  CI[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
              f"  (n={v['n_points']}, {v['n_episodes']} episodes)")
    print(f"\n=== paired contrasts (significant only if the diff CI excludes 0) ===")
    for key, v in bc["contrasts"].items():
        if "note" in v:
            print(f"  {key:<40} {v['note']}")
            continue
        mark = "significant" if v["significant"] else "not significant"
        print(f"  {key:<40} {v['diff']:+.4f}  "
              f"CI[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {mark}")

    if "gamma_scan" in res:
        print(f"\n=== recency gamma scan ===")
        for row in res["gamma_scan"]:
            print(f"  gamma={row['gamma']:.2f}  top3 {row['top3']:.4f}  "
                  f"top10 {row['top10']:.4f}  MRR {row['MRR']:.4f}")

    print(f"\n→ {args.out}")
    return 0


def _norm(w: dict[str, float]) -> dict[str, float]:
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def _auroc(pairs: list[tuple[float, int]]) -> dict:
    """AUROC in Mann-Whitney U form; ties count as 0.5."""
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return {"auroc": None, "n": len(pairs), "base_rate": None,
                "note": "single class, AUROC undefined"}
    s = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return {"auroc": round(s / (len(pos) * len(neg)), 4), "n": len(pairs),
            "base_rate": round(len(pos) / len(pairs), 4),
            "n_pos": len(pos), "n_neg": len(neg)}


def _topk_from_recs(recs: list[tuple[str, dict[int, float]]], k: int) -> float:
    """Closed-form expected top-k: mean of the per-point expected hit probability (no Monte Carlo noise)."""
    if not recs:
        return float("nan")
    return sum(h[k] for _, h in recs) / len(recs)


def _boot_ci(recs: list[tuple[str, dict[int, float]]], k: int, B: int = 2000,
             seed: int = 20260814) -> dict:
    """Episode-level bootstrap (resample whole episodes, since steps within an episode are strongly correlated)."""
    import random
    if not recs:
        return {"point": None, "note": "no samples"}
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
    """Episode-level bootstrap CI for the paired difference (a−b); the two are aligned point by point."""
    import random
    if len(a) != len(b):
        return {"note": f"length mismatch {len(a)} vs {len(b)}"}
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
    """CIs and paired contrasts for the key quantities the paper needs."""
    out: dict = {"note": "episode-level bootstrap B=2000, fixed seed"}
    out["hist_freq_R_top3"] = _boot_ci(acc["hist_freq"]["R"].recs, 3)
    out["hist_freq_R_top1"] = _boot_ci(acc["hist_freq"]["R"].recs, 1)
    out["graph_F_top3"] = _boot_ci(acc["graph_unseen"]["F"].recs, 3)
    out["graph_F_top10"] = _boot_ci(acc["graph_unseen"]["F"].recs, 10)
    out["routed_pred_all_top10"] = _boot_ci(acc["routed_pred"]["all"].recs, 10)
    out["hist_freq_all_top10"] = _boot_ci(acc["hist_freq"]["all"].recs, 10)
    out["contrasts"] = {
        # Paper claim 1: is the graph better than "doing nothing" on first visits -- comparing it
        # to its own top-10 is meaningless, so check the CI upper bound instead against a usable
        # level (see graph_F_top3/top10)
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
        # Sharpest falsification: on the code graph's **exclusive domain** (first visits), does the
        # repository term-frequency baseline -- which uses no code graph at all -- do better? If
        # significantly positive, the graph loses to a trivial baseline even where only it can reach.
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
