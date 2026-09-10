"""X1c：按 next-tool 路由**候选集构造**，并把样本从 71 扩到全部实例。零 API。

两个改动
--------
**改动 1：修 J3 诊断出的误配。**
X1b 证明 P(tool,obj) 不可因子化（I=0.159 nat，perm p=0.0005），但我当时的条件化
只按工具重估了"最近输出内的位置权重"，没有改**候选集的类型**，所以没拿到增益。
真正的依赖来自对象类型随工具切换（见 `parsed_command.py`）：

    read       → object 是**文件路径**
    list_files → object 是**目录**
    search     → object 是 `q:<query>`（无 path 参数时）或路径
    handler    → object 是 "-"（apply_patch / update_plan / view_image ...）

所以正确的模型是 P(obj | tool) 用**按工具不同的候选集**：
    P(u_{t+1}) = P(tool_{t+1} | tool_t) · P(obj_{t+1} | tool_{t+1}, 历史)

**改动 2：去掉代码图依赖，样本 71 → 全部。**
此前 V3g/V3i/V3j/X1b 都被限制在 71 个"已构图"仓库，唯一原因是用 `g.resolve()`
把对象映射到图节点。但**溯源通道根本不需要代码图** —— 它比对的是
「agent 命令里的路径字符串」与「工具输出里的路径字符串」，两者同源。
改用 `T.norm_path` 做纯字符串规范化后，样本可扩到全部 ~496 个实例。

⚠️ 扩样本必须先做一致性检查：在 71 个重叠实例上，图版与无图版的关键指标
应当接近。若相差很大，说明字符串规范化引入了系统偏差，不能直接扩。
本脚本的 `--overlap-check` 就做这件事。

评估设定
--------
被预测的是**完整联合动作** $(tool_{t+1}, obj_{t+1})$，含 obj="-" 的 handler 步，
两个模型共用同一分母（未覆盖记 miss）、闭式期望破并列。
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

from pilot.channels.parsed_command import parse_step        # noqa: E402
from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.analyze.v3g_regime import rank_ge, exp_hit, eff_cand, ece  # noqa: E402
from pilot.analyze.v3_nextstep_dual import fit_tool_trans, p_tool     # noqa: E402
from pilot.analyze.v3j_trace_graph import PATH_RE, boot_ci, paired_ci  # noqa: E402

SEED = 20260814
KS = (1, 2, 3, 5, 10)
HANDLER_OBJ = "-"


# ------------------------------------------------------------ 无图载入

def norm(p: str | None) -> str | None:
    if not p:
        return None
    s = T.norm_path(p)
    return s or None


def out_paths(text: str, cap: int = 400) -> list[str]:
    """从工具输出里按出现顺序抽取规范化路径（去重）。不需要代码图。"""
    if not text:
        return []
    seen, res = set(), []
    for m in PATH_RE.finditer(text[:200_000]):
        s = norm(m.group(1))
        if s and s not in seen:
            seen.add(s)
            res.append(s)
            if len(res) >= cap:
                break
    return res


def load_all(runs_root: Path, require_graph: bool = False) -> list[dict]:
    """载入全部实例的 (tool, obj, 输出) 序列。require_graph 仅用于一致性检查。"""
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        if require_graph:
            from pilot.channels.code_channel import load_graph
            if not load_graph(iid).ok:
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
            path = norm(p.path)
            if path:
                obj = path
            elif p.query:
                obj = f"q:{p.query}"
            else:
                obj = HANDLER_OBJ
            seq.append({"tool": p.kind, "obj": obj, "is_path": bool(path),
                        "out": s.output_text or ""})
        eps.append({"id": iid, "project": inst.project, "seq": seq})
    return eps


# ------------------------------------------------------------ 候选集构造

OBJ_TYPES = ("path", "query", "handler")


def fit_type_prior(train: list[dict], alpha: float = 0.5) -> dict[str, dict[str, float]]:
    """从训练 episode 学 P(对象类型 | 工具)。

    首版把工具硬编码成"handler 名单 vs 其它"，结果 `-` 对象大量来自
    `unknown` / `write_stdin`（不在名单里）→ 它们的候选集里没有 `-` →
    handler 类型覆盖率 0、top-3 恒为 0。学出来才是对的做法。
    """
    cnt: dict[str, Counter] = defaultdict(Counter)
    for e in train:
        for s in e["seq"]:
            ot = ("path" if s["is_path"] else
                  "query" if s["obj"].startswith("q:") else "handler")
            cnt[s["tool"]][ot] += 1
    prior: dict[str, dict[str, float]] = {}
    glob = Counter()
    for tool, row in cnt.items():
        glob.update(row)
    gtot = sum(glob.values()) + alpha * len(OBJ_TYPES)
    gprior = {o: (glob[o] + alpha) / gtot for o in OBJ_TYPES}
    for tool, row in cnt.items():
        tot = sum(row.values()) + alpha * len(OBJ_TYPES)
        prior[tool] = {o: (row[o] + alpha) / tot for o in OBJ_TYPES}
    prior["__default__"] = gprior
    return prior


def _p_path(prov: list[str], visited: list[str]) -> dict[str, float]:
    w: dict[str, float] = {}
    for i, p in enumerate(prov[:40]):
        w[p] = w.get(p, 0.0) + 1.0 / (i + 1)
    cnt = Counter(visited)
    for p, c in cnt.items():
        w[p] = w.get(p, 0.0) + 0.5 * c / max(1, len(visited))
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def _p_query(queries: list[str]) -> dict[str, float]:
    w = {q: 1.0 / (i + 1) for i, q in enumerate(reversed(queries[-10:]))}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def cand_for_tool(tool: str, prov: list[str], visited: list[str],
                  dirs: set[str], queries: list[str],
                  type_prior: dict[str, dict[str, float]] | None = None
                  ) -> dict[str, float]:
    """**按工具**给出对象候选集与权重 —— X1c 的核心。

    P(obj | tool) = Σ_type P(type | tool) · P(obj | type, 历史)
      path    → 溯源路径（按最近输出内位置加权）+ 已访问路径（按频次）
      query   → 历史 query（按新近度）
      handler → 点质量在 "-"
    类型先验从训练集学（见 `fit_type_prior`），不再硬编码工具名单。
    """
    tp = (type_prior or {}).get(tool) or (type_prior or {}).get("__default__") \
        or {"path": 0.5, "query": 0.2, "handler": 0.3}
    w: dict[str, float] = {}
    pp = _p_path(prov, visited)
    pq = _p_query(queries)
    for o, p in pp.items():
        w[o] = w.get(o, 0.0) + tp["path"] * p
    for o, p in pq.items():
        w[o] = w.get(o, 0.0) + tp["query"] * p
    w[HANDLER_OBJ] = w.get(HANDLER_OBJ, 0.0) + tp["handler"]
    # path 类型无候选时（首步没有任何输出），把质量退给 handler
    if not pp and not pq:
        w = {HANDLER_OBJ: 1.0}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def cand_shared(prov: list[str], visited: list[str],
                dirs: set[str], queries: list[str]) -> dict[str, float]:
    """因子化对照：**与工具无关**的单一对象模型（X1b 的做法）。"""
    w: dict[str, float] = {}
    for i, p in enumerate(prov[:40]):
        w[p] = w.get(p, 0.0) + 1.0 / (i + 1)
    cnt = Counter(visited)
    for p, c in cnt.items():
        w[p] = w.get(p, 0.0) + 0.5 * c / max(1, len(visited))
    for d in dirs:
        w[d] = w.get(d, 0.0) + 0.1
    for i, q in enumerate(reversed(queries[-10:])):
        w[q] = w.get(q, 0.0) + 0.2 / (i + 1)
    # 对照模型也给 "-" 一份质量（用全局边际比例 ~0.27），否则对照被人为削弱、
    # 路由的增益会被高估。差别只在于：对照**不按工具**调整这个比例。
    w[HANDLER_OBJ] = w.get(HANDLER_OBJ, 0.0) + 0.27 * max(1.0, sum(w.values()))
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


# ------------------------------------------------------------ 主流程

def evaluate(eps: list[dict], min_group: int = 6) -> dict:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})

    MODELS = ("routed", "factored")
    hits = {m: {k: [] for k in KS} for m in MODELS}
    cov = {m: [] for m in MODELS}
    nll = {m: [] for m in MODELS}
    effc = {m: [] for m in MODELS}
    cal = {m: [] for m in MODELS}
    # 仅对象分量（用于与 V3j 的 71 版对照）
    obj_hits = {k: [] for k in KS}
    # 按**真值对象类型**分层：这是关键分解。V3j/X1b 只在"对象是可解析文件"的点上
    # 评估（过滤掉了新 query 与 handler 步），故其数值不可与本脚本的无条件数直接比。
    OT = ("path", "query", "handler")
    by_ot = {m: {o: {k: [] for k in KS} for o in OT} for m in MODELS}
    ot_count: Counter = Counter()
    nll_cov = {m: [] for m in MODELS}      # 只在覆盖点上的 NLL（真信息量）
    n_pts = 0
    n_eps_used = 0

    for proj, group in by_repo.items():
        if len(group) < min_group:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            tprior = fit_type_prior(train)
            n_eps_used += 1
            seq = held["seq"]
            prov: list[str] = []
            visited: list[str] = []
            dirs: set[str] = set()
            queries: list[str] = []
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["is_path"]:
                    visited.append(cur["obj"])
                    if "/" in cur["obj"]:
                        dirs.add(cur["obj"].rsplit("/", 1)[0])
                if cur["tool"] == "list_files" and cur["is_path"]:
                    dirs.add(cur["obj"])
                if cur["obj"].startswith("q:"):
                    queries.append(cur["obj"])
                op = out_paths(cur["out"])
                if op:
                    prov = op                      # 最近一条有路径的输出
                    for p in op:
                        if "/" in p:
                            dirs.add(p.rsplit("/", 1)[0])

                truth = (nxt["tool"], nxt["obj"])
                ot = ("path" if nxt["is_path"] else
                      "query" if nxt["obj"].startswith("q:") else "handler")
                ot_count[ot] += 1
                n_pts += 1
                pt = {v: p_tool(tm, cur["tool"], v, tool_vocab) for v in tool_vocab}
                zt = sum(pt.values())
                pt = {k: v / zt for k, v in pt.items()}

                # routed：每个工具一套候选集
                jd_r: dict[tuple[str, str], float] = {}
                for a, pa in pt.items():
                    for o, po in cand_for_tool(a, prov, visited, dirs, queries,
                                               tprior).items():
                        jd_r[(a, o)] = jd_r.get((a, o), 0.0) + pa * po
                # factored：共享对象模型
                shared = cand_shared(prov, visited, dirs, queries)
                jd_f = {(a, o): pa * po
                        for a, pa in pt.items() for o, po in shared.items()}

                for m, jd in (("routed", jd_r), ("factored", jd_f)):
                    rs = rank_ge(jd, truth)
                    cov[m].append(1.0 if rs else 0.0)
                    for k in KS:
                        hits[m][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    for k in KS:
                        by_ot[m][ot][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        effc[m].append(eff_cand(jd))
                        pv = jd[truth]
                        nll[m].append(-math.log(max(pv, 1e-12)))
                        nll_cov[m].append(-math.log(max(pv, 1e-12)))
                        cal[m].append((pv, 1))
                        for v in sorted((v for kk, v in jd.items() if kk != truth),
                                        reverse=True)[:20]:
                            cal[m].append((v, 0))
                    else:
                        nll[m].append(-math.log(1e-12))
                # 对象分量单独（与 V3j 对照）
                rso = rank_ge(shared, nxt["obj"])
                for k in KS:
                    obj_hits[k].append(exp_hit(rso[0], rso[1], k) if rso else 0.0)

    out = {
        "n_episodes_total": len(eps), "n_episodes_evaluated": n_eps_used,
        "n_decision_points": n_pts,
        "models": {m: {
            "coverage": boot_ci(cov[m])["mean"],
            "topk": {f"top{k}": boot_ci(hits[m][k]) for k in KS},
            "NLL_all_miss_penalized": round(statistics.mean(nll[m]), 4) if nll[m] else None,
            "NLL_covered_only": (round(statistics.mean(nll_cov[m]), 4)
                                 if nll_cov[m] else None),
            "ECE": round(ece(cal[m]), 4),
            "eff_candidates_median": (round(statistics.median(effc[m]), 2)
                                      if effc[m] else None),
        } for m in MODELS},
        "contrasts": {
            f"routed_minus_factored_top{k}": paired_ci(hits["routed"][k],
                                                       hits["factored"][k])
            for k in (1, 3, 10)},
        "object_only_topk": {f"top{k}": round(statistics.mean(obj_hits[k]), 4)
                             for k in KS},
        "OBJECT_TYPE_COMPOSITION": {o: round(c / max(1, n_pts), 4)
                                    for o, c in ot_count.most_common()},
        "BY_OBJECT_TYPE": {m: {o: {
            "n": len(by_ot[m][o][1]),
            "topk": {f"top{k}": round(statistics.mean(by_ot[m][o][k]), 4)
                     for k in KS} if by_ot[m][o][1] else None}
            for o in OT} for m in MODELS},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x1c_tool_routed.json")
    ap.add_argument("--overlap-check", action="store_true",
                    help="额外在 71 个已构图实例上跑一遍，检查扩样本是否引入偏差")
    args = ap.parse_args()

    eps_all = load_all(Path(args.runs_root))
    print(f"无图载入（全部实例）：{len(eps_all)} episode")
    res = {"ALL_INSTANCES": evaluate(eps_all)}

    if args.overlap_check:
        eps_g = load_all(Path(args.runs_root), require_graph=True)
        print(f"一致性检查（仅已构图 71 个）：{len(eps_g)} episode")
        res["GRAPH_SUBSET_71"] = evaluate(eps_g)

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    for tag, blk in res.items():
        print(f"\n{'='*66}\n=== {tag}  "
              f"（{blk['n_episodes_evaluated']}/{blk['n_episodes_total']} episode，"
              f"{blk['n_decision_points']} 决策点）===")
        print(f"{'模型':<12}{'覆盖':>8}{'top1':>9}{'top3':>9}{'top10':>9}"
              f"{'NLL(覆盖)':>11}{'ECE':>9}{'有效候选':>10}")
        for m in ("routed", "factored"):
            d = blk["models"][m]
            print(f"{m:<12}{d['coverage']:>8.4f}"
                  f"{d['topk']['top1']['mean']:>9.4f}"
                  f"{d['topk']['top3']['mean']:>9.4f}"
                  f"{d['topk']['top10']['mean']:>9.4f}"
                  f"{d['NLL_covered_only']:>11.4f}{d['ECE']:>9.4f}"
                  f"{str(d['eff_candidates_median']):>10}")
        comp = blk["OBJECT_TYPE_COMPOSITION"]
        print(f"  真值对象类型构成: " +
              "  ".join(f"{k}={v:.1%}" for k, v in comp.items()))
        print(f"  {'按对象类型的 top-3 / top-10':<30}" +
              "".join(f"{o:>22}" for o in ("path", "query", "handler")))
        for m in ("routed", "factored"):
            cells = []
            for o in ("path", "query", "handler"):
                d = blk["BY_OBJECT_TYPE"][m][o]
                cells.append(f"{d['topk']['top3']:.3f}/{d['topk']['top10']:.3f} (n={d['n']})"
                             if d["topk"] else "—")
            print(f"    {m:<28}" + "".join(f"{c:>22}" for c in cells))
        print("  配对对比（routed − factored）：")
        for k, v in blk["contrasts"].items():
            if "note" in v:
                print(f"    {k:<34} {v['note']}")
                continue
            print(f"    {k:<34} {v['diff']:+.4f} "
                  f"CI[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
                  f"{'显著' if v['significant'] else '不显著'}")
        print(f"  仅对象分量 top-k: " +
              "  ".join(f"{k}={v}" for k, v in blk["object_only_topk"].items()))

    if "GRAPH_SUBSET_71" in res:
        a = res["ALL_INSTANCES"]["models"]["routed"]["topk"]["top3"]["mean"]
        b = res["GRAPH_SUBSET_71"]["models"]["routed"]["topk"]["top3"]["mean"]
        print(f"\n=== 扩样本一致性 ===")
        print(f"  routed top-3：全部 {a:.4f} vs 已构图 71 个 {b:.4f}"
              f"   相对差 {abs(a-b)/max(b,1e-9):.1%}")
        print(f"  {'✅ 一致，可直接扩样本' if abs(a-b)/max(b,1e-9) < 0.20 else '⚠️ 差异>20%，需查规范化偏差'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
