"""X4b：U^exec 是否有**非机械**结构 —— 决定 X4 的零结果能否发表。零 API。

问题
----
X4 得到「结构熵不预测执行不确定性」（rho=−0.088，CI 含 0），并用两个探针声称功效已确立：
    |Δ| ~ U^exec  rho=+0.534 显著
    v_t ~ U^exec  rho=−0.432 显著
但这两个探针都与 U^exec **机械相关**，不能作为功效证据：
  · v_t ~ U^exec：方差在均值靠近 0/1 时必然小（均值–方差约束），是算术不是结构。
  · |Δ| ~ U^exec：|Δ|=|v_t−v_{t−1}| 由两个 8 次重放均值之差得到，
    方差大 ⇒ Δ 的估计噪声大 ⇒ |Δ| 因噪声而偏大，同样是机械的。

又：U^exec 的**中位数为 0、50.5% 的点方差恰好为零**（m=8 的 recall 只取少数离散值），
这本身就说明该量在 m=8 下很粗。

所以必须换**与重放噪声无机械联系**的探针：
  V1 instance 级 ICC   U^exec 是否有实例层结构（置换零假设打散实例分组）
  V2 步位            U^exec ~ 归一化步位（Spearman）
  V3 工具类型        U^exec 按当前步工具分组是否有差异（置换 tool 标签）
  V4 残差化          先用 v_t 分箱吸收机械的均值–方差关系，再看**残差** U^exec
                     是否仍与 H_tool / 步位 / 工具有关

判据（预注册）
--------------
| V1–V3 结果 | 走向 |
|---|---|
| 至少一项显著 | U^exec 有真结构 ⇒ X4 的零结果是**实质结论**（结构熵与执行不确定性解耦），可发表 |
| 全部不显著 | U^exec 在 m=8 下无可检出结构 ⇒ X4 的零结果**仅反映功效不足**，不可发表；
             需提高 m 或换 U^exec 的定义（如用连续 recall 的 bootstrap 方差） |
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
from pilot.analyze.v3k_trace_causal import _spearman_boot    # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci            # noqa: E402

SEED = 20260814
B_PERM = 2000


def var(xs: list[float]) -> float:
    return statistics.variance(xs) if len(xs) > 1 else 0.0


def icc(groups: dict[str, list[float]]) -> float:
    """组间方差占总方差的比例（简化 ICC）。"""
    allv = [x for v in groups.values() for x in v]
    if len(allv) < 3:
        return float("nan")
    gm = statistics.mean(allv)
    n = len(allv)
    between = sum(len(v) * (statistics.mean(v) - gm) ** 2 for v in groups.values())
    total = sum((x - gm) ** 2 for x in allv)
    return between / total if total > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime", default="pilot/runs/v4prime/attribution.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x4b_uexec_validity.json")
    args = ap.parse_args()

    # ---- 全局工具转移模型（用于 H_tool 协变量）----
    trans: dict[str, Counter] = defaultdict(Counter)
    step_tool: dict[str, list[str]] = {}
    for p in sorted(Path(args.runs_root).glob("shard*/rollouts/*.jsonl")):
        iid = p.stem.split("__k")[0]
        try:
            st = R.steps(R.load(p))
        except (OSError, ValueError):
            continue
        tools = [parse_step(s).kind for s in st]
        step_tool[iid] = tools
        for a, b in zip(tools, tools[1:]):
            trans[a][b] += 1
    vocab = sorted({t for ts in step_tool.values() for t in ts})

    def h_tool(cur: str, alpha: float = 0.5) -> float:
        row = trans.get(cur) or Counter()
        tot = sum(row.values()) + alpha * len(vocab)
        ps = [(row.get(v, 0) + alpha) / tot for v in vocab]
        return -sum(q * math.log(q) for q in ps if q > 0)

    # ---- 逐估值点 ----
    recs = [json.loads(l) for l in open(args.v4prime) if l.strip()]
    rows = []
    for r in recs:
        iid = r["instance_id"]
        tools = step_tool.get(iid) or []
        nst = r.get("n_steps") or max(
            [s["t"] for s in (r.get("causal_segments") or [])] or [1])
        for s in r.get("causal_segments") or []:
            ys = s.get("ys_t")
            if not ys or len(ys) < 2:
                continue
            t = s["t"]
            cur = tools[t - 1] if 1 <= t <= len(tools) else None
            rows.append({
                "iid": iid, "t": t, "pos": t / max(1, nst),
                "u_exec": var([float(y) for y in ys]),
                "v_t": float(s.get("v_t") or 0.0),
                "tool": cur,
                "h_tool": (h_tool(cur) if cur else None),
                "n_gold": r.get("n_gold"),
            })
    print(f"逐点样本 {len(rows)}（来自 {len(recs)} 个 instance）")
    u = [x["u_exec"] for x in rows]
    print(f"U^exec: mean={statistics.mean(u):.6f} median={statistics.median(u):.6f} "
          f"非零率={sum(1 for x in u if x > 1e-12)/len(u):.4f}")

    rng = random.Random(SEED)
    res: dict = {"n_points": len(rows), "n_instances": len(recs),
                 "u_exec": {"mean": round(statistics.mean(u), 6),
                            "median": round(statistics.median(u), 6),
                            "nonzero_rate": round(
                                sum(1 for x in u if x > 1e-12) / len(u), 4)}}

    # ---- V1 instance 级 ICC ----
    by_inst: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        by_inst[x["iid"]].append(x["u_exec"])
    obs_icc = icc(by_inst)
    sizes = [len(v) for v in by_inst.values()]
    nulls = []
    for _ in range(B_PERM):
        pool = u[:]
        rng.shuffle(pool)
        g, i = {}, 0
        for k, sz in zip(by_inst, sizes):
            g[k] = pool[i:i + sz]
            i += sz
        nulls.append(icc(g))
    nulls = sorted(v for v in nulls if not math.isnan(v))
    p_icc = (sum(1 for v in nulls if v >= obs_icc) + 1) / (len(nulls) + 1)
    res["V1_instance_ICC"] = {
        "observed": round(obs_icc, 4),
        "null_mean": round(statistics.mean(nulls), 4),
        "null_p975": round(nulls[int(0.975 * len(nulls))], 4),
        "perm_p": p_icc, "significant": bool(p_icc < 0.05),
        "n_instances": len(by_inst),
    }

    # ---- V2 步位 ----
    res["V2_position"] = _spearman_boot([x["pos"] for x in rows], u)

    # ---- V3 工具类型（置换 tool 标签，统计量=组间方差比）----
    tr = [x for x in rows if x["tool"]]
    by_tool: dict[str, list[float]] = defaultdict(list)
    for x in tr:
        by_tool[x["tool"]].append(x["u_exec"])
    obs_t = icc(by_tool)
    ut = [x["u_exec"] for x in tr]
    tsizes = [len(v) for v in by_tool.values()]
    tn = []
    for _ in range(B_PERM):
        pool = ut[:]
        rng.shuffle(pool)
        g, i = {}, 0
        for k, sz in zip(by_tool, tsizes):
            g[k] = pool[i:i + sz]
            i += sz
        tn.append(icc(g))
    tn = sorted(v for v in tn if not math.isnan(v))
    p_t = (sum(1 for v in tn if v >= obs_t) + 1) / (len(tn) + 1)
    res["V3_tool_type"] = {
        "observed_between_group_ratio": round(obs_t, 4),
        "null_mean": round(statistics.mean(tn), 4),
        "perm_p": p_t, "significant": bool(p_t < 0.05),
        "group_means": {k: round(statistics.mean(v), 5)
                        for k, v in sorted(by_tool.items())},
        "group_n": {k: len(v) for k, v in sorted(by_tool.items())},
    }

    # ---- V4 残差化：先用 v_t 分箱吸收均值–方差的机械关系 ----
    NB = 5
    srt = sorted(rows, key=lambda x: x["v_t"])
    bin_of: dict[int, list[float]] = defaultdict(list)
    for i, x in enumerate(srt):
        x["_bin"] = min(NB - 1, i * NB // len(srt))
        bin_of[x["_bin"]].append(x["u_exec"])
    bmean = {b: statistics.mean(v) for b, v in bin_of.items()}
    for x in rows:
        x["resid"] = x["u_exec"] - bmean[x["_bin"]]
    # **决定性检验**：残差化后的 instance 级 ICC。
    # V1 的原始 ICC 可能仍是机械的 —— v_t 强烈依赖 instance（有些实例恒 v≈1 ⇒
    # 方差必为 0），故组间结构可能只是 v_t 的组间差异。只有在吸收 v_t 之后
    # ICC 仍显著，才能说 U^exec 有真正的、非机械的实例层结构。
    by_inst_r: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        by_inst_r[x["iid"]].append(x["resid"])
    obs_ricc = icc(by_inst_r)
    rvals = [x["resid"] for x in rows]
    rsizes = [len(v) for v in by_inst_r.values()]
    rn = []
    for _ in range(B_PERM):
        pool = rvals[:]
        rng.shuffle(pool)
        g, i = {}, 0
        for k, sz in zip(by_inst_r, rsizes):
            g[k] = pool[i:i + sz]
            i += sz
        rn.append(icc(g))
    rn = sorted(v for v in rn if not math.isnan(v))
    p_ricc = (sum(1 for v in rn if v >= obs_ricc) + 1) / (len(rn) + 1)

    hr = [(x["h_tool"], x["resid"]) for x in rows if x["h_tool"] is not None]
    res["V4_residualized"] = {
        "note": "v_t 五分箱吸收机械的均值–方差关系后，残差 U^exec 的关系",
        "bin_means_of_u_exec_by_v_t": {str(k): round(v, 5)
                                       for k, v in sorted(bmean.items())},
        "resid_vs_h_tool": (_spearman_boot([a for a, _ in hr], [b for _, b in hr])
                            if len(hr) > 20 else {"note": "样本不足"}),
        "resid_vs_position": _spearman_boot([x["pos"] for x in rows],
                                            [x["resid"] for x in rows]),
        "RESIDUALIZED_instance_ICC": {
            "observed": round(obs_ricc, 4),
            "null_mean": round(statistics.mean(rn), 4),
            "null_p975": round(rn[int(0.975 * len(rn))], 4),
            "perm_p": p_ricc, "significant": bool(p_ricc < 0.05),
            "note": "吸收 v_t 后仍显著 ⇒ U^exec 有非机械的实例层结构",
        },
    }

    sig = [k for k in ("V2_position", "V3_tool_type") if res[k].get("significant")]
    if res["V4_residualized"]["RESIDUALIZED_instance_ICC"]["significant"]:
        sig.append("RESIDUALIZED_instance_ICC")
    res["VERDICT"] = {
        "non_mechanical_probes_significant": sig,
        "decision": (
            f"U^exec 有非机械结构（{', '.join(sig)} 显著）⇒ X4 的零结果是**实质结论**，可发表"
            if sig else
            "U^exec 在 m=8 下无可检出的非机械结构 ⇒ X4 的零结果**仅反映功效不足**，"
            "不可作为『结构与执行不确定性解耦』的证据；需提高 m 或改 U^exec 定义"),
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V1 instance 级 ICC（U^exec 是否有实例层结构）===")
    v = res["V1_instance_ICC"]
    print(f"  观测 {v['observed']}  零假设均值 {v['null_mean']}  "
          f"零假设 97.5% {v['null_p975']}  perm p={v['perm_p']:.4g}  "
          f"{'显著' if v['significant'] else '不显著'}  (n_inst={v['n_instances']})")
    print(f"\n=== V2 U^exec ~ 归一化步位 ===")
    v = res["V2_position"]
    print(f"  rho={v['rho']:+.4f} CI{v['ci95']}  n={v['n']}  "
          f"{'显著' if v['significant'] else '不显著'}")
    print(f"\n=== V3 U^exec 按当前步工具分组 ===")
    v = res["V3_tool_type"]
    print(f"  组间方差比 {v['observed_between_group_ratio']}  零假设 {v['null_mean']}  "
          f"perm p={v['perm_p']:.4g}  {'显著' if v['significant'] else '不显著'}")
    for k in v["group_means"]:
        print(f"    {k:<14} mean={v['group_means'][k]:<10} n={v['group_n'][k]}")
    print(f"\n=== V4 残差化（吸收 v_t 的机械效应后）===")
    v4 = res["V4_residualized"]
    print(f"  U^exec 按 v_t 五分箱的均值: {v4['bin_means_of_u_exec_by_v_t']}")
    for key in ("resid_vs_h_tool", "resid_vs_position"):
        d = v4[key]
        if "note" in d:
            print(f"  {key:<20} {d['note']}")
            continue
        print(f"  {key:<20} rho={d['rho']:+.4f} CI{d['ci95']}  "
              f"{'显著' if d['significant'] else '不显著'}")
    ri = res["V4_residualized"]["RESIDUALIZED_instance_ICC"]
    print(f"  **残差化后的 instance ICC**: 观测 {ri['observed']}  "
          f"零假设 {ri['null_mean']} (97.5%={ri['null_p975']})  "
          f"perm p={ri['perm_p']:.4g}  {'显著' if ri['significant'] else '不显著'}")
    print(f"\n=== 裁决 ===\n  {res['VERDICT']['decision']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
