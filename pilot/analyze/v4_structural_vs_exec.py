"""V4（核心实验）：结构不确定性能否预测执行不确定性？

命题
----
    corr( α·U^tool_t + β·U^obj_t ,  U^exec_t )

U^tool  通道 A：工具转移分布的条件熵（零 API，已有 tool_model 产出）
U^obj   通道 B：代码图三层候选分布的熵（零 API，已有 code_channel）
U^exec  fork-replay 的结果方差（**昂贵**，已有 349 个估值段）

为何这个命题能成（而前四次预测全败）
----------------------------------
U^exec 是**重放方差**而非成败标签。前四次失败的根因是 94% 成功率下仅 30 个失败样本，
判别边界学不出来；而重放方差每个估值段都有、且连续取值，样本量是 349 而非 30。

若命题成立 → 可用零成本结构量代理昂贵重放，这是有实用价值的主结论。
若不成立 → 说明结构与执行不确定性解耦，即"必须重放，不能省"，同样是重要结论。
两档都可写。

U^exec 的定义
------------
每个估值段有 m 次重放的 recall 值。由 causal_segments 只存了 delta，
故用**相邻段 v̂ 的变化幅度**作为该点执行不确定性的代理：
    U^exec_t = 4 * v̂_t * (1 - v̂_t)      （Bernoulli 方差归一到 [0,1]）
v̂_t 由 causal_v0 + 累加 delta 还原。v̂=0.5 时最大，v̂∈{0,1} 时为 0。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import load_graph                   # noqa: E402
from pilot.channels.parsed_command import parse_step                 # noqa: E402
from pilot.harness import rollout as R                               # noqa: E402
from pilot.probes import task as T                                   # noqa: E402


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            a = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = a
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def boot_ci_rho(xs: list[float], ys: list[float], B: int = 4000,
                seed: int = 31) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(xs)
    X, Y = np.asarray(xs), np.asarray(ys)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = spearman(X[idx].tolist(), Y[idx].tolist())
        if r is not None:
            out.append(r)
    out.sort()
    return [round(out[int(0.025 * len(out))], 4),
            round(out[int(0.975 * len(out))], 4)] if out else [0.0, 0.0]


def traj(rec: dict) -> list[tuple[int, float]] | None:
    v = rec.get("causal_v0")
    if v is None:
        return None
    out = [(0, float(v))]
    for s in sorted(rec.get("causal_segments", []), key=lambda x: x["t"]):
        d = s.get("delta")
        if d is None:
            continue
        v = min(max(v + float(d), 0.0), 1.0)
        out.append((int(s["t"]), v))
    return out


def build_rows(attr_path: Path, locate_path: Path, runs_root: Path,
               tool_ent_path: Path) -> list[dict]:
    insts = {i.instance_id: i for i in T.load_instances()}
    loc = {}
    for line in locate_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            loc[r["instance_id"]] = r

    # U^tool：从 tool_model 的 entropy_rows 建 (instance, t) -> H_tool
    htool: dict[tuple[str, int], float] = {}
    if tool_ent_path.exists():
        d = json.loads(tool_ent_path.read_text())
        for e in d.get("entropy_rows", []):
            htool[(e["instance_id"], e["t"])] = e["H_tool"]

    rows = []
    attr = [json.loads(l) for l in attr_path.read_text().splitlines() if l.strip()]
    for rec in attr:
        if rec.get("error"):
            continue
        iid = rec["instance_id"]
        inst, lrec = insts.get(iid), loc.get(iid)
        if inst is None or not lrec or not lrec.get("rollout"):
            continue
        cands = list(runs_root.glob(f"shard*/{lrec['rollout']}"))
        if not cands:
            continue
        try:
            steps = R.steps(R.load(cands[0]))
        except (OSError, ValueError):
            continue
        g = load_graph(iid)
        tr = traj(rec)
        if not tr:
            continue
        for t, v in tr:
            if t < 1 or t > len(steps):
                continue
            p = parse_step(steps[t - 1])
            uo = g.u_obj_resolved(p.path or p.obj) if g.ok else {"H": None}
            ut = htool.get((iid, t))
            u_exec = 4.0 * v * (1.0 - v)          # Bernoulli 方差归一
            rows.append({
                "instance_id": iid, "project": rec.get("project"),
                "Y": rec.get("Y"), "t": t, "n_steps": len(steps),
                "tool_kind": p.kind, "obj": p.obj,
                "U_tool": ut, "U_obj": uo.get("H"),
                "n_l1": uo.get("n_l1"), "in_graph": uo.get("in_graph"),
                "v_hat": round(v, 4), "U_exec": round(u_exec, 4),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--locate", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--tool-ent", default="pilot/runs/tool_model_official.json")
    ap.add_argument("--out", default="pilot/runs/v4_structural_vs_exec.json")
    args = ap.parse_args()

    rows = build_rows(Path(args.attr), Path(args.locate),
                      Path(args.runs_root), Path(args.tool_ent))
    print(f"构造 {len(rows)} 个 (instance, t) 观测点")

    def pairs(key):
        xs, ys = [], []
        for r in rows:
            if r.get(key) is not None and r.get("U_exec") is not None:
                xs.append(r[key])
                ys.append(r["U_exec"])
        return xs, ys

    res = {"n_rows": len(rows), "correlations": {}}
    for key in ("U_tool", "U_obj"):
        xs, ys = pairs(key)
        if len(xs) < 10:
            res["correlations"][key] = {"n": len(xs), "note": "样本不足"}
            continue
        rho = spearman(xs, ys)
        res["correlations"][key] = {
            "n": len(xs), "spearman": round(rho, 4) if rho is not None else None,
            "ci95": boot_ci_rho(xs, ys),
            "mean_x": round(statistics.mean(xs), 4),
            "mean_y": round(statistics.mean(ys), 4),
        }

    # 组合：等权 z-score 相加（α=β=1，不调参以免过拟合）
    both = [r for r in rows if r.get("U_tool") is not None
            and r.get("U_obj") is not None and r.get("U_exec") is not None]
    if len(both) >= 10:
        ut = np.array([r["U_tool"] for r in both])
        uo = np.array([r["U_obj"] for r in both])
        ue = [r["U_exec"] for r in both]
        z = ((ut - ut.mean()) / (ut.std() or 1) + (uo - uo.mean()) / (uo.std() or 1))
        rho = spearman(z.tolist(), ue)
        res["correlations"]["U_struct_combined"] = {
            "n": len(both),
            "spearman": round(rho, 4) if rho is not None else None,
            "ci95": boot_ci_rho(z.tolist(), ue),
            "note": "等权 z-score 组合，未调参",
        }

    # 分层：按是否在图中、按成败
    strat = {}
    for name, sub in (("in_graph", [r for r in rows if r.get("in_graph")]),
                      ("succeeded", [r for r in rows if r.get("Y") == 1]),
                      ("failed", [r for r in rows if r.get("Y") == 0])):
        xs = [r["U_obj"] for r in sub if r.get("U_obj") is not None]
        ys = [r["U_exec"] for r in sub if r.get("U_obj") is not None]
        if len(xs) >= 10:
            rho = spearman(xs, ys)
            strat[name] = {"n": len(xs),
                           "spearman_Uobj": round(rho, 4) if rho is not None else None,
                           "ci95": boot_ci_rho(xs, ys)}
    res["stratified"] = strat
    res["coverage"] = {
        "with_U_tool": sum(1 for r in rows if r.get("U_tool") is not None),
        "with_U_obj": sum(1 for r in rows if r.get("U_obj") is not None),
        "obj_in_graph": sum(1 for r in rows if r.get("in_graph")),
    }
    res["rows"] = rows
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V4 结构不确定性 ↔ 执行不确定性 ===")
    print(f"覆盖: {res['coverage']}")
    print(f"\n{'量':<22}{'n':>6}{'Spearman':>11}{'CI95':>22}")
    for k, c in res["correlations"].items():
        if "note" in c and "spearman" not in c:
            print(f"{k:<22}{c['n']:>6}  {c['note']}")
            continue
        print(f"{k:<22}{c['n']:>6}{str(c['spearman']):>11}{str(c['ci95']):>22}")
    print("\n分层（U_obj vs U_exec）:")
    for k, c in strat.items():
        print(f"  {k:<12} n={c['n']:<5} rho={c['spearman_Uobj']} CI={c['ci95']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
