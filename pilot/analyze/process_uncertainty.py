"""过程不确定性量化 + 成功/失败分层归因 —— UQ 主线的正确落点。

设计修正记录
------------
早期两次尝试都把 UQ 绑在"跨 episode 预测二元成败 Y"上：
  (1) v_0（执行前特征）→ LORO AUROC ≈ 0.5
  (2) anytime 前缀特征 → bootstrap CI 全部含 0.5
根本障碍是 94% 成功率下仅 30 个失败样本，判别边界学不出来。

正确落点：不确定性来自**单次执行的重放方差**，而非跨 episode 的成败标签。
v̂_t 由 m 次 fork 重放估计，v̂_t=1 表示从该前缀必定成功（零不确定性），
v̂_t≈0.5 表示一半重放成功（最大不确定性）。每个 episode 都有此量，无论成败。

三个可报告的量
--------------
  U1 过程熵轨迹  H(v̂_t)：不确定性随执行的演化（"第几步收敛"）
  U2 脆弱成功率  Y=1 且 ∃t: v̂_t < 1−ε：成功但重放不稳 —— 结果导向评测的盲区
  U3 收敛点 t*   v̂_t 末次跨越阈值后不再回落的步位

归因分层（成功 case 同样归因）
------------------------------
失败 case：argmin_t Δ̂_t —— 最伤结果的步
成功 case：argmax_t Δ̂_t —— 最关键的正向步
两者都由同一 Theorem 1 分解给出，无需失败样本作为前提。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def binary_entropy(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def reconstruct_traj(rec: dict) -> list[tuple[int, float]] | None:
    """从 causal_v0 + 累加 delta 还原 v̂_t 轨迹。

    causal_segments 只持久化了 delta（实现时的字段缺口），但
    delta_i = v̂_{t_i} − v̂_{t_{i-1}}，故 v̂ 可由 v0 累加还原。
    """
    v = rec.get("causal_v0")
    if v is None:
        return None
    out = [(0, float(v))]
    for s in sorted(rec.get("causal_segments", []), key=lambda x: x["t"]):
        d = s.get("delta")
        if d is None:
            continue
        v = v + float(d)
        out.append((int(s["t"]), round(min(max(v, 0.0), 1.0), 6)))
    return out


def bootstrap_ci(vals: list[float], stat=statistics.mean, B: int = 4000,
                 seed: int = 11) -> tuple[float, float, float]:
    if not vals:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = [stat(arr[rng.integers(0, len(arr), len(arr))].tolist()) for _ in range(B)]
    boots.sort()
    return (round(stat(arr.tolist()), 4),
            round(boots[int(0.025 * B)], 4),
            round(boots[int(0.975 * B)], 4))


def analyze(recs: list[dict], eps: float = 1e-3, theta: float = 0.9) -> dict:
    ok = [r for r in recs if not r.get("error") and reconstruct_traj(r)]
    succ = [r for r in ok if r.get("Y") == 1]
    fail = [r for r in ok if r.get("Y") == 0]

    # ---------------- U1 过程熵轨迹 ----------------
    ent_bins: dict[float, list[float]] = defaultdict(list)
    for r in ok:
        n = max(1, r.get("n_steps", 1))
        for t, v in reconstruct_traj(r):
            b = round(min(1.0, t / n) * 4) / 4
            ent_bins[b].append(binary_entropy(v))
    u1 = []
    for b in sorted(ent_bins):
        pt, lo, hi = bootstrap_ci(ent_bins[b])
        u1.append({"pos": b, "n": len(ent_bins[b]), "mean_H": pt, "ci95": [lo, hi]})

    # ---------------- U2 脆弱成功 ----------------
    frag_flags = []
    frag_min_v = []
    for r in succ:
        vs = [v for _, v in reconstruct_traj(r)]
        is_frag = any(v < 1.0 - eps for v in vs)
        frag_flags.append(1.0 if is_frag else 0.0)
        if is_frag:
            frag_min_v.append(min(vs))
    fr, flo, fhi = bootstrap_ci(frag_flags)

    # ---------------- U3 收敛点 ----------------
    conv = []
    for r in ok:
        tr = reconstruct_traj(r)
        n = max(1, r.get("n_steps", 1))
        last_below = None
        for t, v in tr:
            if v < theta:
                last_below = t
        # 末次低于阈值之后的第一个点即收敛点；若从未低于则 0
        if last_below is None:
            conv.append(0.0)
        else:
            after = [t for t, _ in tr if t > last_below]
            conv.append((after[0] if after else n) / n)
    cpt, clo, chi = bootstrap_ci(conv)

    # ---------------- 归因分层（成功 case 同样归因）----------------
    def top_step(r: dict, mode: str):
        segs = [s for s in r.get("causal_segments", []) if s.get("delta") is not None]
        if not segs:
            return None
        s = (min(segs, key=lambda x: x["delta"]) if mode == "min"
             else max(segs, key=lambda x: x["delta"]))
        return s

    strat = {}
    for name, group, mode in (("failed", fail, "min"), ("succeeded", succ, "max")):
        tops = [top_step(r, mode) for r in group]
        tops = [s for s in tops if s]
        deltas = [s["delta"] for s in tops]
        poss = [s["pos"] for s in tops]
        nz = [d for d in deltas if abs(d) > 1e-9]
        dpt, dlo, dhi = bootstrap_ci(deltas) if deltas else (0, 0, 0)
        ppt, plo, phi = bootstrap_ci(poss) if poss else (0, 0, 0)
        strat[name] = {
            "n": len(group),
            "n_with_segments": len(tops),
            "n_nonzero_top": len(nz),
            "nonzero_frac": round(len(nz) / len(tops), 4) if tops else None,
            "mean_top_delta": dpt, "delta_ci95": [dlo, dhi],
            "mean_top_pos": ppt, "pos_ci95": [plo, phi],
            "mode": "argmin (most harmful step)" if mode == "min"
                    else "argmax (most decisive positive step)",
        }

    # v̂_t 取值分布（暴露 m 的粒度）
    allv = [v for r in ok for _, v in reconstruct_traj(r)]
    return {
        "n_episodes": len(ok),
        "n_succeeded": len(succ),
        "n_failed": len(fail),
        "U1_entropy_trajectory": u1,
        "U2_fragile_success": {
            "rate": fr, "ci95": [flo, fhi],
            "n_fragile": int(sum(frag_flags)), "n_success": len(succ),
            "mean_min_v_among_fragile": (round(statistics.mean(frag_min_v), 4)
                                        if frag_min_v else None),
            "definition": f"Y=1 and exists t with v̂_t < 1-{eps}",
        },
        "U3_convergence_point": {
            "mean_normalized": cpt, "ci95": [clo, chi], "theta": theta,
            "definition": "normalized position of first point after the last dip below theta",
        },
        "attribution_by_outcome": strat,
        "v_hat_value_counts": dict(sorted(Counter(round(v, 3) for v in allv).items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--out", default="pilot/runs/process_uncertainty.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.attr).read_text().splitlines() if l.strip()]
    res = analyze(recs, eps=args.eps, theta=args.theta)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"=== 过程不确定性（n={res['n_episodes']}: "
          f"成功 {res['n_succeeded']}, 失败 {res['n_failed']}）===")
    print("\nU1 熵轨迹 H(v̂_t):")
    for e in res["U1_entropy_trajectory"]:
        print(f"  pos={e['pos']:.2f}  n={e['n']:>3}  H={e['mean_H']:.4f}  CI={e['ci95']}")
    u2 = res["U2_fragile_success"]
    print(f"\nU2 脆弱成功率: {u2['rate']:.1%}  CI={u2['ci95']}  "
          f"({u2['n_fragile']}/{u2['n_success']})")
    print(f"   脆弱者的最低 v̂: {u2['mean_min_v_among_fragile']}")
    u3 = res["U3_convergence_point"]
    print(f"\nU3 收敛点(归一化): {u3['mean_normalized']}  CI={u3['ci95']}  θ={u3['theta']}")
    print("\n归因分层（成功 case 同样归因）:")
    for k, v in res["attribution_by_outcome"].items():
        print(f"  {k:<10} n={v['n']:>2}  非零top占比={v['nonzero_frac']}  "
              f"mean|top Δ|={v['mean_top_delta']} CI={v['delta_ci95']}  "
              f"步位={v['mean_top_pos']} CI={v['pos_ci95']}")
        print(f"             ({v['mode']})")
    print(f"\nv̂_t 取值档位数 {len(res['v_hat_value_counts'])} → m 的粒度限制")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
