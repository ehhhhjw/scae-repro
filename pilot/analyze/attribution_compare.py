"""D1/D2/D3 分析 —— C3 攻击线的判定。

输入：一批 run（每条 rollout 已存），对每条同时算
  - 因果归因 top-1 步（causal.py，fork-resample 估 Δ_t）
  - full-trace judge top-1 步（judge.py）
  - prefix-only judge top-1 步（隔离碰撞偏差）

输出（预注册判据 PILOT §7）：
  D1 = judge-top1 ≠ causal-top1 的比例（bootstrap CI）
  D3 = judge-top1 步位 是否系统性晚于 causal-top1（Wilcoxon 单边）
       + full-trace vs prefix-only judge 步位差（碰撞偏差隔离）
  D2 = 数据流 dead/dfw 与 |Δ_t| 的关系（回归），验证"数据流依赖≠因果效应"

这个脚本消费 run_attribution.py 产出的 per-run 归因结果 jsonl。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def wilcoxon_signed_rank_one_sided(diffs: list[float]) -> dict:
    """单边 Wilcoxon signed-rank：H1 = 中位数 > 0（judge 步位晚于 causal）。

    无 scipy 依赖的最小实现；n 大时用正态近似。零差值剔除。
    """
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n < 6:
        return {"n": n, "note": "样本太少，不做正态近似", "W": None, "p_one_sided": None}
    ranks = _rank([abs(x) for x in d])
    w_plus = sum(r for x, r in zip(d, ranks) if x > 0)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mean_w) / sd_w if sd_w else 0.0
    p = 1 - _norm_cdf(z)          # 单边 H1: W+ 偏大
    return {"n": n, "W_plus": round(w_plus, 1), "z": round(z, 3),
            "p_one_sided": round(p, 4)}


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bootstrap_ci(vals: list[float], stat=statistics.mean, B: int = 2000,
                 seed_stream: list[int] | None = None) -> tuple[float, float, float]:
    """按样本重抽样的 bootstrap 95% CI。用固定 LCG 保证可复现（无 Math.random 依赖）。"""
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = stat(vals)
    state = 12345
    boots = []
    for _ in range(B):
        sample = []
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            sample.append(vals[state % n])
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    return (round(point, 4), round(lo, 4), round(hi, 4))


def analyze(attr_path: Path) -> dict:
    """attr_path: run_attribution 产出的 jsonl，每行含 causal / judge_full / judge_prefix。"""
    recs = [json.loads(l) for l in attr_path.read_text().splitlines() if l.strip()]
    usable = [r for r in recs if r.get("causal_top1_pos") is not None
              and r.get("judge_full_top1_pos") is not None]

    # D1: top-1 步是否不同
    disagree = [int(r["causal_top1_step"] != r["judge_full_top1_step"]) for r in usable]
    d1_point, d1_lo, d1_hi = bootstrap_ci([float(x) for x in disagree])

    # D3: judge 步位 − causal 步位（>0 = judge 更晚）
    pos_diff = [r["judge_full_top1_pos"] - r["causal_top1_pos"] for r in usable]
    d3 = wilcoxon_signed_rank_one_sided(pos_diff)

    # D3 隔离: full vs prefix judge 步位
    both = [r for r in usable if r.get("judge_prefix_top1_pos") is not None]
    fp_diff = [r["judge_full_top1_pos"] - r["judge_prefix_top1_pos"] for r in both]
    d3_iso = wilcoxon_signed_rank_one_sided(fp_diff) if len(both) >= 6 else {"n": len(both)}

    result = {
        "n_runs": len(recs),
        "n_usable": len(usable),
        "D1_disagreement_rate": {"point": d1_point, "ci95": [d1_lo, d1_hi],
                                 "pass": d1_lo > 0.15},
        "D3_judge_later_than_causal": {
            **d3,
            "mean_pos_diff": round(statistics.mean(pos_diff), 4) if pos_diff else None,
            "median_causal_pos": round(statistics.median(
                [r["causal_top1_pos"] for r in usable]), 4) if usable else None,
            "median_judge_pos": round(statistics.median(
                [r["judge_full_top1_pos"] for r in usable]), 4) if usable else None,
            "pass": (d3.get("p_one_sided") is not None and d3["p_one_sided"] < 0.05),
        },
        "D3_collider_isolation_full_vs_prefix": {
            **d3_iso,
            "mean_pos_diff": round(statistics.mean(fp_diff), 4) if fp_diff else None,
        },
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, help="run_attribution 产出的 jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = analyze(Path(args.attr))
    out = Path(args.out) if args.out else Path(args.attr).with_name("d1d3_report.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== D1/D3 分析 ===")
    print(f"可用 run: {res['n_usable']}/{res['n_runs']}")
    d1 = res["D1_disagreement_rate"]
    print(f"D1 归因分歧率: {d1['point']:.1%}  CI95={d1['ci95']}  "
          f"{'PASS' if d1['pass'] else 'FAIL'}(下界>15%)")
    d3 = res["D3_judge_later_than_causal"]
    print(f"D3 步位偏移: causal中位={d3['median_causal_pos']} judge中位={d3['median_judge_pos']} "
          f"Δ={d3['mean_pos_diff']} p={d3.get('p_one_sided')} "
          f"{'PASS' if d3['pass'] else 'FAIL'}(judge显著更晚)")
    iso = res["D3_collider_isolation_full_vs_prefix"]
    print(f"碰撞偏差隔离(full−prefix): Δ={iso.get('mean_pos_diff')} p={iso.get('p_one_sided')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
