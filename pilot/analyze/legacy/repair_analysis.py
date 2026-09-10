"""E1 修复实验分析 —— 主判据为 Δmean recall，非二元 succ。

度量设计修正
------------
E1 前 4 条暴露：succ（要求 recall=1.0）在多文件任务上过于严苛。
django-13344 的 mean recall 从 0.5(none) 升到 0.667(top1) 是实质改善，
但 succ 全为 0，信号被完全丢弃。这重演了早先 outcome 口径的错误
（严格集合相等 vs recall），故主判据必须用连续量。

三个对比
--------
  主判据      mean(top1_missed) − mean(none)
  位置特异性  mean(top1_missed) − mean(rand_missed)
  内容特异性  mean(top1_missed) − mean(top1_irrelev)
另加两个 sham 对比（rand/irrelev vs none），用于判断"任何注入是否都有效"。

关键读法
--------
若三个干预臂彼此接近且都高于 none → 提升来自"注入任何内容"，归因特异性不成立。
若 irrelev 显著低于其他臂 → 内容对照臂在起作用，注入并非等价（sphinx-7590 即此例）。

统计方法（RFEval R4）
--------------------
配对置换检验（翻转符号，B=2e4），小样本下比 bootstrap 可靠且无分布假设；
差值另报 bootstrap CI 便于阅读。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARMS = ["none", "top1_missed", "rand_missed", "top1_irrelev"]


def paired_perm_test(a: list[float], b: list[float], B: int = 20000,
                     seed: int = 23) -> dict:
    """配对置换检验：随机翻转每对差值的符号，检验 mean(a−b) != 0。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 3:
        return {"n": n, "note": "样本不足"}
    obs = statistics.mean(d)
    rng = np.random.default_rng(seed)
    arr = np.asarray(d, dtype=float)
    cnt = sum(1 for _ in range(B)
              if abs(float((arr * rng.choice([-1.0, 1.0], size=n)).mean()))
              >= abs(obs) - 1e-12)
    return {"n": n, "mean_diff": round(obs, 4),
            "p_two_sided": round((cnt + 1) / (B + 1), 4)}


def boot_ci(vals: list[float], B: int = 10000, seed: int = 29) -> list[float]:
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    bs = sorted(float(arr[rng.integers(0, len(arr), len(arr))].mean())
                for _ in range(B))
    return [round(bs[int(0.025 * B)], 4), round(bs[int(0.975 * B)], 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", default="pilot/runs/repair2/repair.jsonl")
    ap.add_argument("--out", default="pilot/runs/repair_analysis.json")
    args = ap.parse_args()

    p = Path(args.repair)
    if not p.exists():
        print("尚无 repair 结果")
        return 0
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("error")
            and all(f"mean_{a}" in r for a in ARMS)]
    if len(recs) < 3:
        print(f"完整四臂记录仅 {len(recs)} 条，等更多数据")
        return 0

    means = {a: [r[f"mean_{a}"] for r in recs] for a in ARMS}
    succs = {a: [r.get(f"succ_{a}", 0.0) for r in recs] for a in ARMS}

    contrasts = {
        "main_top1_vs_none": ("top1_missed", "none"),
        "position_top1_vs_rand": ("top1_missed", "rand_missed"),
        "content_top1_vs_irrelev": ("top1_missed", "top1_irrelev"),
        "sham_rand_vs_none": ("rand_missed", "none"),
        "sham_irrelev_vs_none": ("top1_irrelev", "none"),
    }
    res = {
        "n_cases": len(recs),
        "arm_mean_recall": {a: round(statistics.mean(means[a]), 4) for a in ARMS},
        "arm_mean_recall_ci95": {a: boot_ci(means[a]) for a in ARMS},
        "arm_succ_rate": {a: round(statistics.mean(succs[a]), 4) for a in ARMS},
        "contrasts": {},
    }
    for name, (x, y) in contrasts.items():
        res["contrasts"][name] = {
            **paired_perm_test(means[x], means[y]),
            "diff_ci95": boot_ci([u - v for u, v in zip(means[x], means[y])]),
        }

    iv = {a: statistics.mean(means[a]) for a in
          ("top1_missed", "rand_missed", "top1_irrelev")}
    spread = round(max(iv.values()) - min(iv.values()), 4)
    base = statistics.mean(means["none"])
    res["intervention_arms_spread"] = spread
    res["fragile_failure_cases"] = sum(1 for v in means["none"] if v >= 0.999)
    if spread < 0.02 and iv["top1_missed"] > base + 0.02:
        diag = ("三干预臂差异 <0.02 且均高于 none → 提升来自『注入任何内容』，"
                "归因特异性未获支持")
    elif iv["top1_irrelev"] < min(iv["top1_missed"], iv["rand_missed"]) - 0.05:
        diag = ("无关内容臂显著更低 → 内容对照生效，注入并非等价；"
                "可进一步检验位置特异性")
    else:
        diag = "干预臂间存在差异，需更多样本判定特异性"
    res["diagnosis"] = diag
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"=== E1 修复实验（n={len(recs)}）===")
    print(f"{'臂':<16}{'mean recall':>13}{'CI95':>20}{'succ':>8}")
    for a in ARMS:
        print(f"{a:<16}{res['arm_mean_recall'][a]:>13}"
              f"{str(res['arm_mean_recall_ci95'][a]):>20}"
              f"{res['arm_succ_rate'][a]:>8}")
    print("\n对比（配对置换检验）:")
    for name, c in res["contrasts"].items():
        if "note" in c:
            print(f"  {name:<26} {c['note']}")
            continue
        sig = " *" if c["p_two_sided"] < 0.05 else ""
        print(f"  {name:<26} Δ={c['mean_diff']:>8} "
              f"CI={str(c['diff_ci95']):<18} p={c['p_two_sided']}{sig}")
    print(f"\n干预臂极差 {spread}")
    print(f"none 臂全成功(脆弱失败) {res['fragile_failure_cases']}/{len(recs)}")
    print(f"诊断: {diag}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
