"""汇总 V5 多任务熵扫描 → 验证"早高后低"熵衰减是否普遍（支撑混合估计器动机）。

读 pilot/runs/v5_scan/*/v5_report.json，把每个 fork 点的 (归一步位, 熵, positivity)
聚合，输出：
  - 每任务的 熵 vs 步位 序列
  - 跨任务：高熵步占比、熵与步位的 Spearman 相关（负相关=早高后低）
  - positivity 成立的步位分布（高熵步集中在早期？）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman 秩相关，无 scipy。负值 = 步位越晚熵越低（早高后低）。"""
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="pilot/runs/v5_scan")
    ap.add_argument("--extra", nargs="*", default=[], help="额外的单任务 v5 report 路径")
    args = ap.parse_args()

    reports = list(Path(args.scan_dir).glob("*/v5_report.json"))
    reports += [Path(p) for p in args.extra if Path(p).exists()]

    all_pos: list[float] = []
    all_H: list[float] = []
    per_task = []
    n_high = n_total = 0
    high_positions: list[float] = []

    for rp in reports:
        d = json.loads(rp.read_text())
        n_steps = d.get("rollout_n_steps", 0)
        seq = []
        for s in d.get("per_step", []):
            if s.get("n_samples", 0) < 2:
                continue
            pos = s["fork_step"] / n_steps if n_steps else 0
            H = s["entropy_nats"]
            seq.append({"pos": round(pos, 3), "step": s["fork_step"],
                        "H": H, "high": s["positivity_ok"]})
            all_pos.append(pos)
            all_H.append(H)
            n_total += 1
            if s["positivity_ok"]:
                n_high += 1
                high_positions.append(pos)
        per_task.append({"instance": d["instance"], "n_steps": n_steps, "seq": seq})

    rho = spearman(all_pos, all_H)
    result = {
        "n_tasks": len(reports),
        "n_forkpoints": n_total,
        "high_entropy_fraction": round(n_high / n_total, 3) if n_total else None,
        "spearman_pos_vs_entropy": round(rho, 3),
        "interpretation": (
            "负相关支持'早高后低'：步位越晚熵越低"
            if rho < -0.2 else
            "相关弱：熵衰减非单调，拐点因任务而异" if abs(rho) <= 0.2 else
            "正相关：与预期相反，需检查"
        ),
        "high_entropy_position_mean": (
            round(sum(high_positions) / len(high_positions), 3) if high_positions else None),
        "per_task": per_task,
    }
    out = Path(args.scan_dir) / "entropy_scan_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== V5 多任务熵扫描汇总 ===")
    print(f"任务数 {result['n_tasks']}  fork点数 {result['n_forkpoints']}")
    print(f"高熵步占比: {result['high_entropy_fraction']}")
    print(f"熵 vs 步位 Spearman: {result['spearman_pos_vs_entropy']}  → {result['interpretation']}")
    print(f"高熵步平均步位: {result['high_entropy_position_mean']} (越小=越集中在早期)")
    print("\n各任务 熵-步位序列:")
    for t in per_task:
        seq_str = " ".join(f"{s['pos']:.2f}:{'H' if s['high'] else '.'}({s['H']:.2f})" for s in t["seq"])
        print(f"  {t['instance']} (n={t['n_steps']}): {seq_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
