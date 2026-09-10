"""Summarize the V5 multi-task entropy scan → check whether "high early, low late" entropy decay is general (supports the hybrid-estimator motivation).

Reads pilot/runs/v5_scan/*/v5_report.json, aggregates (normalized step position, entropy, positivity)
for every fork point, and outputs:
  - per-task entropy vs step-position series
  - across tasks: fraction of high-entropy steps, Spearman correlation of entropy with step position (negative = high early, low late)
  - the step-position distribution where positivity holds (are high-entropy steps concentrated early?)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, no scipy. Negative = the later the step position, the lower the entropy (high early, low late)."""
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
    ap.add_argument("--extra", nargs="*", default=[], help="extra single-task v5 report paths")
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
            "negative correlation supports 'high early, low late': the later the step position, the lower the entropy"
            if rho < -0.2 else
            "weak correlation: entropy decay is non-monotonic, the knee varies by task" if abs(rho) <= 0.2 else
            "positive correlation: opposite of expectation, needs checking"
        ),
        "high_entropy_position_mean": (
            round(sum(high_positions) / len(high_positions), 3) if high_positions else None),
        "per_task": per_task,
    }
    out = Path(args.scan_dir) / "entropy_scan_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== V5 multi-task entropy scan summary ===")
    print(f"tasks {result['n_tasks']}  fork points {result['n_forkpoints']}")
    print(f"high-entropy step fraction: {result['high_entropy_fraction']}")
    print(f"entropy vs step position Spearman: {result['spearman_pos_vs_entropy']}  → {result['interpretation']}")
    print(f"mean step position of high-entropy steps: {result['high_entropy_position_mean']} (smaller = more concentrated early)")
    print("\nper-task entropy-vs-position series:")
    for t in per_task:
        seq_str = " ".join(f"{s['pos']:.2f}:{'H' if s['high'] else '.'}({s['H']:.2f})" for s in t["seq"])
        print(f"  {t['instance']} (n={t['n_steps']}): {seq_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
