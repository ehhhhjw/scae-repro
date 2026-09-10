"""R1+R2：三态记账 + manipulation check。零 API，只解析已有 rollout。

痛点
----
现状用单个布尔 `degenerate`（Δ 全 0）把两种语义完全不同的情况混同：
  (a) 干预**无法施加/无法估计** → 应出分母
  (b) 干预**成功施加但无因果效应** → 应入分母且记零效应
导致 `NNondegen=54/70` 无法解释。

RFEval 的 contrast precondition delta + contrast coverage 指出正确记账方式：
delta=0 出分母并并列上报 coverage；kappa=0 入分母记为失败。本模块实现其等价物。

E1 首条实证支持这个区分：django-12155 四臂各 m=4 次全部 recall=0.5、方差为零，
而注入的观测经核验确为真实文件内容（4884 字符）。即 delta!=0 但 kappa=0 ——
失败不是"没看到正确文件"，而是停止/报告策略决定了不报。

三态
----
  INESTIMABLE  无法施加有效干预 / MC1 未通过 → 出分母，计入 coverage
  NO_EFFECT    干预可施加且 MC1 通过，但 Δ_t = 0 → 入分母，零效应
  EFFECT       干预可施加且 Δ_t != 0 → 入分母
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402

_DEFS = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]{3,})")
_IDENTS = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{5,})\b")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 区间 —— 小样本比例的正确区间（R4）。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def trackable_tokens(step) -> set[str]:
    """该步观测里可用于 MC1 追踪的 token。

    优先用 def/class 定义名；若无（例如 sed 只取到 import 区），
    退化为长标识符（>=6 字符），仍具区分力。
    """
    out = step.output_text or ""
    defs = set(_DEFS.findall(out))
    if defs:
        return defs
    idents = Counter(_IDENTS.findall(out))
    return {w for w, c in idents.most_common(12) if c <= 3}


def mc1(steps, t: int) -> dict:
    """MC1：第 t 步观测的内容是否被后续步骤读取/引用。t 为 1-based。"""
    if t < 1 or t > len(steps):
        return {"ok": None, "why": "t 越界"}
    toks = trackable_tokens(steps[t - 1])
    if not toks:
        return {"ok": None, "why": "该步观测无可追踪 token"}
    later = " ".join((s.args_raw or "") + " " + (s.output_text or "")
                     for s in steps[t:])
    hit = sum(1 for x in toks if x in later)
    return {"ok": hit > 0, "n_tokens": len(toks), "n_hit": hit, "why": ""}


def classify(rec: dict, steps) -> list[dict]:
    out = []
    for s in rec.get("causal_segments", []):
        t, d = s.get("t"), s.get("delta")
        if t is None or d is None:
            out.append({"t": t, "state": "INESTIMABLE", "why": "缺 t/delta"})
            continue
        c = mc1(steps, t)
        if c["ok"] is None:
            out.append({"t": t, "state": "INESTIMABLE", "delta": d,
                        "why": "MC1 不可判定：" + c["why"]})
        elif not c["ok"]:
            out.append({"t": t, "state": "INESTIMABLE", "delta": d,
                        "n_tokens": c["n_tokens"],
                        "why": "MC1 未通过（early-assimilation failure）"})
        else:
            out.append({"t": t,
                        "state": "NO_EFFECT" if abs(d) < 1e-9 else "EFFECT",
                        "delta": d, "n_tokens": c["n_tokens"],
                        "n_hit": c["n_hit"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--locate", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/estimability.json")
    args = ap.parse_args()

    loc = {}
    for line in Path(args.locate).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            loc[r["instance_id"]] = r
    attr = [json.loads(l) for l in Path(args.attr).read_text().splitlines() if l.strip()]
    attr = [r for r in attr if not r.get("error")]

    states = Counter()
    by_outcome = {1: Counter(), 0: Counter()}
    early_fail = undecidable = 0
    rows = []
    for r in attr:
        lrec = loc.get(r["instance_id"])
        if not lrec or not lrec.get("rollout"):
            continue
        cands = list(Path(args.runs_root).glob(f"shard*/{lrec['rollout']}"))
        if not cands:
            continue
        try:
            steps = R.steps(R.load(cands[0]))
        except (OSError, ValueError):
            continue
        segs = classify(r, steps)
        y = r.get("Y")
        for s in segs:
            states[s["state"]] += 1
            if y in by_outcome:
                by_outcome[y][s["state"]] += 1
            why = s.get("why", "")
            if "early-assimilation" in why:
                early_fail += 1
            elif "不可判定" in why:
                undecidable += 1
        rows.append({"instance_id": r["instance_id"], "Y": y, "segments": segs})

    tot = sum(states.values())
    est = states["NO_EFFECT"] + states["EFFECT"]
    p, lo, hi = wilson(est, tot)
    pe, ple, phe = wilson(states["EFFECT"], est) if est else (0, 0, 0)

    res = {
        "n_episodes": len(rows), "n_segments": tot,
        "state_counts": dict(states),
        "estimability_coverage": {"point": p, "wilson_ci95": [lo, hi],
                                  "n_estimable": est, "n_total": tot},
        "effect_rate_among_estimable": {"point": pe, "wilson_ci95": [ple, phe],
                                        "n_effect": states["EFFECT"],
                                        "n_estimable": est},
        "inestimable_breakdown": {"early_assimilation_failure": early_fail,
                                  "mc1_undecidable": undecidable},
        "by_outcome": {("succeeded" if k == 1 else "failed"): dict(v)
                       for k, v in by_outcome.items()},
        "note": ("RFEval 式记账：INESTIMABLE 出分母并上报 coverage；"
                 "NO_EFFECT 入分母记零效应。旧 degenerate 字段把两者混同。"),
        "episodes": rows,
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"=== 三态记账（{len(rows)} episode, {tot} 段）===")
    for k in ("EFFECT", "NO_EFFECT", "INESTIMABLE"):
        v = states[k]
        print(f"  {k:<12} {v:>4}  {v/tot:.1%}" if tot else f"  {k}: 0")
    print(f"\n可估计覆盖率 {p:.1%}  Wilson CI {[lo, hi]}  ({est}/{tot})")
    print(f"可估计段中有效应率 {pe:.1%}  Wilson CI {[ple, phe]}")
    print(f"INESTIMABLE 构成: early-assimilation {early_fail}, 不可判定 {undecidable}")
    print("\n按结果分层:")
    for k, v in res["by_outcome"].items():
        print(f"  {k:<10} {dict(v)}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
