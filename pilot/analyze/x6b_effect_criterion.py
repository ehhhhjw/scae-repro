"""X6b：X6 的那个 5 倍差值是真信号，还是**二值判据造出来的**？零 API。

来由
----
X6 报告：低 H_tool 步的 EFFECT 率 0.013 vs 高 H_tool 步 0.066（5 倍），
方向与 X4 的零结果（H_tool ~ U^exec，rho=−0.088 n.s.）**相反**。
但 X6 的 EFFECT 是二值判据（逐点 p<0.05），全样本只有 7 个事件，
所以那个 5 倍完全可能是阈值化的产物。

本脚本的问题
------------
同一批点、同一个 H_tool，把响应变量从**二值 EFFECT** 换成**连续量**后，
关系是否还在？三个响应变量：

  binary   1[p < 0.05]        （X6 的判据）
  absdelta |Δ_t|              （效应量本身）
  neglogp  −log10(p)          （证据强度，不丢信息）

判据（预注册）
--------------
| 结果 | 结论 |
|---|---|
| 二值有关系、两个连续量都无 | 5 倍是**阈值化产物**，X6b 关闭该疑点，X4 的零结果不受挑战 |
| 连续量也有关系 | X4 的零结果需要重新审视（可能是它的聚合方式掩盖了步级关系） |
| 二值的关系本身经不起 bootstrap | 更简单的解释：n=7 的噪声 |

逐点 p 值的算法
---------------
每个估值段有 ys_t（m 次重放的 recall）与 ys_prev。检验 Δ≠0 用两样本置换检验
（把 2m 个值合并后重排，统计量 = 均值差的绝对值），与 X5 一致。
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
from pilot.analyze.v3k_trace_causal import _spearman_boot   # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci, paired_ci  # noqa: E402

SEED = 20260817
B_PERM = 2000


def perm_p(a: list[float], b: list[float], n_perm: int = B_PERM,
           seed: int = SEED) -> float:
    """两样本置换检验的双侧 p（统计量=|均值差|）。"""
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pool = a + b
    na = len(a)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:na]) - statistics.mean(pool[na:])) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime", default="pilot/runs/v4prime/attribution.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x6b_effect_criterion.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    # H_tool 协变量（全局工具转移模型）
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
        for x, y in zip(tools, tools[1:]):
            trans[x][y] += 1
    vocab = sorted({t for ts in step_tool.values() for t in ts})

    def h_tool(cur: str, alpha: float = 0.5) -> float:
        row = trans.get(cur) or Counter()
        tot = sum(row.values()) + alpha * len(vocab)
        ps = [(row.get(v, 0) + alpha) / tot for v in vocab]
        return -sum(q * math.log(q) for q in ps if q > 0)

    recs = [json.loads(l) for l in open(args.v4prime) if l.strip()]
    rows = []
    for r in recs:
        tools = step_tool.get(r["instance_id"]) or []
        for s in r.get("causal_segments") or []:
            yt, yp = s.get("ys_t"), s.get("ys_prev")
            if not yt or not yp or len(yt) < 2 or len(yp) < 2:
                continue
            t = s["t"]
            cur = tools[t - 1] if 1 <= t <= len(tools) else None
            if cur is None:
                continue
            a = [float(x) for x in yt]
            b = [float(x) for x in yp]
            pv = perm_p(a, b)
            rows.append({
                "iid": r["instance_id"], "t": t,
                "h_tool": h_tool(cur),
                "absdelta": abs(statistics.mean(a) - statistics.mean(b)),
                "p": pv,
                "neglogp": -math.log10(max(pv, 1e-4)),
                "binary": 1.0 if pv < args.alpha else 0.0,
            })
    n = len(rows)
    n_eff = int(sum(x["binary"] for x in rows))
    print(f"逐点样本 {n}，EFFECT (p<{args.alpha}) = {n_eff} ({n_eff/n:.1%})")

    # H_tool 中位数分层（复现 X6 的做法）
    hs = sorted(x["h_tool"] for x in rows)
    med = hs[len(hs) // 2]
    lo = [x for x in rows if x["h_tool"] <= med]
    hi = [x for x in rows if x["h_tool"] > med]

    res: dict = {
        "n_points": n, "n_effect": n_eff,
        "effect_rate": round(n_eff / n, 4), "alpha": args.alpha,
        "h_tool_median": round(med, 4),
        "STRATIFIED": {
            "low_H_n": len(lo), "high_H_n": len(hi),
            "low_H_effect_rate": round(statistics.mean(x["binary"] for x in lo), 4),
            "high_H_effect_rate": round(statistics.mean(x["binary"] for x in hi), 4),
        },
        "CONTINUOUS_VS_BINARY": {},
        "SENSITIVITY_TO_ALPHA": {},
    }
    # 二值差的 bootstrap（这是 X6 那个 5 倍的可靠性）
    rng = random.Random(SEED)
    d = []
    for _ in range(4000):
        sl = [lo[rng.randrange(len(lo))]["binary"] for _ in range(len(lo))]
        sh = [hi[rng.randrange(len(hi))]["binary"] for _ in range(len(hi))]
        d.append(statistics.mean(sh) - statistics.mean(sl))
    d.sort()
    res["STRATIFIED"]["binary_diff_high_minus_low"] = {
        "diff": round(statistics.mean(x["binary"] for x in hi)
                      - statistics.mean(x["binary"] for x in lo), 4),
        "ci95": [round(d[100], 4), round(d[3899], 4)],
        "significant": bool(d[100] > 0 or d[3899] < 0),
    }

    # 三个响应变量与 H_tool 的关系
    for resp in ("binary", "absdelta", "neglogp"):
        res["CONTINUOUS_VS_BINARY"][f"h_tool_vs_{resp}"] = _spearman_boot(
            [x["h_tool"] for x in rows], [x[resp] for x in rows])

    # 二值判据对阈值的敏感性
    for a in (0.01, 0.05, 0.10, 0.20):
        b_lo = statistics.mean(1.0 if x["p"] < a else 0.0 for x in lo)
        b_hi = statistics.mean(1.0 if x["p"] < a else 0.0 for x in hi)
        res["SENSITIVITY_TO_ALPHA"][f"alpha={a}"] = {
            "n_effect": int(sum(1 for x in rows if x["p"] < a)),
            "low_H_rate": round(b_lo, 4), "high_H_rate": round(b_hi, 4),
            "ratio": (round(b_hi / b_lo, 2) if b_lo > 0 else None),
        }

    cb = res["CONTINUOUS_VS_BINARY"]
    bin_sig = cb["h_tool_vs_binary"]["significant"]
    cont_sig = (cb["h_tool_vs_absdelta"]["significant"]
                or cb["h_tool_vs_neglogp"]["significant"])
    res["VERDICT"] = (
        "二值有关系而连续量没有 ⇒ X6 的 5 倍是**阈值化产物**，X4 的零结果不受挑战"
        if bin_sig and not cont_sig else
        "连续量也有关系 ⇒ 需重新审视 X4 的零结果"
        if cont_sig else
        "二值与连续量**都无**关系 ⇒ X6 的 5 倍是 n 过小的噪声，"
        "该疑点关闭且不构成对 X4 的挑战")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    st = res["STRATIFIED"]
    print(f"\n=== 复现 X6 的分层（H_tool 中位数 {med:.4f} 分割）===")
    print(f"  低 H_tool EFFECT 率 {st['low_H_effect_rate']} (n={st['low_H_n']})")
    print(f"  高 H_tool EFFECT 率 {st['high_H_effect_rate']} (n={st['high_H_n']})")
    b = st["binary_diff_high_minus_low"]
    print(f"  差值 {b['diff']:+.4f} CI{b['ci95']}  "
          f"{'显著' if b['significant'] else '**不显著**'}")

    print(f"\n=== 换成连续响应变量后（同一批 {n} 个点、同一个 H_tool）===")
    for k, v in cb.items():
        print(f"  {k:<22} rho={v['rho']:+.4f} CI{v['ci95']}  "
              f"{'显著' if v['significant'] else '不显著'}")

    print(f"\n=== 二值判据对阈值的敏感性 ===")
    print(f"{'alpha':<10}{'n_effect':>10}{'低H率':>9}{'高H率':>9}{'比值':>8}")
    for k, v in res["SENSITIVITY_TO_ALPHA"].items():
        print(f"{k:<10}{v['n_effect']:>10}{v['low_H_rate']:>9}{v['high_H_rate']:>9}"
              f"{str(v['ratio']):>8}")

    print(f"\n=== 裁决 ===\n  {res['VERDICT']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
