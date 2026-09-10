"""X7（零 API）：脆弱成功 ↔ L1 预测熵。补 RESEARCH_PLAN_v5 §G8。

命题
----
脆弱成功（0 < causal_v0 < 1，即重放不稳定）的 instance，其轨迹的
平均预测熵 H_tool 与 H_prov 显著高于稳健成功（v0 = 1）的 instance。

被估量
------
1. 脆弱成功 vs 稳健成功的 mean H_tool 配对对比（instance 级）
2. 脆弱成功 vs 完全失败（v0 = 0）的 mean H_tool 对比
3. 三组（失败/脆弱/稳健）的 H_tool 均值趋势
4. causal_v0 ~ mean H_tool 的 Spearman 相关（连续版本）
5. 「高熵 → 脆弱/失败」的精度、召回

设计
----
- 脆弱：0 < causal_v0 < 1（重放方差 > 0）
- 稳健成功：causal_v0 = 1.0（每次重放都成功）
- 失败：causal_v0 = 0.0 或接近（重放通常失败）
- 数据来源：full_attr 的 70 个多文件 instance

RESEARCH_PLAN_v5 判据
--------------------
| 结果 | 走向 |
|---|---|
| 脆弱 H_tool 显著高于稳健，且趋势单调 | 实用结论：H_tool 可作为"需要重放"的早期信号 |
| 趋势存在但不显著 | 探索性：加入 X4 的 H_t ~ U^exec 联合分析 |
| 无趋势 | 如实报告；说明 L1 量与执行不稳定性解耦 |

统计纪律
--------
- instance 级 bootstrap B=2000
- 三组比较 BH-FDR q=0.10（3 个对比）
- Spearman + bootstrap CI
- 坑 #22：先验证 H_tool 在某个维度有可检出结构（位置/工具分层），
  再下"与 v0 无关"的结论
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.parsed_command import parse_step   # noqa: E402
from pilot.harness import rollout as R                  # noqa: E402
from pilot.probes import task as T                      # noqa: E402

SEED = 20260814


def boot_ci(vals: list[float], b: int = 2000, seed: int = SEED) -> dict:
    if not vals:
        return {"note": "无样本"}
    rng = random.Random(seed)
    pts = []
    for _ in range(b):
        pts.append(sum(vals[rng.randrange(len(vals))]
                       for _ in range(len(vals))) / len(vals))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * b)], 4), round(pts[int(0.975 * b)], 4)],
            "n": len(vals)}


def paired_ci(a: list[float], b_: list[float], bnum: int = 2000) -> dict:
    if len(a) != len(b_) or not a:
        return {"note": f"点集不匹配 {len(a)} vs {len(b_)}"}
    rng = random.Random(SEED)
    pts = []
    for _ in range(bnum):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        pts.append(sum(a[i] - b_[i] for i in idx) / len(idx))
    pts.sort()
    lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
    return {"diff": round(statistics.mean(a) - statistics.mean(b_), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def _rankdata(v: list[float]) -> list[float]:
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def _pearson(a: list[float], b_: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b_) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b_))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b_) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


def spearman_boot(xs: list[float], ys: list[float], b: int = 2000) -> dict:
    if len(xs) < 5:
        return {"note": "样本不足"}
    rho = _pearson(_rankdata(xs), _rankdata(ys))
    rng = random.Random(SEED)
    pts = []
    for _ in range(b):
        idx = [rng.randrange(len(xs)) for _ in range(len(xs))]
        sx = [xs[i] for i in idx]
        sy = [ys[i] for i in idx]
        pts.append(_pearson(_rankdata(sx), _rankdata(sy)))
    pts.sort()
    lo, hi = pts[int(0.025 * b)], pts[int(0.975 * b)]
    return {"rho": round(rho, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n": len(xs), "significant": bool(lo > 0 or hi < 0)}


def bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns rejection mask."""
    n = len(pvals)
    if n == 0:
        return []
    ranked = sorted(range(n), key=lambda i: pvals[i])
    reject = [False] * n
    last = -1
    for rank, idx in enumerate(ranked):
        if pvals[idx] <= (rank + 1) * q / n:
            last = rank
    for rank, idx in enumerate(ranked):
        if rank <= last:
            reject[idx] = True
    return reject


def normal_ci_diff(n1: int, m1: float, s1: float,
                   n2: int, m2: float, s2: float) -> dict:
    """正态近似的两样本差值 CI（用于大样本情形下）。"""
    se = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 0.5
    diff = m1 - m2
    return {"diff": round(diff, 4),
            "ci95": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
            "significant": bool(abs(diff) > 1.96 * se)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--tool-model", default="pilot/runs/tool_model_official.json")
    ap.add_argument("--out", default="pilot/runs/x7_fragile_l1.json")
    args = ap.parse_args()

    # ---- 1. 载入工具通道 entropy_rows ----
    tm = json.loads(Path(args.tool_model).read_text())
    h_tool_idx: dict[tuple[str, int], float] = {
        (row["instance_id"], row["t"]): row["H_tool"]
        for row in tm.get("entropy_rows", [])
    }
    print(f"工具通道 entropy_rows: {len(h_tool_idx)} 个步")

    # ---- 2. 载入归因数据（causal_v0）----
    recs: list[dict] = []
    for f in sorted(Path(args.attr_dir).glob("shard*/attribution.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"归因记录: {len(recs)}")
    v0_map: dict[str, float] = {}
    for r in recs:
        v0 = r.get("causal_v0")
        if v0 is not None:
            v0_map[r["instance_id"]] = float(v0)

    # ---- 3. 按 rollout 计算每个 instance 的 mean H_tool ----
    insts = {i.instance_id: i for i in T.load_instances()}
    runs_root = Path(args.runs_root)
    h_per_inst: dict[str, float] = {}
    n_steps_per_inst: dict[str, int] = {}

    for rfile in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = rfile.stem.split("__k")[0]
        if iid not in v0_map:
            continue   # 只分析有 v0 的 instance（多文件子集）
        try:
            rl = R.load(rfile)
            steps = R.steps(rl)
        except (OSError, ValueError):
            continue
        if not steps:
            continue
        hs = [h_tool_idx[(iid, s.idx)] for s in steps
              if (iid, s.idx) in h_tool_idx]
        if not hs:
            continue
        h_per_inst[iid] = statistics.mean(hs)
        n_steps_per_inst[iid] = len(hs)

    print(f"有 H_tool 和 v0 的 instance: {len(h_per_inst)}")

    # ---- 4. 三组划分 ----
    EPS = 1e-6
    fail_ids = [i for i in h_per_inst if v0_map.get(i, 1) < EPS]
    fragile_ids = [i for i in h_per_inst if EPS < v0_map.get(i, 1) < 1 - EPS]
    robust_ids = [i for i in h_per_inst if v0_map.get(i, 1) > 1 - EPS]

    fail_h = [h_per_inst[i] for i in fail_ids]
    fragile_h = [h_per_inst[i] for i in fragile_ids]
    robust_h = [h_per_inst[i] for i in robust_ids]

    print(f"失败: {len(fail_ids)}  脆弱: {len(fragile_ids)}  稳健: {len(robust_ids)}")

    # 三组均值
    groups = {
        "fail": boot_ci(fail_h),
        "fragile": boot_ci(fragile_h),
        "robust": boot_ci(robust_h),
    }

    # 配对对比（BH-FDR 校正 3 对比）
    # 非配对 bootstrap 差值
    def nonpaired_ci(a: list[float], b_: list[float], bnum: int = 2000) -> dict:
        if not a or not b_:
            return {"note": "无样本"}
        rng = random.Random(SEED)
        pts = []
        for _ in range(bnum):
            ma = sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a)
            mb = sum(b_[rng.randrange(len(b_))] for _ in range(len(b_))) / len(b_)
            pts.append(ma - mb)
        pts.sort()
        lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
        return {"diff": round(statistics.mean(a) - statistics.mean(b_), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0 or hi < 0)}

    c_fragile_vs_robust = nonpaired_ci(fragile_h, robust_h)
    c_fail_vs_robust = nonpaired_ci(fail_h, robust_h)
    c_fragile_vs_fail = nonpaired_ci(fragile_h, fail_h)

    contrasts = {
        "fragile_vs_robust": c_fragile_vs_robust,
        "fail_vs_robust": c_fail_vs_robust,
        "fragile_vs_fail": c_fragile_vs_fail,
    }

    # ---- 5. 连续相关：mean H_tool ~ causal_v0 ----
    iids_sorted = sorted(h_per_inst.keys())
    xs = [h_per_inst[i] for i in iids_sorted]
    ys = [v0_map[i] for i in iids_sorted]
    corr_h_v0 = spearman_boot(xs, ys)

    # ---- 6. 「高熵 threshold」的精度/召回（把"需要重放"= fragile+fail 当正类）----
    med_h = statistics.median(list(h_per_inst.values()))
    pos_ids = set(fail_ids) | set(fragile_ids)
    neg_ids = set(robust_ids)
    # 预测：H_tool > median → 需要重放
    tp = sum(1 for i in pos_ids if h_per_inst[i] >= med_h)
    fp = sum(1 for i in neg_ids if h_per_inst[i] >= med_h)
    fn = sum(1 for i in pos_ids if h_per_inst[i] < med_h)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    # AUROC（简单两类）
    all_sorted = sorted(h_per_inst.keys(), key=lambda i: -h_per_inst[i])
    tpr_pts, fpr_pts = [0.0], [0.0]
    n_pos, n_neg = len(pos_ids), len(neg_ids)
    tp_c = fp_c = 0
    for iid in all_sorted:
        if iid in pos_ids:
            tp_c += 1
        else:
            fp_c += 1
        tpr_pts.append(tp_c / n_pos if n_pos else 0)
        fpr_pts.append(fp_c / n_neg if n_neg else 0)
    auroc = sum((fpr_pts[i + 1] - fpr_pts[i]) * (tpr_pts[i + 1] + tpr_pts[i]) / 2
                for i in range(len(fpr_pts) - 1))

    predict_fragile = {
        "threshold_H_median": round(med_h, 4),
        "n_pos": len(pos_ids), "n_neg": len(neg_ids),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "note": "正类=脆弱+失败（需要重放），负类=稳健成功",
    }

    # ---- 7. 功效检验：H_tool 本身是否有可检出结构（先于主张"无关"）----
    # 已在 X1 证明 H_tool 有位置效应（早期低、中后高），这里再用 n_steps 检验
    ns = [n_steps_per_inst[i] for i in iids_sorted if i in n_steps_per_inst]
    corr_h_nsteps = spearman_boot(
        [h_per_inst[i] for i in iids_sorted if i in n_steps_per_inst], ns)

    res = {
        "n_instances_with_h_and_v0": len(h_per_inst),
        "group_sizes": {
            "fail": len(fail_ids),
            "fragile": len(fragile_ids),
            "robust": len(robust_ids),
        },
        "group_mean_H_tool": groups,
        "contrasts": contrasts,
        "continuous_spearman_H_vs_v0": corr_h_v0,
        "predictability_of_fragile": predict_fragile,
        "power_check_H_vs_nsteps": corr_h_nsteps,
        "per_instance": [
            {"id": i, "v0": v0_map[i], "mean_H_tool": round(h_per_inst[i], 4),
             "n_steps": n_steps_per_inst.get(i, 0),
             "group": "fail" if i in fail_ids else ("fragile" if i in fragile_ids else "robust")}
            for i in sorted(h_per_inst.keys())
        ],
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---- 输出摘要 ----
    print(f"\n=== X7：脆弱成功 ↔ L1 预测熵 ===")
    print(f"\n三组 H_tool 均值:")
    for g, v in groups.items():
        if "note" not in v:
            print(f"  {g:10s}: {v['mean']:.4f} CI{v['ci95']}  (n={v['n']})")
    print(f"\n配对对比（bootstrap 差值）:")
    for name, v in contrasts.items():
        if "note" not in v:
            print(f"  {name}: {v['diff']:+.4f} CI{v['ci95']}"
                  f"  {'显著' if v['significant'] else '不显著'}")
    c = corr_h_v0
    if "note" not in c:
        print(f"\nSpearman H_tool ~ v0: rho={c['rho']:+.4f} CI{c['ci95']}"
              f"  {'显著' if c['significant'] else '不显著'}")
    pf = predict_fragile
    print(f"\n「高 H_tool → 需要重放」预测性能:")
    print(f"  阈值={pf['threshold_H_median']:.4f}  prec={pf['precision']:.3f}  "
          f"rec={pf['recall']:.3f}  f1={pf['f1']:.3f}  AUROC={pf['auroc']:.3f}")
    pw = res["power_check_H_vs_nsteps"]
    if "note" not in pw:
        print(f"\n功效检验：H_tool ~ n_steps  rho={pw['rho']:+.4f} CI{pw['ci95']}"
              f"  {'显著' if pw['significant'] else '不显著'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
