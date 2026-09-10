"""V3k：用执行溯源图解释**因果归因**。零 API（消费已有的 full_attr 结果）。

可证伪的结构预测
----------------
outcome 是 gold 文件的 recall。若 trace 溯源图刻画了因果结构，则：

    因果效应 Δ_t 应集中在「把 gold 文件送进 agent 观测史」的那些步上。

具体地，把每一步按它与 gold 的溯源关系分三类：
  visit_gold   该步直接操作了某个 gold 文件
  mention_gold 该步的**输出**里提到了某个 gold 文件（agent 由此得知它存在）
  first_gold   该步是**首次**让某个 gold 文件进入观测史的步（溯源图上 gold 的来源）
  none         与 gold 无溯源关系

预测：|Δ_t| 在 first_gold / visit_gold 上显著大于 none。

为何这是有力的检验
------------------
它把**廉价的结构量**（溯源关系，读 rollout 即可算）与**昂贵的干预估计**
（fork-replay 的 Δ_t，每点 m=8 次续跑）对上。三档走向都有意义：

| 结果 | 含义 |
|---|---|
| Δ 集中在 gold 溯源步 | 溯源图**解释**了因果结构 → 可用结构量做廉价归因筛选 |
| Δ 与溯源无关 | 效应经由别的通道（如停止策略）传导 → 支持"必须干预"的主线 |
| Δ 反向集中 | 溯源关系构造有误，需回查 |

统计纪律
--------
基线用**闭式期望**而非采样（踩坑 #15）：对每个 instance，
"随机取一个估值点命中 gold 溯源步"的概率 = 该 instance 内 gold 溯源步占比，
再对 instance 平均。CI 用 instance 级 bootstrap。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.parsed_command import parse_step        # noqa: E402
from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.analyze.v3j_trace_graph import PATH_RE           # noqa: E402

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


def paired_ci(a: list[float], b: list[float], bnum: int = 2000) -> dict:
    if len(a) != len(b) or not a:
        return {"note": f"点集不匹配 {len(a)} vs {len(b)}"}
    rng = random.Random(SEED)
    pts = []
    for _ in range(bnum):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        pts.append(sum(a[i] - b[i] for i in idx) / len(idx))
    pts.sort()
    lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
    return {"diff": round(statistics.mean(a) - statistics.mean(b), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def _rankdata(v: list[float]) -> list[float]:
    """平均秩（处理并列）。"""
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


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


def _spearman_boot(xs: list[float], ys: list[float], b: int = 2000) -> dict:
    """Spearman rho + bootstrap CI。"""
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


def gold_relation(steps, gold: set[str]) -> dict[int, dict]:
    """每步与 gold 的溯源关系。返回 {step_idx(1-based): flags}。"""
    rel: dict[int, dict] = {}
    gold_seen: set[str] = set()          # 已进入观测史的 gold
    for s in steps:
        p = parse_step(s)
        obj = (p.path or p.obj or "").lstrip("./")
        visit = any(obj and (obj == gp or obj.endswith("/" + gp)
                             or gp.endswith("/" + obj)) for gp in gold)
        ment: set[str] = set()
        txt = s.output_text or ""
        if txt:
            found = {m.group(1) for m in PATH_RE.finditer(txt[:200_000])}
            for gp in gold:
                base = gp.split("/")[-1]
                if gp in found or any(f == gp or f.endswith("/" + gp)
                                      or f.endswith("/" + base) for f in found):
                    ment.add(gp)
        first = ment - gold_seen
        gold_seen |= ment
        rel[s.idx] = {"visit_gold": bool(visit), "mention_gold": bool(ment),
                      "first_gold": bool(first), "n_first": len(first),
                      "any_gold": bool(visit or ment)}
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3k_trace_causal.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    roll = {p.stem.split("__k")[0]: p
            for p in Path(args.runs_root).glob("shard*/rollouts/*.jsonl")}

    recs = []
    for f in sorted(Path(args.attr_dir).glob("shard*/attribution.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"载入 {len(recs)} 条归因记录")

    # 逐点：|Δ| 按溯源类别
    abs_by_cls: dict[str, list[float]] = defaultdict(list)
    # instance 级配对：instance 内 gold 溯源步的平均|Δ| vs 非 gold 步的平均|Δ|
    pair_gold, pair_none = [], []
    # top-1 是否落在 gold 溯源步（vs 闭式期望）
    top1_hit, top1_chance = [], []
    judge_hit, judge_chance = [], []
    n_used = n_skip = 0
    cls_counts: Counter = Counter()
    pos_delta: list[tuple[float, float]] = []      # (归一化位置, |Δ|) 供功效检验
    inst_means: list[float] = []

    for r in recs:
        iid = r["instance_id"]
        inst = insts.get(iid)
        segs = r.get("causal_segments") or []
        if inst is None or not inst.gold or len(segs) < 2 or iid not in roll:
            n_skip += 1
            continue
        try:
            steps = R.steps(R.load(roll[iid]))
        except (OSError, ValueError):
            n_skip += 1
            continue
        rel = gold_relation(steps, set(inst.gold))
        pts = [(s["t"], abs(float(s.get("delta") or 0.0))) for s in segs
               if s.get("t") in rel]
        if len(pts) < 2:
            n_skip += 1
            continue
        n_used += 1
        g_vals = [d for t, d in pts if rel[t]["any_gold"]]
        n_vals = [d for t, d in pts if not rel[t]["any_gold"]]
        nst = max(1, r.get("n_steps") or max(t for t, _ in pts))
        for t, d in pts:
            pos_delta.append((t / nst, d))
        inst_means.append(statistics.mean(d for _, d in pts))
        for t, d in pts:
            k = ("first_gold" if rel[t]["first_gold"] else
                 "visit_gold" if rel[t]["visit_gold"] else
                 "mention_gold" if rel[t]["mention_gold"] else "none")
            abs_by_cls[k].append(d)
            cls_counts[k] += 1
        if g_vals and n_vals:
            pair_gold.append(statistics.mean(g_vals))
            pair_none.append(statistics.mean(n_vals))
        # top-1：因果 vs judge
        share = sum(1 for t, _ in pts if rel[t]["any_gold"]) / len(pts)
        t1 = r.get("causal_top1_step")
        if t1 in rel:
            top1_hit.append(1.0 if rel[t1]["any_gold"] else 0.0)
            top1_chance.append(share)           # 闭式期望，不采样
        jt = r.get("judge_full_top1_step")
        if jt in rel:
            judge_hit.append(1.0 if rel[jt]["any_gold"] else 0.0)
            judge_chance.append(share)

    # ---- 功效检验：先证明 |Δ| 不是纯噪声，否则"不集中"只是检不出 ----
    # 踩坑 #17：负面结果发表前必须先证明测量工具本身有效。
    # 若 |Δ| 与位置等别的量有显著关系，说明它有可被检出的结构，
    # 那么"与 gold 溯源无关"才是实质结论而非功效不足。
    pos_pairs = [(p, d) for p, d in pos_delta]
    power = {}
    if len(pos_pairs) >= 20:
        xs = [p for p, _ in pos_pairs]
        ys = [d for _, d in pos_pairs]
        power["spearman_absdelta_vs_position"] = _spearman_boot(xs, ys)
    # 实例间异质性：各 instance 的 mean|Δ| 是否比"同一总体"预期更分散
    if len(inst_means) >= 10:
        power["between_instance_mean_absdelta"] = boot_ci(inst_means)
        power["between_instance_sd"] = round(statistics.stdev(inst_means), 4)
    power["nonzero_delta_share"] = boot_ci(
        [1.0 if d > 1e-9 else 0.0 for _, d in pos_pairs])
    # 功效更高的检验：|Δ|>0 这个二元事件是否与 gold 溯源有关
    # （比比较均值敏感，因为不受少数大 Δ 主导）
    nz_gold = [1.0 if d > 1e-9 else 0.0 for d in abs_by_cls["first_gold"]
               + abs_by_cls["visit_gold"] + abs_by_cls["mention_gold"]]
    nz_none = [1.0 if d > 1e-9 else 0.0 for d in abs_by_cls["none"]]
    power["nonzero_rate_gold_provenance"] = boot_ci(nz_gold)
    power["nonzero_rate_none"] = boot_ci(nz_none)
    if nz_gold and nz_none:
        # 非配对两比例差的 bootstrap CI
        rng2 = random.Random(SEED)
        dd = []
        for _ in range(2000):
            a = sum(nz_gold[rng2.randrange(len(nz_gold))]
                    for _ in range(len(nz_gold))) / len(nz_gold)
            b2 = sum(nz_none[rng2.randrange(len(nz_none))]
                     for _ in range(len(nz_none))) / len(nz_none)
            dd.append(a - b2)
        dd.sort()
        lo2, hi2 = dd[50], dd[1949]
        power["nonzero_rate_diff_gold_minus_none"] = {
            "diff": round(statistics.mean(nz_gold) - statistics.mean(nz_none), 4),
            "ci95": [round(lo2, 4), round(hi2, 4)],
            "significant": bool(lo2 > 0 or hi2 < 0)}

    res = {
        "n_records": len(recs), "n_used": n_used, "n_skipped": n_skip,
        "POWER_CHECK": power,
        "step_class_counts": dict(cls_counts),
        "abs_delta_by_class": {k: boot_ci(v) for k, v in abs_by_cls.items()},
        "within_instance_paired": {
            "gold_provenance_steps": boot_ci(pair_gold),
            "non_gold_steps": boot_ci(pair_none),
            "diff": paired_ci(pair_gold, pair_none),
            "n_instances": len(pair_gold),
        },
        "causal_top1_lands_on_gold_provenance": {
            "observed": boot_ci(top1_hit),
            "closed_form_chance": boot_ci(top1_chance),
            "diff": paired_ci(top1_hit, top1_chance),
        },
        "judge_top1_lands_on_gold_provenance": {
            "observed": boot_ci(judge_hit),
            "closed_form_chance": boot_ci(judge_chance),
            "diff": paired_ci(judge_hit, judge_chance),
        },
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n可用 {n_used} 个 instance（跳过 {n_skip}）")
    pw = res["POWER_CHECK"]
    print(f"\n=== 功效检验：|Δ| 是否具有可检出的结构（先于任何负面结论）===")
    sp = pw.get("spearman_absdelta_vs_position")
    if sp:
        print(f"  |Δ| ~ 归一化步位  Spearman rho {sp['rho']:+.4f} "
              f"CI{sp['ci95']}  n={sp['n']}  "
              f"{'显著' if sp['significant'] else '不显著'}")
    nz = pw.get("nonzero_delta_share")
    if nz and "note" not in nz:
        print(f"  |Δ|>0 的估值点占比 {nz['mean']:.4f} CI{nz['ci95']}")
    for k in ("nonzero_rate_gold_provenance", "nonzero_rate_none"):
        v = pw.get(k)
        if v and "note" not in v:
            print(f"  {k:<32} {v['mean']:.4f} CI{v['ci95']} (n={v['n']})")
    dv = pw.get("nonzero_rate_diff_gold_minus_none")
    if dv:
        print(f"  |Δ|>0 率之差(gold−none) {dv['diff']:+.4f} "
              f"CI[{dv['ci95'][0]:+.4f}, {dv['ci95'][1]:+.4f}]  "
              f"{'显著' if dv['significant'] else '不显著'}")
    if "between_instance_sd" in pw:
        print(f"  instance 间 mean|Δ| 的标准差 {pw['between_instance_sd']}"
              f"（均值 {pw['between_instance_mean_absdelta']['mean']}）")

    print(f"\n=== 各类步的 |Δ|（逐点）===")
    print(f"{'类别':<16}{'n':>6}{'mean|Δ|':>10}   CI")
    for k in ("first_gold", "visit_gold", "mention_gold", "none"):
        v = res["abs_delta_by_class"].get(k)
        if not v or "note" in v:
            print(f"{k:<16} 无样本")
            continue
        print(f"{k:<16}{v['n']:>6}{v['mean']:>10.4f}   CI{v['ci95']}")

    p = res["within_instance_paired"]
    print(f"\n=== instance 内配对：gold 溯源步 vs 非 gold 步（n={p['n_instances']}）===")
    if "note" not in p["gold_provenance_steps"]:
        print(f"  gold 溯源步 mean|Δ| {p['gold_provenance_steps']['mean']:.4f} "
              f"CI{p['gold_provenance_steps']['ci95']}")
        print(f"  非 gold 步   mean|Δ| {p['non_gold_steps']['mean']:.4f} "
              f"CI{p['non_gold_steps']['ci95']}")
        d = p["diff"]
        print(f"  差值 {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
              f"  {'显著' if d['significant'] else '不显著'}")

    for key, label in (("causal_top1_lands_on_gold_provenance", "因果 top-1"),
                       ("judge_top1_lands_on_gold_provenance", "LLM judge top-1")):
        b = res[key]
        if "note" in b["observed"]:
            continue
        d = b["diff"]
        print(f"\n=== {label} 落在 gold 溯源步的比例（闭式期望为基线）===")
        print(f"  观测 {b['observed']['mean']:.4f} CI{b['observed']['ci95']}   "
              f"闭式期望 {b['closed_form_chance']['mean']:.4f} "
              f"CI{b['closed_form_chance']['ci95']}")
        print(f"  差值 {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
              f"  {'显著' if d['significant'] else '不显著'}")

    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
