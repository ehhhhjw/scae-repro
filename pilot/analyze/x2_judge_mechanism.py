"""X2：LLM judge 的选择能否被一条**零 LLM 的结构规则**复现。零 API。

动机
----
V3k 发现 judge top-1 有 90.0% 落在"与 gold 有溯源关系"的步上（因果 top-1 仅 28.6%）。
若一条平凡结构规则就能复现 judge 的选择，则**观测型过程归因未超出结构信息**，
这是对该类方法的强否证，且零成本。

必须先修的选择空间不匹配
------------------------
V3k 对 judge 与因果 top-1 用了**同一个**基线（估值点中 gold 溯源步的占比 36.2%）。
但 judge 读整条 trace，从**全部步**里选；因果 top-1 只在 ~5 个**等距估值点**里选。
两者选择空间不同，基线必须各算：

    judge 的基线 = gold 溯源步 / 全部步
    因果的基线   = gold 溯源步 / 估值点

本脚本并列上报两个空间，并给出**限制到共同空间**的公平对比。

候选规则（全部零 LLM）
----------------------
  R1 first_gold      首次让 gold 进入观测史的步
  R2 visit_gold      首个直接操作 gold 文件的步
  R3 last_gold       最后一个与 gold 有溯源关系的步（因 judge 已知偏后 A12）
  R4 last_step       最后一步（纯位置基线：检验"偏后"本身能否解释 judge）
  R5 max_gold        提到 gold 最多的步
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                          # noqa: E402
from pilot.probes import task as T                              # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402

SEED = 20260814
RULES = ("R1_first_gold", "R2_visit_gold", "R3_last_gold",
         "R4_last_step", "R5_max_gold")


def rule_pick(rel: dict[int, dict], all_steps: list[int]) -> dict[str, int | None]:
    """各规则的 top-1 选择（步号）。"""
    gold_steps = [t for t in all_steps if rel[t]["any_gold"]]
    firsts = [t for t in all_steps if rel[t]["first_gold"]]
    visits = [t for t in all_steps if rel[t]["visit_gold"]]
    return {
        "R1_first_gold": firsts[0] if firsts else None,
        "R2_visit_gold": visits[0] if visits else None,
        "R3_last_gold": gold_steps[-1] if gold_steps else None,
        "R4_last_step": all_steps[-1] if all_steps else None,
        "R5_max_gold": (max(all_steps, key=lambda t: rel[t]["n_first"])
                        if all_steps else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x2_judge_mechanism.json")
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

    # 选择空间诊断
    judge_in_estpts = []
    share_allsteps, share_estpts = [], []
    judge_hit_all, judge_hit_est = [], []
    causal_hit_est = []
    # 规则一致率
    agree = {r: [] for r in RULES}
    agree_chance = {r: [] for r in RULES}
    # 公平对比（限制到 judge 落在估值点内的 instance）
    fair_judge, fair_causal = [], []
    n_used = 0
    judge_pos, causal_pos = [], []
    residual_when_rule_misses: Counter = Counter()
    # 类内位置：judge 在 gold 溯源步**之中**的相对位置（1=最靠后），闭式零假设 0.5
    within_gold_pos, within_gold_null = [], []
    n_gold_steps = []

    for rec in recs:
        iid = rec["instance_id"]
        inst = insts.get(iid)
        segs = rec.get("causal_segments") or []
        if inst is None or not inst.gold or iid not in roll or not segs:
            continue
        try:
            steps = R.steps(R.load(roll[iid]))
        except (OSError, ValueError):
            continue
        rel = gold_relation(steps, set(inst.gold))
        all_steps = sorted(rel)
        est_pts = sorted({s["t"] for s in segs if s.get("t") in rel})
        if len(all_steps) < 3 or len(est_pts) < 2:
            continue
        n_used += 1

        jt = rec.get("judge_full_top1_step")
        ct = rec.get("causal_top1_step")
        s_all = sum(1 for t in all_steps if rel[t]["any_gold"]) / len(all_steps)
        s_est = sum(1 for t in est_pts if rel[t]["any_gold"]) / len(est_pts)
        share_allsteps.append(s_all)
        share_estpts.append(s_est)

        if jt in rel:
            judge_in_estpts.append(1.0 if jt in est_pts else 0.0)
            judge_hit_all.append(1.0 if rel[jt]["any_gold"] else 0.0)
            judge_pos.append(jt / len(all_steps))
        if ct in rel:
            causal_hit_est.append(1.0 if rel[ct]["any_gold"] else 0.0)
            causal_pos.append(ct / len(all_steps))
        if jt in est_pts and ct in rel:
            fair_judge.append(1.0 if rel[jt]["any_gold"] else 0.0)
            fair_causal.append(1.0 if rel[ct]["any_gold"] else 0.0)

        # 类内位置：给定 judge 挑了 gold 溯源步，它在这些步里排第几（从后数）
        gsteps = [t for t in all_steps if rel[t]["any_gold"]]
        n_gold_steps.append(len(gsteps))
        if jt in rel and rel.get(jt, {}).get("any_gold") and len(gsteps) >= 2:
            # 归一化"靠后程度"：1.0=最后一个 gold 步，0.0=第一个
            r_idx = gsteps.index(jt)
            within_gold_pos.append(r_idx / (len(gsteps) - 1))
            within_gold_null.append(0.5)          # 均匀期望，闭式

        # 规则 vs judge 的一致率；chance = 1/n_steps（随机规则命中同一步）
        picks = rule_pick(rel, all_steps)
        for r in RULES:
            p = picks[r]
            if p is None or jt not in rel:
                continue
            agree[r].append(1.0 if p == jt else 0.0)
            agree_chance[r].append(1.0 / len(all_steps))
        # 残差：R3 没命中时 judge 选了什么样的步
        if jt in rel and picks["R3_last_gold"] is not None and picks["R3_last_gold"] != jt:
            residual_when_rule_misses[
                "gold" if rel[jt]["any_gold"] else "non_gold"] += 1

    res = {
        "n_instances": n_used,
        "SELECTION_SPACE": {
            "note": "judge 从全部步选，因果 top-1 只从等距估值点选 → 基线必须各算",
            "judge_pick_inside_estimation_points": boot_ci(judge_in_estpts),
            "gold_provenance_share_of_all_steps": boot_ci(share_allsteps),
            "gold_provenance_share_of_estimation_points": boot_ci(share_estpts),
        },
        "judge_vs_own_baseline": {
            "observed": boot_ci(judge_hit_all),
            "correct_baseline_all_steps": boot_ci(share_allsteps),
            "diff": paired_ci(judge_hit_all, share_allsteps[:len(judge_hit_all)]),
        },
        "causal_vs_own_baseline": {
            "observed": boot_ci(causal_hit_est),
            "correct_baseline_estimation_points": boot_ci(share_estpts),
            "diff": paired_ci(causal_hit_est, share_estpts[:len(causal_hit_est)]),
        },
        "fair_comparison_same_space": {
            "note": "仅取 judge 的选择恰落在估值点内的 instance",
            "n": len(fair_judge),
            "judge": boot_ci(fair_judge), "causal": boot_ci(fair_causal),
            "diff": paired_ci(fair_judge, fair_causal),
        },
        "position": {"judge_mean_pos": boot_ci(judge_pos),
                     "causal_mean_pos": boot_ci(causal_pos)},
        "RULE_AGREEMENT_WITH_JUDGE": {
            r: {"agreement": boot_ci(agree[r]),
                "closed_form_chance": boot_ci(agree_chance[r]),
                "diff": paired_ci(agree[r], agree_chance[r])}
            for r in RULES},
        "residual_when_R3_misses": dict(residual_when_rule_misses),
        "WITHIN_CLASS_position": {
            "note": "给定 judge 挑了 gold 溯源步，它在这些步中的归一化靠后程度"
                    "（1=最后一个 gold 步）；闭式零假设 0.5",
            "observed": boot_ci(within_gold_pos),
            "uniform_null": 0.5,
            "diff": paired_ci(within_gold_pos, within_gold_null),
            "mean_n_gold_steps_per_instance": boot_ci(
                [float(x) for x in n_gold_steps]),
            "uniform_within_class_agreement": boot_ci(
                [1.0 / x for x in n_gold_steps if x > 0]),
        },
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    ss = res["SELECTION_SPACE"]
    print(f"\n可用 {n_used} instance")
    print(f"\n=== 选择空间诊断（V3k 基线错配的修正）===")
    print(f"  judge 的选择落在估值点内的比例 "
          f"{ss['judge_pick_inside_estimation_points']['mean']:.4f} "
          f"CI{ss['judge_pick_inside_estimation_points']['ci95']}")
    print(f"  gold 溯源步占**全部步** {ss['gold_provenance_share_of_all_steps']['mean']:.4f} "
          f"CI{ss['gold_provenance_share_of_all_steps']['ci95']}")
    print(f"  gold 溯源步占**估值点** {ss['gold_provenance_share_of_estimation_points']['mean']:.4f} "
          f"CI{ss['gold_provenance_share_of_estimation_points']['ci95']}")

    for key, label in (("judge_vs_own_baseline", "LLM judge（基线=全部步）"),
                       ("causal_vs_own_baseline", "因果 top-1（基线=估值点）")):
        b = res[key]
        d = b["diff"]
        obs = b["observed"]
        base = b.get("correct_baseline_all_steps") or b.get(
            "correct_baseline_estimation_points")
        if "note" in obs:
            continue
        print(f"\n=== {label} 落在 gold 溯源步 ===")
        print(f"  观测 {obs['mean']:.4f} CI{obs['ci95']}   "
              f"正确基线 {base['mean']:.4f} CI{base['ci95']}")
        if "note" not in d:
            print(f"  差值 {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'显著' if d['significant'] else '不显著'}")

    fc = res["fair_comparison_same_space"]
    if "note" not in fc["judge"]:
        print(f"\n=== 同空间公平对比（n={fc['n']}）===")
        print(f"  judge {fc['judge']['mean']:.4f}   因果 {fc['causal']['mean']:.4f}")
        d = fc["diff"]
        if "note" not in d:
            print(f"  差值 {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'显著' if d['significant'] else '不显著'}")

    print(f"\n=== 结构规则复现 judge 的 top-1（精确同步命中）===")
    print(f"{'规则':<18}{'一致率':>9}{'闭式chance':>12}   差值 CI")
    for r in RULES:
        b = res["RULE_AGREEMENT_WITH_JUDGE"][r]
        if "note" in b["agreement"]:
            print(f"{r:<18} 无样本")
            continue
        d = b["diff"]
        ds = (f"{d['diff']:+.4f} CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
              f"{'显著' if d['significant'] else '不显著'}") if "note" not in d else "—"
        print(f"{r:<18}{b['agreement']['mean']:>9.4f}"
              f"{b['closed_form_chance']['mean']:>12.4f}   {ds}")

    p = res["position"]
    print(f"\n  judge 平均相对位置 {p['judge_mean_pos']['mean']:.4f}   "
          f"因果 {p['causal_mean_pos']['mean']:.4f}")
    print(f"  R3 未命中时 judge 的选择类别 {res['residual_when_R3_misses']}")
    wc = res["WITHIN_CLASS_position"]
    if "note" not in wc["observed"]:
        d = wc["diff"]
        print(f"\n=== 类内位置：judge 在 gold 溯源步之中挑哪一个 ===")
        print(f"  归一化靠后程度 {wc['observed']['mean']:.4f} CI{wc['observed']['ci95']}"
              f"   均匀零假设 0.5")
        if "note" not in d:
            print(f"  差值 {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'显著' if d['significant'] else '不显著'}")
        print(f"  每 instance 平均 {wc['mean_n_gold_steps_per_instance']['mean']:.2f} "
              f"个 gold 溯源步 → 类内随机猜中率 "
              f"{wc['uniform_within_class_agreement']['mean']:.4f}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
