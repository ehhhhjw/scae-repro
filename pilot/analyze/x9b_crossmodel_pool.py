"""X9b：把所有 judge 模型的结果并到一张表，回答"机制在几个 judge 上成立"。零 API。

为什么要单独做
--------------
X9 每跑一个模型只给"该模型 vs 原模型"的两两对比。但要主张的是
**"碰撞偏差是 full-trace judging 这一类做法的性质"**，需要的是跨多个 judge 的
一致性证据，而非若干条两两比较。本脚本给三样东西：

  1. 每个模型的位移（点估计 + CI + 符号检验），并列在同一张表
  2. **模型间异质性**：位移在模型间是否存在超出抽样误差的差异
     （对同一批 episode 做"模型"这一因子的置换检验）
  3. **内容 vs 机制的分裂**：把"位移"与"gold 溯源富集度"两列并排，
     显示前者一致、后者不一致

⚠️ 只在**所有模型共有的 episode** 上比较，否则模型间差异会与 episode 组成混淆。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402

SEED = 20260817
B_PERM = 4000


def sign_p(diffs: list[float]) -> float | None:
    nz = [d for d in diffs if d != 0]
    if not nz:
        return None
    k = sum(1 for d in nz if d > 0)
    return min(1.0, 2 * sum(comb(len(nz), i) for i in range(k, len(nz) + 1))
               / 2 ** len(nz))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="每个 judge 模型一个 x9 的 jsonl")
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x9b_crossmodel_pool.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    roll = {p.stem.split("__k")[0]: p
            for p in Path(args.runs_root).glob("shard*/rollouts/*.jsonl")}
    orig = {}
    for f in sorted(Path(args.attr_dir).glob("shard*/attribution.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                orig[r["instance_id"]] = r

    # 载入各模型
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in args.jsonl:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            f_, p_ = d.get("full") or {}, d.get("prefix") or {}
            if f_.get("parsed") and p_.get("parsed"):
                by_model[d["model"]][d["instance_id"]] = d
    models = sorted(by_model)
    print(f"载入 {len(models)} 个 judge 模型：{', '.join(models)}")
    for m in models:
        print(f"  {m}: {len(by_model[m])} 个成对 episode")

    # 只用所有模型共有的 episode，否则模型差异与 episode 组成混淆
    common = set.intersection(*[set(by_model[m]) for m in models])
    common &= set(orig) & set(roll)
    common = sorted(common)
    print(f"共有 episode（用于跨模型比较）：{len(common)}")
    if len(common) < 5:
        print("共有 episode 太少，无法比较")
        return 1

    # gold 溯源关系（每 episode 算一次）
    rel_cache: dict[str, tuple[dict, list[int], float]] = {}
    for iid in common:
        st = R.steps(R.load(roll[iid]))
        rel = gold_relation(st, set(insts[iid].gold))
        allst = sorted(rel)
        share = sum(1 for t in allst if rel[t]["any_gold"]) / len(allst)
        rel_cache[iid] = (rel, allst, share)

    rows: dict[str, dict[str, list[float]]] = {}
    for m in models:
        disp, enrich, base = [], [], []
        for iid in common:
            d = by_model[m][iid]
            disp.append(d["full"]["top1_pos"] - d["prefix"]["top1_pos"])
            rel, allst, share = rel_cache[iid]
            s = d["full"]["top1_step"]
            if s in rel:
                enrich.append(1.0 if rel[s]["any_gold"] else 0.0)
                base.append(share)
        rows[m] = {"disp": disp, "enrich": enrich, "base": base}
    # 原模型作为第四个 judge
    odisp, oenrich, obase = [], [], []
    for iid in common:
        r = orig[iid]
        odisp.append(r["judge_full_top1_pos"] - r["judge_prefix_top1_pos"])
        rel, allst, share = rel_cache[iid]
        s = r.get("judge_full_top1_step")
        if s in rel:
            oenrich.append(1.0 if rel[s]["any_gold"] else 0.0)
            obase.append(share)
    rows["gpt-5.5-0424-global (original)"] = {
        "disp": odisp, "enrich": oenrich, "base": obase}
    all_models = list(rows)

    # ---- 模型间异质性：置换"模型"标签（同一 episode 内重排）----
    # 对**两个**量都做：位移（期望一致）与富集度（期望异质）。
    # 只有"位移不异质 + 富集度异质"同时成立，才构成"机制一致、内容不一致"。
    rng = random.Random(SEED)

    def heterogeneity(key: str) -> dict:
        mats = [rows[m][key] for m in all_models]
        n = min(len(v) for v in mats)
        if n < 3:
            return {"note": "样本不足"}
        mats = [v[:n] for v in mats]
        obs = statistics.variance([statistics.mean(v) for v in mats])
        nl = []
        for _ in range(B_PERM):
            sh = [[0.0] * n for _ in all_models]
            for j in range(n):
                col = [mats[i][j] for i in range(len(all_models))]
                rng.shuffle(col)
                for i in range(len(all_models)):
                    sh[i][j] = col[i]
            nl.append(statistics.variance([statistics.mean(v) for v in sh]))
        nl.sort()
        pv = (sum(1 for v in nl if v >= obs) + 1) / (len(nl) + 1)
        return {"observed_variance": round(obs, 6),
                "null_mean": round(statistics.mean(nl), 6),
                "null_p975": round(nl[int(0.975 * len(nl))], 6),
                "perm_p": pv, "significant_heterogeneity": bool(pv < 0.05),
                "model_means": {m: round(statistics.mean(rows[m][key][:n]), 4)
                                for m in all_models}}

    het_disp = heterogeneity("disp")
    het_enr = heterogeneity("enrich")
    obs_var = het_disp.get("observed_variance", float("nan"))
    p_het = het_disp.get("perm_p", 1.0)
    nulls = [het_disp.get("null_mean", 0.0)]

    res = {
        "n_common_episodes": len(common),
        "models": all_models,
        "PER_MODEL": {
            m: {
                "displacement": boot_ci(rows[m]["disp"]),
                "n_later": sum(1 for d in rows[m]["disp"] if d > 0),
                "n_tied": sum(1 for d in rows[m]["disp"] if d == 0),
                "sign_p": sign_p(rows[m]["disp"]),
                "enrichment": boot_ci(rows[m]["enrich"]),
                "enrichment_baseline": boot_ci(rows[m]["base"]),
                "enrichment_diff": paired_ci(rows[m]["enrich"], rows[m]["base"]),
            } for m in all_models},
        "HETEROGENEITY": {
            "note": "置换同一 episode 内的模型标签；统计量=各模型均值的方差。"
                    "位移希望 p 大（机制一致），富集度希望 p 小（内容异质）。"
                    "⚠️ 位移那一侧是在'接受零假设'，故必须同时看效应量而非只看 p。",
            "displacement": het_disp,
            "enrichment": het_enr,
        },
    }
    n_pos = sum(1 for m in all_models
                if (res["PER_MODEL"][m]["displacement"]["ci95"][0] > 0))
    res["VERDICT"] = {
        "n_models_with_positive_displacement_CI": n_pos,
        "n_models": len(all_models),
        "n_models_with_significant_enrichment": sum(
            1 for m in all_models
            if res["PER_MODEL"][m]["enrichment_diff"].get("significant")
            and res["PER_MODEL"][m]["enrichment_diff"]["diff"] > 0),
        "statement": (
            f"位移：{n_pos}/{len(all_models)} 个 judge 的 CI 严格为正，"
            f"模型间异质性 perm p={het_disp.get('perm_p', float('nan')):.3g}"
            f"（{'显著' if het_disp.get('significant_heterogeneity') else '不显著'}）。"
            f"富集度：在 "
            f"{sum(1 for m in all_models if res['PER_MODEL'][m]['enrichment_diff'].get('significant') and res['PER_MODEL'][m]['enrichment_diff']['diff'] > 0)}"
            f"/{len(all_models)} 个上显著，模型间异质性 perm "
            f"p={het_enr.get('perm_p', float('nan')):.3g}"
            f"（{'显著' if het_enr.get('significant_heterogeneity') else '不显著'}）"),
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"\n=== 各 judge 的碰撞位移与富集度（共有 {len(common)} 个 episode）===")
    hdr = (f"{'judge 模型':<32}{'位移':>9} {'位移 95% CI':<20}"
           f"{'变晚':>8}{'符号 p':>11}  {'富集':>7}{'基线':>7} {'富集差 CI':<20}")
    print(hdr)
    print("-" * len(hdr))
    for m in all_models:
        d = res["PER_MODEL"][m]
        dp, en, ed = d["displacement"], d["enrichment"], d["enrichment_diff"]
        sp = d["sign_p"]
        ci = f"[{dp['ci95'][0]:+.3f},{dp['ci95'][1]:+.3f}]"
        eci = (f"[{ed['ci95'][0]:+.3f},{ed['ci95'][1]:+.3f}]"
               + ("*" if ed.get("significant") else " ")) if "ci95" in ed else "—"
        print(f"{m:<32}{dp['mean']:>+9.4f} {ci:<20}"
              f"{d['n_later']}/{len(common):<6}"
              f"{(f'{sp:.2g}' if sp is not None else '—'):>11}  "
              f"{en['mean']:>7.3f}{d['enrichment_baseline']['mean']:>7.3f} {eci:<20}")
    print("  * = 富集度显著高于基线")
    print(f"\n=== 模型间异质性（置换同一 episode 内的模型标签）===")
    for key, label, want in (("displacement", "位移", "希望不显著=机制一致"),
                             ("enrichment", "富集度", "希望显著=内容异质")):
        h = res["HETEROGENEITY"][key]
        if "note" in h:
            print(f"  {label}: {h['note']}"); continue
        print(f"  {label:<6} 观测方差 {h['observed_variance']:<10} "
              f"零假设 {h['null_mean']:<10} (97.5%={h['null_p975']})  "
              f"perm p={h['perm_p']:.4g}  "
              f"{'显著' if h['significant_heterogeneity'] else '不显著'}  [{want}]")
        print(f"         各模型均值 {h['model_means']}")
    print(f"\n=== 裁决 ===\n  {res['VERDICT']['statement']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
