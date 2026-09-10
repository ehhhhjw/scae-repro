"""X9b: pool all judge models into one table, answering "on how many judges does the mechanism hold". Zero API.

Why this is a separate script
-----------------------------
Each X9 run gives only the pairwise "this model vs the original model" contrast. But the claim to
be made is **"collision bias is a property of full-trace judging as a class of practice"**, which
requires consistency evidence across several judges rather than a handful of pairwise comparisons.
This script provides three things:

  1. per-model displacement (point estimate + CI + sign test), laid out in a single table
  2. **between-model heterogeneity**: whether displacement varies across models beyond sampling
     error (a permutation test on the "model" factor over the same set of episodes)
  3. **the content vs mechanism split**: displacement and gold-provenance enrichment side by side,
     showing the former is consistent and the latter is not

⚠️ Compare only on **episodes shared by all models**, otherwise between-model differences get
confounded with episode composition.
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
                    help="one x9 jsonl per judge model")
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

    # load each model
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
    print(f"loaded {len(models)} judge models: {', '.join(models)}")
    for m in models:
        print(f"  {m}: {len(by_model[m])} paired episodes")

    # use only episodes shared by all models, else model differences confound with episode composition
    common = set.intersection(*[set(by_model[m]) for m in models])
    common &= set(orig) & set(roll)
    common = sorted(common)
    print(f"shared episodes (used for cross-model comparison): {len(common)}")
    if len(common) < 5:
        print("too few shared episodes to compare")
        return 1

    # gold provenance relation (computed once per episode)
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
    # the original model as a fourth judge
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

    # ---- between-model heterogeneity: permute the "model" label (reshuffled within each episode) ----
    # Applied to **both** quantities: displacement (expected consistent) and enrichment (expected heterogeneous).
    # Only "displacement not heterogeneous + enrichment heterogeneous" holding together
    # constitutes "consistent mechanism, inconsistent content".
    rng = random.Random(SEED)

    def heterogeneity(key: str) -> dict:
        mats = [rows[m][key] for m in all_models]
        n = min(len(v) for v in mats)
        if n < 3:
            return {"note": "insufficient samples"}
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
            "note": "permute model labels within each episode; statistic = variance of the per-model means. "
                    "For displacement we want a large p (consistent mechanism), for enrichment a small p (heterogeneous content). "
                    "⚠️ The displacement side amounts to 'accepting the null hypothesis', so the effect size must be read alongside p, not p alone.",
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
            f"displacement: {n_pos}/{len(all_models)} judges have a strictly positive CI, "
            f"between-model heterogeneity perm p={het_disp.get('perm_p', float('nan')):.3g}"
            f" ({'significant' if het_disp.get('significant_heterogeneity') else 'not significant'}). "
            f"enrichment: significant on "
            f"{sum(1 for m in all_models if res['PER_MODEL'][m]['enrichment_diff'].get('significant') and res['PER_MODEL'][m]['enrichment_diff']['diff'] > 0)}"
            f"/{len(all_models)}, between-model heterogeneity perm "
            f"p={het_enr.get('perm_p', float('nan')):.3g}"
            f" ({'significant' if het_enr.get('significant_heterogeneity') else 'not significant'})"),
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"\n=== collision displacement and enrichment per judge ({len(common)} shared episodes) ===")
    hdr = (f"{'judge model':<32}{'disp':>9} {'disp 95% CI':<20}"
           f"{'later':>8}{'sign p':>11}  {'enrich':>7}{'base':>7} {'enrich diff CI':<20}")
    print(hdr)
    print("-" * len(hdr))
    for m in all_models:
        d = res["PER_MODEL"][m]
        dp, en, ed = d["displacement"], d["enrichment"], d["enrichment_diff"]
        sp = d["sign_p"]
        ci = f"[{dp['ci95'][0]:+.3f},{dp['ci95'][1]:+.3f}]"
        eci = (f"[{ed['ci95'][0]:+.3f},{ed['ci95'][1]:+.3f}]"
               + ("*" if ed.get("significant") else " ")) if "ci95" in ed else "-"
        print(f"{m:<32}{dp['mean']:>+9.4f} {ci:<20}"
              f"{d['n_later']}/{len(common):<6}"
              f"{(f'{sp:.2g}' if sp is not None else '-'):>11}  "
              f"{en['mean']:>7.3f}{d['enrichment_baseline']['mean']:>7.3f} {eci:<20}")
    print("  * = enrichment significantly above baseline")
    print(f"\n=== between-model heterogeneity (permuting model labels within each episode) ===")
    for key, label, want in (("displacement", "disp", "want not significant = consistent mechanism"),
                             ("enrichment", "enrich", "want significant = heterogeneous content")):
        h = res["HETEROGENEITY"][key]
        if "note" in h:
            print(f"  {label}: {h['note']}"); continue
        print(f"  {label:<6} observed variance {h['observed_variance']:<10} "
              f"null {h['null_mean']:<10} (97.5%={h['null_p975']})  "
              f"perm p={h['perm_p']:.4g}  "
              f"{'significant' if h['significant_heterogeneity'] else 'not significant'}  [{want}]")
        print(f"         per-model means {h['model_means']}")
    print(f"\n=== Verdict ===\n  {res['VERDICT']['statement']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
