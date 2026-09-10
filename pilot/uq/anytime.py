"""Anytime uncertainty quantification: AUROC(t) / ECE(t) -- the main UQ experiment.

Answers the core request: "quantify uncertainty over the AI execution **process**".

Design motivation
-----------------
Early on we wrongly anchored UQ at v_0 (pre-execution prediction from the issue text
plus static repository features alone), measuring LORO AUROC ≈ 0.5 (chance). That only
shows "pre-execution information is insufficient", not that the process is unquantifiable.
The right anchor is the evolution of v_t with t: **at which step can we reliably foresee
the outcome**.

Strict anti-leakage constraint
------------------------------
Features may only use information visible to the agent itself within tau_{1:t}.
Any gold-dependent quantity (e.g. "has a gold file been read yet") is **strictly
forbidden** -- it would inflate AUROC and render it meaningless.
Every feature below is computed only from (tool name, target path, output text),
independent of F*.

Survivorship bias handling
--------------------------
As t grows only long episodes survive (>=16 steps: only 48%). Therefore:
  1. report n(t) at every t
  2. additionally run a **fixed-cohort** analysis: use only episodes with >= T_max steps,
     evaluated over t=1..T_max, removing the bias from a changing composition.
     Both curves are reported.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.uq import predictor as P                         # noqa: E402

_SEARCHY = ("grep", "rg", "find", "ls", "glob", "search")
_READY = ("sed", "cat", "head", "tail", "read", "view", "open")

PREFIX_FEATURES = [
    "bias",
    "log_t",              # steps executed so far (log)
    "n_distinct_targets", # number of distinct targets touched
    "n_distinct_dirs",    # number of distinct directories touched
    "search_ratio",       # share of search-type actions (exploring vs close reading)
    "log_empty_outputs",  # count of empty/very short outputs (search miss = wrong direction)
    "log_out_chars",      # cumulative observation length (information intake)
    "action_entropy",     # entropy of the action-bucket distribution (behavioral focus)
    "repeat_rate",        # share of repeated actions (stalling signal)
    "max_dir_depth",      # deepest directory level
    "new_target_rate",    # rate of new targets in the last few steps (still exploring vs converged)
]


def _cls(tool: str, args_raw: str) -> str:
    """Coarsely bucket an action into search / read / other -- looks only at the tool name and command head, never at gold."""
    blob = f"{tool} {args_raw}".lower()
    head = blob
    m = re.search(r'"cmd"\s*:\s*"([^"\s]+)', blob)
    if m:
        head = m.group(1)
    if any(k in head for k in _SEARCHY):
        return "search"
    if any(k in head for k in _READY):
        return "read"
    if any(k in blob for k in _SEARCHY):
        return "search"
    if any(k in blob for k in _READY):
        return "read"
    return "other"


def prefix_features(steps: list, t: int, repo_root: str | Path | None) -> np.ndarray:
    """Extract features at the step-t cross section. Uses only agent-visible information from the first t steps."""
    pre = steps[:t]
    buckets, dirs, out_len, empties, kinds = [], set(), 0, 0, []
    for s in pre:
        tool, target = R.action_bucket(s, repo_root)
        buckets.append(f"{tool}#{target}")
        if target and target != "-":
            d = target.rsplit("/", 1)[0] if "/" in target else "."
            dirs.add(d)
        o = s.output_text or ""
        out_len += len(o)
        if len(o.strip()) < 8:
            empties += 1
        kinds.append(_cls(s.tool, s.args_raw))

    cnt = Counter(buckets)
    n = len(pre)
    ent = -sum((c / n) * math.log(c / n) for c in cnt.values()) if n else 0.0
    n_search = sum(1 for k in kinds if k == "search")
    repeats = n - len(cnt)
    depth = max((tg.count("/") for tg in
                 (b.split("#", 1)[1] for b in buckets) if tg and tg != "-"), default=0)
    # rate of new targets appearing in the last min(5,n) steps
    tail = buckets[-5:]
    seen_before = set(buckets[: max(0, n - len(tail))])
    new_rate = sum(1 for b in tail if b not in seen_before) / len(tail) if tail else 0.0

    return np.array([
        1.0,
        math.log1p(n),
        math.log1p(len(cnt)),
        math.log1p(len(dirs)),
        n_search / n if n else 0.0,
        math.log1p(empties),
        math.log1p(out_len),
        ent,
        repeats / n if n else 0.0,
        float(depth),
        new_rate,
    ])


def build_anytime_dataset(results_path: Path, runs_root: Path,
                          cutoffs: list[int], multifile_only: bool = False) -> dict:
    """Build (X, y, groups) for each cutoff t. Only episodes with n_steps >= t are included.

    multifile_only: use only instances with |F*|>=2. On the full corpus, at a 94% base
    rate there are only 30 failures and 5 repos have zero failures, so the discriminative
    signal per LORO fold is single-digit; the multi-file subset has a 27% failure rate,
    is far better balanced, and the paper already established that the discriminative
    signal concentrates in that subset.
    """
    insts = {i.instance_id: i for i in T.load_instances()}
    recs = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if multifile_only:
        recs = [r for r in recs if r.get("n_gold", 1) >= 2]
    per_t: dict[int, dict] = {t: {"X": [], "y": [], "g": [], "ids": []} for t in cutoffs}
    n_loaded = 0
    steplen: dict[str, int] = {}

    for r in recs:
        if not r.get("ok") or not r.get("rollout"):
            continue
        inst = insts.get(r["instance_id"])
        if inst is None:
            continue
        # the rollout may live under any shard
        cands = list(runs_root.glob(f"shard*/{r['rollout']}")) or \
                list(runs_root.glob(r["rollout"]))
        if not cands:
            continue
        try:
            steps = R.steps(R.load(cands[0]))
        except (OSError, ValueError):
            continue
        n_loaded += 1
        pred = frozenset(r["answer"]) if r.get("answer") else None
        y = T.outcome(pred, inst.gold)["Y"]
        steplen[r["instance_id"]] = len(steps)
        for t in cutoffs:
            if len(steps) < t:
                continue
            per_t[t]["X"].append(prefix_features(steps, t, inst.repo))
            per_t[t]["y"].append(float(y))
            per_t[t]["g"].append(inst.project)
            per_t[t]["ids"].append(r["instance_id"])
    return {"per_t": per_t, "n_loaded": n_loaded, "steplen": steplen}


def loro_eval(X: np.ndarray, y: np.ndarray, groups: list[str],
              l2: float = 2.0) -> dict:
    """leave-one-repository-out evaluation. Returns AUROC/ECE/Brier/AURC."""
    projects = sorted(set(groups))
    preds = np.full(len(y), np.nan)
    for p in projects:
        te = np.array([g == p for g in groups])
        tr = ~te
        if tr.sum() < 12 or te.sum() == 0:
            continue
        if len(set(y[tr].tolist())) < 2:
            preds[te] = float(y[tr].mean())
            continue
        m = P.BayesianLogisticLaplace(l2=l2).fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])
    ok = ~np.isnan(preds)
    if ok.sum() < 15 or len(set(y[ok].tolist())) < 2:
        return {"n": int(ok.sum()), "note": "insufficient samples or classes"}
    pp, yy = preds[ok], y[ok]
    _, risk, aurc = P.risk_coverage(pp, yy)
    return {
        "n": int(ok.sum()),
        "n_pos": int(yy.sum()),
        "base_rate": round(float(yy.mean()), 4),
        "AUROC": round(P.auroc(pp, yy), 4),
        "ECE": round(P.ece(pp, yy), 4),
        "Brier": round(P.brier(pp, yy), 4),
        "AURC": round(aurc, 4),
        "pred_std": round(float(pp.std()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--cutoffs", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 16, 20])
    ap.add_argument("--cohort-t", type=int, default=16,
                    help="fixed cohort: use only episodes with >= this many steps to remove survivorship bias")
    ap.add_argument("--multifile-only", action="store_true")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--out", default="pilot/runs/anytime.json")
    args = ap.parse_args()

    ds = build_anytime_dataset(Path(args.results), Path(args.runs_root), args.cutoffs,
                               multifile_only=args.multifile_only)
    print(f"loaded {ds['n_loaded']} rollouts")

    # --- full curve (sample shrinks with t, so report n(t)) ---
    curve = []
    for t in args.cutoffs:
        d = ds["per_t"][t]
        if len(d["y"]) < args.min_n:
            curve.append({"t": t, "n": len(d["y"]), "note": "insufficient samples"})
            continue
        res = loro_eval(np.array(d["X"]), np.array(d["y"]), d["g"])
        res["t"] = t
        curve.append(res)
        print(f"  t={t:>2}  n={res.get('n')}  AUROC={res.get('AUROC')}  "
              f"ECE={res.get('ECE')}  AURC={res.get('AURC')}")

    # --- fixed-cohort curve (composition held constant, survivorship bias excluded) ---
    T0 = args.cohort_t
    cohort_ids = {i for i, L in ds["steplen"].items() if L >= T0}
    cohort = []
    print(f"\nfixed cohort (steps>={T0}): {len(cohort_ids)} episodes")
    for t in [x for x in args.cutoffs if x <= T0]:
        d = ds["per_t"][t]
        keep = [j for j, iid in enumerate(d["ids"]) if iid in cohort_ids]
        if len(keep) < args.min_n:
            cohort.append({"t": t, "n": len(keep), "note": "insufficient samples"})
            continue
        X = np.array([d["X"][j] for j in keep])
        y = np.array([d["y"][j] for j in keep])
        g = [d["g"][j] for j in keep]
        res = loro_eval(X, y, g)
        res["t"] = t
        cohort.append(res)
        print(f"  t={t:>2}  n={res.get('n')}  AUROC={res.get('AUROC')}  "
              f"ECE={res.get('ECE')}")

    out = {
        "n_rollouts": ds["n_loaded"],
        "features": PREFIX_FEATURES,
        "leakage_note": ("features are computed only from (tool name, target path, "
                         "output text) and contain no gold-dependent quantity"),
        "curve_all": curve,
        "cohort_t": T0,
        "cohort_size": len(cohort_ids),
        "curve_cohort": cohort,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
