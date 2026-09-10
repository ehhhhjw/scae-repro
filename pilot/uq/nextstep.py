"""Next-step action prediction -- completing the "compute probabilities over the Next Step" goal.

Task
----
Given the prefix tau_{1:t}, predict the equivalence class (tool, target node in G) of the
next action A_{t+1}. This is an evaluable proxy for the policy pi(.|tau_{1:t}): RQ1 only
measured that distribution's **entropy** (the overlap diagnostic); here we add its
**predictability**, and together the two characterize the policy uncertainty of the
execution process.

Candidate set construction (no gold leakage)
--------------------------------------------
For each decision point, candidates = action buckets already seen in this episode
              ∪ high-frequency action buckets from this repo's corpus (top-K, counted on
                the **training folds**)
A decision point counts only when the label is inside the candidate set (coverage is
reported separately); otherwise no model could possibly hit it.

Scoring features (all depend only on tau_{1:t} and corpus statistics)
---------------------------------------------------------------------
  f1 global frequency of the candidate on the training folds (log)
  f2 frequency of the candidate on the same-repo training folds (log)
  f3 whether it already appeared in the current episode
  f4 same tool as the previous step
  f5 same directory as the previous step
  f6 step distance since the candidate was last used (log; closer = more likely repeated)
  f7 whether the candidate is the previous action itself (self-repeat)

Evaluation: leave-one-repository-out, metrics top-1 / top-3 accuracy and MRR.
Baselines: (a) marginal frequency (f1 only) (b) repeat the previous step (c) random.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.uq import predictor as P                         # noqa: E402

FEATURES = ["bias", "log_global_freq", "log_repo_freq", "seen_in_episode",
            "same_tool_as_prev", "same_dir_as_prev", "log_recency", "is_prev_action"]


def load_episodes(results_path: Path, runs_root: Path) -> list[dict]:
    insts = {i.instance_id: i for i in T.load_instances()}
    recs = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    eps = []
    for r in recs:
        if not r.get("ok") or not r.get("rollout"):
            continue
        inst = insts.get(r["instance_id"])
        if inst is None:
            continue
        cands = list(runs_root.glob(f"shard*/{r['rollout']}")) or \
                list(runs_root.glob(r["rollout"]))
        if not cands:
            continue
        try:
            steps = R.steps(R.load(cands[0]))
        except (OSError, ValueError):
            continue
        seq = [f"{a}#{b}" for a, b in
               (R.action_bucket(s, inst.repo) for s in steps)]
        if len(seq) >= 3:
            eps.append({"id": r["instance_id"], "project": inst.project, "seq": seq})
    return eps


def _dirof(bucket: str) -> str:
    tgt = bucket.split("#", 1)[1] if "#" in bucket else ""
    return tgt.rsplit("/", 1)[0] if "/" in tgt else "."


def _toolof(bucket: str) -> str:
    return bucket.split("#", 1)[0] if "#" in bucket else bucket


def featurize(cand: str, prefix: list[str], gfreq: Counter, rfreq: Counter,
              n_g: int, n_r: int) -> np.ndarray:
    prev = prefix[-1] if prefix else ""
    seen = cand in prefix
    recency = 0
    if seen:
        for k in range(len(prefix) - 1, -1, -1):
            if prefix[k] == cand:
                recency = len(prefix) - k
                break
    return np.array([
        1.0,
        math.log1p(gfreq.get(cand, 0) / max(1, n_g) * 1000),
        math.log1p(rfreq.get(cand, 0) / max(1, n_r) * 1000),
        1.0 if seen else 0.0,
        1.0 if prev and _toolof(cand) == _toolof(prev) else 0.0,
        1.0 if prev and _dirof(cand) == _dirof(prev) else 0.0,
        math.log1p(recency),
        1.0 if cand == prev else 0.0,
    ])


def build_pairs(eps: list[dict], topk_repo: int = 40) -> dict:
    """Build (decision point, candidate set, label) triples, grouped by repo for LORO."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    return by_repo


def evaluate_loro(by_repo: dict, topk_repo: int = 40, l2: float = 2.0) -> dict:
    projects = sorted(by_repo)
    tot = hit1 = hit3 = 0
    rr_sum = 0.0
    cov_num = cov_den = 0
    base_hit1 = {"marginal": 0, "repeat_prev": 0, "random": 0}
    per_repo = {}

    for held in projects:
        train_eps = [e for p in projects if p != held for e in by_repo[p]]
        test_eps = by_repo[held]
        if len(train_eps) < 20 or not test_eps:
            continue

        # corpus statistics from the training folds (using the test fold is strictly forbidden)
        gfreq = Counter(b for e in train_eps for b in e["seq"])
        n_g = sum(gfreq.values())
        rfreq_by = defaultdict(Counter)
        for e in train_eps:
            rfreq_by[e["project"]].update(e["seq"])
        # the held-out repo has no statistics of its own, so use the global top-K as candidates
        global_top = [b for b, _ in gfreq.most_common(topk_repo)]

        # --- train the scoring model: positive = the true next step, negatives = the rest of the candidate set ---
        X, y = [], []
        for e in train_eps:
            seq = e["seq"]
            rf = rfreq_by[e["project"]]
            n_r = sum(rf.values())
            for t in range(1, len(seq)):
                prefix, label = seq[:t], seq[t]
                cands = set(prefix) | set(b for b, _ in rf.most_common(topk_repo))
                if label not in cands:
                    continue
                for c in cands:
                    X.append(featurize(c, prefix, gfreq, rf, n_g, n_r))
                    y.append(1.0 if c == label else 0.0)
        if len(X) < 200 or sum(y) < 20:
            continue
        m = P.BayesianLogisticLaplace(l2=l2).fit(np.array(X), np.array(y))

        # --- test ---
        r_tot = r_h1 = r_h3 = 0
        for e in test_eps:
            seq = e["seq"]
            n_r = n_g
            for t in range(1, len(seq)):
                prefix, label = seq[:t], seq[t]
                cands = sorted(set(prefix) | set(global_top))
                cov_den += 1
                if label not in cands:
                    continue
                cov_num += 1
                feats = np.array([featurize(c, prefix, gfreq, gfreq, n_g, n_r)
                                  for c in cands])
                scores = m.predict_map(feats)
                order = np.argsort(-scores)
                ranked = [cands[i] for i in order]
                pos = ranked.index(label) + 1
                tot += 1; r_tot += 1
                if pos == 1: hit1 += 1; r_h1 += 1
                if pos <= 3: hit3 += 1; r_h3 += 1
                rr_sum += 1.0 / pos
                # baselines
                mb = max(cands, key=lambda c: gfreq.get(c, 0))
                if mb == label: base_hit1["marginal"] += 1
                if prefix[-1] == label: base_hit1["repeat_prev"] += 1
                base_hit1["random"] += 1.0 / len(cands)
        if r_tot:
            per_repo[held] = {"n": r_tot, "top1": round(r_h1 / r_tot, 4),
                              "top3": round(r_h3 / r_tot, 4)}

    if not tot:
        return {"note": "no valid decision points"}
    return {
        "n_decision_points": tot,
        "candidate_coverage": round(cov_num / cov_den, 4) if cov_den else None,
        "top1": round(hit1 / tot, 4),
        "top3": round(hit3 / tot, 4),
        "MRR": round(rr_sum / tot, 4),
        "baselines_top1": {k: round(v / tot, 4) for k, v in base_hit1.items()},
        "per_repo": per_repo,
    }


def coarsen(bucket: str) -> str:
    """Coarse action class: structural features that transfer across repos, without concrete paths.

    Under LORO the exact bucket (which includes the file path) does not transfer -- the path
    does not exist in another repo. Here we keep four transferable dimensions:
    (tool, is-test-file, directory-depth bin, has-concrete-target).
    """
    tool = _toolof(bucket)
    tgt = bucket.split("#", 1)[1] if "#" in bucket else ""
    if not tgt or tgt == "-":
        return f"{tool}|notarget"
    is_test = "test" in tgt.lower()
    depth = tgt.count("/")
    dbin = "0" if depth == 0 else ("1-2" if depth <= 2 else "3+")
    return f"{tool}|test={int(is_test)}|d={dbin}"


def evaluate_within_repo(by_repo: dict, topk: int = 40, l2: float = 2.0) -> dict:
    """Within-repo leave-one-episode-out -- the correct evaluation setting for next-step prediction.

    The actual use case is "predict the agent's next step inside this repository", which
    needs no cross-repository transfer, so the unit of generalization is the episode, not the repo.
    """
    tot = hit1 = hit3 = 0
    rr = 0.0
    cov_num = cov_den = 0
    base = {"marginal": 0, "repeat_prev": 0}
    for proj, eps in by_repo.items():
        if len(eps) < 6:
            continue
        for held_i, held in enumerate(eps):
            train = [e for j, e in enumerate(eps) if j != held_i]
            gfreq = Counter(b for e in train for b in e["seq"])
            n_g = sum(gfreq.values())
            if n_g < 50:
                continue
            X, y = [], []
            for e in train:
                seq = e["seq"]
                for t in range(1, len(seq)):
                    prefix, label = seq[:t], seq[t]
                    cands = set(prefix) | set(b for b, _ in gfreq.most_common(topk))
                    if label not in cands:
                        continue
                    for c in cands:
                        X.append(featurize(c, prefix, gfreq, gfreq, n_g, n_g))
                        y.append(1.0 if c == label else 0.0)
            if len(X) < 150 or sum(y) < 15:
                continue
            m = P.BayesianLogisticLaplace(l2=l2).fit(np.array(X), np.array(y))
            seq = held["seq"]
            for t in range(1, len(seq)):
                prefix, label = seq[:t], seq[t]
                cands = sorted(set(prefix) | set(b for b, _ in gfreq.most_common(topk)))
                cov_den += 1
                if label not in cands:
                    continue
                cov_num += 1
                feats = np.array([featurize(c, prefix, gfreq, gfreq, n_g, n_g)
                                  for c in cands])
                order = np.argsort(-m.predict_map(feats))
                ranked = [cands[i] for i in order]
                pos = ranked.index(label) + 1
                tot += 1
                if pos == 1: hit1 += 1
                if pos <= 3: hit3 += 1
                rr += 1.0 / pos
                if max(cands, key=lambda c: gfreq.get(c, 0)) == label:
                    base["marginal"] += 1
                if prefix[-1] == label:
                    base["repeat_prev"] += 1
    if not tot:
        return {"note": "no valid decision points"}
    return {
        "setting": "within-repo, leave-one-episode-out",
        "n_decision_points": tot,
        "candidate_coverage": round(cov_num / cov_den, 4) if cov_den else None,
        "top1": round(hit1 / tot, 4), "top3": round(hit3 / tot, 4),
        "MRR": round(rr / tot, 4),
        "baselines_top1": {k: round(v / tot, 4) for k, v in base.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--topk-repo", type=int, default=40)
    ap.add_argument("--coarse", action="store_true",
                    help="use coarse action classes (transferable across repos)")
    ap.add_argument("--within-repo", action="store_true",
                    help="within-repo leave-one-episode-out (the correct setting)")
    ap.add_argument("--out", default="pilot/runs/nextstep.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.results), Path(args.runs_root))
    print(f"loaded {len(eps)} episodes, "
          f"{sum(len(e['seq'])-1 for e in eps)} step transitions in total")
    if args.coarse:
        for e in eps:
            e["seq"] = [coarsen(b) for b in e["seq"]]
        print("action classes coarsened (transferable across repos)")
    by_repo = build_pairs(eps, args.topk_repo)
    print(f"repos: {len(by_repo)}")
    res = (evaluate_within_repo(by_repo, args.topk_repo) if args.within_repo
           else evaluate_loro(by_repo, args.topk_repo))
    res["n_episodes"] = len(eps)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\n=== Next-step prediction (leave-one-repository-out) ===")
    print(f"  decision points {res.get('n_decision_points')}  "
          f"candidate coverage {res.get('candidate_coverage')}")
    print(f"  top-1 {res.get('top1')}  top-3 {res.get('top3')}  MRR {res.get('MRR')}")
    print(f"  baseline top-1: {res.get('baselines_top1')}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
