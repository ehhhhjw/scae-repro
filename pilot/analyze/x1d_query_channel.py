"""X1d: is the query type genuinely unpredictable, or is it an artifact of parsing and modelling? Zero API.

Background
----------
X1c reported top-3 = 0.029 for the query type (21.6% of steps) and on that basis wrote
"free-text queries" up as a hard upper bound on next-action prediction. But two things are
suspicious:

**(a) Broken parsing.** `parsed_command.parse_segment` uses `shlex.split`, which raises
ValueError on unclosed quotes such as `grep "a\\|b"`, and the fallback to `seg.split()`
**keeps the quotes and truncates on whitespace**. Measured: `shlex.split` fails on 26.6% of
command segments; among the query objects **75.4% carry residual quotes or a trailing
backslash**, and **25.5% degenerate into generic fragments like `"def` / `"class`**.
`"def` alone occurs 251 times -- a high-frequency object that ought to be easy to predict.

**(b) Weak model.** The query component of x1c only uses "the last 10 queries weighted by
recency". But the vast majority of queries are **identifiers** (`in_bulk`,
`formfield_for_foreignkey`, `supports_logical_xor`), whose sources should be the **problem
statement** and **symbol names appearing in earlier tool output**.

What this script does
---------------------
  1. Build a more robust query extractor (not dependent on shlex succeeding) and quantify the
     difference before/after the repair
  2. Swap in more appropriate candidate generation: frequency + problem-statement identifiers +
     symbol names in the output
  3. Compare on **the same set of points**, and determine how much of the 0.029 is attributable
     to parsing, how much to modelling, and how much is genuinely unpredictable

Criterion (pre-registered)
--------------------------
| Result | Conclusion |
|---|---|
| top-3 rises significantly after the repair | the original 0.029 is a **measurement and modelling artifact**; the paper's phrasing about "free text is a hard upper bound" must be narrowed |
| the parsing repair alone lifts it | attributable to parsing; `parsed_command` needs fixing, and it affects every analysis that uses query |
| still ≈0.03 after changing both | query really is unpredictable, the original conclusion holds and is more solid |
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.parsed_command import parse_step        # noqa: E402
from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.analyze.v3g_regime import rank_ge, exp_hit, eff_cand  # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci, paired_ci      # noqa: E402

KS = (1, 3, 5, 10, 20)
SEARCH_HEADS = {"grep", "rg", "egrep", "fgrep", "ag", "ack"}
# identifier: Python-style name, including dotted attribute access
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
# definition lines in the output
DEFN = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
STOP = {"def", "class", "import", "from", "return", "if", "for", "while", "self",
        "None", "True", "False", "and", "or", "not", "in", "is", "as", "with",
        "try", "except", "raise", "pass", "print", "len", "str", "int", "list",
        "dict", "set", "type", "object", "super", "test", "tests"}


def repair_query(cmd: str) -> str | None:
    """Robustly pull the search pattern out of a command, without relying on shlex succeeding.

    Prefers **the full content inside quotes** (exactly what shlex drops when it fails),
    otherwise takes the first non-option token. Then strips regex metacharacters and escapes,
    leaving the core identifiers.
    """
    for seg in re.split(r"\|\||&&|;", cmd):
        toks = seg.strip().split()
        if not toks:
            continue
        head = toks[0].split("/")[-1]
        if head not in SEARCH_HEADS:
            continue
        # quoted content: tolerates unclosed syntax such as \| inside
        m = re.search(r'"([^"]*)"|\'([^\']*)\'', seg)
        raw = (m.group(1) or m.group(2)) if m else None
        if raw is None:
            for t in toks[1:]:
                if not t.startswith("-"):
                    raw = t
                    break
        if raw is None:
            continue
        return raw.strip()
    return None


def core_idents(raw: str) -> list[str]:
    """Reduce a search pattern to the identifiers it contains (dropping regex and stop words)."""
    out = []
    for m in IDENT.finditer(raw):
        w = m.group(0)
        if w in STOP or len(w) < 3:
            continue
        if w not in out:
            out.append(w)
    return out


def norm_query(raw: str) -> str:
    """Normalized query key: prefer joining the core identifiers, otherwise the cleaned raw string."""
    ci = core_idents(raw)
    if ci:
        return "q:" + "|".join(ci)
    return "q:" + raw.strip('"\'\\ ').lower()


def load(runs_root: Path) -> list[dict]:
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        try:
            steps = R.steps(R.load(f))
        except (OSError, ValueError):
            continue
        if len(steps) < 3:
            continue
        seq = []
        for s in steps:
            p = parse_step(s)
            a = s.args
            cmd = a.get("cmd") or a.get("command")
            if isinstance(cmd, list):
                cmd = " ".join(str(c) for c in cmd)
            cmd = cmd if isinstance(cmd, str) else ""
            rq = repair_query(cmd) if cmd else None
            seq.append({
                "tool": p.kind,
                "old_q": (f"q:{p.query}" if (p.query and not p.path) else None),
                "new_q": (norm_query(rq) if (rq and not p.path) else None),
                "raw_q": rq,
                "out": s.output_text or "",
            })
        eps.append({"id": iid, "project": inst.project, "seq": seq,
                    "problem": inst.problem_statement or ""})
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x1d_query_channel.json")
    args = ap.parse_args()

    eps = load(Path(args.runs_root))
    print(f"loaded {len(eps)} episodes")

    # ---- scale of the parsing repair ----
    n_old = n_new = n_damaged = 0
    old_ctr: Counter = Counter()
    new_ctr: Counter = Counter()
    for e in eps:
        for s in e["seq"]:
            if s["old_q"]:
                n_old += 1
                old_ctr[s["old_q"]] += 1
                if s["old_q"][2:3] in ('"', "'") or s["old_q"].endswith("\\"):
                    n_damaged += 1
            if s["new_q"]:
                n_new += 1
                new_ctr[s["new_q"]] += 1

    MODELS = ("old_recency", "new_recency", "new_freq",
              "new_freq_plus_problem", "new_freq_plus_problem_plus_symbols")
    hits = {m: {k: [] for k in KS} for m in MODELS}
    cov = {m: [] for m in MODELS}
    effc = {m: [] for m in MODELS}
    n_pts = 0

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            # marginal query frequencies in the training set (LOO)
            tr_new = Counter(s["new_q"] for x in train for s in x["seq"] if s["new_q"])
            tr_old = Counter(s["old_q"] for x in train for s in x["seq"] if s["old_q"])
            # identifiers in the problem statement
            pids = [w for w in core_idents(held["problem"])][:200]
            seq = held["seq"]
            hist_new: list[str] = []
            hist_old: list[str] = []
            syms: list[str] = []
            for t in range(len(seq) - 1):
                if seq[t]["new_q"]:
                    hist_new.append(seq[t]["new_q"])
                if seq[t]["old_q"]:
                    hist_old.append(seq[t]["old_q"])
                for m in DEFN.finditer(seq[t]["out"][:200_000]):
                    if m.group(1) not in syms:
                        syms.append(m.group(1))
                nxt = seq[t + 1]
                if not nxt["new_q"]:
                    continue
                n_pts += 1

                def recency(h: list[str]) -> dict[str, float]:
                    w = {q: 1.0 / (i + 1) for i, q in enumerate(reversed(h[-10:]))}
                    z = sum(w.values())
                    return {k: v / z for k, v in w.items()} if z else {}

                def freq(h: list[str], prior: Counter) -> dict[str, float]:
                    w: dict[str, float] = {}
                    for q, c in Counter(h).items():
                        w[q] = w.get(q, 0.0) + 2.0 * c
                    for q, c in prior.items():
                        w[q] = w.get(q, 0.0) + 1.0 * c / max(1, sum(prior.values())) * 10
                    z = sum(w.values())
                    return {k: v / z for k, v in w.items()} if z else {}

                base = freq(hist_new, tr_new)
                plus_p = dict(base)
                for i, w in enumerate(pids):
                    key = f"q:{w}"
                    plus_p[key] = plus_p.get(key, 0.0) + 0.30 / (i + 1) ** 0.5
                plus_ps = dict(plus_p)
                for i, w in enumerate(reversed(syms[-60:])):
                    key = f"q:{w}"
                    plus_ps[key] = plus_ps.get(key, 0.0) + 0.40 / (i + 1) ** 0.5

                def nrm(d):
                    z = sum(d.values())
                    return {k: v / z for k, v in d.items()} if z else {}

                dists = {
                    "old_recency": recency(hist_old),
                    "new_recency": recency(hist_new),
                    "new_freq": base,
                    "new_freq_plus_problem": nrm(plus_p),
                    "new_freq_plus_problem_plus_symbols": nrm(plus_ps),
                }
                truth_new, truth_old = nxt["new_q"], nxt["old_q"]
                for nm, d in dists.items():
                    truth = truth_old if nm == "old_recency" else truth_new
                    rs = rank_ge(d, truth) if truth else None
                    cov[nm].append(1.0 if rs else 0.0)
                    for k in KS:
                        hits[nm][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        effc[nm].append(eff_cand(d))

    res: dict = {
        "n_episodes": len(eps), "n_query_points": n_pts,
        "PARSING_REPAIR": {
            "n_query_objects_old": n_old,
            "n_query_objects_new": n_new,
            "damaged_share_old": round(n_damaged / max(1, n_old), 4),
            "distinct_old": len(old_ctr), "distinct_new": len(new_ctr),
            "top_old_fragments": old_ctr.most_common(6),
            "top_new_keys": new_ctr.most_common(6),
            "note": "old is parsed_command as it stands (falls back to splitting on whitespace "
                    "when shlex fails), new is this script's robust extraction + identifier "
                    "normalization",
        },
        "MODELS": {
            m: {"coverage": round(statistics.mean(cov[m]), 4),
                "topk": {f"top{k}": boot_ci(hits[m][k]) for k in KS},
                "eff_candidates": (round(statistics.median(effc[m]), 2)
                                   if effc[m] else None)}
            for m in MODELS},
        "CONTRASTS": {
            "parsing_only(new_recency - old_recency) top3":
                paired_ci(hits["new_recency"][3], hits["old_recency"][3]),
            "modelling_only(new_freq - new_recency) top3":
                paired_ci(hits["new_freq"][3], hits["new_recency"][3]),
            "plus_problem top3":
                paired_ci(hits["new_freq_plus_problem"][3], hits["new_freq"][3]),
            "plus_symbols top3":
                paired_ci(hits["new_freq_plus_problem_plus_symbols"][3],
                          hits["new_freq_plus_problem"][3]),
            "total(best - old) top3":
                paired_ci(hits["new_freq_plus_problem_plus_symbols"][3],
                          hits["old_recency"][3]),
        },
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    pr = res["PARSING_REPAIR"]
    print(f"\n=== scale of the parsing repair ===")
    print(f"  query objects old={pr['n_query_objects_old']} new={pr['n_query_objects_new']}")
    print(f"  damaged share in old {pr['damaged_share_old']:.1%}")
    print(f"  distinct values old={pr['distinct_old']} new={pr['distinct_new']}")
    print(f"  most common old fragments: {pr['top_old_fragments'][:4]}")
    print(f"  most common new keys:      {pr['top_new_keys'][:4]}")

    print(f"\n=== model comparison (query steps, n={n_pts}) ===")
    print(f"{'model':<38}{'coverage':>8}{'top1':>8}{'top3':>8}{'top10':>8}{'eff_cand':>9}")
    for m in MODELS:
        d = res["MODELS"][m]
        print(f"{m:<38}{d['coverage']:>8.4f}{d['topk']['top1']['mean']:>8.4f}"
              f"{d['topk']['top3']['mean']:>8.4f}{d['topk']['top10']['mean']:>8.4f}"
              f"{str(d['eff_candidates']):>9}")

    print(f"\n=== attribution decomposition (paired, top-3) ===")
    for k, v in res["CONTRASTS"].items():
        if "note" in v:
            print(f"  {k:<52} {v['note']}")
            continue
        print(f"  {k:<52} {v['diff']:+.4f} CI[{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]  {'significant' if v['significant'] else 'not significant'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
