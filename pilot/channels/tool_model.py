"""V1 通道 A：工具关系的数学模型。零 API。

状态空间的选择（实测决定）
------------------------
codex 只暴露 3 个名义工具，其中 exec_command 占 8159/8200 —— 名义 API 层是
单点分布，熵恒为 0，不携带信息。真正的语义工具在 shell 命令头里：
  sed 2763 / grep 2266 / git 1862 / ls 470 / find 212 / python 145 / rg 143 ...
因此工具状态空间建在**命令头**上。这本身是一个可写进论文的观察：
工具关系必须在语义层建模，名义 API 层退化。

模型
----
  M0 边际      P(tool)                                基线
  M1 一阶      P(tool_{t+1} | tool_t)                 + Dirichlet 平滑
  M2 条件      P(tool_{t+1} | tool_t, ctx_t)          ctx = (上一步是否命中, 步位分档)
  M3 repeat    P = 1[tool_{t+1} = tool_t]             trivial 基线（曾打败我们的模型）

评估
----
leave-one-repository-out 的 perplexity 与 top-1/top-3；
另报每个决策点的条件熵 H(P(tool_{t+1}|·))，它是 U^tool_t 的定义。
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

from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402

_CMD = re.compile(r'"cmd"\s*:\s*"([^"\s]+)')
# 命令头归一：环境变量前缀、路径、别名
_ENVPREFIX = re.compile(r"^[A-Z_][A-Z0-9_]*=")


def tool_of(step) -> str:
    """把一步映射到语义工具（命令头）。"""
    m = _CMD.search(step.args_raw or "")
    if not m:
        return step.tool
    head = m.group(1)
    # 去掉 VAR=x 前缀形式
    if _ENVPREFIX.match(head):
        return "envset"
    head = head.split("/")[-1]
    alias = {"python3": "python", "rg": "grep", "egrep": "grep", "fgrep": "grep",
             "nl": "cat", "head": "cat", "tail": "cat", "wc": "cat"}
    return alias.get(head, head)


def ctx_of(step, t: int, n_steps: int) -> str:
    """少量可观测上下文：上一步是否有实质输出 × 步位分档。"""
    out = step.output_text or ""
    hit = "hit" if len(out.strip()) >= 40 else "miss"
    frac = t / max(1, n_steps)
    pos = "early" if frac <= 1 / 3 else ("mid" if frac <= 2 / 3 else "late")
    return f"{hit}|{pos}"


def load_sequences(runs_root: Path) -> list[dict]:
    """每条 rollout -> 工具序列 + 上下文序列。"""
    insts = {i.instance_id: i for i in T.load_instances()}
    out = []
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
        n = len(steps)
        out.append({
            "instance_id": iid, "project": inst.project,
            "tools": [tool_of(s) for s in steps],
            "ctxs": [ctx_of(s, i + 1, n) for i, s in enumerate(steps)],
        })
    return out


def fit_marginal(seqs: list[dict], alpha: float = 0.5) -> dict:
    c = Counter(t for s in seqs for t in s["tools"])
    return {"counts": c, "alpha": alpha}


def fit_first_order(seqs: list[dict], alpha: float = 0.5) -> dict:
    tr: dict[str, Counter] = defaultdict(Counter)
    for s in seqs:
        for a, b in zip(s["tools"], s["tools"][1:]):
            tr[a][b] += 1
    return {"trans": tr, "alpha": alpha}


def fit_conditional(seqs: list[dict], alpha: float = 0.5) -> dict:
    tr: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for s in seqs:
        for i in range(len(s["tools"]) - 1):
            tr[(s["tools"][i], s["ctxs"][i])][s["tools"][i + 1]] += 1
    return {"trans": tr, "alpha": alpha}


def prob_marginal(m: dict, nxt: str, vocab: list[str]) -> float:
    a, c = m["alpha"], m["counts"]
    return (c.get(nxt, 0) + a) / (sum(c.values()) + a * len(vocab))


def prob_first(m: dict, cur: str, nxt: str, vocab: list[str],
               fallback: dict | None = None) -> float:
    a = m["alpha"]
    row = m["trans"].get(cur)
    if not row or sum(row.values()) == 0:
        return prob_marginal(fallback, nxt, vocab) if fallback else 1 / len(vocab)
    return (row.get(nxt, 0) + a) / (sum(row.values()) + a * len(vocab))


def prob_cond(m: dict, cur: str, ctx: str, nxt: str, vocab: list[str],
              back1: dict | None, back0: dict | None) -> float:
    a = m["alpha"]
    row = m["trans"].get((cur, ctx))
    if not row or sum(row.values()) < 5:          # 样本不足则退化到一阶
        return (prob_first(m=back1, cur=cur, nxt=nxt, vocab=vocab, fallback=back0)
                if back1 else 1 / len(vocab))
    return (row.get(nxt, 0) + a) / (sum(row.values()) + a * len(vocab))


def entropy(ps: list[float]) -> float:
    z = sum(ps) or 1.0
    return -sum((p / z) * math.log(p / z) for p in ps if p > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/tool_model.json")
    args = ap.parse_args()

    seqs = load_sequences(Path(args.runs_root))
    vocab = sorted({t for s in seqs for t in s["tools"]})
    n_trans = sum(len(s["tools"]) - 1 for s in seqs)
    print(f"载入 {len(seqs)} 条序列，{n_trans} 个步转移，工具词表 {len(vocab)}")
    print(f"工具词表: {vocab}")

    projects = sorted({s["project"] for s in seqs})
    stats = {k: {"logp": 0.0, "n": 0, "hit1": 0, "hit3": 0}
             for k in ("M0_marginal", "M1_first", "M2_cond", "M3_repeat")}
    ent_rows = []

    for held in projects:
        tr = [s for s in seqs if s["project"] != held]
        te = [s for s in seqs if s["project"] == held]
        if len(tr) < 20 or not te:
            continue
        m0 = fit_marginal(tr)
        m1 = fit_first_order(tr)
        m2 = fit_conditional(tr)

        for s in te:
            tools, ctxs = s["tools"], s["ctxs"]
            for i in range(len(tools) - 1):
                cur, ctx, nxt = tools[i], ctxs[i], tools[i + 1]
                dists = {
                    "M0_marginal": [prob_marginal(m0, v, vocab) for v in vocab],
                    "M1_first": [prob_first(m1, cur, v, vocab, m0) for v in vocab],
                    "M2_cond": [prob_cond(m2, cur, ctx, v, vocab, m1, m0)
                                for v in vocab],
                    "M3_repeat": [(0.9 if v == cur else 0.1 / max(1, len(vocab) - 1))
                                  for v in vocab],
                }
                for name, ps in dists.items():
                    z = sum(ps)
                    p = ps[vocab.index(nxt)] / z
                    st = stats[name]
                    st["logp"] += math.log(max(p, 1e-12))
                    st["n"] += 1
                    order = sorted(range(len(vocab)), key=lambda j: -ps[j])
                    ranked = [vocab[j] for j in order]
                    if ranked[0] == nxt:
                        st["hit1"] += 1
                    if nxt in ranked[:3]:
                        st["hit3"] += 1
                # U^tool_t：M2 分布的条件熵
                ent_rows.append({
                    "instance_id": s["instance_id"], "project": s["project"],
                    "t": i + 1, "cur_tool": cur, "ctx": ctx,
                    "H_tool": round(entropy(dists["M2_cond"]), 4),
                })

    res = {"n_sequences": len(seqs), "n_transitions": n_trans,
           "vocab": vocab, "n_projects": len(projects), "models": {}}
    for name, st in stats.items():
        if not st["n"]:
            continue
        res["models"][name] = {
            "n": st["n"],
            "perplexity": round(math.exp(-st["logp"] / st["n"]), 4),
            "top1": round(st["hit1"] / st["n"], 4),
            "top3": round(st["hit3"] / st["n"], 4),
        }
    Hs = [r["H_tool"] for r in ent_rows]
    res["U_tool"] = {
        "n": len(Hs),
        "mean": round(statistics.mean(Hs), 4) if Hs else None,
        "median": round(statistics.median(Hs), 4) if Hs else None,
        "p10": round(sorted(Hs)[len(Hs) // 10], 4) if Hs else None,
        "p90": round(sorted(Hs)[9 * len(Hs) // 10], 4) if Hs else None,
        "max_possible": round(math.log(len(vocab)), 4),
    }
    res["entropy_rows"] = ent_rows
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V1 工具转移模型（LORO over {len(projects)} repos）===")
    print(f"{'模型':<14}{'perplexity':>12}{'top-1':>9}{'top-3':>9}")
    for name in ("M0_marginal", "M1_first", "M2_cond", "M3_repeat"):
        if name in res["models"]:
            m = res["models"][name]
            print(f"{name:<14}{m['perplexity']:>12}{m['top1']:>9}{m['top3']:>9}")
    u = res["U_tool"]
    print(f"\nU^tool 条件熵: mean={u['mean']} median={u['median']} "
          f"p10={u['p10']} p90={u['p90']}  (上界 log|V|={u['max_possible']})")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
