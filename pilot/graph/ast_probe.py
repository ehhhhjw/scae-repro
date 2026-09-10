"""AST 特征的判别性检验 —— 先验证再投入。

在为 499 个仓库全量建索引（预计 20+ 分钟）之前，先在子集上回答一个问题：
**AST 语义特征能否区分成功与失败的定位任务？**

若不能，AST 路线与之前的行为表层特征同样无效，不值得全量投入。
若能，则说明前三次预测失败确实源于「缺语义」而非「任务不可预测」。

检验设计
--------
取全部失败 case（30 个）+ 等量分层抽样的成功 case，对每个实例：
  1. 从 issue 文本抽候选符号（大写驼峰 / snake_case 标识符）
  2. 用 AST 索引测这些符号的歧义度（定义点数、最大重名、命中文件数、目录分散度）
  3. 额外测 gold 文件在依赖图中的结构位置（入度、出度、是否叶子）
     —— 注意：gold 结构特征**仅用于解释性分析**，不能进预测特征（会泄漏）
  4. 对比成功组 vs 失败组的分布，报 Mann-Whitney U 与效应量

严格区分两类特征
----------------
  可预测用（execution-time available）：仅由 issue 文本 + 仓库 AST 计算，不含 gold
  仅解释用（post-hoc only）：涉及 F* 的结构位置，用于理解失败机制，不进模型
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.graph.ast_index import build_index, summarize     # noqa: E402
from pilot.probes import task as T                           # noqa: E402

_CAMEL = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")
_SNAKE = re.compile(r"\b[a-z_][a-z0-9_]{3,}\b")
_STOP = {"self", "none", "true", "false", "return", "import", "class", "print",
         "value", "error", "result", "should", "would", "there", "which",
         "python", "django", "test", "tests", "code", "line", "file", "with",
         "this", "that", "from", "have", "when", "then", "will", "using"}


def issue_symbols(text: str, limit: int = 25) -> list[str]:
    """从 issue 文本抽候选符号。只用文本，不看仓库、不看 gold。"""
    cams = _CAMEL.findall(text or "")
    snakes = [w for w in _SNAKE.findall(text or "") if w not in _STOP]
    cnt = Counter(cams + snakes)
    return [w for w, _ in cnt.most_common(limit)]


def mann_whitney_u(a: list[float], b: list[float]) -> dict:
    """Mann-Whitney U（正态近似）+ 效应量 rank-biserial r。无 scipy 依赖。"""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return {"n_a": na, "n_b": nb, "note": "样本不足"}
    allv = [(v, 0) for v in a] + [(v, 1) for v in b]
    allv.sort(key=lambda x: x[0])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    ra = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    ua = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (ua - mu) / sd if sd else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    r_rb = 2 * ua / (na * nb) - 1          # rank-biserial，范围 [-1,1]
    return {"n_a": na, "n_b": nb,
            "median_a": round(statistics.median(a), 3),
            "median_b": round(statistics.median(b), 3),
            "U": round(ua, 1), "z": round(z, 3), "p_two_sided": round(p, 4),
            "rank_biserial_r": round(r_rb, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--n-success", type=int, default=40,
                    help="抽多少成功 case 作对照")
    ap.add_argument("--out", default="pilot/runs/ast_probe.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    recs = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("ok")]
    fails = [r for r in ok if r.get("Y") == 0]
    succs = [r for r in ok if r.get("Y") == 1]
    # 分层抽成功对照：按 repo 与 gold 数尽量匹配失败组
    key = lambda r: (r.get("project"), min(r.get("n_gold", 1), 3))          # noqa: E731
    want = Counter(key(r) for r in fails)
    picked, pool = [], sorted(succs, key=lambda r: r["instance_id"])
    for k, need in want.items():
        cands = [r for r in pool if key(r) == k and r not in picked]
        picked += cands[: need * 2]
    for r in pool:
        if len(picked) >= args.n_success:
            break
        if r not in picked:
            picked.append(r)
    picked = picked[: args.n_success]

    target = fails + picked
    print(f"检验样本: 失败 {len(fails)} + 成功 {len(picked)} = {len(target)}")

    rows = []
    for i, r in enumerate(target, 1):
        inst = insts.get(r["instance_id"])
        if inst is None or not inst.repo.is_dir():
            continue
        try:
            idx = build_index(inst.repo)
        except (OSError, RecursionError):
            continue
        syms = issue_symbols(inst.problem_statement)
        amb = idx.ambiguity(syms)
        # 仅解释用：gold 在依赖图中的位置
        gold_in = gold_out = 0
        gold_found = 0
        for g in inst.gold:
            if g in idx.file_syms or g in idx.edges:
                gold_found += 1
            gold_out += len(idx.edges.get(g, set()))
            gold_in += sum(1 for f, outs in idx.edges.items() if g in outs)
        rows.append({
            "instance_id": r["instance_id"], "project": r.get("project"),
            "Y": r.get("Y"), "n_gold": r.get("n_gold"),
            "n_issue_syms": len(syms),
            # --- 可预测用（不含 gold）---
            "n_defsites": amb["n_defsites"],
            "max_fanout": amb["max_fanout"],
            "n_files_hit": amb["n_files_hit"],
            "spread": amb["spread"],
            "repo_files": idx.n_files,
            "hit_ratio": round(amb["n_files_hit"] / max(1, idx.n_files), 5),
            # --- 仅解释用（含 gold，禁止进模型）---
            "gold_indeg": gold_in, "gold_outdeg": gold_out,
            "gold_indexed": gold_found,
        })
        if i % 10 == 0:
            print(f"  已处理 {i}/{len(target)}", flush=True)

    A = [x for x in rows if x["Y"] == 0]     # 失败
    B = [x for x in rows if x["Y"] == 1]     # 成功
    tests = {}
    for f in ("n_defsites", "max_fanout", "n_files_hit", "spread", "hit_ratio",
              "repo_files", "gold_indeg", "gold_outdeg"):
        tests[f] = mann_whitney_u([x[f] for x in A], [x[f] for x in B])

    res = {"n_rows": len(rows), "n_fail": len(A), "n_success": len(B),
           "predictive_features": ["n_defsites", "max_fanout", "n_files_hit",
                                   "spread", "hit_ratio", "repo_files"],
           "explanatory_only": ["gold_indeg", "gold_outdeg"],
           "tests": tests, "rows": rows}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== AST 特征判别性（失败 {len(A)} vs 成功 {len(B)}）===")
    print(f"{'特征':<16}{'失败中位':>10}{'成功中位':>10}{'z':>8}{'p':>9}{'r':>8}  类别")
    for f, t in tests.items():
        if "note" in t:
            print(f"  {f:<14} {t['note']}")
            continue
        cat = "可预测" if f in res["predictive_features"] else "仅解释"
        star = " *" if t["p_two_sided"] < 0.05 else ""
        print(f"{f:<16}{t['median_a']:>10}{t['median_b']:>10}"
              f"{t['z']:>8}{t['p_two_sided']:>9}{t['rank_biserial_r']:>8}  {cat}{star}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
