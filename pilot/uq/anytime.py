"""Anytime 不确定性量化：AUROC(t) / ECE(t) —— UQ 主线实验。

回答用户的核心诉求："对 AI 执行**过程**进行不确定性量化"。

设计动机
--------
早期我们把 UQ 的落点错误地定在 v_0（仅凭 issue 文本 + 仓库静态特征做执行前预测），
实测 LORO AUROC ≈ 0.5（随机）。但这只说明"执行前信息量不足"，不说明过程不可量化。
正确的落点是 v_t 随 t 的演化：**执行到第几步，我们才能可靠预知结果**。

严格的反泄漏约束
----------------
特征只允许使用 tau_{1:t} 中 agent 自己可见的信息。
**绝对禁止**任何依赖 gold 的量（如"是否已读过 gold 文件"）——那会让 AUROC 虚高且无意义。
下面每个特征都只由 (工具名, 目标路径, 输出文本) 计算，与 F* 无关。

生存偏差处理
------------
随 t 增大只有长 episode 存活（步数>=16 仅 48%）。因此：
  1. 每个 t 都报 n(t)
  2. 额外做**固定队列**分析：只用步数 >= T_max 的 episode，在 t=1..T_max 上评估，
     消除组成变化带来的偏差。两条曲线都报。
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
    "log_t",              # 已执行步数（log）
    "n_distinct_targets", # 触达的不同目标数
    "n_distinct_dirs",    # 触达的不同目录数
    "search_ratio",       # 搜索类动作占比（探索 vs 精读）
    "log_empty_outputs",  # 空/极短输出次数（搜索无命中 = 方向不对的信号）
    "log_out_chars",      # 累计观测长度（信息摄入量）
    "action_entropy",     # 动作桶分布熵（行为集中度）
    "repeat_rate",        # 重复动作占比（停滞信号）
    "max_dir_depth",      # 最深目录层级
    "new_target_rate",    # 最近若干步引入新目标的比率（仍在探索 vs 已收敛）
]


def _cls(tool: str, args_raw: str) -> str:
    """把动作粗分为 search / read / other —— 只看工具名与命令头，不看 gold。"""
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
    """在第 t 步截面上抽取特征。仅使用前 t 步的 agent 可见信息。"""
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
    # 最近 min(5,n) 步里出现的新目标比率
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
    """对每个 cutoff t 构造 (X, y, groups)。只纳入 n_steps >= t 的 episode。

    multifile_only：只用 |F*|>=2 的实例。全量语料 94% 基线率下仅 30 个失败且
    5 个 repo 零失败，LORO 每折的判别信号是个位数；多文件子集失败率 27%，
    类别平衡得多，且论文已认定判别信号集中在该子集。
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
        # rollout 可能在任一 shard 下
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
    """leave-one-repository-out 评估。返回 AUROC/ECE/Brier/AURC。"""
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
        return {"n": int(ok.sum()), "note": "样本或类别不足"}
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
                    help="固定队列：只用步数>=该值的 episode 消除生存偏差")
    ap.add_argument("--multifile-only", action="store_true")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--out", default="pilot/runs/anytime.json")
    args = ap.parse_args()

    ds = build_anytime_dataset(Path(args.results), Path(args.runs_root), args.cutoffs,
                               multifile_only=args.multifile_only)
    print(f"载入 {ds['n_loaded']} 条 rollout")

    # --- 全体曲线（样本随 t 衰减，报 n(t)）---
    curve = []
    for t in args.cutoffs:
        d = ds["per_t"][t]
        if len(d["y"]) < args.min_n:
            curve.append({"t": t, "n": len(d["y"]), "note": "样本不足"})
            continue
        res = loro_eval(np.array(d["X"]), np.array(d["y"]), d["g"])
        res["t"] = t
        curve.append(res)
        print(f"  t={t:>2}  n={res.get('n')}  AUROC={res.get('AUROC')}  "
              f"ECE={res.get('ECE')}  AURC={res.get('AURC')}")

    # --- 固定队列曲线（组成不变，排除生存偏差）---
    T0 = args.cohort_t
    cohort_ids = {i for i, L in ds["steplen"].items() if L >= T0}
    cohort = []
    print(f"\n固定队列（步数>={T0}）: {len(cohort_ids)} episode")
    for t in [x for x in args.cutoffs if x <= T0]:
        d = ds["per_t"][t]
        keep = [j for j, iid in enumerate(d["ids"]) if iid in cohort_ids]
        if len(keep) < args.min_n:
            cohort.append({"t": t, "n": len(keep), "note": "样本不足"})
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
        "leakage_note": ("特征仅由 (工具名, 目标路径, 输出文本) 计算，"
                         "不含任何依赖 gold 的量"),
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
