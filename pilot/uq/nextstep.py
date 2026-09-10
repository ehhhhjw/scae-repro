"""Next-step 动作预测 —— 补齐"对 Next Step 进行概率计算"这一目标。

任务
----
给定前缀 tau_{1:t}，预测下一动作 A_{t+1} 的等价类 (tool, target node in G)。
这是 policy pi(.|tau_{1:t}) 的可评估代理：RQ1 只测了该分布的**熵**（overlap 诊断），
这里补上它的**可预测性**，两者共同刻画执行过程的策略不确定性。

候选集构造（不泄漏 gold）
------------------------
对每个决策点，候选 = 该 episode 已出现过的动作桶
              ∪ 该 repo 语料中的高频动作桶（top-K，从**训练折**统计）
标签在候选集内时才计入（覆盖率单独报告），否则任何模型都不可能命中。

打分特征（全部只依赖 tau_{1:t} 与语料统计）
-----------------------------------------
  f1 该候选在训练折的全局频率（log）
  f2 该候选在同 repo 训练折的频率（log）
  f3 是否在当前 episode 已出现过
  f4 与上一步同工具
  f5 与上一步同目录
  f6 距上次使用该候选的步距（log，越近越可能重复）
  f7 该候选是否为上一步动作本身（自重复）

评估：leave-one-repository-out，指标 top-1 / top-3 accuracy 与 MRR。
基线：(a) 边际频率（只用 f1）(b) 重复上一步 (c) 随机。
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
    """构造 (决策点, 候选集, 标签) 三元组，按 repo 分组以便 LORO。"""
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

        # 训练折的语料统计（严禁用测试折）
        gfreq = Counter(b for e in train_eps for b in e["seq"])
        n_g = sum(gfreq.values())
        rfreq_by = defaultdict(Counter)
        for e in train_eps:
            rfreq_by[e["project"]].update(e["seq"])
        # held-out repo 没有自身训练统计，用全局 top-K 作为候选补充
        global_top = [b for b, _ in gfreq.most_common(topk_repo)]

        # --- 训练打分模型：正样本=真实下一步，负样本=同候选集其余 ---
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

        # --- 测试 ---
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
        return {"note": "无有效决策点"}
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
    """粗粒度动作类：跨 repo 可迁移的结构特征，不含具体路径。

    LORO 下精确桶（含文件路径）不可迁移——路径在别的 repo 不存在。
    这里保留 (工具, 是否测试文件, 目录深度档, 是否有具体目标) 四个可迁移维度。
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
    """同 repo 内 leave-one-episode-out —— next-step 预测的正确评估设置。

    实际用途是"在这个仓库里预测 agent 下一步"，不需要跨仓库迁移，
    因此泛化单元是 episode 而非 repo。
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
        return {"note": "无有效决策点"}
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
                    help="用粗粒度动作类（跨 repo 可迁移）")
    ap.add_argument("--within-repo", action="store_true",
                    help="同 repo 内 leave-one-episode-out（正确设置）")
    ap.add_argument("--out", default="pilot/runs/nextstep.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.results), Path(args.runs_root))
    print(f"载入 {len(eps)} episode，"
          f"共 {sum(len(e['seq'])-1 for e in eps)} 个步转移")
    if args.coarse:
        for e in eps:
            e["seq"] = [coarsen(b) for b in e["seq"]]
        print("已粗粒度化动作类（跨 repo 可迁移）")
    by_repo = build_pairs(eps, args.topk_repo)
    print(f"repo 数: {len(by_repo)}")
    res = (evaluate_within_repo(by_repo, args.topk_repo) if args.within_repo
           else evaluate_loro(by_repo, args.topk_repo))
    res["n_episodes"] = len(eps)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\n=== Next-step 预测（leave-one-repository-out）===")
    print(f"  决策点 {res.get('n_decision_points')}  "
          f"候选覆盖率 {res.get('candidate_coverage')}")
    print(f"  top-1 {res.get('top1')}  top-3 {res.get('top3')}  MRR {res.get('MRR')}")
    print(f"  基线 top-1: {res.get('baselines_top1')}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
