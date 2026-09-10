"""X1d：query 类型真的不可预测，还是解析与建模造成的？零 API。

来由
----
X1c 报告 query 类型（21.6% 的步）top-3 = 0.029，并据此把"自由文本查询"写成
next-action 预测的硬上界。但两处可疑：

**(a) 解析损坏。** `parsed_command.parse_segment` 用 `shlex.split`，它在
`grep "a\\|b"` 这类未闭合引号上抛 ValueError，回退到 `seg.split()` 后**保留引号并按
空格截断**。实测：`shlex.split` 在 26.6% 的命令分段上失败；query 对象里
**75.4% 带残留引号或尾反斜杠**，**25.5% 退化成 `"def` / `"class` 这种通用片段**。
其中 `"def` 单独出现 251 次 —— 一个高频对象本该好预测。

**(b) 模型偏弱。** x1c 的 query 分量只用"最近 10 个 query 按新近度加权"。
但 query 绝大多数是**标识符**（`in_bulk`、`formfield_for_foreignkey`、
`supports_logical_xor`），其来源应是**问题描述**与**此前工具输出里出现的符号名**。

本脚本做三件事
--------------
  1. 修一个更稳健的 query 抽取器（不依赖 shlex 成功），量化修复前后的差异
  2. 换更合适的候选生成：频次 + 问题描述标识符 + 输出中的符号名
  3. 在**同一批点**上比较，判定 0.029 有多少归因于解析、多少归因于建模、
     多少是真正的不可预测

判据（预注册）
--------------
| 结果 | 结论 |
|---|---|
| 修复后 top-3 显著上升 | 原 0.029 是**测量与建模产物**，论文关于"自由文本是硬上界"的表述须收窄 |
| 仅解析修复就上升 | 归因于解析；`parsed_command` 需修，且影响所有用到 query 的分析 |
| 两者都改后仍 ≈0.03 | query 确实不可预测，原结论成立且更稳固 |
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
# 标识符：Python 风格的名字，含点号属性访问
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
# 输出里的定义行
DEFN = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
STOP = {"def", "class", "import", "from", "return", "if", "for", "while", "self",
        "None", "True", "False", "and", "or", "not", "in", "is", "as", "with",
        "try", "except", "raise", "pass", "print", "len", "str", "int", "list",
        "dict", "set", "type", "object", "super", "test", "tests"}


def repair_query(cmd: str) -> str | None:
    """从命令里稳健地取出搜索模式，不依赖 shlex 成功。

    优先取**引号内的完整内容**（这是 shlex 失败时丢掉的东西），
    否则取第一个非选项 token。再去掉正则元字符与转义，留下核心标识符。
    """
    for seg in re.split(r"\|\||&&|;", cmd):
        toks = seg.strip().split()
        if not toks:
            continue
        head = toks[0].split("/")[-1]
        if head not in SEARCH_HEADS:
            continue
        # 引号内内容：允许内部含 \| 等未闭合语法
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
    """把一个搜索模式化成它包含的标识符（去掉正则与停用词）。"""
    out = []
    for m in IDENT.finditer(raw):
        w = m.group(0)
        if w in STOP or len(w) < 3:
            continue
        if w not in out:
            out.append(w)
    return out


def norm_query(raw: str) -> str:
    """规范化后的 query 键：优先用核心标识符拼接，否则用清理后的原串。"""
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
    print(f"载入 {len(eps)} episode")

    # ---- 解析修复的规模 ----
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
            # 训练集里的 query 边际频次（LOO）
            tr_new = Counter(s["new_q"] for x in train for s in x["seq"] if s["new_q"])
            tr_old = Counter(s["old_q"] for x in train for s in x["seq"] if s["old_q"])
            # 问题描述里的标识符
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
            "note": "old 是 parsed_command 现状（shlex 失败即回退按空格切），"
                    "new 是本脚本的稳健抽取 + 标识符规范化",
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
    print(f"\n=== 解析修复的规模 ===")
    print(f"  query 对象 old={pr['n_query_objects_old']} new={pr['n_query_objects_new']}")
    print(f"  old 中受损比例 {pr['damaged_share_old']:.1%}")
    print(f"  不同取值 old={pr['distinct_old']} new={pr['distinct_new']}")
    print(f"  old 最常见片段: {pr['top_old_fragments'][:4]}")
    print(f"  new 最常见键:   {pr['top_new_keys'][:4]}")

    print(f"\n=== 模型对比（query 步，n={n_pts}）===")
    print(f"{'模型':<38}{'覆盖':>8}{'top1':>8}{'top3':>8}{'top10':>8}{'有效候选':>9}")
    for m in MODELS:
        d = res["MODELS"][m]
        print(f"{m:<38}{d['coverage']:>8.4f}{d['topk']['top1']['mean']:>8.4f}"
              f"{d['topk']['top3']['mean']:>8.4f}{d['topk']['top10']['mean']:>8.4f}"
              f"{str(d['eff_candidates']):>9}")

    print(f"\n=== 归因分解（配对，top-3）===")
    for k, v in res["CONTRASTS"].items():
        if "note" in v:
            print(f"  {k:<52} {v['note']}")
            continue
        print(f"  {k:<52} {v['diff']:+.4f} CI[{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]  {'显著' if v['significant'] else '不显著'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
