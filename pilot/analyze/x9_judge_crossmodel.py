"""X9：碰撞隔离与 judge 机制的**跨模型复现**。API，但预算极小。

为什么是这个实验
----------------
论文自己写着的外部效度限制：「结果来自单一 agent 与单一模型，把位移归因于
full-trace judging 的一般性之前需要跨模型复现」。本实验正是补它。

为什么不是 X3
-------------
新 key 的速率上限实测为 **10 次调用 / 60 分钟**（端点直接返回
`您当前调用的API超过了10次/60.0分钟`）。X3 需要 1680 次续跑，每次续跑要 agent
再跑 15–20 步 ≈ 15–20 次调用 → 约 3400 小时。**差三个数量级，不可行。**
而 judge 判决**每次只需 1 次调用**（不用工具、一次性 thread），于是

    25 episode × 2 条件（full / prefix）= 50 次调用 ≈ 5.5 小时

同一批判决还能**零额外成本**地复现"judge 是否也在挑 gold 溯源步"。

被估量（与原实验同构）
----------------------
  C = pos(judge_full) − pos(judge_prefix)      碰撞隔离对比
  prefix 截断规则沿用原实验：upto = max(2, round(n_steps × 0.6))
  judge 富集度 = judge top-1 落在 gold 溯源步的比例 vs 全部步中的占比

⚠️ 模型不同带来的限定
---------------------
原实验的 judge 是 gpt-5.5-0424-global；本实验是 qwen 系列。这**正是**跨模型复现
的意义所在。但注意：被评分的 trace 仍由 gpt-5.5 生成，所以这里换的是**评分者**
而非**被评者**——恰好是论文那条限制说的那件事。

判据（预注册）
--------------
| 结果 | 结论 |
|---|---|
| C 显著 > 0 且量级相当 | 碰撞偏差**不是单一 judge 的特性**，论文的外部效度限制解除 |
| C 显著 > 0 但量级小很多 | 机制存在但强度依模型而异，如实报告两个数字 |
| C ≈ 0 或反向 | 原结果**不可跨模型推广**，必须收窄为"关于该 judge 的陈述" |
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.analyze.judge_http import judge_http              # noqa: E402
from pilot.harness import rollout as R                        # noqa: E402
from pilot.probes import task as T                            # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402


class RateLimiter:
    """滑动窗口速率限制。实测端点为 10 次 / 60 分钟。

    每次调用前阻塞到窗口内可用；遇到端点报速率超限时可用 `penalize()` 追加退避。
    """

    def __init__(self, max_calls: int, window_s: float, safety_s: float = 20.0,
                 assume_full: bool = False):
        self.max_calls = max_calls
        self.window_s = window_s
        self.safety_s = safety_s
        self.stamps: deque[float] = deque()
        if assume_full:
            # 启动时窗口可能已被之前的探测占满；保守起见按"刚刚用满"起算，
            # 于是首次调用会先等一个完整窗口，而不是立刻撞限制。
            now = time.time()
            for _ in range(max_calls):
                self.stamps.append(now)

    def acquire(self) -> None:
        while True:
            now = time.time()
            while self.stamps and now - self.stamps[0] > self.window_s:
                self.stamps.popleft()
            if len(self.stamps) < self.max_calls:
                self.stamps.append(now)
                return
            wait = self.window_s - (now - self.stamps[0]) + self.safety_s
            print(f"    [rate] 窗口已满 {len(self.stamps)}/{self.max_calls}，"
                  f"等待 {wait/60:.1f} 分钟", flush=True)
            time.sleep(max(5.0, wait))

    def penalize(self, extra_s: float) -> None:
        print(f"    [rate] 端点报超限，追加退避 {extra_s/60:.1f} 分钟", flush=True)
        time.sleep(extra_s)


def rate_limited_error(res: dict | None) -> bool:
    raw = json.dumps(res or {}, ensure_ascii=False)
    return ("超过了" in raw) or ("IRC-001" in raw) or ("rate" in raw.lower()
                                                       and "limit" in raw.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x9_judge_crossmodel.jsonl")
    ap.add_argument("--report", default="pilot/runs/x9_judge_crossmodel.json")
    ap.add_argument("--model", default="qwen3.6-plus")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--wire-api", default="responses",
                    help="codex 0.145 已移除 chat 支持（`wire_api = \"chat\"` is no "
                         "longer supported），故只能用 responses")
    ap.add_argument("--workdir", default="pilot/runs/x9_work")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--limit", type=int, default=25,
                    help="episode 数；每个耗 2 次调用")
    ap.add_argument("--prefix-frac", type=float, default=0.6)
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--turn-timeout", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=4,
                    help="遇到端点速率限制时的重试次数（速率限制不算故障）")
    ap.add_argument("--backoff-min", type=float, default=20.0,
                    help="速率限制后的退避分钟数")
    ap.add_argument("--assume-window-full", action="store_true",
                    help="启动时假定速率窗口已满（先等一个窗口再发第一次调用）")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    roll = {p.stem.split("__k")[0]: p
            for p in Path(args.runs_root).glob("shard*/rollouts/*.jsonl")}
    recs = []
    for f in sorted(Path(args.attr_dir).glob("shard*/attribution.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    # 只取原实验两个 judge 条件都成功解析的 episode，保证配对可比
    usable = [r for r in recs
              if r.get("judge_full_top1_pos") is not None
              and r.get("judge_prefix_top1_pos") is not None
              and r["instance_id"] in roll]
    usable.sort(key=lambda r: r["instance_id"])
    print(f"原实验可配对 episode {len(usable)}，本次取前 {args.limit}")

    out_path = Path(args.out)
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["instance_id"]] = d
        print(f"断点续跑：已完成 {len(done)}")

    todo = [r for r in usable[:args.limit] if r["instance_id"] not in done]

    if todo and not args.analyze_only:
        import os
        if not os.environ.get("IDEALAB_API_KEY"):
            print("❌ 未设置 IDEALAB_API_KEY，judge 调用必然失败，已中止。")
            return 2
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
        rl = RateLimiter(args.max_calls, args.window_min * 60,
                         assume_full=args.assume_window_full)
        n_fail = 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for i, r in enumerate(todo, 1):
                iid = r["instance_id"]
                inst = insts[iid]
                base = R.load(roll[iid])
                st = R.steps(base)
                upto = max(2, int(round(len(st) * args.prefix_frac)))
                rec = {"instance_id": iid, "model": args.model,
                       "n_steps": len(st), "prefix_upto": upto}
                for mode in ("full", "prefix"):
                    print(f"  [{i}/{len(todo)}] {iid} judge={mode}", flush=True)
                    res = None
                    for attempt in range(1, args.max_retries + 1):
                        rl.acquire()
                        try:
                            res = judge_http(
                                base, inst, r.get("Y", 0), args.model,
                                mode=mode,
                                prefix_upto=(upto if mode == "prefix" else None),
                                timeout=args.turn_timeout)
                        except Exception as exc:  # noqa: BLE001
                            res = {"parsed": False,
                                   "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                        # 速率限制不是故障：退避后重试，不计入解析失败
                        if rate_limited_error(res) and attempt < args.max_retries:
                            rl.penalize(args.backoff_min * 60)
                            continue
                        break
                    rec[mode] = res
                    if not (res or {}).get("parsed"):
                        n_fail += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()          # 逐条落盘，崩溃不丢已花掉的配额
                done[iid] = rec
                # 首个 episode 若两条都失败，立刻停手
                rate_only = all(rate_limited_error(rec.get(m))
                                for m in ("full", "prefix"))
                if (i == 1 and not rate_only
                        and not any((rec.get(m) or {}).get("parsed")
                                    for m in ("full", "prefix"))):
                    print("❌ FAIL-FAST：首个 episode 两个条件都未解析，已中止。")
                    for m in ("full", "prefix"):
                        print(f"     {m}: {json.dumps(rec.get(m), ensure_ascii=False)[:220]}")
                    return 3
        print(f"\n完成，解析失败 {n_fail} 次调用")

    # ------------------------------------------------------------- 分析
    orig = {r["instance_id"]: r for r in usable}
    pairs = []
    for iid, d in done.items():
        f_, p_ = d.get("full") or {}, d.get("prefix") or {}
        if not (f_.get("parsed") and p_.get("parsed")):
            continue
        o = orig.get(iid)
        if o is None:
            continue
        pairs.append({
            "instance_id": iid,
            "new_full_pos": f_["top1_pos"], "new_prefix_pos": p_["top1_pos"],
            "new_full_step": f_["top1_step"], "new_prefix_step": p_["top1_step"],
            "old_full_pos": o["judge_full_top1_pos"],
            "old_prefix_pos": o["judge_prefix_top1_pos"],
            "old_full_step": o.get("judge_full_top1_step"),
        })
    if not pairs:
        print("\n无可分析样本（尚未产生成功判决）")
        return 0

    nd = [x["new_full_pos"] - x["new_prefix_pos"] for x in pairs]
    od = [x["old_full_pos"] - x["old_prefix_pos"] for x in pairs]
    from math import comb
    nz = [d for d in nd if d != 0]
    k = sum(1 for d in nz if d > 0)
    sign_p = (min(1.0, 2 * sum(comb(len(nz), i) for i in range(k, len(nz) + 1))
                  / 2 ** len(nz)) if nz else None)

    # judge 富集度（同一批判决，零额外调用）
    enrich, base_share = [], []
    for x in pairs:
        inst = insts[x["instance_id"]]
        st = R.steps(R.load(roll[x["instance_id"]]))
        rel = gold_relation(st, set(inst.gold))
        allst = sorted(rel)
        s = x["new_full_step"]
        if s in rel:
            enrich.append(1.0 if rel[s]["any_gold"] else 0.0)
            base_share.append(sum(1 for t in allst if rel[t]["any_gold"]) / len(allst))

    rep = {
        "model": args.model, "n_paired": len(pairs),
        "COLLIDER_REPLICATION": {
            "new_model_displacement": boot_ci(nd),
            "orig_model_displacement_same_episodes": boot_ci(od),
            "new_minus_orig": paired_ci(nd, od),
            "n_later": sum(1 for d in nd if d > 0), "n_tied": len(nd) - len(nz),
            "sign_test_p": sign_p,
            "new_full_pos_mean": round(statistics.mean(
                x["new_full_pos"] for x in pairs), 4),
            "new_prefix_pos_mean": round(statistics.mean(
                x["new_prefix_pos"] for x in pairs), 4),
        },
        "JUDGE_ENRICHMENT": {
            "observed": boot_ci(enrich),
            "baseline_all_steps": boot_ci(base_share),
            "diff": paired_ci(enrich, base_share),
        },
        "pairs": pairs,
    }
    Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    cr = rep["COLLIDER_REPLICATION"]
    print(f"\n=== 碰撞隔离的跨模型复现（{args.model}，n={len(pairs)}）===")
    print(f"  新模型 full−prefix 位移 {cr['new_model_displacement']['mean']:+.4f} "
          f"CI{cr['new_model_displacement']['ci95']}")
    print(f"  原模型（同一批 episode）  {cr['orig_model_displacement_same_episodes']['mean']:+.4f} "
          f"CI{cr['orig_model_displacement_same_episodes']['ci95']}")
    d = cr["new_minus_orig"]
    if "note" not in d:
        print(f"  两模型之差 {d['diff']:+.4f} CI{d['ci95']}  "
              f"{'显著' if d['significant'] else '不显著'}")
    print(f"  变晚 {cr['n_later']}/{len(nd)}（并列 {cr['n_tied']}），"
          f"符号检验 p={cr['sign_test_p']}")
    print(f"  平均位置 full {cr['new_full_pos_mean']} vs prefix {cr['new_prefix_pos_mean']}")
    je = rep["JUDGE_ENRICHMENT"]
    if "note" not in je["observed"]:
        print(f"\n=== judge 富集度（同一批判决，零额外调用）===")
        print(f"  落在 gold 溯源步 {je['observed']['mean']:.4f} "
              f"CI{je['observed']['ci95']}   基线 {je['baseline_all_steps']['mean']:.4f}")
        dd = je["diff"]
        if "note" not in dd:
            print(f"  差值 {dd['diff']:+.4f} CI{dd['ci95']}  "
                  f"{'显著' if dd['significant'] else '不显著'}")
    print(f"\n→ {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
