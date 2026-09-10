"""归因实验编排 —— C3 的入口。

对 runner 产出的 run（每条有 rollout 副本），对选定子集同时做：
  1) 因果归因（causal.py）：fork-resample 估 Δ_t → top-1 根因步
  2) full-trace judge 归因（judge.py）
  3) prefix-only judge 归因（隔离碰撞偏差；prefix 取到 causal top-1 之前，避免泄露）

产出 per-run jsonl，交给 analyze/attribution_compare.py 出 D1/D3。

选谁做归因
----------
优先失败 run（Y=0，有明确"哪步害的"）。若失败样本不足，
对成功 run 也做（问"哪步最关键"），C3 变为"关键步是否一致"。
两种都记录 outcome_Y，分析时可分层。

成本控制
--------
- 因果归因用窗口（默认在 trace 上等距取 W 个点，避免尾段偏置——D3 的方法学要求）
- m 默认 4；judge 每 run 2 次调用（full + prefix）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer                            # noqa: E402
from pilot.harness.runner import write_codex_config                     # noqa: E402
from pilot.probes import task as T                                       # noqa: E402
from pilot.attribution import causal as C                                # noqa: E402
from pilot.attribution import judge as J                                 # noqa: E402


def equidistant_window_points(n_steps: int, w: int) -> list[int]:
    """在 [1, n] 上等距取 w 个点（含首尾附近），避免尾段偏置污染 D3。"""
    if n_steps <= w:
        return list(range(1, n_steps + 1))
    return sorted({max(1, round(1 + i * (n_steps - 1) / (w - 1))) for i in range(w)})


def causal_top1_over_points(
    srv, base_items, inst, workdir, model, provider, m, points, turn_timeout,
    target_file: str | None = None,
) -> dict:
    """在指定离散点上估 v_t，取相邻点差分近似 Δ，返回 top-1（最负）步。

    target_file 非空 → per-file 归因：估 v_t^(f)，回答"哪一步决定了 f 是否被找到"
    （DESIGN_v2 §4.1.1；被漏文件是最强负向信号）。

    注意：离散点上的差分是"段效应"近似，正式实验用连续窗口；pilot 为控成本用等距点。
    """
    st = R.steps(base_items)
    vals = {}
    pts = sorted(set([0] + points))
    for t in pts:
        vals[t] = C.estimate_v(srv, base_items, t, inst, workdir, model, provider,
                               m, turn_timeout, target_file=target_file)
    seg = []
    for i in range(1, len(pts)):
        t0, t1 = pts[i - 1], pts[i]
        d = vals[t1].v_hat - vals[t0].v_hat
        # 段效应归到段末点 t1（该段最后执行的步）
        a, b = R.action_bucket(st[t1 - 1], inst.repo)
        # 持久化每个估值点的 m 次原始 recall（V4 诊断的数据缺口修复）：
        # 只存 delta 会导致 U^exec 只能由 v̂ 反推 Bernoulli 方差，而 recall 是
        # [0,1] 连续值，反推失真 —— 实测推导值被 v̂ 粗粒度锁死，真方差却是细粒度。
        seg.append({"t": t1, "seg_from": t0, "delta": round(d, 4),
                    "action": f"{a}#{b}", "pos": round(t1 / len(st), 4),
                    "v_t": round(vals[t1].v_hat, 4),
                    "v_prev": round(vals[t0].v_hat, 4),
                    "ys_t": [round(y, 4) for y in vals[t1].ys],
                    "ys_prev": [round(y, 4) for y in vals[t0].ys]})
    if not seg:
        return {"top1_step": None}
    top1 = min(seg, key=lambda s: s["delta"])       # 最伤结果的段
    return {"top1_step": top1["t"], "top1_pos": top1["pos"],
            "top1_delta": top1["delta"], "segments": seg,
            "v0": round(vals[0].v_hat, 4),
            "ys_v0": [round(y, 4) for y in vals[0].ys],
            "n_steps": len(st)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="runner 输出目录（含 results.jsonl 与 rollouts/）")
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--m", type=int, default=4, help="fork-resample 次数")
    ap.add_argument("--window-points", type=int, default=6, help="等距估值点数")
    ap.add_argument("--only-failed", action="store_true", help="只对 Y=0 的 run 归因")
    ap.add_argument("--per-file", action="store_true",
                    help="per-file 召回归因：对被漏 gold 文件估 v_t^(f)（DESIGN §4.1.1）")
    ap.add_argument("--multifile-only", action="store_true",
                    help="只跑多文件任务（gold>=2）——单文件任务实测 100% 无信号")
    ap.add_argument("--outcome", choices=["binary", "per_file", "recall"], default="per_file",
                    help="结果变量：binary=任务级Y；per_file=单个gold文件命中；"
                         "recall=连续召回率（修 Δ≡0 退化，推荐）")
    ap.add_argument("--prefix-frac", type=float, default=0.6,
                    help="prefix-only judge 的截断比例（固定值，避免依赖 causal 结果的构造性偏差）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--turn-timeout", type=float, default=600.0)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    results = [json.loads(l) for l in (runs_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    results = [r for r in results if r.get("ok") and r.get("rollout")]
    if args.only_failed:
        results = [r for r in results if r.get("Y") == 0]
    if args.multifile_only:
        results = [r for r in results if r.get("n_gold", 1) >= 2]
    if args.limit > 0:
        results = results[: args.limit]

    out_path = Path(args.out) if args.out else runs_dir / "attribution.jsonl"
    workroot = runs_dir / "attr_work"
    codex_home = runs_dir / "attr_codex_home"
    write_codex_config(codex_home, args.model)

    insts = {i.instance_id: i for i in T.load_instances()}
    srv = AppServer(codex_bin=args.codex_bin, codex_home=codex_home,
                    log_path=runs_dir / "attr_appserver.log", experimental_api=True)
    n = 0
    t0 = time.time()
    try:
        with open(out_path, "a", encoding="utf-8") as fh:
            for r in results:
                iid = r["instance_id"]
                inst = insts[iid]
                rollout_file = runs_dir / r["rollout"]
                if not rollout_file.exists():
                    continue
                base_items = R.load(rollout_file)
                st = R.steps(base_items)
                if len(st) < 3:
                    continue
                # per-file 归因目标：优先被漏的 gold 文件（最强负向信号），
                # 其次多文件任务的首个 gold；单文件任务退回任务级二元 Y。
                pred = frozenset(r["answer"]) if r.get("answer") else None
                missed = T.missed_files(pred, inst.gold)
                target_file = None
                if args.outcome == "recall":
                    # 连续结果变量，修 Δ≡0 退化（实测 50% 的 per-file case 失效）
                    target_file = "__recall__"
                elif args.per_file:
                    if missed:
                        target_file = missed[0]
                    elif len(inst.gold) >= 2:
                        target_file = sorted(inst.gold)[0]

                rec = {"instance_id": iid, "project": r.get("project"),
                       "Y": r["Y"], "n_steps": len(st),
                       "n_gold": len(inst.gold), "recall": r.get("recall"),
                       "missed_files": missed, "target_file": target_file,
                       "attr_mode": "per_file" if target_file else "task_binary"}
                try:
                    pts = equidistant_window_points(len(st), args.window_points)
                    wd = workroot / iid
                    caus = causal_top1_over_points(
                        srv, base_items, inst, wd, args.model, args.model_provider,
                        args.m, pts, args.turn_timeout, target_file=target_file)
                    rec["causal_top1_step"] = caus.get("top1_step")
                    rec["causal_top1_pos"] = caus.get("top1_pos")
                    rec["causal_top1_delta"] = caus.get("top1_delta")
                    rec["causal_v0"] = caus.get("v0")
                    # 退化检测：Δ 全为 0 ⇒ 归因无法区分任何步骤（须如实统计，不可隐藏）
                    segs = caus.get("segments", [])
                    rec["causal_segments"] = segs
                    rec["degenerate"] = bool(segs) and all(
                        abs(s.get("delta", 0.0)) < 1e-9 for s in segs)

                    jf = J.judge_attribution(srv, base_items, inst, r["Y"],
                                             args.model, args.model_provider,
                                             mode="full", turn_timeout=args.turn_timeout)
                    if jf and jf.get("parsed"):
                        rec["judge_full_top1_step"] = jf["top1_step"]
                        rec["judge_full_top1_pos"] = jf["top1_pos"]

                    # prefix-only：截断到**固定比例**（不依赖 causal 结果）。
                    # ⚠️ 曾用"截到 causal top1 之前"，那会人为压低 prefix judge 的可选步位，
                    # 制造"full 晚于 prefix"的假象 —— 构造性偏差，已修（PREREGISTRATION §3）。
                    upto = max(2, int(round(len(st) * args.prefix_frac)))
                    jp = J.judge_attribution(srv, base_items, inst, r["Y"],
                                             args.model, args.model_provider,
                                             mode="prefix", prefix_upto=upto,
                                             turn_timeout=args.turn_timeout)
                    if jp and jp.get("parsed"):
                        rec["judge_prefix_top1_step"] = jp["top1_step"]
                        rec["judge_prefix_top1_pos"] = jp["top1_pos"]
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    rec["trace"] = traceback.format_exc()[-500:]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                print(f"[{n}/{len(results)}] {iid} Y={r['Y']} "
                      f"causal_top1={rec.get('causal_top1_step')} "
                      f"judge_top1={rec.get('judge_full_top1_step')}", flush=True)
    finally:
        srv.close()
    print(f"\n完成 {n} 条归因，用时 {(time.time()-t0)/60:.1f}min → {out_path}")
    print(f"下一步: python -m pilot.analyze.attribution_compare --attr {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
