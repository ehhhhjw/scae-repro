"""X3（API，最高优先）溯源掩蔽干预：把 A3 因果化。

命题
----
Agent 走向"刚在输出里看到的路径"这一溯源边是**因果的**：
掩蔽它会降低 recall。

被估量（路径特异效应，path-specific effect）
-----------------------------------------
PSE(t*) = E[Y|τ_{1:t*}] - E[Y|τ_{1:t*}, do(output_{t*} 中 gold 路径被掩蔽)]

三臂设计（instance 内配对）
--------------------------
  C 对照  : 输出不改（v_t* 的标准估计）
  M 掩蔽  : 删掉输出中提到 gold 的行（do(observation)）
  P 安慰剂: 删掉等量的非 gold 行（内部效度检查）

Fork 点 t* = first_gold 步（首次让 gold 进入 agent 观测史的步）
  - 71 个候选 instance，median first_gold_t = 2（步序号，非比例）
  - 每臂 m=8 forks × ~2.5min = 20min/instance；6 并行 → ~8.3h 总计

三档判据
--------
(i)  M ≪ C 且 P ≈ C → 溯源边因果有效；论文主结论
(ii) M ≈ C 且 P ≈ C → 信息冗余；做剂量–反应（掩蔽 k=1..K 条 gold）
(iii)P 也下降      → 算子扰动；修 mask_output_lines 再重跑

使用方法
--------
  # 生成干预清单（零 API）
  python3 -m pilot.analyze.x3_prov_mask --mode plan --out pilot/runs/x3_plan.json

  # 运行干预（API，~8.3h）
  python3 -m pilot.analyze.x3_prov_mask --mode run \
      --plan pilot/runs/x3_plan.json \
      --out pilot/runs/x3_pse.json

  # 分析结果（零 API）
  python3 -m pilot.analyze.x3_prov_mask --mode analyze \
      --out pilot/runs/x3_pse.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                          # noqa: E402
from pilot.harness.appserver import AppServer                   # noqa: E402
from pilot.harness.runner import write_codex_config             # noqa: E402
from pilot.probes import task as T                              # noqa: E402
from pilot.attribution.causal import estimate_v                 # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402
from pilot.analyze.v3j_trace_graph import PATH_RE                # noqa: E402

SEED = 20260814
M_FORKS = 8          # fork 次数（每臂）
N_BOOT = 4000        # bootstrap 次数


def _candidate_instances(
    runs_root: str,
    runs_root2: str | None = None,
) -> list[dict]:
    """找到有 first_gold 步的多文件 instance 及其 first_gold_t。

    返回列表：[{"instance_id", "rollout_path", "first_gold_t", "n_gold", "gold", "n_steps"}]
    """
    insts = {i.instance_id: i for i in T.load_instances()}
    # 收集 rollout 路径（优先 full_locate，再找 v4prime）
    roll_map: dict[str, Path] = {}
    for root in ([runs_root] + ([runs_root2] if runs_root2 else [])):
        for p in Path(root).glob("shard*/rollouts/*.jsonl"):
            iid = p.stem.split("__k")[0]
            if iid not in roll_map:
                roll_map[iid] = p

    candidates = []
    for iid, rollout_path in roll_map.items():
        inst = insts.get(iid)
        if inst is None or len(inst.gold) < 2:
            continue
        try:
            base_items = R.load(rollout_path)
        except Exception:
            continue
        st = R.steps(base_items)
        if len(st) < 3:
            continue
        rel = gold_relation(st, inst.gold)
        first_gold_t = None
        for tidx in sorted(rel.keys()):
            if rel[tidx].get("first_gold"):
                first_gold_t = tidx
                break
        if first_gold_t is None:
            continue
        candidates.append({
            "instance_id": iid,
            "rollout_path": str(rollout_path),
            "first_gold_t": first_gold_t,
            "n_gold": len(inst.gold),
            "gold": sorted(inst.gold),
            "n_steps": len(st),
        })
    return sorted(candidates, key=lambda x: x["instance_id"])


def plan_mode(args: argparse.Namespace) -> int:
    """生成干预清单（零 API）。"""
    print("计算 first_gold 候选清单...")
    cands = _candidate_instances(args.runs_root, args.runs_root2)
    print(f"候选 instance: {len(cands)}")

    # 统计
    fg_ts = [c["first_gold_t"] for c in cands]
    fg_ratios = [c["first_gold_t"] / c["n_steps"] for c in cands]
    within_5 = sum(1 for t in fg_ts if t <= 5)

    summary = {
        "n_candidates": len(cands),
        "first_gold_t": {
            "median": statistics.median(fg_ts),
            "p25": fg_ts[len(fg_ts) // 4],
            "p75": fg_ts[3 * len(fg_ts) // 4],
            "within_5_steps": within_5,
            "within_5_rate": round(within_5 / max(1, len(cands)), 3),
        },
        "first_gold_ratio": {
            "median": round(statistics.median(fg_ratios), 3),
            "mean": round(statistics.mean(fg_ratios), 3),
        },
        "estimated_budget": {
            "forks": len(cands) * 3 * M_FORKS,  # 3 arms * m
            "parallel_workers": 6,
            "min_per_fork": 2.5,
            "estimated_hours": len(cands) * 3 * M_FORKS * 2.5 / 60 / 6,
        },
        "candidates": cands,
    }

    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n=== X3 干预清单 ===")
    print(f"候选 instance: {len(cands)}")
    print(f"first_gold_t 中位数: {statistics.median(fg_ts):.0f}")
    print(f"first_gold_t ≤5: {within_5}/{len(cands)} = {within_5/max(1,len(cands))*100:.0f}%")
    print(f"预估预算: {len(cands)*3*M_FORKS} forks, "
          f"≈{len(cands)*3*M_FORKS*2.5/60/6:.1f}h (6 并行)")
    print(f"\n→ {args.out}")
    return 0


def _estimate_arm(
    srv: AppServer,
    base_items: list[dict],
    inst: T.Instance,
    workdir: Path,
    model: str,
    provider: str,
    t_star: int,
    arm: str,             # "control" / "mask" / "placebo"
    gold_paths: list[str],
    m: int,
    turn_timeout: float,
) -> dict:
    """给定前缀到 t_star（含），按臂修改输出，估计 v_{t*}。

    control : 原始前缀 (truncate_including_step(t_star))
    mask    : 掩蔽 gold 行
    placebo : 掩蔽等量非 gold 行（行数 = gold 行数，从非 gold 行随机采样）
    """
    prefix = R.truncate_including_step(base_items, t_star)
    R.assert_no_dangling_calls(prefix)
    diag = {}

    if arm == "mask":
        try:
            prefix, diag = R.mask_output_lines(base_items, t_star, gold_paths, mode="drop")
        except ValueError as e:
            return {"error": str(e), "arm": arm}
    elif arm == "placebo":
        # 先数 gold 行数
        n_gold_lines = R.count_output_lines_matching(base_items, t_star, gold_paths)
        if n_gold_lines == 0:
            return {"error": "no gold lines to match for placebo", "arm": arm}
        st = R.steps(base_items)
        if not (1 <= t_star <= len(st)) or st[t_star - 1].out_line is None:
            return {"error": "placebo: no output at t_star", "arm": arm}
        txt = R.payload(base_items[st[t_star - 1].out_line]).get("output", "") or ""
        all_lines = txt.split("\n")
        non_gold = [ln for ln in all_lines
                    if not any(gp in ln for gp in gold_paths) and ln.strip()]
        # 修 ①：安慰剂应移除**同样含路径**的非 gold 行，否则它删掉的是
        # `Chunk ID:` / `Output:` 这类无信息头部行，信息量远小于掩蔽臂，
        # 不构成公平对照（"扰动等量、信息不同"才是安慰剂的本意）。
        pathy = [ln for ln in non_gold if PATH_RE.search(ln)]
        pool = pathy if len(pathy) >= n_gold_lines else (
            pathy + [ln for ln in non_gold if ln not in pathy])
        # 修 ②：`hash()` 受 PYTHONHASHSEED 随机化，跨进程不稳定 → 不可复现。
        # 改用字符串种子（random.Random 接受 str，确定性），同时做 per-instance 播种。
        rng = random.Random(f"{SEED}:{inst.instance_id}")
        rng.shuffle(pool)
        placebo_targets = pool[:n_gold_lines]
        if not placebo_targets:
            return {"error": "placebo: no non-gold lines available", "arm": arm}
        try:
            prefix, diag = R.mask_output_lines(
                base_items, t_star, placebo_targets, mode="placebo"
            )
        except ValueError as e:
            return {"error": str(e), "arm": arm}
        # 修 ③：显式记录配平情况。行数不足时不静默通过 —— 主分析只用配平样本，
        # 未配平样本单独报告（它们仍可喂剂量–反应分析）。
        diag["n_gold_lines_target"] = n_gold_lines
        diag["n_pathy_available"] = len(pathy)
        diag["balanced"] = bool(diag.get("n_lines_dropped") == n_gold_lines)
        diag["placebo_all_pathy"] = bool(len(pathy) >= n_gold_lines)

    # 写前缀
    workdir.mkdir(parents=True, exist_ok=True)
    fork_file = workdir / f"prefix_t{t_star}_{arm}.jsonl"
    R.dump(prefix, fork_file)

    # m 次 fork
    ys: list[float] = []
    errs: list[str] = []
    for i in range(m):
        try:
            T.reset_repo(inst)
            from pilot.harness.appserver import thread_rollout_path
            fk = srv.thread_fork(
                path=fork_file, cwd=inst.repo, model=model,
                model_provider=provider,
                sandbox="read-only", approval_policy="never",
                base_instructions=T.BASE_INSTRUCTIONS,
            )
            tid = fk["thread"]["id"]
            tr = srv.turn_run(tid, T.CONTINUE_PROMPT, timeout=turn_timeout)
            # ⚠️ 关键护栏：turn 失败时 rollout 里没有 answer，`answer_from_rollout`
            # 返回 None，而 `T.outcome(None, gold)["recall"] == 0.0` —— 于是
            # **任何 API/环境失败都会静默变成"recall=0"并被当作有效观测**。
            # 若失败间歇性地更多命中 mask 臂，就会**制造出虚假的正向 PSE**。
            # 故必须先判定 turn 是否真成功，失败记 NaN（排除）而不是 0。
            err_msgs = []
            for nt in (tr.get("notifications") or []):
                if nt.get("method") == "error":
                    pr_ = nt.get("params") or {}
                    # willRetry=true 是"正在重连"的瞬时通知，不是终态失败。
                    # 只有 turn/completed.status=="failed" 才是终态（实测：
                    # Reconnecting 1..5 之后仍可能成功）。
                    if pr_.get("willRetry"):
                        continue
                    err_msgs.append(str((pr_.get("error") or {}).get("message"))[:120])
                elif nt.get("method") == "turn/completed":
                    tn = (nt.get("params") or {}).get("turn") or {}
                    if tn.get("status") == "failed":
                        err_msgs.append("turn failed: " +
                                        str((tn.get("error") or {}).get("message"))[:120])
            if err_msgs:
                ys.append(float("nan")); errs.append("TURN_FAILED " + err_msgs[0]); continue
            pth = thread_rollout_path(fk)
            if not (pth and Path(pth).exists()):
                ys.append(float("nan")); errs.append("NO_ROLLOUT_FILE"); continue
            items = R.load(pth)
            # 续跑里连一条 assistant 消息都没有 ⇒ 无效（真 agent 总会说点什么），
            # 而不是"答了但 recall=0"
            has_assistant = any(
                it.get("type") == "response_item"
                and (it.get("payload") or {}).get("role") == "assistant"
                for it in items)
            ans = T.answer_from_rollout(items)
            if ans is None and not has_assistant:
                ys.append(float("nan")); errs.append("NO_ASSISTANT_MESSAGE"); continue
            ys.append(T.outcome(ans, inst.gold)["recall"])
        except Exception as exc:
            # 不静默：记录异常类型，便于区分"环境失败"与"真的 recall=0"
            ys.append(float("nan"))
            errs.append(f"{type(exc).__name__}: {str(exc)[:120]}")

    ys_clean = [y for y in ys if not math.isnan(y)]
    return {
        "arm": arm,
        "t_star": t_star,
        "ys": ys,
        "v_hat": sum(ys_clean) / len(ys_clean) if ys_clean else float("nan"),
        "n_valid": len(ys_clean),
        "n_errors": len(errs),
        "errors": errs[:5],
        "diag": diag,
    }


def run_mode(args: argparse.Namespace) -> int:
    """执行三臂干预（API，~8.3h）。"""
    plan = json.loads(Path(args.plan).read_text())
    candidates = plan["candidates"]
    if args.shard_count > 0 and args.shard_index >= 0:
        # 按 project 轮转分片，保证各片仓库多样性与负载均衡（同 attr 的做法）
        by_proj = {}
        for c in candidates:
            by_proj.setdefault(c["instance_id"].split("__")[0], []).append(c)
        ordered = []
        pools = list(by_proj.values())
        while any(pools):
            for pl in pools:
                if pl:
                    ordered.append(pl.pop(0))
        candidates = [c for i, c in enumerate(ordered)
                      if i % args.shard_count == args.shard_index]
        print(f"[shard {args.shard_index}/{args.shard_count}] 分到 {len(candidates)} 个 instance")
    if args.limit > 0:
        candidates = candidates[:args.limit]
    print(f"将处理 {len(candidates)} 个 instance (limit={args.limit or 'none'})")

    out_path = Path(args.out)
    workroot = Path(args.workdir)
    workroot.mkdir(parents=True, exist_ok=True)
    codex_home = workroot / "codex_home"
    write_codex_config(codex_home, args.model)

    # 前置检查：缺 API key 会让每个 fork 都失败，而失败曾被静默记为 recall=0
    import os
    if not os.environ.get("IDEALAB_API_KEY"):
        print("❌ 环境变量 IDEALAB_API_KEY 未设置。X3 每个 fork 都会失败，"
              "且失败曾被静默记为 recall=0（伪装成干预效应）。请先 export 后重跑。")
        return 2

    insts = {i.instance_id: i for i in T.load_instances()}
    srv = AppServer(
        codex_bin=args.codex_bin, codex_home=codex_home,
        log_path=workroot / "x3_appserver.log", experimental_api=True,
    )
    results = []
    n = 0
    t0 = time.time()
    try:
        for c in candidates:
            iid = c["instance_id"]
            inst = insts.get(iid)
            if inst is None:
                continue
            rollout_path = Path(c["rollout_path"])
            if not rollout_path.exists():
                print(f"[SKIP] {iid}: rollout not found at {rollout_path}")
                continue
            try:
                base_items = R.load(rollout_path)
            except Exception as exc:
                print(f"[SKIP] {iid}: load error {exc}")
                continue

            t_star = c["first_gold_t"]
            gold_paths = c["gold"]
            wd = workroot / iid

            rec = {
                "instance_id": iid,
                "first_gold_t": t_star,
                "n_gold": c["n_gold"],
                "gold": gold_paths,
            }
            arms = {}
            for arm in ("control", "mask", "placebo"):
                print(f"  [{n+1}/{len(candidates)}] {iid} arm={arm} t*={t_star}...",
                      flush=True)
                try:
                    res = _estimate_arm(
                        srv, base_items, inst, wd,
                        args.model, args.model_provider,
                        t_star, arm, gold_paths,
                        args.m, args.turn_timeout,
                    )
                except Exception as exc:
                    res = {"arm": arm, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                           "trace": traceback.format_exc()[-300:]}
                arms[arm] = res
            rec["arms"] = arms
            # fail-fast：第一个 instance 若任一臂全部无效，说明环境/配置坏了，
            # 立刻中止。否则 12h 后会得到一份全零数据，而它**看起来正好像
            # 分支 (ii)「信息冗余」** —— 配置错误伪装成科学结论。
            if n == 0:
                bad = [a for a, r in arms.items() if not r.get("n_valid")]
                if bad:
                    print(f"❌ FAIL-FAST：首个 instance 的臂 {bad} 全部无效重放。")
                    for a in bad:
                        for e in (arms[a].get("errors") or [])[:3]:
                            print(f"     {a}: {e}")
                    print("   已中止，未浪费预算。修好后重跑。")
                    return 3

            # 计算 PSE 和 placebo 效应
            c_v = arms.get("control", {}).get("v_hat", float("nan"))
            m_v = arms.get("mask", {}).get("v_hat", float("nan"))
            p_v = arms.get("placebo", {}).get("v_hat", float("nan"))
            rec["PSE"] = round(c_v - m_v, 4) if not math.isnan(c_v - m_v) else None
            rec["placebo_effect"] = round(c_v - p_v, 4) if not math.isnan(c_v - p_v) else None
            rec["control_v"] = round(c_v, 4) if not math.isnan(c_v) else None
            rec["mask_v"] = round(m_v, 4) if not math.isnan(m_v) else None
            rec["placebo_v"] = round(p_v, 4) if not math.isnan(p_v) else None

            results.append(rec)
            n += 1
            print(f"  → PSE={rec['PSE']} placebo_eff={rec['placebo_effect']} "
                  f"(control={rec['control_v']}, mask={rec['mask_v']}, placebo={rec['placebo_v']})",
                  flush=True)

            # 追加写入（中间断点也可恢复）
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        srv.close()

    elapsed = (time.time() - t0) / 60
    print(f"\n完成 {n} 条 X3 干预，用时 {elapsed:.1f}min → {out_path}")
    return 0


def analyze_mode(args: argparse.Namespace) -> int:
    """分析 X3 结果（零 API）。"""
    out_path = Path(args.out)
    if not out_path.exists():
        print(f"⚠️ 结果文件不存在：{out_path}")
        return 1

    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    if not recs:
        print("⚠️ 没有记录")
        return 1

    print(f"分析 {len(recs)} 条 X3 结果")

    # PSE 和 placebo 效应
    pse_vals = [r["PSE"] for r in recs if r.get("PSE") is not None]
    plac_vals = [r["placebo_effect"] for r in recs if r.get("placebo_effect") is not None]
    control_vs = [r["control_v"] for r in recs if r.get("control_v") is not None]
    mask_vs = [r["mask_v"] for r in recs if r.get("mask_v") is not None]

    pse_ci = boot_ci(pse_vals) if pse_vals else {"note": "无样本"}
    plac_ci = boot_ci(plac_vals) if plac_vals else {"note": "无样本"}

    # 配对比较（instance 内）
    pse_paired = paired_ci(
        [r["control_v"] for r in recs if r.get("control_v") is not None and r.get("mask_v") is not None],
        [r["mask_v"] for r in recs if r.get("control_v") is not None and r.get("mask_v") is not None],
    )
    plac_paired = paired_ci(
        [r["control_v"] for r in recs if r.get("control_v") is not None and r.get("placebo_v") is not None],
        [r["placebo_v"] for r in recs if r.get("control_v") is not None and r.get("placebo_v") is not None],
    )

    # 三档判据
    pse_sig = pse_paired.get("significant", False)
    plac_sig = plac_paired.get("significant", False)
    pse_dir = (pse_paired.get("diff", 0.0) or 0.0) > 0   # control > mask

    if pse_sig and not plac_sig and pse_dir:
        verdict = "(i) STRONG_POSITIVE: 溯源边因果有效 M≪C 且 P≈C"
    elif not pse_sig and not plac_sig:
        verdict = "(ii) REDUNDANCY: 信息冗余 M≈C，建议剂量–反应实验"
    elif plac_sig:
        verdict = "(iii) OPERATOR_DISTURBANCE: P 也下降，修 mask 算子后重跑"
    else:
        verdict = "AMBIGUOUS: 仅部分条件满足，查看详细数字"

    # 剂量–反应分层（按 n_gold）
    by_ngold: dict[int, list[float]] = {}
    for r in recs:
        ng = r.get("n_gold", 0)
        if r.get("PSE") is not None:
            by_ngold.setdefault(ng, []).append(r["PSE"])
    dose_response = {str(ng): boot_ci(vals) for ng, vals in sorted(by_ngold.items())}

    analysis = {
        "n_instances": len(recs),
        "PSE": {
            "mean_PSE": pse_ci,
            "paired_control_minus_mask": pse_paired,
        },
        "PLACEBO": {
            "mean_placebo_effect": plac_ci,
            "paired_control_minus_placebo": plac_paired,
        },
        "VERDICT": verdict,
        "DOSE_RESPONSE_BY_N_GOLD": dose_response,
        "error_rate": round(sum(1 for r in recs
                               if any("error" in (r.get("arms") or {}).get(a, {})
                                      for a in ("control", "mask", "placebo")))
                            / max(1, len(recs)), 3),
    }

    # 写分析文件（追加后缀 _analysis.json）
    ana_path = out_path.with_suffix("") .parent / (out_path.stem + "_analysis.json")
    ana_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n=== X3：溯源掩蔽干预分析 ===")
    print(f"n={len(recs)}")
    if "note" not in pse_paired:
        pse_d = pse_paired.get("diff", "?")
        pse_c = pse_paired.get("ci95", "?")
        sig_str = "SIGNIFICANT" if pse_sig else "n.s."
        print(f"  PSE (control-mask): {pse_d:+.4f} CI{pse_c} [{sig_str}]")
    if "note" not in plac_paired:
        plac_d = plac_paired.get("diff", "?")
        plac_c = plac_paired.get("ci95", "?")
        sig_str = "SIGNIFICANT" if plac_sig else "n.s."
        print(f"  安慰剂 (control-placebo): {plac_d:+.4f} CI{plac_c} [{sig_str}]")
    print(f"\n判据: {verdict}")
    print(f"\n→ {ana_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "run", "analyze"], default="plan")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate",
                    help="第一个 rollout 来源（全量定位结果）")
    ap.add_argument("--runs-root2", default=None,
                    help="第二个 rollout 来源（可选，补充覆盖）")
    ap.add_argument("--plan", default="pilot/runs/x3_plan.json",
                    help="干预清单路径（plan 模式产出，run 模式读取）")
    ap.add_argument("--out", default="pilot/runs/x3_pse.json",
                    help="run 模式的输出路径 / analyze 模式的读取路径")
    ap.add_argument("--workdir", default="pilot/runs/x3_work")
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--m", type=int, default=M_FORKS)
    ap.add_argument("--turn-timeout", type=float, default=900.0)
    ap.add_argument("--shard-index", type=int, default=-1,
                    help="分片索引（0-based）。串行 1680 fork ≈70h，必须分片并行。")
    ap.add_argument("--shard-count", type=int, default=0,
                    help="分片总数。每片用独立 --workdir → 独立 CODEX_HOME+AppServer")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个 instance（调试用，0=全部）")
    args = ap.parse_args()

    if args.mode == "plan":
        return plan_mode(args)
    elif args.mode == "run":
        return run_mode(args)
    elif args.mode == "analyze":
        return analyze_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
