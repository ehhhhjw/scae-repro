"""因果归因：用 fork-resample 估计 anytime 价值 v_t，导出 Δ_t 与根因步。

v_t = E[Y | τ_1:t]，用同前缀续跑 m 次的成功率估计。
Δ_t = v_t − v_{t-1}（Theorem 1，相对策略基线的因果贡献 + 观测项）。

对失败 run（Y=0），"根因步" = 使结果最受损的步 = argmin_t Δ_t
（Δ_t 最负 = 走到这一步后成功概率掉得最多）。
对成功 run，argmax_t Δ_t = 最关键的正确步。

配对设计（CRN）：所有 v_t 共享同一条被截断的前缀，环境噪声由确定性工具面钉死，
因此 v_t 与 v_{t-1} 的差主要反映第 t 步的效应而非采样噪声。

成本：连续窗口上 v_t 复用为 v_{t-1} 的后项，K 个连续步只需 (K+1) 个 fork 点 × m 次。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer, thread_rollout_path       # noqa: E402
from pilot.probes import task as T                                       # noqa: E402


@dataclass
class StepValue:
    t: int                       # 步序（v_t 中的 t）；t=0 表示执行前先验 v_0
    v_hat: float                 # 成功率估计
    n: int                       # 采样次数 m
    successes: int
    ys: list[int] = field(default_factory=list)


def estimate_v(
    srv: AppServer,
    base_items: list[dict],
    t: int,
    inst: T.Instance,
    workdir: Path,
    model: str | None,
    provider: str | None,
    m: int,
    turn_timeout: float,
    target_file: str | None = None,
) -> StepValue:
    """估计 v_t = E[Y | τ_1:t]：保留前 t 步，续跑 m 次到底，取成功率。

    t=0：不保留任何步（只留 meta/user prompt），估计执行前先验 v_0。
    target_file 非空时估计 per-file 值 v_t^(f) = E[Y_f | τ_1:t]，
    其中 Y_f = 1[f ∈ F̂]（见 DESIGN_v2 §4.1.1：单文件任务全对，
    信号在多文件任务的"漏文件"上，故把归因目标细化到单个 gold 文件）。
    """
    if t == 0:
        prefix = _prefix_zero(base_items)
    else:
        prefix = R.truncate_including_step(base_items, t)
    R.assert_no_dangling_calls(prefix)
    tag = f"t{t}" if target_file is None else f"t{t}_f{abs(hash(target_file)) % 100000}"
    fork_file = workdir / f"v_prefix_{tag}.jsonl"
    R.dump(prefix, fork_file)

    ys: list[int] = []
    for _ in range(m):
        T.reset_repo(inst)
        fk = srv.thread_fork(
            path=fork_file, cwd=inst.repo, model=model, model_provider=provider,
            sandbox="read-only", approval_policy="never",
            base_instructions=T.BASE_INSTRUCTIONS,
        )
        tid = fk["thread"]["id"]
        srv.turn_run(tid, T.CONTINUE_PROMPT, timeout=turn_timeout)
        pth = thread_rollout_path(fk)
        ans = T.answer_from_rollout(R.load(pth)) if pth and Path(pth).exists() else None
        if target_file == "__recall__":
            # 连续结果变量：recall ∈ [0,1]。
            # 修 Δ≡0 退化（实测 50% 的 case：被漏文件在所有前缀下 v_t^(f)≡0，
            # 单文件二元结果无法区分任何步骤）。recall 保留其他 gold 文件的命中变化，
            # 信号不消失。Theorem 1 的可加分解对有界连续结果同样成立。
            ys.append(T.outcome(ans, inst.gold)["recall"])
        elif target_file is None:
            ys.append(T.outcome(ans, inst.gold)["Y"])
        else:
            ys.append(T.file_hit(ans, target_file))
    return StepValue(t=t, v_hat=sum(ys) / len(ys) if ys else 0.0,
                     n=len(ys), successes=sum(ys), ys=ys)


def _prefix_zero(items: list[dict]) -> list[dict]:
    """v_0 的前缀：保留到第一个 function_call 之前（即只有 meta + user prompt）。"""
    st = R.steps(items)
    if not st:
        return items
    return items[: st[0].call_line]


def causal_attribution(
    srv: AppServer,
    base_items: list[dict],
    inst: T.Instance,
    workdir: Path,
    model: str | None,
    provider: str | None,
    m: int = 4,
    window: tuple[int, int] | None = None,
    turn_timeout: float = 600.0,
) -> dict:
    """对一条 trace 做因果归因。

    window: (lo, hi) 只估计这段连续步的 v_t（含 v_{lo-1} 作为基线）。None=全轨迹。
    返回 v_t 序列、Δ_t 序列、根因步（top-1/top-3）、可加性残差。
    """
    st = R.steps(base_items)
    n = len(st)
    lo, hi = (window or (1, n))
    lo, hi = max(1, lo), min(n, hi)

    workdir.mkdir(parents=True, exist_ok=True)
    values: dict[int, StepValue] = {}
    for t in range(lo - 1, hi + 1):          # 含 v_{lo-1} 作为第一个差分的基线
        values[t] = estimate_v(srv, base_items, t, inst, workdir,
                               model, provider, m, turn_timeout)

    deltas: list[dict] = []
    for t in range(lo, hi + 1):
        d = values[t].v_hat - values[t - 1].v_hat
        a, b = R.action_bucket(st[t - 1], inst.repo)
        deltas.append({
            "t": t, "delta": round(d, 4),
            "v_t": round(values[t].v_hat, 4), "v_prev": round(values[t - 1].v_hat, 4),
            "action": f"{a}#{b}", "tool": st[t - 1].tool,
        })

    # 根因步：失败最受损 = argmin Δ；也给 |Δ| 最大
    by_min = sorted(deltas, key=lambda d: d["delta"])
    by_absmax = sorted(deltas, key=lambda d: -abs(d["delta"]))
    return {
        "n_steps": n,
        "window": [lo, hi],
        "m": m,
        "v0": round(values[0].v_hat, 4) if 0 in values else round(values[lo - 1].v_hat, 4),
        "v_final": round(values[hi].v_hat, 4),
        "deltas": deltas,
        "rootcause_top1_min": by_min[0] if by_min else None,      # 最伤结果的步
        "rootcause_top3_min": by_min[:3],
        "rootcause_top1_absmax": by_absmax[0] if by_absmax else None,
        "sum_delta": round(sum(d["delta"] for d in deltas), 4),
    }


def normalized_position(t: int, n_steps: int) -> float:
    """步位归一化到 (0,1]，供 D3 步位偏移检验。"""
    return t / n_steps if n_steps else 0.0


# ---------------------------------------------------------------------------
# 混合估计器（V5 发现 positivity 分步 → 按熵自适应）DESIGN_v2 §3.5.1
# ---------------------------------------------------------------------------

def estimate_policy_at(
    srv: AppServer, base_items: list[dict], t: int, inst: T.Instance,
    workdir: Path, model: str | None, provider: str | None,
    m: int, turn_timeout: float,
) -> dict:
    """估计第 t 步的策略分布 π̂_t（同前缀 fork m 次，只采第 t 步动作），返回动作桶计数与熵。

    用于判断该步是高熵（fork-resample 可用）还是低熵（需软干预）。
    """
    import math
    from collections import Counter
    prefix = R.truncate_before_step(base_items, t)
    R.assert_no_dangling_calls(prefix)
    ff = workdir / f"pi_prefix_t{t}.jsonl"
    R.dump(prefix, ff)
    acts: list[str] = []
    for _ in range(m):
        T.reset_repo(inst)
        fk = srv.thread_fork(path=ff, cwd=inst.repo, model=model, model_provider=provider,
                             sandbox="read-only", approval_policy="never",
                             base_instructions=T.BASE_INSTRUCTIONS)
        seen: list = []

        def _stop(p, _s=seen):
            it = p.get("item", p)
            k = str(it.get("itemType") or it.get("type") or "")
            if any(x in k for x in ("ommand", "oolCall", "tool_call", "unction")):
                _s.append(1); return True
            return False

        srv.turn_run(fk["thread"]["id"], T.CONTINUE_PROMPT, timeout=turn_timeout, on_item=_stop)
        pth = thread_rollout_path(fk)
        st = R.steps(R.load(pth)) if pth and Path(pth).exists() else []
        if len(st) >= t:
            a, b = R.action_bucket(st[t - 1], inst.repo)
            acts.append(f"{a}#{b}")
    counts = Counter(acts)
    n = sum(counts.values())
    H = -sum((c / n) * math.log(c / n) for c in counts.values() if c) if n else 0.0
    return {"counts": dict(counts), "n": n, "entropy": round(H, 4),
            "high_entropy": len(counts) >= 2}


def hybrid_delta_at(
    srv: AppServer, base_items: list[dict], t: int, inst: T.Instance,
    workdir: Path, model: str | None, provider: str | None,
    m: int, turn_timeout: float, tau_H: float = 0.0,
) -> dict:
    """混合估计第 t 步的因果贡献 Δ_t。

    高熵步：观测式 fork-resample（v_t − v_{t-1}）。
    低熵步：软干预 —— 用 substitute_action 注入 oracle 动作（读 gold 文件），
            真实执行拿真 output，续跑 m 次，估相对当前（确定性）动作的效应。
    """
    from pilot.harness import action_exec as AE

    pi = estimate_policy_at(srv, base_items, t, inst, workdir, model, provider,
                            max(2, m), turn_timeout)
    st = R.steps(base_items)

    if pi["high_entropy"] and pi["entropy"] > tau_H:
        v_t = estimate_v(srv, base_items, t, inst, workdir, model, provider, m, turn_timeout)
        v_p = estimate_v(srv, base_items, t - 1, inst, workdir, model, provider, m, turn_timeout)
        return {"t": t, "mode": "fork_resample", "entropy": pi["entropy"],
                "delta": round(v_t.v_hat - v_p.v_hat, 4),
                "v_t": round(v_t.v_hat, 4), "v_prev": round(v_p.v_hat, 4)}

    # 低熵步：软干预。替代动作 = oracle（读 gold），估 do(a_oracle) 相对 do(a_observed)
    tool, args, cmd = AE.oracle_action(inst)
    T.reset_repo(inst)
    real_out = AE.exec_readonly_action(inst.repo, cmd)
    sub_items = R.substitute_action(base_items, t, tool, args, real_out)
    R.assert_no_dangling_calls(sub_items)
    ff = workdir / f"steer_t{t}.jsonl"
    R.dump(sub_items, ff)
    ys = []
    for _ in range(m):
        T.reset_repo(inst)
        fk = srv.thread_fork(path=ff, cwd=inst.repo, model=model, model_provider=provider,
                             sandbox="read-only", approval_policy="never",
                             base_instructions=T.BASE_INSTRUCTIONS)
        srv.turn_run(fk["thread"]["id"], T.CONTINUE_PROMPT, timeout=turn_timeout)
        pth = thread_rollout_path(fk)
        ans = T.answer_from_rollout(R.load(pth)) if pth and Path(pth).exists() else None
        ys.append(T.outcome(ans, inst.gold)["Y"])
    v_steer = sum(ys) / len(ys) if ys else 0.0
    # 观测动作的价值 = v_t（保留原第 t 步）
    v_obs = estimate_v(srv, base_items, t, inst, workdir, model, provider, m, turn_timeout)
    return {"t": t, "mode": "soft_intervention", "entropy": pi["entropy"],
            "delta_steer_minus_obs": round(v_steer - v_obs.v_hat, 4),
            "v_steer_oracle": round(v_steer, 4), "v_observed": round(v_obs.v_hat, 4),
            "note": "低熵步：正的 delta 表示注入 oracle 动作能提升成功率（该步是瓶颈）"}
