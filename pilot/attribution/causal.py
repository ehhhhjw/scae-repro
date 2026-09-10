"""Causal attribution: estimate the anytime value v_t by fork-resample, then derive Δ_t and the root-cause step.

v_t = E[Y | τ_1:t], estimated by the success rate of m continuations sharing the same prefix.
Δ_t = v_t − v_{t-1} (Theorem 1, causal contribution relative to the policy baseline + observation term).

For a failed run (Y=0), the "root-cause step" = the step that hurt the outcome most = argmin_t Δ_t
(most negative Δ_t = success probability drops the most after reaching this step).
For a successful run, argmax_t Δ_t = the most decisive correct step.

Paired design (CRN): all v_t share the same truncated prefix and environment noise is pinned down by a deterministic
tool surface, so the gap between v_t and v_{t-1} mainly reflects the effect of step t rather than sampling noise.

Cost: over a contiguous window v_t is reused as the trailing term of v_{t-1}, so K contiguous steps need only (K+1) fork points × m runs.
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
    t: int                       # step index (the t in v_t); t=0 is the pre-execution prior v_0
    v_hat: float                 # success-rate estimate
    n: int                       # number of samples m
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
    """Estimate v_t = E[Y | τ_1:t]: keep the first t steps, run m continuations to completion, take the success rate.

    t=0: keep no steps (only meta/user prompt), estimating the pre-execution prior v_0.
    When target_file is set, estimate the per-file value v_t^(f) = E[Y_f | τ_1:t],
    where Y_f = 1[f ∈ F̂] (see DESIGN_v2 §4.1.1: single-file tasks are all correct,
    the signal lies in the "missed files" of multi-file tasks, so the attribution target is refined to a single gold file).
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
            # Continuous outcome variable: recall ∈ [0,1].
            # Fixes the Δ≡0 degeneracy (measured on 50% of cases: v_t^(f)≡0 for the missed file under every prefix,
            # so a single-file binary outcome cannot separate any step). recall retains hit changes on the other gold
            # files, so the signal does not vanish. Theorem 1's additive decomposition also holds for bounded continuous outcomes.
            ys.append(T.outcome(ans, inst.gold)["recall"])
        elif target_file is None:
            ys.append(T.outcome(ans, inst.gold)["Y"])
        else:
            ys.append(T.file_hit(ans, target_file))
    return StepValue(t=t, v_hat=sum(ys) / len(ys) if ys else 0.0,
                     n=len(ys), successes=sum(ys), ys=ys)


def _prefix_zero(items: list[dict]) -> list[dict]:
    """Prefix for v_0: keep everything before the first function_call (i.e. only meta + user prompt)."""
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
    """Causal attribution for one trace.

    window: (lo, hi) estimates v_t only over this contiguous range (including v_{lo-1} as baseline). None = whole trace.
    Returns the v_t sequence, the Δ_t sequence, root-cause steps (top-1/top-3), and the additivity residual.
    """
    st = R.steps(base_items)
    n = len(st)
    lo, hi = (window or (1, n))
    lo, hi = max(1, lo), min(n, hi)

    workdir.mkdir(parents=True, exist_ok=True)
    values: dict[int, StepValue] = {}
    for t in range(lo - 1, hi + 1):          # include v_{lo-1} as the baseline for the first difference
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

    # root-cause step: most harmful on failure = argmin Δ; also report the largest |Δ|
    by_min = sorted(deltas, key=lambda d: d["delta"])
    by_absmax = sorted(deltas, key=lambda d: -abs(d["delta"]))
    return {
        "n_steps": n,
        "window": [lo, hi],
        "m": m,
        "v0": round(values[0].v_hat, 4) if 0 in values else round(values[lo - 1].v_hat, 4),
        "v_final": round(values[hi].v_hat, 4),
        "deltas": deltas,
        "rootcause_top1_min": by_min[0] if by_min else None,      # the step that hurt the outcome most
        "rootcause_top3_min": by_min[:3],
        "rootcause_top1_absmax": by_absmax[0] if by_absmax else None,
        "sum_delta": round(sum(d["delta"] for d in deltas), 4),
    }


def normalized_position(t: int, n_steps: int) -> float:
    """Normalize step position to (0,1], for the D3 step-position shift test."""
    return t / n_steps if n_steps else 0.0


# ---------------------------------------------------------------------------
# Hybrid estimator (V5 found positivity varies by step → adapt on entropy) DESIGN_v2 §3.5.1
# ---------------------------------------------------------------------------

def estimate_policy_at(
    srv: AppServer, base_items: list[dict], t: int, inst: T.Instance,
    workdir: Path, model: str | None, provider: str | None,
    m: int, turn_timeout: float,
) -> dict:
    """Estimate the policy distribution π̂_t at step t (fork m times on the same prefix, sampling only the step-t action); returns action-bucket counts and entropy.

    Used to decide whether the step is high-entropy (fork-resample usable) or low-entropy (needs soft intervention).
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
    """Hybrid estimate of the causal contribution Δ_t of step t.

    High-entropy step: observational fork-resample (v_t − v_{t-1}).
    Low-entropy step: soft intervention -- inject an oracle action (read the gold file) via substitute_action,
            execute it for real to get a real output, run m continuations, and estimate the effect relative to the current (deterministic) action.
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

    # low-entropy step: soft intervention. Substitute action = oracle (read gold), estimating do(a_oracle) vs do(a_observed)
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
    # value of the observed action = v_t (keeping the original step t)
    v_obs = estimate_v(srv, base_items, t, inst, workdir, model, provider, m, turn_timeout)
    return {"t": t, "mode": "soft_intervention", "entropy": pi["entropy"],
            "delta_steer_minus_obs": round(v_steer - v_obs.v_hat, 4),
            "v_steer_oracle": round(v_steer, 4), "v_observed": round(v_obs.v_hat, 4),
            "note": "low-entropy step: a positive delta means injecting the oracle action raises the success rate (this step is the bottleneck)"}
