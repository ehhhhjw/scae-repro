#!/usr/bin/env python3
"""V5 独立探针：同前缀重复 fork 的动作熵。

这是整个方法的单点故障。给定一条已有 rollout，截到第 k 步之前，
反复 fork + 续跑，只采样"续跑产生的第一个动作 A_k"，统计其分布的熵。

判据
----
  H > 0（≥2 个不同动作）→ 策略在该前缀上有随机性 → fork-resample 能估计 v_t
                          → C2 归因 / C4 预测的估计路径成立
  H ≈ 0（只有 1 个动作） → positivity 失效 → Δ_t 的策略基线项恒为 0
                          → 必须改用软干预(turn/steer)，估计量重写为"相对扰动策略的效应"

为什么单独做
------------
smoke.py 的 V5 依赖先跑完整任务。这里直接吃已有 rollout，可对多个 (仓库, k)
快速扫描，且能复用同一个 AppServer 子进程降低握手开销。

用法
----
    python -m pilot.smoke.probe_v5 \\
        --rollout pilot/runs/smoke/astropy__astropy-12907/base.rollout.jsonl \\
        --instance astropy__astropy-12907 \\
        --model gpt-5.5-0424-global --model-provider idealab \\
        --fork-steps 2 4 6 --samples 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer, thread_rollout_path       # noqa: E402
from pilot.probes import task as T                                       # noqa: E402


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c) if n else 0.0


def norm_entropy(counts: Counter) -> float:
    k = len(counts)
    return entropy(counts) / math.log(k) if k > 1 else 0.0


def first_action_after(items_after: list[dict], k: int, repo: Path) -> str | None:
    """续跑轨迹里第 k 步的动作桶（前 k-1 步是被固定的前缀）。"""
    st = R.steps(items_after)
    if len(st) < k:
        return None
    a, b = R.action_bucket(st[k - 1], repo)
    return f"{a}#{b}"


def probe_one_step(
    srv: AppServer, base_items: list[dict], k: int, inst: T.Instance,
    outdir: Path, model: str | None, provider: str | None,
    samples: int, turn_timeout: float,
) -> dict:
    prefix = R.truncate_before_step(base_items, k)
    R.assert_no_dangling_calls(prefix)
    fork_file = outdir / f"prefix_k{k}.jsonl"
    R.dump(prefix, fork_file)

    orig_first = None
    st0 = R.steps(base_items)
    if len(st0) >= k:
        a, b = R.action_bucket(st0[k - 1], inst.repo)
        orig_first = f"{a}#{b}"

    actions: list[str] = []
    errors: list[str] = []
    for i in range(samples):
        T.reset_repo(inst)                       # 每次续跑前回干净态
        try:
            fk = srv.thread_fork(
                path=fork_file, cwd=inst.repo, model=model, model_provider=provider,
                sandbox="read-only", approval_policy="never",
                base_instructions=T.BASE_INSTRUCTIONS,
            )
            tid = fk["thread"]["id"]

            captured: dict = {"hit": False}

            def _stop(p: dict, _c=captured) -> bool:
                it = p.get("item", p)
                kind = str(it.get("itemType") or it.get("type") or "")
                if "ommand" in kind or "oolCall" in kind or "tool_call" in kind or "unction" in kind:
                    _c["hit"] = True
                    return True
                return False

            srv.turn_run(tid, T.CONTINUE_PROMPT, timeout=turn_timeout, on_item=_stop)
            pth = thread_rollout_path(fk)
            fa = first_action_after(R.load(pth), k, inst.repo) if pth and Path(pth).exists() else None
            if fa:
                actions.append(fa)
            else:
                errors.append(f"s{i}: 未采到第{k}步动作")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"s{i}: {type(exc).__name__}: {str(exc)[:150]}")

    counts = Counter(actions)
    return {
        "fork_step": k,
        "orig_first_action": orig_first,
        "n_samples": len(actions),
        "n_distinct": len(counts),
        "entropy_nats": round(entropy(counts), 4),
        "normalized_entropy": round(norm_entropy(counts), 4),
        "distribution": dict(counts),
        "positivity_ok": len(counts) >= 2,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--fork-steps", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default="pilot/runs/v5")
    ap.add_argument("--turn-timeout", type=float, default=600.0)
    args = ap.parse_args()

    inst = {i.instance_id: i for i in T.load_instances()}[args.instance]
    base_items = R.load(args.rollout)
    n_steps = len(R.steps(base_items))
    steps_to_probe = [k for k in args.fork_steps if 2 <= k <= n_steps]

    outdir = Path(args.out) / inst.instance_id
    outdir.mkdir(parents=True, exist_ok=True)

    from pilot.harness.runner import write_codex_config
    codex_home = outdir / "codex_home"
    write_codex_config(codex_home, args.model)

    srv = AppServer(codex_bin=args.codex_bin, codex_home=codex_home,
                    log_path=outdir / "appserver.log", experimental_api=True)
    report = {"instance": inst.instance_id, "rollout_n_steps": n_steps,
              "samples_per_step": args.samples, "per_step": []}
    t0 = time.time()
    try:
        for k in steps_to_probe:
            res = probe_one_step(srv, base_items, k, inst, outdir,
                                 args.model, args.model_provider,
                                 args.samples, args.turn_timeout)
            report["per_step"].append(res)
            print(f"[k={k}] distinct={res['n_distinct']}/{res['n_samples']} "
                  f"H={res['entropy_nats']} nats  positivity_ok={res['positivity_ok']}",
                  flush=True)
            for a, c in sorted(res["distribution"].items(), key=lambda kv: -kv[1]):
                print(f"      {c:>2}x  {a}")
    finally:
        srv.close()

    report["wall_s"] = round(time.time() - t0, 1)
    any_ok = any(s["positivity_ok"] for s in report["per_step"])
    report["verdict"] = {
        "any_step_has_positivity": any_ok,
        "conclusion": (
            "至少一个 fork 点有动作随机性 → fork-resample 可估计 v_t，主线成立"
            if any_ok else
            "所有 fork 点动作熵均为 0 → positivity 失效 → 改软干预路线，估计量需重写"
        ),
    }
    (outdir / "v5_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n裁决: {report['verdict']['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
