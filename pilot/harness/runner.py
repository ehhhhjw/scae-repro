"""批量文件定位 runner —— S1 的主体。

对全部 500 个 SWE-bench 加工实例跑文件定位：
  reset 仓库 → thread/start → turn(problem_statement) → 从 rollout 解析 <answer> → 判定 Y

每个任务可跑 K 次（不同随机性），用于估计 U_policy 与 avg@K。

设计
----
- **per-task reset**：数据集工作区叠了 gold patch，每个任务跑前强制回 base_commit。
- **答案从 rollout 解析**（answer_from_rollout），不依赖通知字段结构。
- **断点续跑**：已完成的 (instance, k) 记录在 results.jsonl，重启自动跳过。
- **每任务独立 CODEX_HOME**：隔离 rollout，避免互相污染；含 idealab config.toml。
- **并发**：每个 worker 一个 AppServer 子进程。默认串行（并发放大 API 压力，pilot 阶段先稳）。
- **落盘**：results.jsonl 每行一条 (instance, k) 结果 + rollout 副本路径，供后续归因复用。

用法
----
    python -m pilot.harness.runner \\
        --model gpt-5.5-0424-global --model-provider idealab \\
        --k 1 --limit 5 --out pilot/runs/s1
    # 全量：--limit 0（或不传）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer, thread_rollout_path       # noqa: E402
from pilot.probes import task as T                                       # noqa: E402

IDEALAB_CONFIG = """\
model = "{model}"
model_provider = "idealab"

[model_providers.idealab]
name = "Idealab AI"
base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
env_key = "IDEALAB_API_KEY"
wire_api = "{wire_api}"
"""


def write_codex_config(codex_home: Path, model: str,
                       wire_api: str = "responses") -> None:
    """写 codex 配置。

    wire_api 默认 "responses"（gpt-5.5 系列用），但 qwen 系列在该端点上只支持
    "chat"：用 responses 会在流式返回中途断连，表现为
    `stream disconnected before completion`，且 turn 以 status=failed 结束。
    实测 /chat/completions 对 qwen3.6-plus 返回 200，/responses 不可用。
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        IDEALAB_CONFIG.format(model=model, wire_api=wire_api), encoding="utf-8")


def done_keys(results_path: Path) -> set[tuple[str, int]]:
    """已完成的 (instance_id, k)，用于断点续跑。"""
    done: set[tuple[str, int]] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec["instance_id"], rec["k"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_one(
    srv: AppServer,
    inst: T.Instance,
    k: int,
    outdir: Path,
    model: str | None,
    model_provider: str | None,
    turn_timeout: float,
) -> dict:
    """跑单个 (instance, k)。返回结构化结果记录。"""
    t0 = time.time()
    reset = T.reset_repo(inst)
    if not reset["ok"]:
        return {
            "instance_id": inst.instance_id, "k": k, "ok": False,
            "error": f"reset 失败: {reset['reason']}", "wall_s": round(time.time() - t0, 1),
        }

    started = srv.thread_start(
        cwd=inst.repo, model=model, model_provider=model_provider,
        sandbox="read-only", approval_policy="never",
        base_instructions=T.BASE_INSTRUCTIONS, timeout=120,
    )
    tid = started["thread"]["id"]
    rpath = thread_rollout_path(started)

    run = srv.turn_run(tid, T.build_prompt(inst), timeout=turn_timeout)

    # 优先从 rollout 文件解析（可靠），退化到通知文本
    items: list[dict] = []
    if rpath and Path(rpath).exists():
        items = R.load(rpath)
        ans = T.answer_from_rollout(items)
    else:
        ans = T.parse_answer(T.final_message_text(run["items"]))

    steps = R.steps(items) if items else []
    oc = T.outcome(ans, inst.gold)

    # 保存 rollout 副本供后续归因复用
    saved = ""
    if rpath and Path(rpath).exists():
        dst = outdir / "rollouts" / f"{inst.instance_id}__k{k}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rpath, dst)
        saved = str(dst.relative_to(outdir))

    tools = {}
    for s in steps:
        tools[s.tool] = tools.get(s.tool, 0) + 1

    return {
        "instance_id": inst.instance_id,
        "project": inst.project,
        "k": k,
        "ok": True,
        "Y": oc["Y"],
        "f1": round(oc["f1"], 4),
        "precision": round(oc["precision"], 4),
        "recall": round(oc["recall"], 4),
        "parsed": oc["parsed"],
        "n_gold": len(inst.gold),
        "gold": sorted(inst.gold),
        "answer": sorted(ans) if ans else None,
        "n_steps": len(steps),
        "tools": tools,
        "compaction": len(R.compaction_lines(items)) if items else 0,
        "interrupted": run["interrupted"],
        "rollout": saved,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--k", type=int, default=1, help="每任务重复次数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部 500）")
    ap.add_argument("--instances", nargs="*", default=None, help="指定 instance_id 列表")
    ap.add_argument("--out", default="pilot/runs/s1")
    ap.add_argument("--turn-timeout", type=float, default=900.0)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    if not os.environ.get("IDEALAB_API_KEY"):
        print("警告: 未设置 IDEALAB_API_KEY", file=sys.stderr)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    codex_home = outdir / "codex_home"
    write_codex_config(codex_home, args.model)

    all_insts = T.load_instances()
    if args.instances:
        want = set(args.instances)
        insts = [i for i in all_insts if i.instance_id in want]
    else:
        insts = all_insts
        if args.limit > 0:
            insts = insts[: args.limit]

    done = done_keys(results_path)
    todo = [(i, k) for i in insts for k in range(args.k) if (i.instance_id, k) not in done]
    print(f"实例 {len(insts)} × K={args.k} = {len(insts) * args.k} 次；已完成 {len(done)}；待跑 {len(todo)}")

    srv = AppServer(
        codex_bin=args.codex_bin, codex_home=codex_home,
        log_path=outdir / "appserver.log", experimental_api=True,
    )
    n_done = 0
    n_ok = 0
    n_y = 0
    t_start = time.time()
    try:
        with open(results_path, "a", encoding="utf-8") as fh:
            for inst, k in todo:
                try:
                    rec = run_one(srv, inst, k, outdir, args.model,
                                  args.model_provider, args.turn_timeout)
                except Exception as exc:  # noqa: BLE001
                    rec = {
                        "instance_id": inst.instance_id, "k": k, "ok": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "trace": traceback.format_exc()[-800:],
                    }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if rec.get("ok"):
                    n_ok += 1
                    n_y += rec.get("Y", 0)
                if n_done % args.progress_every == 0 or n_done == len(todo):
                    rate = n_y / n_ok if n_ok else 0
                    eta = (time.time() - t_start) / n_done * (len(todo) - n_done)
                    print(f"[{n_done}/{len(todo)}] ok={n_ok} 成功定位率={rate:.2%} "
                          f"ETA={eta/60:.1f}min", flush=True)
    finally:
        srv.close()

    # 汇总
    _summarize(results_path, outdir)
    return 0


def _summarize(results_path: Path, outdir: Path) -> None:
    from collections import defaultdict
    recs = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("ok")]
    by_proj: dict[str, list] = defaultdict(list)
    for r in ok:
        by_proj[r.get("project", "?")].append(r)

    summary = {
        "n_total": len(recs),
        "n_ok": len(ok),
        "n_error": len(recs) - len(ok),
        "solve_rate": round(sum(r["Y"] for r in ok) / len(ok), 4) if ok else None,
        "parse_rate": round(sum(r["parsed"] for r in ok) / len(ok), 4) if ok else None,
        "mean_steps": round(sum(r["n_steps"] for r in ok) / len(ok), 2) if ok else None,
        "mean_wall_s": round(sum(r["wall_s"] for r in ok) / len(ok), 1) if ok else None,
        "n_with_compaction": sum(1 for r in ok if r.get("compaction", 0) > 0),
        "by_project": {
            p: {"n": len(rs), "solve_rate": round(sum(r["Y"] for r in rs) / len(rs), 4)}
            for p, rs in sorted(by_proj.items())
        },
        "by_n_gold": {},
    }
    from collections import Counter
    ng = defaultdict(list)
    for r in ok:
        ng[r["n_gold"]].append(r["Y"])
    summary["by_n_gold"] = {
        str(k): {"n": len(v), "solve_rate": round(sum(v) / len(v), 4)}
        for k, v in sorted(ng.items())
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 汇总 ===")
    print(f"完成 {summary['n_ok']}/{summary['n_total']}  错误 {summary['n_error']}")
    print(f"成功定位率 (Y=1): {summary['solve_rate']}")
    print(f"答案解析率: {summary['parse_rate']}")
    print(f"平均步数: {summary['mean_steps']}  平均墙钟: {summary['mean_wall_s']}s")
    print(f"含压缩事件的 run: {summary['n_with_compaction']}")
    print("按 repo 成功率:")
    for p, s in summary["by_project"].items():
        print(f"  {p:<14} n={s['n']:<4} solve={s['solve_rate']}")


if __name__ == "__main__":
    raise SystemExit(main())
