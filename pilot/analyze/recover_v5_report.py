#!/usr/bin/env python3
"""从磁盘残留数据回收崩溃掉的 V5 熵探针报告（只读、零 API 成本）。

背景
----
`pilot/smoke/probe_v5.py` 只在 main() 末尾才写 `v5_report.json`，所以进程中途崩溃
会把已经花掉的 fork 采样全部丢失。但采样本身是持久化的：每次 `thread_fork` +
`turn_run` 都会在 `<outdir>/codex_home/sessions/YYYY/MM/DD/rollout-*.jsonl` 落一条
完整的续跑 rollout。本脚本把这些残留 rollout 重新归组到各个 fork 点，复用
`pilot.harness.rollout` 的 `steps()` / `action_bucket()` 重算分布与熵，产出与
正常跑完时同构的 `v5_report.json`。

归组判据（结构性，不用时间戳）
------------------------------
`probe_v5` 在 fork 点 k 处用 `truncate_before_step(base, k)` 造前缀，该前缀恰好含
k-1 个步骤。fork 出来的续跑 rollout 会原样带上这段前缀，然后追加第 k 步。于是：

    m := 该 rollout 与 base rollout 的**最长公共步骤前缀长度**（按 call_id 逐一比对）
    k  = m + 1

call_id 是模型侧生成的随机串，续跑不可能重现 base 的 call_id，所以 m 是良定义的。
再加两道硬校验：
  1. 每个 `prefix_kN.jsonl` 必须逐行等于 `truncate_before_step(base, N)`
     —— 确认我们找对了 base rollout；
  2. 每条 rollout 的前 k-1 步对应的 response_item 序列必须等于 `prefix_k{k}.jsonl`
     的 response_item 序列（忽略 codex 重新序列化 `reasoning.content` 造成的差异）。

任何一条 rollout 归不到已声明的 fork 点，或校验不过，脚本直接失败退出而**不写
任何输出**。宁可回收失败，也不产出猜测出来的数字。

时间戳只用于同一 fork 点内部的样本排序（probe_v5 的采样循环是严格串行的），
不参与归组。

用法
----
    python3 -m pilot.analyze.recover_v5_report \\
        --run-dir pilot/runs/v5_scan/django__django-10097
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R      # noqa: E402
from pilot.probes import task as T          # noqa: E402


class Unrecoverable(RuntimeError):
    """数据不足以可靠归组。调用方应如实报告失败，不得回退到猜测。"""


# ---------------------------------------------------------------------------
# 与 probe_v5 逐字一致的统计口径（不要在这里"改进"，否则新旧报告不可比）
# ---------------------------------------------------------------------------

def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c) if n else 0.0


def norm_entropy(counts: Counter) -> float:
    k = len(counts)
    return entropy(counts) / math.log(k) if k > 1 else 0.0


def first_action_after(items_after: list[dict], k: int, repo: Path) -> str | None:
    st = R.steps(items_after)
    if len(st) < k:
        return None
    a, b = R.action_bucket(st[k - 1], repo)
    return f"{a}#{b}"


# ---------------------------------------------------------------------------
# 结构比对
# ---------------------------------------------------------------------------

def _canon_response_items(items: list[dict]) -> list[dict]:
    """取出 response_item 的 payload，抹掉 fork 重新序列化引入的噪声字段。

    实测唯一差异：`reasoning` payload 在 fork 后的 rollout 里多一个 `content` 键
    （原 rollout 里没有）。其余字段逐字相同。
    """
    out = []
    for it in items:
        if R.line_type(it) != "response_item":
            continue
        p = R.payload(it)
        if p.get("type") == "reasoning":
            p = {k: v for k, v in p.items() if k != "content"}
        out.append(p)
    return out


def _common_step_prefix(fork_ids: list[str | None], base_ids: list[str | None]) -> int:
    m = 0
    while m < len(fork_ids) and m < len(base_ids):
        a, b = fork_ids[m], base_ids[m]
        if a is None or b is None or a != b:
            break
        m += 1
    return m


_TS_RE = re.compile(r"rollout-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-")


def _sort_key(p: Path) -> tuple:
    m = _TS_RE.search(p.name)
    return (m.group(1) if m else "", p.name)


# ---------------------------------------------------------------------------
# base rollout 定位（带校验的发现，不是猜测）
# ---------------------------------------------------------------------------

def _prefixes_match(base_items: list[dict], prefix_files: dict[int, Path]) -> bool:
    for k, pf in prefix_files.items():
        try:
            trunc = R.truncate_before_step(base_items, k)
        except (IndexError, ValueError):
            return False
        if json.dumps(trunc, sort_keys=True) != json.dumps(R.load(pf), sort_keys=True):
            return False
    return True


def discover_base(repo_root: Path, instance: str, prefix_files: dict[int, Path],
                  explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise Unrecoverable(f"--base 指向的文件不存在: {p}")
        if not _prefixes_match(R.load(p), prefix_files):
            raise Unrecoverable(f"--base {p} 与 prefix_k*.jsonl 不一致，不是这次探针用的 base rollout")
        return p

    cands = sorted(set(repo_root.glob(f"pilot/runs/**/{instance}__k*.jsonl"))
                   | set(repo_root.glob(f"pilot/runs/**/{instance}/base.rollout.jsonl")))
    hits = [c for c in cands if _prefixes_match(R.load(c), prefix_files)]
    if not hits:
        raise Unrecoverable(
            f"在 pilot/runs 下找不到与 {instance} 的 prefix_k*.jsonl 一致的 base rollout"
            f"（试过 {len(cands)} 个候选）。请用 --base 显式指定。")
    if len({json.dumps(R.load(h), sort_keys=True) for h in hits}) > 1:
        raise Unrecoverable(f"找到多个内容不同的 base rollout 候选: {[str(h) for h in hits]}")
    return hits[0]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def recover(run_dir: Path, repo_root: Path, instance: str | None,
            base_path: str | None, samples_expected: int | None) -> dict:
    instance = instance or run_dir.name

    prefix_files = {}
    for pf in run_dir.glob("prefix_k*.jsonl"):
        m = re.fullmatch(r"prefix_k(\d+)\.jsonl", pf.name)
        if m:
            prefix_files[int(m.group(1))] = pf
    if not prefix_files:
        raise Unrecoverable(f"{run_dir} 下没有 prefix_k*.jsonl，无法知道探测过哪些 fork 点")
    fork_steps = sorted(prefix_files)

    base = discover_base(repo_root, instance, prefix_files, base_path)
    base_items = R.load(base)
    base_steps = R.steps(base_items)
    base_ids = [s.call_id for s in base_steps]
    if any(i is None for i in base_ids):
        raise Unrecoverable("base rollout 里存在没有 call_id 的步骤，无法做结构归组")
    if len(set(base_ids)) != len(base_ids):
        raise Unrecoverable("base rollout 里 call_id 有重复，最长公共前缀不再良定义")

    repo = T.PROJECTS_ROOT / instance

    sessions_dir = run_dir / "codex_home" / "sessions"
    if not sessions_dir.is_dir():
        raise Unrecoverable(f"没有 {sessions_dir}，fork 出来的续跑 rollout 不在磁盘上，不可回收")
    session_files = sorted(sessions_dir.rglob("rollout-*.jsonl"), key=_sort_key)
    if not session_files:
        raise Unrecoverable(f"{sessions_dir} 下没有 rollout-*.jsonl，不可回收")

    prefix_canon = {k: _canon_response_items(R.load(p)) for k, p in prefix_files.items()}

    groups: dict[int, list[Path]] = {k: [] for k in fork_steps}
    for sp in session_files:
        items = R.load(sp)
        st = R.steps(items)
        m = _common_step_prefix([s.call_id for s in st], base_ids)
        k = m + 1
        if k not in groups:
            raise Unrecoverable(
                f"{sp} 与 base 的公共步骤前缀长度 {m} → k={k}，不在已声明的 fork 点 "
                f"{fork_steps} 里。无法可靠归组，拒绝猜测。")
        # 硬校验：前 k-1 步对应的 response_item 序列必须等于 prefix_k{k}
        want = prefix_canon[k]
        got = _canon_response_items(items)[: len(want)]
        if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
            raise Unrecoverable(
                f"{sp} 归到 k={k}，但其前缀 response_item 序列与 prefix_k{k}.jsonl 不一致，"
                f"拒绝采信。")
        groups[k].append(sp)

    per_step = []
    for k in fork_steps:
        orig_first = None
        if len(base_steps) >= k:
            a, b = R.action_bucket(base_steps[k - 1], repo)
            orig_first = f"{a}#{b}"

        actions: list[str] = []
        errors: list[str] = []
        for i, sp in enumerate(groups[k]):
            fa = first_action_after(R.load(sp), k, repo)
            if fa:
                actions.append(fa)
            else:
                errors.append(f"s{i}: 未采到第{k}步动作")
        n_found = len(groups[k])
        if samples_expected and n_found < samples_expected:
            errors.append(
                f"[recovered] 磁盘上只有 {n_found}/{samples_expected} 个样本"
                f"（探针在此 fork 点崩溃）")

        counts = Counter(actions)
        per_step.append({
            "fork_step": k,
            "orig_first_action": orig_first,
            "n_samples": len(actions),
            "n_distinct": len(counts),
            "entropy_nats": round(entropy(counts), 4),
            "normalized_entropy": round(norm_entropy(counts), 4),
            "distribution": dict(counts),
            "positivity_ok": len(counts) >= 2,
            "errors": errors,
        })

    # wall_s 无法从崩溃的进程恢复：用残留 rollout 的时间戳跨度做下界估计。
    stamps = [_sort_key(p)[0] for p in session_files if _sort_key(p)[0]]
    wall_s = None
    if len(stamps) >= 2:
        import datetime as _dt
        fmt = "%Y-%m-%dT%H-%M-%S"
        t0 = _dt.datetime.strptime(min(stamps), fmt)
        t1 = _dt.datetime.strptime(max(stamps), fmt)
        wall_s = round((t1 - t0).total_seconds(), 1)

    any_ok = any(s["positivity_ok"] for s in per_step)
    return {
        "instance": instance,
        "rollout_n_steps": len(base_steps),
        "samples_per_step": samples_expected or max(
            (len(groups[k]) for k in fork_steps), default=0),
        "per_step": per_step,
        "wall_s": wall_s,
        "verdict": {
            "any_step_has_positivity": any_ok,
            "conclusion": (
                "至少一个 fork 点有动作随机性 → fork-resample 可估计 v_t，主线成立"
                if any_ok else
                "所有 fork 点动作熵均为 0 → positivity 失效 → 改软干预路线，估计量需重写"
            ),
        },
        "recovery": {
            "recovered_from_disk": True,
            "reason": "probe_v5 崩溃，未写出 v5_report.json；从残留 fork rollout 重算",
            "base_rollout": str(base),
            "n_session_rollouts": len(session_files),
            "grouping": {str(k): [p.name for p in groups[k]] for k in fork_steps},
            "wall_s_is_timestamp_span_lower_bound": True,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="探针输出目录，如 pilot/runs/v5_scan/<instance>")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--instance", default=None, help="默认取 run-dir 目录名")
    ap.add_argument("--base", default=None, help="base rollout 路径；不传则在 pilot/runs 下带校验搜索")
    ap.add_argument("--samples", type=int, default=None,
                    help="原探针的 --samples；用于标注缺样本的 fork 点")
    ap.add_argument("--out", default=None, help="默认 <run-dir>/v5_report.json")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 v5_report.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "v5_report.json"
    if out.exists() and not args.force:
        print(f"[拒绝] {out} 已存在。本脚本不覆盖已有报告（加 --force 才允许）。", file=sys.stderr)
        return 2

    try:
        report = recover(run_dir, Path(args.repo_root), args.instance, args.base, args.samples)
    except Unrecoverable as exc:
        print(f"[不可回收] {exc}", file=sys.stderr)
        return 1

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已回收 → {out}")
    print(f"base rollout: {report['recovery']['base_rollout']}  "
          f"n_steps={report['rollout_n_steps']}  "
          f"残留 rollout {report['recovery']['n_session_rollouts']} 条")
    for s in report["per_step"]:
        print(f"[k={s['fork_step']}] distinct={s['n_distinct']}/{s['n_samples']} "
              f"H={s['entropy_nats']} nats  positivity_ok={s['positivity_ok']}")
        for a, c in sorted(s["distribution"].items(), key=lambda kv: -kv[1]):
            print(f"      {c:>2}x  {a}")
        for e in s["errors"]:
            print(f"      ! {e}")
    print(f"\n裁决: {report['verdict']['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
