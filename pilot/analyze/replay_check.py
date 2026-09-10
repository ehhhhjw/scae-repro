"""A3 重放一致性检验 —— 零 API 成本，为论文的假设 A3 提供实证支撑。

论文 Method §Identification 声称 A3（环境可重放）由构造保证：工具输出全序排序、
截断位置固定、locale/TZ/终端宽度/hash seed 全部钉死。这是**可检验的断言**，
不应只作声明。

做法：从已有 rollout 中取出真实执行过的只读动作，在同一仓库（reset 到 base_commit）
重复执行 R 次，检查输出是否逐字节一致。

report:
  - consistency_rate: 逐字节一致的动作占比
  - 不一致的动作及其差异来源（用于定位残余非确定性）

这直接对应论文里应报告的 "A3 replay consistency rate"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.harness import action_exec as AE                 # noqa: E402
from pilot.probes import task as T                          # noqa: E402


def extract_shell_cmd(step) -> str | None:
    """从 step 的参数里取出 shell 命令（codex 的 exec_command 工具）。"""
    a = step.args
    cmd = a.get("cmd") or a.get("command")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) and cmd.strip() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True,
                    help="rollout jsonl 路径（可用 glob 展开）")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-actions", type=int, default=120)
    ap.add_argument("--out", default="pilot/runs/replay_check.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    checked = 0
    consistent = 0
    skipped_nonreadonly = 0
    mismatches: list[dict] = []
    per_repo: Counter = Counter()

    for rp in args.rollouts:
        p = Path(rp)
        if not p.exists():
            continue
        # rollout 文件名形如 <instance_id>__k0.jsonl
        iid = p.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None or not inst.repo.is_dir():
            continue
        state = T.reset_repo(inst)
        if not state.get("ok"):
            continue
        steps = R.steps(R.load(p))
        for s in steps:
            if checked >= args.max_actions:
                break
            cmd = extract_shell_cmd(s)
            if not cmd:
                continue
            if not AE.is_readonly(cmd):
                skipped_nonreadonly += 1
                continue
            digests = set()
            for _ in range(args.repeats):
                out = AE.exec_readonly_action(inst.repo, cmd)
                digests.add(hashlib.sha256(out.encode("utf-8", "ignore")).hexdigest())
            checked += 1
            per_repo[inst.project] += 1
            if len(digests) == 1:
                consistent += 1
            else:
                mismatches.append({"instance": iid, "step": s.idx,
                                   "cmd": cmd[:200], "n_distinct": len(digests)})

    rate = consistent / checked if checked else None
    report = {
        "n_actions_checked": checked,
        "repeats_each": args.repeats,
        "n_consistent": consistent,
        "consistency_rate": round(rate, 4) if rate is not None else None,
        "n_skipped_non_readonly": skipped_nonreadonly,
        "by_project": dict(per_repo),
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:20],
        "interpretation": (
            "A3 在检验范围内成立（100% 逐字节一致）"
            if rate == 1.0 else
            f"A3 存在残余非确定性：{len(mismatches)} 个动作输出不稳定，需定位并钉死"
            if rate is not None else "无可检验动作"),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("=== A3 重放一致性检验 ===")
    print(f"  检验动作数 {checked}（每个重复 {args.repeats} 次）")
    print(f"  逐字节一致 {consistent} → 一致率 {report['consistency_rate']}")
    print(f"  跳过非只读 {skipped_nonreadonly}")
    print(f"  不一致 {len(mismatches)}")
    for m in mismatches[:5]:
        print(f"    {m['instance']} step{m['step']}: {m['cmd'][:80]} ({m['n_distinct']} 种输出)")
    print(f"  结论: {report['interpretation']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
