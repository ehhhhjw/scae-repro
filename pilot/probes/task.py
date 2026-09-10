"""文件定位任务：实例加载、prompt、结果判定。

数据
----
instance_summary.csv 字段：instance_id, base_commit, problem_statement, patch_file
patch_file 用 "\\n" 分隔多个 gold 文件（已核实：71/500 为多文件，最多 21 个）。
仓库位于 <PROJECTS_ROOT>/<instance_id>，已 checkout 到 base_commit。
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 数据集位置：默认指向仓库内 dataset/，可用环境变量覆盖指向本地 SWE-bench-Verified 快照。
# 复现包本身不携带录制数据与仓库快照（见 REPRODUCE.md 第 1 步）。
_REPO_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = Path(
    os.environ.get("SCAE_CSV", str(_REPO_ROOT / "dataset" / "instance_summary.csv"))
)
PROJECTS_ROOT = Path(
    os.environ.get("SCAE_PROJECTS", str(_REPO_ROOT / "dataset" / "projects"))
)


@dataclass
class Instance:
    instance_id: str
    base_commit: str
    problem_statement: str
    gold: frozenset[str]

    @property
    def repo(self) -> Path:
        return PROJECTS_ROOT / self.instance_id

    @property
    def project(self) -> str:
        """repo 家族，用于 leave-one-repo-out 分层。"""
        return self.instance_id.split("__")[0]


def load_instances(csv_path: str | Path | None = None) -> list[Instance]:
    path = Path(csv_path) if csv_path else CSV_PATH
    out: list[Instance] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            gold = frozenset(
                norm_path(p) for p in row["patch_file"].split("\n") if p.strip()
            )
            out.append(
                Instance(
                    instance_id=row["instance_id"].strip(),
                    base_commit=row["base_commit"].strip(),
                    problem_statement=row["problem_statement"],
                    gold=gold,
                )
            )
    return out


def norm_path(p: str) -> str:
    s = p.strip().strip("'\"`").replace("\\", "/")
    s = re.sub(r"^\./", "", s)
    return s.lstrip("/")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# base_instructions：收窄工具面 + 固定输出格式。
# 注意：这是 S0 的 [V2] 待验证项 —— 我们只能"要求"，能否真正抑制内置 shell 需实测。
# 违规调用不删不藏，一律记录并在论文里报告违规率。
BASE_INSTRUCTIONS = """\
You are a code localization agent. Your ONLY task is to identify which file(s) in the \
repository must be modified to resolve the issue described by the user.

Hard constraints:
- This is a READ-ONLY task. Never create, modify, delete, or move any file.
- Never run tests, builds, installers, or any command that mutates state.
- Restrict yourself to reading and searching the repository.
- Do not guess from the issue text alone; verify against the actual repository contents.

When you are done, emit your answer as the LAST thing in your final message, in exactly \
this format and nothing else after it:

<answer>
path/to/first_file.py
path/to/second_file.py
</answer>

Paths must be relative to the repository root. List every file that must be modified, and \
no others.
"""

PROMPT_TEMPLATE = """\
Resolve the following issue by identifying the file(s) that must be modified.

--- ISSUE ---
{problem_statement}
--- END ISSUE ---

Search and read the repository as needed, then emit the <answer> block.
"""

# fork 之后用于推进的续跑语。必须在所有 arm 中逐字一致，
# 否则它本身就成了一个干预，会污染 Δ_t 的估计。
CONTINUE_PROMPT = "Continue."


def build_prompt(inst: Instance) -> str:
    return PROMPT_TEMPLATE.format(problem_statement=inst.problem_statement.strip())


# ---------------------------------------------------------------------------
# 结果判定
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_answer(text: str) -> frozenset[str] | None:
    """取最后一个 <answer> 块。返回 None 表示 agent 没有按格式作答。"""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    body = matches[-1]
    paths = {norm_path(line) for line in body.splitlines() if line.strip()}
    paths = {p for p in paths if p and not p.startswith("<")}
    return frozenset(paths)


# 非源文件模式：SWE-bench 的 gold patch 通常只含源文件，不含 test/docs。
# agent 常合理地把 test 也报出来，因此主口径用 recall（gold ⊆ answer）。
_NONSOURCE_RE = re.compile(
    r"(^|/)(tests?|testing|docs?|doc)/|(^|/)test_[^/]+\.py$|_test\.py$|\.(rst|md|txt)$",
    re.IGNORECASE,
)


def _filter_source(paths: frozenset[str]) -> frozenset[str]:
    return frozenset(p for p in paths if not _NONSOURCE_RE.search(p))


def outcome(pred: frozenset[str] | None, gold: frozenset[str]) -> dict:
    """主口径 Y = (gold ⊆ pred)，即 recall==1（找全所有 gold 源文件即算成功）。

    附带两个敏感性口径：
      Y_exact    = 严格集合相等
      Y_filtered = 过滤掉 pred 中的 test/docs 后再严格相等
    以及 precision/recall/f1 供细粒度分析。
    """
    if pred is None:
        return {"Y": 0, "Y_exact": 0, "Y_filtered": 0,
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "parsed": False}
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    pred_src = _filter_source(pred)
    return {
        "Y": int(gold <= pred),                       # 主口径：recall == 1
        "Y_exact": int(pred == gold),
        "Y_filtered": int(pred_src == gold or gold <= pred_src),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "parsed": True,
    }


def file_hit(pred: frozenset[str] | None, target: str) -> int:
    """per-file 命中指示 Y_f = 1[target ∈ pred]。

    pilot 实测（20样本）：单文件任务 100% 全对，失败全部集中在多文件任务
    （recall=0.5，找到一个漏一个）。二元 Y 信号太稀疏，故把归因目标细化到
    单个 gold 文件：问"哪一步决定了文件 f 是否被定位到"。
    Theorem 1 的可加分解对每个 f 独立成立（Y_f 也是二元结果）。
    """
    if pred is None:
        return 0
    return int(norm_path(target) in {norm_path(p) for p in pred})


def missed_files(pred: frozenset[str] | None, gold: frozenset[str]) -> list[str]:
    """被漏掉的 gold 文件 —— 最强的负向归因信号来源。"""
    if pred is None:
        return sorted(gold)
    return sorted(gold - pred)


def final_message_text(items: list[dict]) -> str:
    """从 turn_run 返回的 item/completed 通知列表里拼出 agent 的最终文本。

    通知字段结构随 codex 版本变化，这里尽量宽松地捞所有可能的文本载荷。
    更可靠的路径是 answer_from_rollout()（直接读 rollout 文件）。
    """
    chunks: list[str] = []
    for p in items:
        item = p.get("item", p)
        kind = str(item.get("itemType") or item.get("type") or "")
        if "gentMessage" in kind or "agent_message" in kind or kind == "message":
            txt = item.get("text") or item.get("content") or ""
            if isinstance(txt, list):
                txt = "".join(
                    seg.get("text", "") for seg in txt
                    if isinstance(seg, dict)
                )
            if txt:
                chunks.append(str(txt))
    return "\n".join(chunks)


def answer_from_rollout(rollout_items: list[dict]) -> frozenset[str] | None:
    """从 rollout JSONL 直接解析最终 <answer>。比 final_message_text 可靠。

    取所有 assistant message 的文本拼起来，再用 parse_answer 取最后一个 <answer> 块。
    """
    chunks: list[str] = []
    for it in rollout_items:
        if it.get("type") != "response_item":
            continue
        p = it.get("payload", {}) or {}
        if p.get("type") == "message" and p.get("role") == "assistant":
            content = p.get("content", [])
            if isinstance(content, list):
                chunks.append("".join(
                    seg.get("text", "") for seg in content
                    if isinstance(seg, dict)
                ))
            elif isinstance(content, str):
                chunks.append(content)
    return parse_answer("\n".join(chunks))


# ---------------------------------------------------------------------------
# 仓库前置检查
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=False,
    )


def reset_repo(inst: Instance) -> dict:
    """把仓库强制回到干净的 base_commit。

    数据集现状（已核实）：499 个仓库工作区都叠着 gold patch（=已修复状态），
    还带 procbench-* 的 stash。定位任务必须跑在 buggy 的 base_commit 上，
    否则 agent 会读到已修复代码 = 泄题。

    做法：checkout base_commit（分离 HEAD）→ 丢弃工作区改动 → 清未跟踪文件。
    不动 stash（它们与工作区改动同源，是 gold patch，可从 SWE-bench 再生）。
    """
    repo = inst.repo
    if not repo.is_dir():
        return {"ok": False, "reason": f"仓库不存在: {repo}"}
    _git(repo, "checkout", "-f", inst.base_commit)
    _git(repo, "reset", "--hard", inst.base_commit)
    _git(repo, "clean", "-fdq")
    return check_repo(inst)


def check_repo(inst: Instance) -> dict:
    """跑任务前必须通过：仓库存在、工作区干净、HEAD 就是 base_commit。

    任何一条不满足都会让重放不可比（agent 会看到上一次运行留下的痕迹）。
    """
    repo = inst.repo
    if not repo.is_dir():
        return {"ok": False, "reason": f"仓库不存在: {repo}"}
    dirty = _git(repo, "status", "--porcelain").stdout.strip()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    ok = not dirty and head.startswith(inst.base_commit[:10])
    return {
        "ok": ok,
        "dirty": bool(dirty),
        "head": head,
        "expected": inst.base_commit,
        "reason": "" if ok else ("工作区不干净" if dirty else f"HEAD={head} != base_commit"),
    }


def n_py_files(inst: Instance) -> int:
    """仓库规模的粗代理，用于难度分层与 U_task 特征。"""
    return sum(1 for _ in inst.repo.rglob("*.py"))
