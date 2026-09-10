"""File localization task: instance loading, prompt, outcome scoring.

Data
----
instance_summary.csv fields: instance_id, base_commit, problem_statement, patch_file
patch_file separates multiple gold files with "\\n" (verified: 71/500 are multi-file, at most 21).
Repositories live at <PROJECTS_ROOT>/<instance_id>, already checked out to base_commit.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Dataset location: defaults to dataset/ inside the repository; override via env var to point at a local SWE-bench-Verified snapshot.
# The reproduction package itself ships neither recorded data nor repository snapshots (see REPRODUCE.md step 1).
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
        """repo family, used for leave-one-repo-out stratification."""
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

# base_instructions: narrow the tool surface + fix the output format.
# Note: this is S0's [V2] open item -- we can only "request" it; whether the built-in shell
# is really suppressed has to be measured.
# Violating calls are neither dropped nor hidden: all are logged and the violation rate is
# reported in the paper.
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

# Continuation utterance used to advance after a fork. It must be verbatim identical
# across all arms, otherwise it becomes an intervention itself and contaminates the
# estimate of Δ_t.
CONTINUE_PROMPT = "Continue."


def build_prompt(inst: Instance) -> str:
    return PROMPT_TEMPLATE.format(problem_statement=inst.problem_statement.strip())


# ---------------------------------------------------------------------------
# Outcome scoring
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_answer(text: str) -> frozenset[str] | None:
    """Take the last <answer> block. None means the agent did not answer in the required format."""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    body = matches[-1]
    paths = {norm_path(line) for line in body.splitlines() if line.strip()}
    paths = {p for p in paths if p and not p.startswith("<")}
    return frozenset(paths)


# Non-source file patterns: SWE-bench gold patches usually contain only source files, no test/docs.
# The agent often reasonably reports tests as well, so the primary definition uses recall (gold ⊆ answer).
_NONSOURCE_RE = re.compile(
    r"(^|/)(tests?|testing|docs?|doc)/|(^|/)test_[^/]+\.py$|_test\.py$|\.(rst|md|txt)$",
    re.IGNORECASE,
)


def _filter_source(paths: frozenset[str]) -> frozenset[str]:
    return frozenset(p for p in paths if not _NONSOURCE_RE.search(p))


def outcome(pred: frozenset[str] | None, gold: frozenset[str]) -> dict:
    """Primary definition Y = (gold ⊆ pred), i.e. recall==1 (finding every gold source file counts as success).

    Plus two sensitivity definitions:
      Y_exact    = strict set equality
      Y_filtered = strict equality after filtering test/docs out of pred
    and precision/recall/f1 for fine-grained analysis.
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
        "Y": int(gold <= pred),                       # primary definition: recall == 1
        "Y_exact": int(pred == gold),
        "Y_filtered": int(pred_src == gold or gold <= pred_src),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "parsed": True,
    }


def file_hit(pred: frozenset[str] | None, target: str) -> int:
    """per-file hit indicator Y_f = 1[target ∈ pred].

    Measured in the pilot (20 samples): single-file tasks are 100% correct, failures are entirely
    concentrated in multi-file tasks (recall=0.5, one found and one missed). The binary Y signal
    is too sparse, so the attribution target is refined down to a single gold file: asking
    "which step decided whether file f got localized".
    Theorem 1's additive decomposition holds independently for each f (Y_f is binary too).
    """
    if pred is None:
        return 0
    return int(norm_path(target) in {norm_path(p) for p in pred})


def missed_files(pred: frozenset[str] | None, gold: frozenset[str]) -> list[str]:
    """Missed gold files -- the strongest source of negative attribution signal."""
    if pred is None:
        return sorted(gold)
    return sorted(gold - pred)


def final_message_text(items: list[dict]) -> str:
    """Assemble the agent's final text from the item/completed notification list returned by turn_run.

    The notification field layout varies across codex versions, so this scrapes every plausible
    text payload as loosely as possible. The more reliable path is answer_from_rollout()
    (which reads the rollout file directly).
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
    """Parse the final <answer> straight from the rollout JSONL. More reliable than final_message_text.

    Concatenates the text of all assistant messages, then uses parse_answer to take the last
    <answer> block.
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
# Repository preflight checks
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=False,
    )


def reset_repo(inst: Instance) -> dict:
    """Force the repository back to a clean base_commit.

    Dataset state as shipped (verified): all 499 repository worktrees have the gold patch applied
    (= already-fixed state), plus a procbench-* stash. The localization task must run on the buggy
    base_commit, otherwise the agent reads already-fixed code = answer leakage.

    Procedure: checkout base_commit (detached HEAD) → discard worktree changes → clean untracked files.
    Stashes are left alone (same origin as the worktree changes, they are the gold patch, and can be
    regenerated from SWE-bench).
    """
    repo = inst.repo
    if not repo.is_dir():
        return {"ok": False, "reason": f"repository does not exist: {repo}"}
    _git(repo, "checkout", "-f", inst.base_commit)
    _git(repo, "reset", "--hard", inst.base_commit)
    _git(repo, "clean", "-fdq")
    return check_repo(inst)


def check_repo(inst: Instance) -> dict:
    """Must pass before running a task: repository exists, worktree clean, HEAD is base_commit.

    Violating any one of these makes replays incomparable (the agent would see traces left by the
    previous run).
    """
    repo = inst.repo
    if not repo.is_dir():
        return {"ok": False, "reason": f"repository does not exist: {repo}"}
    dirty = _git(repo, "status", "--porcelain").stdout.strip()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    ok = not dirty and head.startswith(inst.base_commit[:10])
    return {
        "ok": ok,
        "dirty": bool(dirty),
        "head": head,
        "expected": inst.base_commit,
        "reason": "" if ok else ("worktree not clean" if dirty else f"HEAD={head} != base_commit"),
    }


def n_py_files(inst: Instance) -> int:
    """Coarse proxy for repository size, used for difficulty stratification and the U_task feature."""
    return sum(1 for _ in inst.repo.rglob("*.py"))
