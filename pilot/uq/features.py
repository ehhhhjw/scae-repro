"""Pre-execution features for v_0 = E[Y | q, G] -- the input to the C4 predictor.

Key constraint: v_0 is a **pre-execution** prediction, so features may only use
problem_statement(q) and the repository graph(G); **never the trace** (that is
post-execution information, and using it is data leakage).

pilot features (purely local, zero API, zero third-party dependencies):
  text side (q):
    - description length, number of code blocks, presence of traceback, number of explicit filename mentions
    - error keyword density (Error/Exception/Traceback/assert)
  graph side (G, proxied by the distribution in the repo of symbols mentioned in problem_statement):
    - number of .py files in the repo (scale -> difficulty), directory depth
    - number of candidate files in the repo for identifiers mentioned in q (ambiguity -> proxy for U_task)
  gold side (used only for stratification/labels, not as features): n_gold

The production version builds true graph features with a tree-sitter AST + LSP index;
pilot uses a lightweight regex proxy, which suffices to validate the chain behind the
C4 claim "is success/failure predictable before execution".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")
_FILEMENTION = re.compile(r"\b[\w/]+\.py\b")
_CODEBLOCK = re.compile(r"```|>>>|\.\.\.|Traceback")
_ERRWORDS = re.compile(r"\b(Error|Exception|Traceback|assert|raise|fail|bug)\b", re.I)

FEATURE_NAMES = [
    "bias",
    "log_ps_len",          # description length (log)
    "n_file_mentions",     # number of explicitly mentioned .py files
    "has_traceback",       # contains a traceback/code block
    "err_density",         # error word density
    "log_repo_pyfiles",    # number of .py files in the repo (log, scale)
    "log_ident_candidates",# number of candidate files in the repo for identifiers in q (log, ambiguity)
    "n_distinct_idents",   # number of distinct identifiers in q (log)
]


def _repo_py_count(repo: Path) -> int:
    try:
        return sum(1 for _ in repo.rglob("*.py"))
    except OSError:
        return 0


def _grep_candidate_files(repo: Path, idents: list[str], cap: int = 20,
                          max_files: int = 4000) -> int:
    """Number of distinct .py files in the repo hit by identifiers mentioned in q (ambiguity proxy -> U_task).

    Pure Python traversal (does not rely on external rg -- in this environment rg is a shell function that subprocess cannot find).
    Pre-execution information (looks only at repo source and q, never the trace). File count is capped to control cost.
    """
    if not idents:
        return 0
    needles = [x for x in idents[:cap] if len(x) >= 4]
    if not needles:
        return 0
    hit = 0
    scanned = 0
    for pyf in repo.rglob("*.py"):
        if scanned >= max_files:
            break
        if any(p in (".git", "node_modules", ".tox", "build", "dist") for p in pyf.parts):
            continue
        scanned += 1
        try:
            text = pyf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(nd in text for nd in needles):
            hit += 1
    return hit


def extract_features(problem_statement: str, repo: Path) -> dict:
    q = problem_statement or ""
    idents = _IDENT.findall(q)
    # Keep frequent ones, filter out stopword-like short words
    from collections import Counter
    common = {"this", "that", "with", "from", "have", "will", "when", "then",
              "should", "which", "would", "there", "class", "self", "None",
              "True", "False", "return", "import"}
    idc = Counter(x for x in idents if x not in common)
    top_idents = [w for w, _ in idc.most_common(30)]

    n_pyfiles = _repo_py_count(repo)
    n_cand = _grep_candidate_files(repo, top_idents)

    return {
        "log_ps_len": float(np.log1p(len(q))),
        "n_file_mentions": float(len(set(_FILEMENTION.findall(q)))),
        "has_traceback": 1.0 if _CODEBLOCK.search(q) else 0.0,
        "err_density": float(len(_ERRWORDS.findall(q)) / (len(q.split()) + 1) * 100),
        "log_repo_pyfiles": float(np.log1p(n_pyfiles)),
        "log_ident_candidates": float(np.log1p(n_cand)),
        "n_distinct_idents": float(np.log1p(len(idc))),
        "_raw_n_pyfiles": n_pyfiles,
        "_raw_n_candidates": n_cand,
    }


def feature_vector(feat: dict) -> np.ndarray:
    """Assemble in FEATURE_NAMES order (including the bias intercept)."""
    return np.array([
        1.0,
        feat["log_ps_len"],
        feat["n_file_mentions"],
        feat["has_traceback"],
        feat["err_density"],
        feat["log_repo_pyfiles"],
        feat["log_ident_candidates"],
        feat["n_distinct_idents"],
    ])


def build_matrix(instances, repos_ok_only: bool = True):
    """Build feature matrix X, labels y and groups(repo) for a set of (instance, Y).

    instances: list of dict {instance_id, project, Y, problem_statement, repo_path}
    Returns (X, y, groups, ids).
    """
    X, y, groups, ids = [], [], [], []
    for r in instances:
        feat = extract_features(r["problem_statement"], Path(r["repo_path"]))
        X.append(feature_vector(feat))
        y.append(int(r["Y"]))
        groups.append(r["project"])
        ids.append(r["instance_id"])
    return np.array(X), np.array(y, dtype=float), groups, ids
