"""v_0 = E[Y | q, G] 的执行前特征 —— C4 预测器的输入。

关键约束：v_0 是**执行前**预测，特征只能用 problem_statement(q) 与仓库图(G)，
**绝不能用 trace**（那是执行后信息，用了就是数据泄露）。

pilot 特征（纯本地、零 API、零第三方依赖）：
  文本侧（q）：
    - 描述长度、代码块数、traceback 有无、明确文件名提及数
    - 报错关键词密度（Error/Exception/Traceback/assert）
  图侧（G，用 problem_statement 提到的符号在仓库里的分布代理）：
    - 仓库 .py 文件数（规模 → 难度）、目录深度
    - q 中提到的标识符在仓库里的候选文件数（歧义度 → U_task 代理）
  gold 侧（仅用于分层/标签，不进特征）：n_gold

正式版用 tree-sitter AST + LSP 索引构造真图特征；pilot 用轻量正则代理，
足以验证"执行前是否可预测成败"这条 C4 主张的链路。
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
    "log_ps_len",          # 描述长度（log）
    "n_file_mentions",     # 明确提到的 .py 文件数
    "has_traceback",       # 是否含 traceback/代码块
    "err_density",         # 报错词密度
    "log_repo_pyfiles",    # 仓库 .py 文件数（log，规模）
    "log_ident_candidates",# q 中标识符在仓库的候选文件数（log，歧义度）
    "n_distinct_idents",   # q 中不同标识符数（log）
]


def _repo_py_count(repo: Path) -> int:
    try:
        return sum(1 for _ in repo.rglob("*.py"))
    except OSError:
        return 0


def _grep_candidate_files(repo: Path, idents: list[str], cap: int = 20,
                          max_files: int = 4000) -> int:
    """q 中提到的标识符在仓库里命中的不同 .py 文件数（歧义度代理 → U_task）。

    纯 Python 遍历实现（不依赖外部 rg —— 本环境 rg 是 shell 函数，subprocess 找不到）。
    执行前信息（只看仓库源码与 q，不看 trace）。为控成本对文件数封顶。
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
    # 取高频、去停用词式的短词过滤
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
    """按 FEATURE_NAMES 顺序组装（含 bias 截距）。"""
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
    """对一组 (instance, Y) 构造特征矩阵 X 与标签 y 与分组 groups(repo)。

    instances: list of dict {instance_id, project, Y, problem_statement, repo_path}
    返回 (X, y, groups, ids)。
    """
    X, y, groups, ids = [], [], [], []
    for r in instances:
        feat = extract_features(r["problem_statement"], Path(r["repo_path"]))
        X.append(feature_vector(feat))
        y.append(int(r["Y"]))
        groups.append(r["project"])
        ids.append(r["instance_id"])
    return np.array(X), np.array(y, dtype=float), groups, ids
