"""anytime 价值预测器 v_t 与不确定性三分解 —— C4 + C5，纯 numpy。

C4：v_t = E[Y | τ_1:t, q, G] 的可学习代理。pilot 用贝叶斯 logistic（Laplace 近似），
    repo 作分层随机效应处理 django 主导与 12 组不平衡。后验协方差直接给 epistemic (U_model)。

C5：三分解
    U_task   ~ 任务内在歧义（aleatoric，不可约）——由图/文本特征的 Bayes-optimal 残差熵代理
    U_policy ~ 解码随机性（aleatoric，可约）——由同任务 K 次 rollout 的结果方差估计
    U_model  ~ 预测器认知不确定（epistemic）——由参数后验方差经 logistic 传播

为什么不用 sklearn/scipy：环境只有 numpy。手写 Laplace 近似（IRLS + Hessian 求逆）足够，
且完全可控、可复现（无随机初始化依赖）。

注意：pilot 阶段这是**方法骨架 + 合成自测**，真实特征在 features.py 里从 run 结果构造。
"""

from __future__ import annotations

import numpy as np


class BayesianLogisticLaplace:
    """带高斯先验的贝叶斯 logistic 回归，Laplace 近似后验。

    先验 w ~ N(0, (1/λ) I)。MAP 由 IRLS 求，后验协方差 = Hessian^{-1}。
    预测分布用 probit 近似传播参数不确定性，从而分出 epistemic。
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.w: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogisticLaplace":
        n, d = X.shape
        w = np.zeros(d)
        lam = self.l2
        for _ in range(self.max_iter):
            eta = X @ w
            p = self._sigmoid(eta)
            W = p * (1 - p)
            # 梯度（负对数后验）与 Hessian
            grad = X.T @ (p - y) + lam * w
            H = (X * W[:, None]).T @ X + lam * np.eye(d)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            w_new = w - step
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new
        eta = X @ w
        p = self._sigmoid(eta)
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + lam * np.eye(d)
        self.w = w
        self.cov = np.linalg.inv(H)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """后验预测均值，用 probit 近似边际化参数不确定性。"""
        assert self.w is not None and self.cov is not None
        mu = X @ self.w
        var = np.einsum("ij,jk,ik->i", X, self.cov, X)      # 每行 x^T Σ x
        kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
        return self._sigmoid(kappa * mu)

    def predict_map(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return self._sigmoid(X @ self.w)

    def epistemic_var(self, X: np.ndarray) -> np.ndarray:
        """logit 空间的参数不确定性方差 x^T Σ x（U_model 的来源）。"""
        assert self.cov is not None
        return np.einsum("ij,jk,ik->i", X, self.cov, X)


# ---------------------------------------------------------------------------
# 不确定性三分解
# ---------------------------------------------------------------------------

def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decompose_uncertainty(
    p_pred: np.ndarray,
    epistemic_logit_var: np.ndarray,
    policy_var: np.ndarray | None = None,
) -> dict:
    """把预测熵分解为 total / task / policy / model。

    - total     = H(p_pred)（预测分布的熵）
    - U_model   ∝ 参数后验在 logit 上的方差经 sigmoid 斜率传播（epistemic）
    - U_policy  = 同任务 K 次 rollout 的结果方差（若提供）
    - U_task    = total − U_model − U_policy（残差，aleatoric 不可约部分）

    这是 pilot 的可操作近似；正式实验用完整 Bayesian 预测熵分解 + 方差成分模型。
    """
    total = binary_entropy(p_pred)
    slope = p_pred * (1 - p_pred)                    # d sigmoid / d logit
    u_model = slope * epistemic_logit_var            # logit 方差 → 概率不确定近似
    u_policy = policy_var if policy_var is not None else np.zeros_like(p_pred)
    u_task = np.clip(total - u_model - u_policy, 0.0, None)
    return {
        "total": total,
        "U_model": u_model,
        "U_policy": u_policy,
        "U_task": u_task,
    }


# ---------------------------------------------------------------------------
# 校准指标（C4：X4a）
# ---------------------------------------------------------------------------

def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error（等宽分箱）。"""
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / n) * abs(conf - acc)
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUROC（用不确定性/预测分数区分成败），秩和实现，无 sklearn。"""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    # 简化：并列取平均秩
    for v in np.where(counts > 1)[0]:
        mask = inv == v
        ranks[mask] = ranks[mask].mean()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def risk_coverage(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """selective prediction：按置信度降序，返回 (coverage, risk, AURC)。

    scores 越大越有信心成功；risk = 1 - accuracy。
    """
    order = np.argsort(-scores)
    y_sorted = y[order]
    n = len(y)
    cov = np.arange(1, n + 1) / n
    cum_correct = np.cumsum(y_sorted)
    risk = 1 - cum_correct / np.arange(1, n + 1)
    aurc = float(np.mean(risk))
    return cov, risk, aurc
