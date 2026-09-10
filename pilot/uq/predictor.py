"""Anytime value predictor v_t and the uncertainty three-way decomposition -- C4 + C5, pure numpy.

C4: a learnable proxy for v_t = E[Y | τ_1:t, q, G]. The pilot uses Bayesian logistic regression
    (Laplace approximation), with repo as a hierarchical random effect to handle django dominance
    and the imbalance across the 12 groups. The posterior covariance directly gives epistemic
    uncertainty (U_model).

C5: three-way decomposition
    U_task   ~ intrinsic task ambiguity (aleatoric, irreducible) -- proxied by the Bayes-optimal
               residual entropy of the graph/text features
    U_policy ~ decoding randomness (aleatoric, reducible) -- estimated from the outcome variance
               of K rollouts on the same task
    U_model  ~ predictor epistemic uncertainty (epistemic) -- propagated from the parameter
               posterior variance through the logistic link

Why not sklearn/scipy: the environment only has numpy. A hand-written Laplace approximation
(IRLS + Hessian inversion) suffices, and is fully controllable and reproducible (no dependence
on random initialization).

Note: at the pilot stage this is a **method skeleton + synthetic self-test**; the real features are
constructed from run results in features.py.
"""

from __future__ import annotations

import numpy as np


class BayesianLogisticLaplace:
    """Bayesian logistic regression with a Gaussian prior, Laplace-approximated posterior.

    Prior w ~ N(0, (1/λ) I). The MAP is found by IRLS, posterior covariance = Hessian^{-1}.
    The predictive distribution propagates parameter uncertainty with a probit approximation,
    which is what isolates the epistemic part.
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
            # gradient (negative log posterior) and Hessian
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
        """Posterior predictive mean, marginalizing parameter uncertainty with a probit approximation."""
        assert self.w is not None and self.cov is not None
        mu = X @ self.w
        var = np.einsum("ij,jk,ik->i", X, self.cov, X)      # x^T Σ x per row
        kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
        return self._sigmoid(kappa * mu)

    def predict_map(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return self._sigmoid(X @ self.w)

    def epistemic_var(self, X: np.ndarray) -> np.ndarray:
        """Parameter uncertainty variance x^T Σ x in logit space (the source of U_model)."""
        assert self.cov is not None
        return np.einsum("ij,jk,ik->i", X, self.cov, X)


# ---------------------------------------------------------------------------
# Uncertainty three-way decomposition
# ---------------------------------------------------------------------------

def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decompose_uncertainty(
    p_pred: np.ndarray,
    epistemic_logit_var: np.ndarray,
    policy_var: np.ndarray | None = None,
) -> dict:
    """Decompose the predictive entropy into total / task / policy / model.

    - total     = H(p_pred) (entropy of the predictive distribution)
    - U_model   ∝ logit-space parameter posterior variance propagated through the sigmoid slope (epistemic)
    - U_policy  = outcome variance across K rollouts of the same task (if provided)
    - U_task    = total − U_model − U_policy (residual, the irreducible aleatoric part)

    This is the pilot's operational approximation; the full experiment uses the complete Bayesian
    predictive entropy decomposition + a variance components model.
    """
    total = binary_entropy(p_pred)
    slope = p_pred * (1 - p_pred)                    # d sigmoid / d logit
    u_model = slope * epistemic_logit_var            # logit variance → probability uncertainty approximation
    u_policy = policy_var if policy_var is not None else np.zeros_like(p_pred)
    u_task = np.clip(total - u_model - u_policy, 0.0, None)
    return {
        "total": total,
        "U_model": u_model,
        "U_policy": u_policy,
        "U_task": u_task,
    }


# ---------------------------------------------------------------------------
# Calibration metrics (C4: X4a)
# ---------------------------------------------------------------------------

def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width binning)."""
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
    """AUROC (separating success/failure by uncertainty/prediction score), rank-sum implementation, no sklearn."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    # simplification: ties take the average rank
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
    """Selective prediction: sort by descending confidence, return (coverage, risk, AURC).

    Larger scores mean more confidence in success; risk = 1 - accuracy.
    """
    order = np.argsort(-scores)
    y_sorted = y[order]
    n = len(y)
    cov = np.arange(1, n + 1) / n
    cum_correct = np.cumsum(y_sorted)
    risk = 1 - cum_correct / np.arange(1, n + 1)
    aurc = float(np.mean(risk))
    return cov, risk, aurc
