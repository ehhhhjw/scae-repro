#!/usr/bin/env python3
"""UQ 自检：在**已知生成过程**上验证 Laplace 后验与三分解的行为是否正确。

为什么需要它
------------
论文对 v0 报了两件不同的事，必须分开验证：
  (a) 我们的贝叶斯机制是否产出行为正确的后验 —— 只能在已知真值的合成数据上检验；
  (b) 手工前置协变量能否预测真实仓库上的定位成功 —— 这是 LORO 实验（结论为负）。
本脚本只做 (a)。它不碰真实语料，不调任何 API。

判据（epistemic 项的决定性检验）
--------------------------------
把协变量推到训练支撑之外，U_model 必须显著上升。若不上升，说明 Laplace 后验只是
在重新缩放概率，而没有真正传播参数不确定性 —— 那么 U_model 不配叫 epistemic。

用法
----
    python3 -m pilot.analyze.uq_selftest --out pilot/runs/uq_selftest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pilot.uq.predictor import (
    BayesianLogisticLaplace,
    binary_entropy,
    decompose_uncertainty,
)

SEED = 20260810


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot/runs/uq_selftest.json")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--k-rollouts", type=int, default=4,
                    help="每个测试点的重复 rollout 次数，用于经验估计 U_policy")
    ap.add_argument("--ood-scale", type=float, default=5.0,
                    help="把协变量推离训练支撑的倍数")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    d = args.d

    # ---- 已知生成过程 ----
    w_true = rng.normal(0.0, 1.0, size=d)
    X_tr = rng.normal(0.0, 1.0, size=(args.n_train, d))
    p_tr = 1.0 / (1.0 + np.exp(-(X_tr @ w_true)))
    y_tr = (rng.random(args.n_train) < p_tr).astype(float)

    X_te = rng.normal(0.0, 1.0, size=(args.n_test, d))
    p_te_true = 1.0 / (1.0 + np.exp(-(X_te @ w_true)))

    model = BayesianLogisticLaplace(l2=1.0).fit(X_tr, y_tr)

    # ---- 参数恢复：MAP 与真值的相关（机制是否学到东西）----
    w_hat = model.w
    w_corr = float(np.corrcoef(w_true, w_hat)[0, 1])

    # ---- 支撑内：三分解 ----
    p_in = model.predict_proba(X_te)
    ev_in = model.epistemic_var(X_te)
    # U_policy：同一任务 K 次 rollout 的结果方差（用真 p 模拟重复 rollout）
    draws = (rng.random((args.k_rollouts, args.n_test)) < p_te_true).astype(float)
    policy_var = draws.var(axis=0, ddof=1)
    dec_in = decompose_uncertainty(p_in, ev_in, policy_var=policy_var)

    # ---- 支撑外：epistemic 必须上升 ----
    X_ood = X_te * args.ood_scale
    p_ood = model.predict_proba(X_ood)
    ev_ood = model.epistemic_var(X_ood)
    dec_ood = decompose_uncertainty(p_ood, ev_ood)

    um_in = float(dec_in["U_model"].mean())
    um_ood = float(dec_ood["U_model"].mean())

    # 校准（支撑内，对真 p 而非标签，避免有限样本噪声主导）
    mae_p = float(np.abs(p_in - p_te_true).mean())

    res = {
        "seed": SEED,
        "setup": {
            "n_train": args.n_train, "n_test": args.n_test, "d": d,
            "k_rollouts": args.k_rollouts, "ood_scale": args.ood_scale,
            "prior": "w ~ N(0, I), l2=1.0",
        },
        "parameter_recovery": {
            "corr_w_true_vs_map": round(w_corr, 4),
            "mean_abs_prob_error_in_support": round(mae_p, 4),
        },
        "in_support_decomposition": {
            "predictive_entropy": round(float(dec_in["total"].mean()), 4),
            "U_task": round(float(dec_in["U_task"].mean()), 4),
            "U_policy": round(float(dec_in["U_policy"].mean()), 4),
            "U_model": round(um_in, 4),
        },
        "epistemic_inflation_test": {
            "U_model_in_support": round(um_in, 4),
            "U_model_out_of_support": round(um_ood, 4),
            "ratio": round(um_ood / um_in, 3) if um_in > 0 else None,
            "pass": bool(um_ood > 2.0 * um_in),
            "criterion": "支撑外 U_model > 2x 支撑内 U_model",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== UQ 自检（已知生成过程）===")
    pr = res["parameter_recovery"]
    print(f"参数恢复 corr(w_true, MAP) = {pr['corr_w_true_vs_map']}  "
          f"支撑内概率 MAE = {pr['mean_abs_prob_error_in_support']}")
    dd = res["in_support_decomposition"]
    print(f"预测熵 {dd['predictive_entropy']} = U_task {dd['U_task']} "
          f"+ U_policy {dd['U_policy']} + U_model {dd['U_model']}")
    ei = res["epistemic_inflation_test"]
    print(f"epistemic 膨胀检验: {ei['U_model_in_support']} -> "
          f"{ei['U_model_out_of_support']} (x{ei['ratio']})  pass={ei['pass']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
