#!/usr/bin/env python3
"""UQ self-check: verify on a **known generative process** that the Laplace posterior and the three-way decomposition behave correctly.

Why it is needed
----------------
The paper reports two different things about v0, which must be verified separately:
  (a) whether our Bayesian machinery yields a behaviourally correct posterior -- only testable on synthetic data with known truth;
  (b) whether hand-built pre-step covariates predict localization success on real repositories -- that is the LORO experiment (negative result).
This script only does (a). It does not touch the real corpus and calls no API.

Criterion (decisive test for the epistemic term)
------------------------------------------------
Push the covariates outside the training support and U_model must rise significantly. If it does not, the Laplace
posterior is merely rescaling probabilities without actually propagating parameter uncertainty -- then U_model does not deserve the name epistemic.

Usage
-----
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
                    help="repeat rollouts per test point, used to empirically estimate U_policy")
    ap.add_argument("--ood-scale", type=float, default=5.0,
                    help="factor by which the covariates are pushed off the training support")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    d = args.d

    # ---- known generative process ----
    w_true = rng.normal(0.0, 1.0, size=d)
    X_tr = rng.normal(0.0, 1.0, size=(args.n_train, d))
    p_tr = 1.0 / (1.0 + np.exp(-(X_tr @ w_true)))
    y_tr = (rng.random(args.n_train) < p_tr).astype(float)

    X_te = rng.normal(0.0, 1.0, size=(args.n_test, d))
    p_te_true = 1.0 / (1.0 + np.exp(-(X_te @ w_true)))

    model = BayesianLogisticLaplace(l2=1.0).fit(X_tr, y_tr)

    # ---- parameter recovery: correlation of MAP with truth (did the machinery learn anything) ----
    w_hat = model.w
    w_corr = float(np.corrcoef(w_true, w_hat)[0, 1])

    # ---- in support: three-way decomposition ----
    p_in = model.predict_proba(X_te)
    ev_in = model.epistemic_var(X_te)
    # U_policy: outcome variance over K rollouts of the same task (repeats simulated with the true p)
    draws = (rng.random((args.k_rollouts, args.n_test)) < p_te_true).astype(float)
    policy_var = draws.var(axis=0, ddof=1)
    dec_in = decompose_uncertainty(p_in, ev_in, policy_var=policy_var)

    # ---- out of support: epistemic must rise ----
    X_ood = X_te * args.ood_scale
    p_ood = model.predict_proba(X_ood)
    ev_ood = model.epistemic_var(X_ood)
    dec_ood = decompose_uncertainty(p_ood, ev_ood)

    um_in = float(dec_in["U_model"].mean())
    um_ood = float(dec_ood["U_model"].mean())

    # calibration (in support, against the true p rather than labels, to avoid finite-sample noise dominating)
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
            "criterion": "out-of-support U_model > 2x in-support U_model",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== UQ self-check (known generative process) ===")
    pr = res["parameter_recovery"]
    print(f"parameter recovery corr(w_true, MAP) = {pr['corr_w_true_vs_map']}  "
          f"in-support probability MAE = {pr['mean_abs_prob_error_in_support']}")
    dd = res["in_support_decomposition"]
    print(f"predictive entropy {dd['predictive_entropy']} = U_task {dd['U_task']} "
          f"+ U_policy {dd['U_policy']} + U_model {dd['U_model']}")
    ei = res["epistemic_inflation_test"]
    print(f"epistemic inflation test: {ei['U_model_in_support']} -> "
          f"{ei['U_model_out_of_support']} (x{ei['ratio']})  pass={ei['pass']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
