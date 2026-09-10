# Method Implementation Notes

Maps the paper's formalism onto concrete code locations, so that the implementation can be checked against the definitions.

---

## 1. Structural causal model and identifiability

The paper's SCM (Eqs. 2-4):

```
A_t := argmax_k ( log π_k(S_t) + G^(t)_{π,k} ),   G^(t)_{π,k} ~ Gumbel(0,1)
O_t := E(A_t, G; U^(t)_env)
S_{t+1} := T(S_t, A_t, O_t),   S_1 := σ(q, sys)
F̂ := φ(S_T),   Y := 1[F* ⊆ F̂]
```

**Proposition 1 (constructive sequential randomization)**: under Assumptions 1-3,
`A_t ⊥ (Z, U^(≥t)) | τ_{1:t−1}`, hence

```
E[Y | τ_{1:t−1}, do(A_t = a)] = E[Y | τ_{1:t−1}, A_t = a]
```

Code correspondence:

| Formalism | Code |
|---|---|
| Conditioning on the realized prefix τ_{1:t−1} | `rollout.truncate_before_step(items, k)` |
| Exogeneity of Gumbel-max decoding noise | provided by server-side sampling; **not testable from traces** (the paper states plainly that this is a structural assumption) |
| Environment noise U_env pinned | `config/env_pin.sh` + `appserver.PINNED_ENV` |
| Assumption 3 (replayable) | `analyze/replay_check.py` measures 1.00 / 100 actions |
| Assumption 1 (context observable) | `rollout.compaction_lines()` measures 0 compactions |

**Key point**: no "no unmeasured confounding" assumption is needed here. The policy depends only on context we also record,
and decoding noise is a decoder property -- so conditioning on the prefix fixes S_t, and the remaining variation is exogenous.
The paper therefore says the trace is **by construction** a sequentially randomized experiment.

---

## 2. Anytime value and exact additive decomposition

Define `v_t := E[Y | τ_{1:t}, q, G]`, with incremental gain `Δ_t := v_t − v_{t−1}`.

**Theorem 1**: `Σ_{t=1}^T Δ_t = v_T − v_0 = Y − E[Y | q, G]`

That is, the episode's "surprise relative to the pre-execution expectation" is allocated across steps with no residual. Attribution and prediction are two ends of the same object:
the left endpoint v_0 is exactly the pre-execution prediction.

Code: `attribution/causal.py::causal_attribution`; `estimate_v` estimates each v_t with m replays.

**An honest point about the implementation**: `v_t` is estimated only at **5 equally spaced points**, not at every step.
What is reported is therefore **segment effects**, and Σ Δ_t does not exactly equal v_T − v_0.
The paper explicitly does **not** report additivity as evidence (Appendix E.1), only as an internal consistency check.

Choosing equally spaced points rather than a "contiguous tail window" is deliberate: windows concentrated in the late part of a trace would contaminate the positional comparison that §5.7 relies on.

---

## 3. Hybrid estimator and positivity

The estimator is only valid where the **local action distribution still has overlap**. The paper does not assume positivity, it **measures** it:

```
Ĥ(π̂_t) = − Σ_c π̂_t(c) log π̂_t(c),   π̂_t(c) = m⁻¹ Σ_i 1[bucket(A_t^(i)) = c]
```

`bucket` operates at the (tool, target object) granularity, implemented in `rollout.action_bucket()`.

Hence the two branches of Eq. 8 (`causal.py::hybrid_delta_at`):

```python
if pi["high_entropy"] and pi["entropy"] > tau_H:
    # Observational branch: replays still have overlap ⇒ Prop.1 applies directly
    delta = v_t - v_prev
else:
    # Interventional branch: policy is near-deterministic at this point ⇒ substitute an adjacent action a' and execute for real
    delta = E[Y | τ, do(A_t=a')] - E[Y | τ, do(A_t=a_t)]
```

**τ_H = 0** is the loosest possible threshold: as long as ≥2 distinct actions appear among the m replays, the observational branch is taken.
This makes the paper's reported "50% of probe points observationally identified" an **upper bound**.

**The interventional branch is not observational identification under the original policy** (Assumption 4). It estimates the effect of "moving to an adjacent action
that remains plausible under the perturbed policy π̃". The paper **labels** each estimate with the branch it took,
and in Appendix B.4 places the two kinds of quantity on different rungs of the causal ladder (II.5 vs III).

---

## 4. Why the outcome variable is recall

With binary `Y`, single-target instances are almost always solved ⇒ most substitute actions do not change the indicator ⇒ Δ_t ≡ 0.
The paper screens six candidates by **intervention sensitivity** (paired effect / standard deviation, `analyze/x8_outcome_sensitivity.py`):

| Candidate | Sensitivity | Movable fraction |
|---|---|---|
| **recall over F\*** | **2.505** | **98.6%** |
| F₁ | 2.165 | -- |
| Binary success | 1.643 | 73.2% |

Theorem 1 only requires Y to be **bounded**, so switching to `recall ∈ [0,1]` or a per-file indicator `Y_f`
changes none of the formal conclusions (Appendix B.6).

Code: `probes/task.py::outcome()` returns `Y / Y_exact / Y_filtered / precision / recall / f1` together.

---

## 5. Three-way uncertainty decomposition

```
H[Y | q, G] = U_task + U_policy + U_model
```

| Component | Type | Estimation | Implied action |
|---|---|---|---|
| `U_task` | Aleatoric, irreducible | residual; graph-ambiguity covariates | ask the user for clarification |
| `U_policy` | Aleatoric, reducible | between-rollout variance at fixed q | resample and aggregate |
| `U_model` | Epistemic | posterior variance propagated through the link | abstain / escalate / collect data |

Code: `uq/predictor.py::decompose_uncertainty`; v₀ uses
`BayesianLogisticLaplace` (IRLS for the MAP, inverse Hessian as the Laplace covariance,
probit approximation to propagate parameter uncertainty).

**Read the paper's self-imposed limits on this section carefully**: the three-way decomposition is validated only on a **synthetic generative process** (D.7),
and on real traces only `U_policy` is directly estimated (0.049, from 23 instances with K=4 repeated rollouts).
**Identifiability of the three terms on real traces is not established.** `analyze/uq_selftest.py` is "a specification with a self-check",
not a validation against real data.

The 7 pre-execution covariates for v₀ (`uq/features.py`): issue description length, number of explicitly named files,
whether a traceback is present, error term density, repository size, number of distinct identifiers, number of repository files containing those identifiers
(as a proxy for `U_task` ambiguity). **None of them read the trace** -- otherwise execution-time information would leak into a "pre-execution" predictor.

---

## 6. Identification design for collider bias

This is the only comparison in the paper that holds without requiring our own estimator to be correct.

Same model, same rubric; only the information set visible to the judge is changed:

| Condition | Conditions on | Consequence |
|---|---|---|
| full-trace | prefix + **subsequent steps** + **realized outcome Y** | conditions on descendants of A_t ⇒ opens non-causal paths |
| prefix-only | fixed 60% prefix, Y hidden | prefix-conditioned control arm |

If the full-trace judge were approximating a prefix-conditioned step-local quantity, this manipulation **should not** move the blame position.
Measured displacement is **+0.537** (sign test p = 1.1×10⁻¹⁶).

**A design flaw and its correction** (faithfully recorded in paper E.2): the original design truncated prefix-only just before
"the step selected by the causal estimate". Because the judge cannot select a step it cannot see, this pushes the positions reachable under the prefix-only condition
below the comparison point, **producing a positive displacement even if conditioning on descendants had no effect at all**.
The corrected design uses a truncation fraction (60%) that is **fixed in advance and independent of any estimate**.

Code: `attribution/judge.py` (the two conditions) + `analyze/attribution_compare.py` (displacement statistics).
`--prefix-frac 0.6` is hard-coded in `harness/parallel.py` to prevent accidental changes.

---

## 7. Statistical conventions (Appendix C.5)

Conventions fixed up front and applied throughout; picking rules after seeing results is not allowed:

- **Confidence intervals** = episode-level or instance-level bootstrap
- **Null hypothesis** = within-episode permutation or closed-form expectation, **never** a sampled random baseline
- **Ranking metrics** use the closed-form expectation under random tie-breaking (count-based scorers have very many ties)
- **Common denominator**: for predictors with different candidate-set sizes, a realized object falling outside the set counts as a miss rather than being dropped
- **Multiple comparisons** BH-FDR, q = 0.10, distinguishing exploratory / confirmatory
- **Wilcoxon** refuses to give a p-value when n < 6

The implementation is scattered across the individual analysis scripts (deliberately not factored into a shared library: each test constructs its null hypothesis differently,
and abstraction would obscure the differences). `v3g_regime.py` exports a few shared helpers
(`rank_ge`, `exp_hit`, `exp_recip`, `eff_cand`, `ece`).

---

## 8. Dataflow reachability ≠ causal effect

Paper Appendix B.7 clarifies this specifically, because it is a common substitute criterion:

- **No dataflow yet causally powerful**: a `grep` that returns nothing contributes no content to the subsequent context,
  and is "inert" by the dataflow criterion; but it may make the agent abandon the correct directory -- **an empty result is itself information**.
  Under this paper's estimand it shows up as a large **\|Δ_t\|**.
- **Dataflow present yet causally inert**: a long file read injects thousands of tokens but changes no subsequent action.
  Fix the prefix, delete that read, and verify by replay: the continuation is unchanged ⇒ the step's contribution is 0.

So taxonomies that score steps by downstream dataflow will **systematically misrank** precisely those steps that the causal criterion identifies as decisive.
