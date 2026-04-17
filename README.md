# Selective Prediction Under Uncertainty

> *A prediction interval of width 18 is not a prediction. It is a refusal to predict, dressed in interval notation. The model should say so explicitly.*

---

## What this is

A principled framework for deciding **which predictions a model should make** — and which it should withhold — based on interval width as an uncertainty signal grounded in conformal prediction theory.

This is not abstention in the vague sense of "low-confidence examples." It is a measurable, curve-based framework: the **Coverage-Abstention-Precision (CAP) curve** quantifies, for every abstention budget, the exact effective coverage and precision the model can guarantee on accepted predictions.

Part of a series on measurable reliability in ML. See also: [Mathematical Reliability](https://github.com/mariam-elelidy/Mathematical-Reliability-for-ML-Predictions) · [Assumption Stress Harness](https://github.com/mariam-elelidy/Assumption-Stress-Harness) · [Influence & Stability](https://github.com/mariam-elelidy/Influence-Stability-Analysis-for-ML-Predictions) · [Calibration](https://github.com/mariam-elelidy/Calibration-as-a-Measurable-Reliability-Constraint)

---

## Core result (n=2000, d=8, α=0.10, seed=42)

| Abstention | Eff. coverage | Δ coverage | Mean width | Actionable |
|---|---|---|---|---|
| 0% (baseline) | 0.8947 | — | 4.24 | 67.7% |
| 10% | 0.9028 | +0.008 | 3.35 | 75.0% |
| 20% | 0.9219 | +0.027 | 2.82 | 84.4% |
| **30%** | **0.9500** | **+0.055** | **2.58** | **96.4%** |
| 35% | 0.9614 | +0.066 | 2.54 | 100.0% |

**Subgroup finding at 30% abstention:**

| Group | Coverage | Width |
|---|---|---|
| Accepted (n=280) | **0.9500** | 2.58 |
| Rejected (n=120) | **0.7667** | 8.26 |

The marginal conformal guarantee (0.895 overall) was masking a 0.183 coverage gap between easy and hard cases. Selective prediction exposes — and addresses — it.

---

## Quick start

```bash
pip install numpy scipy

python selective_prediction.py                          # defaults
python selective_prediction.py --clinical-width 2.0     # tighter threshold
python selective_prediction.py --alpha 0.05 --n 4000    # higher coverage target
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 2000 | Dataset size |
| `--d` | 8 | Feature dimension |
| `--alpha` | 0.10 | Miscoverage level (1-α = coverage target) |
| `--seed` | 42 | Random seed |
| `--lam` | 0.001 | Ridge λ |
| `--clinical-width` | 3.0 | Width threshold for "actionable" prediction |

---

## How it works

```
Training data
      │
      ├──► Stage 1: ridge fit ŵ = (X'X + λI)⁻¹ X'y
      │
      ├──► Stage 2: variance model ŵ_σ fit on |training residuals|
      │             σ̂_i = clip(x_i' ŵ_σ, 0.3, 3.0)
      │
      └──► Calibration: conformity score s_i = |y_i - ŷ_i| / σ̂_i
                        conformal quantile q = k-th order statistic

Test time — per-point intervals
      width_j = 2 * q * σ̂_j          ← varies per point!
      accept if width_j ≤ τ

CAP curve — sweep τ from min(width) to max(width):
      x-axis: abstention rate = 1 - |accepted| / n_test
      y-axis: effective coverage on accepted subset
              actionability = fraction with width < clinical threshold
```

The key property: width $\propto \hat{\sigma}_j$. Abstaining on wide-interval cases is equivalent to abstaining on cases with high estimated noise — a direct measure of irreducible uncertainty.

---

## Why this matters clinically

Consider a risk-score model deployed across patients who vary in how predictable they are. Standard conformal prediction gives one number: 90% coverage overall. This can mean:

- 95% coverage on low-variability patients (confident, actionable)
- 76% coverage on high-variability patients (below target, clinically dangerous)
- Average: 90% — the guarantee appears met

Selective prediction converts "90% overall" into "95% on the cases we make predictions for" — a more meaningful and deployable reliability contract.

---

## Key findings

**The marginal guarantee hides a subgroup failure.** 0.895 overall, 0.767 on rejected cases. The conformal theorem guarantees marginal coverage, not subgroup coverage. Selective prediction makes the distinction visible and measurable.

**Width is a principled, not ad hoc, abstention signal.** Width ∝ sigma_hat. Abstaining on wide-interval cases is abstaining on genuinely uncertain cases — the subgroup audit confirms this: rejected cases have 6.35× higher true sigma than accepted cases.

**Global q cannot be selectively improved.** With constant width, every test case looks the same. Locally adaptive intervals are a prerequisite for the CAP curve to have any shape at all. The global vs adaptive comparison table makes this concrete.

---

## Outputs

| Output | What it answers |
|---|---|
| Var model correlation | "How well does the uncertainty signal track true difficulty?" |
| Baseline metrics (global vs adaptive) | "What does the model look like before any abstention?" |
| CAP curve | "For each abstention budget: coverage, width, actionability?" |
| Key operating points table | "Where is the best coverage-precision trade-off?" |
| Subgroup audit (accepted vs rejected) | "Does abstaining actually improve coverage on accepted cases?" |
| Global vs adaptive comparison | "Does selective prediction require adaptive intervals?" |
| Summary tensor `[abst, eff_cov, mean_w, actionable]` | Machine-readable curve for comparison across settings |

---

## Repository layout

```
├── README.md                  ← this file
├── selective_prediction.py    ← implementation
├── output.txt                 ← annotated run output with cross-observations
└── writeup.md                 ← full technical writeup
```

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.
- Geifman, Y., & El-Yaniv, R. (2017). Selective prediction in deep neural networks. *NeurIPS*.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*.
- Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for nonparametric regression. *JRSS-B*.

---
