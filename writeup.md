# Selective Prediction Under Uncertainty

**Author:** Mariam Mohamed Elelidy  
**Topic:** Selective Prediction · Conformal Theory · Decision Reliability · Clinical AI

---

## TL;DR

Conformal prediction provides a *marginal* coverage guarantee across all test points. But marginal coverage masks a structural problem in heterogeneous populations: hard cases may be systematically under-covered while easy cases are over-covered, and the average hides both.

This artifact introduces **selective prediction** — allowing the model to withhold predictions where its intervals are too wide to be useful. The result is a Coverage-Abstention-Precision (CAP) curve: for every abstention budget, the exact coverage and precision the model can guarantee on the cases it accepts.

**Core result:** At 30% abstention, effective coverage rises 0.895 → 0.950 (+6.1%), mean width falls 4.24 → 2.58 (−39%), actionability rises 67.7% → 96.4%. The rejected cases have only 0.767 coverage — the marginal guarantee hid a subgroup reliability failure that selective prediction both exposes and addresses.

---

## 1. Problem

Standard conformal prediction gives one guarantee: the interval covers the truth ≥90% of the time *on average*. This does not tell you:

- How reliable the model is on the high-variability subgroup specifically
- Whether any individual prediction is informative enough to act on
- Which predictions should be flagged for specialist review

A prediction interval of width 15 on a quantity measured in [0, 10] is not a prediction — it is an admission of ignorance wearing interval clothing. Reporting it as "covered" inflates confidence and misrepresents reliability.

**Selective prediction asks:** given a budget for how many predictions to make, which should the model make, and what can it guarantee about those it accepts?

---

## 2. Testable Claims

**Primary:** Under heteroscedastic noise with a well-estimated variance model, interval width is a valid uncertainty signal. Abstaining on wide-interval predictions increases effective coverage and precision on the accepted subset.

**Subgroup claim:** The marginal conformal coverage guarantee masks systematic subgroup coverage inequality. At 30% abstention, rejected cases have coverage 0.767 — 13 points below the promised 0.90.

**Global-vs-adaptive claim:** Selective prediction only adds value with locally adaptive intervals. A global-q model (constant width) cannot distinguish easy from hard cases — its CAP curve is flat.

---

## 3. Method

### Heteroscedastic data

$$X \sim \mathcal{N}(0, I_{n \times d}), \quad \sigma_i = 0.3 + 2.5 \cdot \text{relu}(x_{i,1}), \quad y_i = x_i^\top w^* + \sigma_i \varepsilon_i$$

Half the dataset has $\sigma \approx 0.3$ (easy); the other half has $\sigma$ up to 2.8 (hard). This bimodal structure is realistic: clinical populations routinely have clearly differentiated patient subgroups.

### Two-stage locally adaptive conformal

**Stage 1 — prediction model:** ridge regression ŵ.

**Stage 2 — variance model:** ridge on |training residuals| → $\hat{\sigma}_i = \text{clip}(x_i^\top \hat{w}_\sigma, 0.3, 3.0)$.
Estimated sigma correlates 0.965 with true sigma.

**Locally adaptive conformity score:** $s_i = |y_i - \hat{y}_i| / \hat{\sigma}_i$ for $i \in \text{cal}$

**Conformal quantile:** $q = r_{(k)},\quad k = \lceil (n_{\text{cal}}+1)(1-\alpha) \rceil$

**Per-point test interval:**
$$\hat{C}(x_j) = [\hat{y}_j \pm q \cdot \hat{\sigma}_j], \quad \text{width}_j = 2q\hat{\sigma}_j$$

Width varies per test point. This is what makes abstention-by-width principled.

### Abstention rule

Accept test point $j$ if and only if $\text{width}_j \leq \tau$.

### CAP curve

Sweep $\tau$ from $\min(\text{width})$ to $\max(\text{width})$. At each $\tau$:

- **Effective coverage** $= \mathbb{P}(y_j \in \hat{C}(x_j) \mid \text{width}_j \leq \tau)$
- **Actionability** $= \mathbb{P}(\text{width}_j < \delta_{\text{clinical}} \mid \text{width}_j \leq \tau)$, where $\delta_{\text{clinical}} = 3.0$

---

## 4. Results

### Baseline

| Method | Coverage | Mean Width | Actionable |
|---|---|---|---|
| Non-adaptive (global q) | 0.9300 | 6.8753 | 0% |
| Locally adaptive | 0.8950 | 4.2826 | 67.5% |

Non-adaptive coverage (0.930) appears higher but is an artifact: global $q$ is set conservatively for the hardest cases, over-covering the easy majority.

### CAP curve

| Abstention | Eff. coverage | Δ cov | Mean width | Δ width | Actionable |
|---|---|---|---|---|---|
| 0% | 0.8947 | — | 4.2401 | — | 67.7% |
| 10% | 0.9028 | +0.008 | 3.3539 | −0.93 | 75.0% |
| 20% | 0.9219 | +0.027 | 2.8205 | −1.46 | 84.4% |
| **30%** | **0.9500** | **+0.055** | **2.5763** | **−1.71** | **96.4%** |
| 35% | 0.9614 | +0.066 | 2.5444 | −1.74 | **100%** |

### Subgroup audit at 30% abstention

| Group | n | Coverage | Width | True σ |
|---|---|---|---|---|
| Accepted | 280 | **0.9500** | 2.58 | 0.47 |
| Rejected | 120 | **0.7667** | 8.26 | 3.00 |

Width ratio 3.21×. Sigma ratio 6.35×. Coverage gap **+0.183**.

### Summary tensor `[abst_rate, eff_coverage, mean_width, actionable_frac]`

```
[[0.0025  0.8947  4.2401  0.6767],
 [0.1000  0.9028  3.3539  0.7500],
 [0.2000  0.9219  2.8205  0.8438],
 [0.3000  0.9500  2.5763  0.9643],
 [0.3500  0.9615  2.5444  1.0000]]
```

---

## 5. Analysis

### Marginal ≠ subgroup

The conformal theorem guarantees $\mathbb{P}(y \in \hat{C}(x)) \geq 1-\alpha$ over a randomly drawn test point. It does not guarantee coverage on any specific subgroup. Under heteroscedasticity, easy cases are over-covered and hard cases under-covered — the average hits the target while the subgroup guarantee fails silently.

Selective prediction does not improve calibration. It restricts prediction to cases where the guarantee actually holds.

### Width as a principled abstention signal

Width $\propto \hat{\sigma}_j$. Abstaining on high-width cases is equivalent to abstaining on cases with high estimated noise — a direct measure of irreducible uncertainty. And the subgroup audit confirms this is not just heuristic: abstained cases have empirical coverage 0.767, validating that the width signal correctly identifies cases where the guarantee fails.

### Global q cannot be selectively improved

With constant width, "abstain on cases with width > τ" is vacuous — every case has the same width. A non-adaptive model cannot distinguish easy from hard cases. Locally adaptive intervals are a prerequisite for meaningful selective prediction.

### Clinical interpretation

| Case type | Width | Coverage | Action |
|---|---|---|---|
| Accepted | 2.58 avg | 0.950 | Make prediction, inform clinical decision |
| Rejected | 8.26 avg | 0.767 | Withhold — flag for specialist review |

"I estimate your risk score is in [3.2, 5.8]" (width 2.6) is actionable.  
"I estimate your risk score is in [−2.0, 16.0]" (width 18) is not — and reporting it at 90% coverage when it actually achieves 76.7% is actively misleading.

---

## 6. Connections to the Series

| Artifact | Reliability question |
|---|---|
| Mathematical reliability | Does the interval contain the truth ≥90%? |
| Assumption stress harness | Does coverage hold when assumptions break? |
| Influence & stability | Which training points drive predictions? |
| Calibration decomposition | Are predicted probabilities trustworthy? |
| **This artifact** | On which cases should the model make predictions at all? |

Conformal prediction provides the *coverage* guarantee. Selective prediction enforces the *usefulness* constraint. A reliability-first pipeline requires both.

---

## 7. Limitations

| Limitation | Detail |
|---|---|
| **Variance model quality** | CAP curve quality degrades if sigma estimates are poor (here rho = 0.965) |
| **Fixed clinical threshold** | δ_clinical = 3.0 requires domain input; it is not derivable from the data |
| **No formal abstention guarantee** | Effective coverage after abstention is empirically measured, not theoretically derived |
| **Distribution shift** | If deployment sigma distribution differs from calibration, the abstention rule degrades |
| **Individual vs group abstention** | Abstention is per-point; population-optimal policies may do better |

---

## 8. Reproducibility

```bash
pip install numpy scipy

python selective_prediction.py                     # defaults
python selective_prediction.py --clinical-width 2.0 --alpha 0.05
```

All results are deterministic given `--seed`. No plotting libraries required.

---

## 9. Takeaways

> **A prediction interval of width 18 is not a prediction. It is a refusal to predict, dressed in interval notation. The model should say so explicitly.**

Three shifts from building this:

1. **Marginal coverage is the wrong unit for heterogeneous populations.** 0.895 overall, 0.767 on hard cases. A deployed model in a clinical setting should report subgroup coverage — or better, should not make predictions on the subgroup where coverage fails.

2. **The abstention decision and the prediction are inseparable.** Width is not metadata about the interval — it is the decision criterion for whether to make the prediction at all. Treating abstention as a first-class output, not a post-hoc consideration, changes what a reliability-first pipeline looks like.

3. **Usefulness is a constraint, not a property.** The right question is not "how confident is this prediction?" but "is this prediction narrow enough to support the decision it's meant to inform?" Actionability — the fraction of predictions within the clinical threshold — directly answers this.

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.
- Geifman, Y., & El-Yaniv, R. (2017). Selective prediction in deep neural networks. *NeurIPS*.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*.
- Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for nonparametric regression. *JRSS-B*.
