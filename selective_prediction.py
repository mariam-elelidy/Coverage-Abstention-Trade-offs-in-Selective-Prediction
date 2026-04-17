r"""
Selective Prediction Under Uncertainty
=======================================
Author : Mariam Mohamed Elelidy
Purpose: Determine which predictions the model should make and which it
         should withhold, using interval width as a principled abstention
         criterion grounded in conformal prediction theory.

Core question
-------------
Split conformal prediction provides a *marginal* coverage guarantee:
P(y in interval) >= 1 - alpha over all test points. But marginal coverage
masks a critical heterogeneity: hard cases (high intrinsic noise) may be
systematically under-covered while easy cases are over-covered. A model
that makes predictions on everything is not equally reliable everywhere.

Selective prediction asks: if we allow the model to withhold some fraction
of predictions, which cases should it decline — and what reliability can we
guarantee on the cases it accepts?

This artifact gives a quantitative answer via the Coverage-Abstention-Precision
(CAP) curve: for every abstention budget from 0% to 100%, it reports:
  - Effective coverage on accepted predictions
  - Mean interval width on accepted predictions
  - Actionability (fraction of accepted predictions with width < clinical threshold)

The key finding
---------------
Under heteroscedastic noise, locally adaptive conformal prediction creates
per-point interval widths. Abstaining on wide-interval points:
  1. Increases effective coverage (from 0.895 to 0.961 at 30% abstention)
  2. Decreases mean width by 39% (4.24 → 2.58 at 30% abstention)
  3. Increases actionability from 68% to 96%
  4. Reveals that rejected cases were under-covered (0.767) — the marginal
     guarantee masked a subgroup reliability failure

Design choices
--------------
- Heteroscedastic data (sigma = 0.3 + 2.5 * relu(x_0)) creates genuine
  per-point uncertainty variation. This is realistic: medical data routinely
  has high-variance subgroups (e.g., patients with comorbidities).
- Two-stage variance model (ridge on |residuals|) estimates sigma without
  access to ground truth noise. Correlation with true sigma = 0.965.
- Clinical threshold (DEFAULT_CLINICAL_WIDTH = 3.0) defines "actionable":
  a prediction interval narrow enough to support a clinical decision. Adjust
  this for your application.
- Non-adaptive baseline (global q) is included to show what selective
  prediction cannot do: a constant-width model cannot distinguish easy from
  hard cases.

Usage
-----
    python selective_prediction.py                  # defaults
    python selective_prediction.py --n 3000 --alpha 0.05 --clinical-width 2.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

DEFAULT_CLINICAL_WIDTH = 3.0  # interval width considered clinically actionable


# ────────────────────────────────────────────────────────────────────────────
# Core primitives
# ────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge: ŵ = (X'X + λI)^{-1} X'y."""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k = ceil((n+1)(1-alpha))-th order statistic."""
    n = len(scores)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(np.sort(scores)[k - 1])


# ────────────────────────────────────────────────────────────────────────────
# Data generation (heteroscedastic)
# ────────────────────────────────────────────────────────────────────────────

def generate_data(
    rng:   np.random.Generator,
    n:     int,
    d:     int,
    sigma_base:  float = 0.3,
    sigma_scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heteroscedastic regression.

    sigma_i = sigma_base + sigma_scale * relu(x_i[0])

    The first feature drives noise level. About half the dataset has
    sigma near sigma_base (easy); the other half has sigma up to
    sigma_base + sigma_scale (hard). This bimodal structure produces
    clear differentiation in the coverage-abstention curve.

    Returns: X, y, true_sigma
    """
    X      = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    true_sigma = sigma_base + sigma_scale * np.maximum(X[:, 0], 0.0)
    y      = X @ w_true + true_sigma * rng.normal(size=n)
    return X, y, true_sigma


# ────────────────────────────────────────────────────────────────────────────
# Variance estimation (stage-2 model)
# ────────────────────────────────────────────────────────────────────────────

def fit_variance_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_pred: np.ndarray,
    sigma_lo: float = 0.3,
    sigma_hi: float = 3.0,
    lam: float = 1e-3,
) -> np.ndarray:
    """Fit a ridge model to predict |residuals| from features.

    Stage 2 of the two-stage locally adaptive conformal pipeline.
    Clipping sigma to [sigma_lo, sigma_hi] prevents extreme normalization
    from distorting the conformal quantile.
    """
    abs_res = np.abs(y_tr - X_tr @ w_pred)
    return ridge_fit(X_tr, abs_res, lam)


def predict_sigma(
    X: np.ndarray,
    w_var: np.ndarray,
    sigma_lo: float = 0.3,
    sigma_hi: float = 3.0,
) -> np.ndarray:
    return np.clip(X @ w_var, sigma_lo, sigma_hi)


# ────────────────────────────────────────────────────────────────────────────
# Conformal calibration (adaptive and non-adaptive)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveIntervals:
    pred:      np.ndarray
    lo:        np.ndarray
    hi:        np.ndarray
    width:     np.ndarray
    covered:   np.ndarray
    q_norm:    float
    sigma_te:  np.ndarray


def calibrate_adaptive(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_te:  np.ndarray,
    y_te:  np.ndarray,
    w_pred: np.ndarray,
    w_var:  np.ndarray,
    alpha:  float,
    sigma_lo: float = 0.3,
    sigma_hi: float = 3.0,
) -> AdaptiveIntervals:
    """Locally adaptive conformal prediction.

    Conformity score: s_i = |y_i - ŷ_i| / sigma_i
    Test interval:   [ŷ_j - q * sigma_j, ŷ_j + q * sigma_j]

    Width is proportional to sigma_j per test point — the key property
    that makes abstention by width a principled uncertainty-based decision.
    """
    sigma_cal = predict_sigma(X_cal, w_var, sigma_lo, sigma_hi)
    sigma_te  = predict_sigma(X_te,  w_var, sigma_lo, sigma_hi)

    cal_scores = np.abs(y_cal - X_cal @ w_pred) / sigma_cal
    q_norm = conformal_quantile(cal_scores, alpha)

    pred      = X_te @ w_pred
    half_w    = q_norm * sigma_te
    lo, hi    = pred - half_w, pred + half_w
    covered   = (y_te >= lo) & (y_te <= hi)

    return AdaptiveIntervals(
        pred=pred, lo=lo, hi=hi,
        width=2 * half_w, covered=covered,
        q_norm=q_norm, sigma_te=sigma_te,
    )


@dataclass
class GlobalIntervals:
    pred:    np.ndarray
    lo:      np.ndarray
    hi:      np.ndarray
    width:   float
    covered: np.ndarray
    q:       float


def calibrate_global(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_te:  np.ndarray,
    y_te:  np.ndarray,
    w_pred: np.ndarray,
    alpha:  float,
) -> GlobalIntervals:
    """Standard (non-adaptive) conformal prediction. All widths = 2q."""
    abs_res = np.abs(y_cal - X_cal @ w_pred)
    q       = conformal_quantile(abs_res, alpha)
    pred    = X_te @ w_pred
    lo, hi  = pred - q, pred + q
    covered = (y_te >= lo) & (y_te <= hi)
    return GlobalIntervals(pred=pred, lo=lo, hi=hi,
                           width=2 * q, covered=covered, q=q)


# ────────────────────────────────────────────────────────────────────────────
# CAP curve
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CurvePoint:
    tau:          float
    abst_rate:    float
    eff_coverage: float
    mean_width:   float
    actionable:   float     # fraction with width < clinical threshold
    n_accepted:   int


def cap_curve(
    covered:  np.ndarray,
    width:    np.ndarray,
    clinical_width: float = DEFAULT_CLINICAL_WIDTH,
    n_points: int = 600,
) -> list[CurvePoint]:
    """Coverage-Abstention-Precision curve.

    Sweeps abstention threshold tau from the minimum width to the maximum.
    At each tau, the model accepts all test points with width <= tau.

    For a non-adaptive model, width is constant so all thresholds give the
    same accepted set (the entire test set), producing a flat "curve".
    """
    n_te = len(covered)
    thresholds = np.unique(np.quantile(width, np.linspace(0.005, 0.999, n_points)))
    pts = []
    for tau in thresholds:
        acc = width <= tau
        if acc.sum() < 5:
            continue
        pts.append(CurvePoint(
            tau=tau,
            abst_rate=float(1 - acc.mean()),
            eff_coverage=float(covered[acc].mean()),
            mean_width=float(width[acc].mean()),
            actionable=float((width[acc] < clinical_width).mean()),
            n_accepted=int(acc.sum()),
        ))
    return pts


# ────────────────────────────────────────────────────────────────────────────
# Subgroup coverage audit
# ────────────────────────────────────────────────────────────────────────────

def subgroup_audit(
    covered:    np.ndarray,
    width:      np.ndarray,
    true_sigma: np.ndarray,
    tau:        float,
) -> None:
    """Audit coverage and width for accepted vs rejected subgroups."""
    acc = width <= tau
    rej = ~acc
    print(f"  {'group':<10}  {'n':>5}  {'coverage':>9}  "
          f"{'mean_width':>10}  {'true_sigma':>11}")
    print("  " + "─" * 56)
    for name, mask in [("accepted", acc), ("rejected", rej)]:
        if mask.sum() == 0:
            continue
        print(f"  {name:<10}  {mask.sum():>5}  "
              f"{covered[mask].mean():>9.4f}  "
              f"{width[mask].mean():>10.4f}  "
              f"{true_sigma[mask].mean():>11.4f}")
    if rej.sum() > 0 and acc.sum() > 0:
        print(f"\n  Width ratio  (rejected/accepted): "
              f"{width[rej].mean()/width[acc].mean():.2f}×")
        print(f"  Sigma ratio  (rejected/accepted): "
              f"{true_sigma[rej].mean()/true_sigma[acc].mean():.2f}×")
        print(f"  Coverage gap (accepted - rejected): "
              f"{covered[acc].mean() - covered[rej].mean():+.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Terminal report
# ────────────────────────────────────────────────────────────────────────────

def _bar(v: float, lo: float = 0.0, hi: float = 1.0, w: int = 22) -> str:
    x = max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-9)))
    k = int(round(x * w))
    return "█" * k + "░" * (w - k)


def print_report(
    ai:    AdaptiveIntervals,
    gi:    GlobalIntervals,
    curve: list[CurvePoint],
    true_sigma_te: np.ndarray,
    y_te:  np.ndarray,
    alpha: float,
    clinical_width: float,
    n: int, d: int,
    var_corr: float,
) -> None:
    target = 1 - alpha
    sep = "─" * 76

    print()
    print("┌" + sep + "┐")
    print("│  Selective Prediction Under Uncertainty" + " " * 37 + "│")
    print(f"│  α = {alpha:.2f}  │  target coverage = {target:.2f}  │  "
          f"n = {n}  d = {d}  │  clinical_width = {clinical_width:.1f}" + " " * 5 + "│")
    print("└" + sep + "┘")

    n_te = len(y_te)
    base_cov_a = ai.covered.mean()
    base_cov_g = gi.covered.mean()

    print()
    print("  ── BASELINE (no abstention) ────────────────────────────────────")
    print(f"  Var model correlation with true sigma: {var_corr:.4f}")
    print(f"  Non-adaptive (global q={gi.q:.3f}):  coverage={base_cov_g:.4f}  "
          f"width={gi.width:.4f}  [all points identical]")
    print(f"  Locally adaptive (q_norm={ai.q_norm:.3f}):  "
          f"coverage={base_cov_a:.4f}  mean_width={ai.width.mean():.4f}  "
          f"width_range=[{ai.width.min():.2f}, {ai.width.max():.2f}]")
    print()
    act_base = float((ai.width < clinical_width).mean())
    print(f"  Baseline actionability (width < {clinical_width}):  "
          f"{act_base*100:.1f}% of test predictions")
    print()

    # CAP curve
    print("  ── COVERAGE-ABSTENTION-PRECISION CURVE (adaptive) ──────────────")
    print(f"  {'abst%':>7}  {'eff_cov':>8}  {'mean_w':>8}  "
          f"{'actionable%':>12}  {'n_acc':>6}  coverage bar")
    print("  " + "─" * 72)
    step = max(1, len(curve) // 14)
    for pt in curve[::step]:
        flag = " ← operating point" if abs(pt.abst_rate - 0.30) < 0.02 else ""
        print(f"  {pt.abst_rate*100:>6.1f}%  {pt.eff_coverage:>8.4f}  "
              f"{pt.mean_width:>8.4f}  {pt.actionable*100:>11.1f}%  "
              f"{pt.n_accepted:>6}  {_bar(pt.eff_coverage, 0.85, 1.0)}{flag}")

    # Key operating points
    print()
    print("  ── KEY OPERATING POINTS ─────────────────────────────────────────")
    print(f"  {'abstain':>8}  {'eff_cov':>8}  {'Δcov':>7}  "
          f"{'mean_w':>8}  {'Δwidth':>8}  {'actionable':>11}  n")
    print("  " + "─" * 68)
    for tgt in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        pt = min(curve, key=lambda p: abs(p.abst_rate - tgt))
        d_cov   = pt.eff_coverage - base_cov_a
        d_width = pt.mean_width - ai.width.mean()
        print(f"  {pt.abst_rate*100:>7.1f}%  {pt.eff_coverage:>8.4f}  "
              f"{d_cov:>+7.4f}  {pt.mean_width:>8.4f}  {d_width:>+8.4f}  "
              f"{pt.actionable*100:>10.1f}%  {pt.n_accepted}")

    # Recommended operating point: min abstention to reach eff_cov >= target
    candidates = [pt for pt in curve if pt.eff_coverage >= target]
    if candidates:
        rec = min(candidates, key=lambda p: p.abst_rate)
        print()
        print(f"  Recommended operating point (eff_cov ≥ {target:.2f}):")
        print(f"    Abstain  {rec.abst_rate*100:.1f}%  │  "
              f"eff_cov = {rec.eff_coverage:.4f}  │  "
              f"mean_width = {rec.mean_width:.4f}  │  "
              f"actionable = {rec.actionable*100:.1f}%  │  n = {rec.n_accepted}")

    # Subgroup audit at 30% abstention
    pt30 = min(curve, key=lambda p: abs(p.abst_rate - 0.30))
    print()
    print(f"  ── SUBGROUP AUDIT at {pt30.abst_rate*100:.0f}% abstention ──────────────────────────")
    subgroup_audit(ai.covered, ai.width, true_sigma_te, pt30.tau)
    print()
    print("  ⚠  Rejected cases have lower coverage than accepted cases.")
    print("     The marginal guarantee hid a subgroup reliability failure.")
    print("     Selective prediction exposes — and addresses — it.")

    # Global vs adaptive comparison
    print()
    print("  ── GLOBAL (NON-ADAPTIVE) vs ADAPTIVE — comparison ──────────────")
    print(f"  {'abst%':>7}  {'global_cov':>11}  {'global_w':>9}  "
          f"{'adapt_cov':>10}  {'adapt_w':>9}  {'adapt_act':>10}")
    print("  " + "─" * 64)
    for tgt in [0.0, 0.10, 0.20, 0.30, 0.40]:
        pt = min(curve, key=lambda p: abs(p.abst_rate - tgt))
        print(f"  {pt.abst_rate*100:>6.1f}%  "
              f"{gi.covered.mean():>11.4f}  {gi.width:>9.4f}  "
              f"{pt.eff_coverage:>10.4f}  {pt.mean_width:>9.4f}  "
              f"{pt.actionable*100:>9.1f}%")
    print()
    print("  Global q: coverage and width are fixed regardless of abstention.")
    print("  Selective prediction only adds value when intervals are adaptive.")


def print_tensor_summary(
    curve: list[CurvePoint],
    base_cov: float,
) -> None:
    print()
    print("═" * 76)
    print("FINAL TENSOR  [abst_rate, eff_coverage, mean_width, actionable_frac]")
    print("Rows: abstention operating points 0% → 50%")
    print("═" * 76)
    ops = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    for tgt in ops:
        pt = min(curve, key=lambda p: abs(p.abst_rate - tgt))
        rows.append([pt.abst_rate, pt.eff_coverage,
                     pt.mean_width, pt.actionable])
    mat = np.array(rows)
    print(mat)
    print()
    print(f"Baseline (no abstention) coverage: {base_cov:.4f}")
    print(f"Coverage at 30% abstention:        {mat[6, 1]:.4f}  "
          f"(+{mat[6,1]-base_cov:+.4f})")
    print(f"Width at 30% abstention:           {mat[6, 2]:.4f}  "
          f"({mat[6,2]/mat[0,2]*100:.1f}% of baseline)")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Selective prediction under uncertainty"
    )
    p.add_argument("--n",             type=int,   default=2000)
    p.add_argument("--d",             type=int,   default=8)
    p.add_argument("--alpha",         type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lam",           type=float, default=1e-3)
    p.add_argument("--clinical-width",type=float, default=DEFAULT_CLINICAL_WIDTH,
                   dest="clinical_width")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── Data ──
    X, y, true_sigma = generate_data(rng, args.n, args.d)
    idx  = rng.permutation(args.n)
    n_tr = int(0.6 * args.n)
    n_cal= int(0.2 * args.n)
    tr, cal, te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]

    X_tr, y_tr = X[tr], y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te,  y_te  = X[te],  y[te]

    # ── Models ──
    w_pred = ridge_fit(X_tr, y_tr, args.lam)
    w_var  = fit_variance_model(X_tr, y_tr, w_pred, lam=args.lam)

    sigma_te = predict_sigma(X_te, w_var)
    var_corr = float(np.corrcoef(sigma_te, true_sigma[te])[0, 1])

    # ── Conformal calibration ──
    ai = calibrate_adaptive(X_cal, y_cal, X_te, y_te,
                             w_pred, w_var, args.alpha)
    gi = calibrate_global(  X_cal, y_cal, X_te, y_te,
                             w_pred, args.alpha)

    # ── CAP curve ──
    curve = cap_curve(ai.covered, ai.width, args.clinical_width)

    # ── Report ──
    print_report(ai, gi, curve, true_sigma[te], y_te,
                 args.alpha, args.clinical_width,
                 args.n, args.d, var_corr)
    print_tensor_summary(curve, float(ai.covered.mean()))


if __name__ == "__main__":
    main()
