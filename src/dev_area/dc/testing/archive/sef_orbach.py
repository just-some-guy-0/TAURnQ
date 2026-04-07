# sef_to_orbach_moments.py
#
# Given SEF/KWW parameters (tau_star, beta) at multiple temperatures T,
# compute intrinsic moments of q = log10(tau) at each T, then perform a
# weighted linear regression of:
#
#   q_mean(T) = A + (Ueff / ln(10)) * (1/T)
#
# consistent with Eq (10) Orbach term:
#   tau_orbach^{-1} = 10^{-A} * exp(-Ueff/T)   (Ueff in Kelvin, T in Kelvin)
# => log10(tau_orbach) = A + Ueff/(T ln 10)
#
# Outputs: mean and SD of A and Ueff, plus covariance and correlation.

import numpy as np

EULER_GAMMA = 0.5772156649015328606
LN10 = np.log(10.0)

# -----------------------------
# 1) SEF intrinsic moments for q = log10(tau)
# -----------------------------
def sef_moments_q(tau_star: np.ndarray, beta: np.ndarray):
    """
    For SEF/KWW intrinsic distribution:
      E[ln tau]  = ln(tau*) + (1/beta - 1)*gamma_E
      Var(ln tau)= (pi^2/6)*(1/beta^2 - 1)

    Transform to q = log10(tau) = ln(tau)/ln(10):
      E[q] = E[ln tau]/ln(10)
      Var(q)=Var(ln tau)/(ln(10))^2
    """
    tau_star = np.asarray(tau_star, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if np.any(tau_star <= 0):
        raise ValueError("All tau_star must be > 0.")
    if np.any((beta <= 0) | (beta > 1)):
        raise ValueError("All beta must be in (0, 1].")

    mu_ln = np.log(tau_star) + (1.0 / beta - 1.0) * EULER_GAMMA
    var_ln = (np.pi**2 / 6.0) * (1.0 / beta**2 - 1.0)

    mu_q = mu_ln / LN10
    sigma_q = np.sqrt(var_ln) / LN10
    return mu_q, sigma_q


# -----------------------------
# 2) Weighted least squares for Orbach parameters
# -----------------------------
def fit_orbach_from_sef(T: np.ndarray, tau_star: np.ndarray, beta: np.ndarray,
                        weight_mode: str = "sigma_q",
                        sigma_floor: float = 1e-6):
    """
    Fit: mu_q(T) = A + (Ueff/ln10)*(1/T)

    Inputs:
      T (K): array shape (N,)
      tau_star (s): array shape (N,)
      beta: array shape (N,)
      weight_mode:
        - "sigma_q": weights = 1/sigma_q^2 (default; uses intrinsic width as proxy)
        - "none": unweighted OLS
      sigma_floor: prevents infinite weights if sigma_q ~ 0

    Returns:
      results dict with A, Ueff (K), their SDs, covariance, corr, fit diagnostics
    """
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0):
        raise ValueError("All T must be > 0 K.")
    if len(T) < 3:
        raise ValueError("Need at least 3 temperatures for a stable regression (N>=3).")

    mu_q, sigma_q = sef_moments_q(tau_star, beta)

    x = 1.0 / T
    y = mu_q

    # Design matrix: y = b0 + b1*x, where b0=A, b1=Ueff/ln10
    X = np.column_stack([np.ones_like(x), x])

    if weight_mode.lower() == "none":
        W = np.eye(len(T))
    elif weight_mode.lower() == "sigma_q":
        s = np.maximum(sigma_q, sigma_floor)
        w = 1.0 / (s**2)
        W = np.diag(w)
    else:
        raise ValueError("weight_mode must be 'sigma_q' or 'none'.")

    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y

    # Solve for params
    theta = np.linalg.solve(XtWX, XtWy)  # [A, slope]
    A_hat = float(theta[0])
    slope_hat = float(theta[1])

    # Convert slope to Ueff (K): slope = Ueff/ln10
    Ueff_hat = slope_hat * LN10

    # Residuals and (weighted) RSS
    y_hat = X @ theta
    r = y - y_hat
    RSS = float(r.T @ W @ r)

    n = len(y)
    p = 2
    dof = n - p
    if dof <= 0:
        raise ValueError("Not enough degrees of freedom.")

    # Estimate residual variance (weighted). This is a model-misfit term.
    s2 = RSS / dof

    # Covariance of [A, slope]
    Cov_theta = s2 * np.linalg.inv(XtWX)

    # Convert to covariance for [A, Ueff]
    # Ueff = LN10 * slope
    J = np.array([[1.0, 0.0],
                  [0.0, LN10]])
    Cov_AU = J @ Cov_theta @ J.T

    sd_A = float(np.sqrt(Cov_AU[0, 0]))
    sd_Ueff = float(np.sqrt(Cov_AU[1, 1]))
    corr_AU = float(Cov_AU[0, 1] / (sd_A * sd_Ueff)) if (sd_A > 0 and sd_Ueff > 0) else np.nan

    # A basic (unweighted) R^2 on y vs y_hat for interpretability
    ss_tot = float(np.sum((y - y.mean())**2))
    ss_res = float(np.sum((y - y_hat)**2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "A_mean": A_hat,
        "A_sd": sd_A,
        "Ueff_mean_K": Ueff_hat,
        "Ueff_sd_K": sd_Ueff,
        "Cov_AU": Cov_AU,
        "Corr_AU": corr_AU,
        "mu_q": mu_q,
        "sigma_q": sigma_q,
        "x_1_over_T": x,
        "y_fit": y_hat,
        "residuals": r,
        "RSS_weighted": RSS,
        "dof": dof,
        "s2": s2,
        "R2_unweighted": R2,
    }


# -----------------------------
# Example usage (replace arrays)
# -----------------------------
if __name__ == "__main__":
    # Provide arrays at multiple temperatures
    T = np.array([8, 10, 12, 15, 20, 25], dtype=float)              # K
    tau_star = np.array([1e5, 2e4, 6e3, 1e3, 2e2, 80], dtype=float) # s
    beta = np.array([0.35, 0.40, 0.45, 0.55, 0.65, 0.75], dtype=float)

    res = fit_orbach_from_sef(T, tau_star, beta, weight_mode="sigma_q")

    print("Orbach parameter moments from SEF across temperatures")
    print(f"A (mean)        = {res['A_mean']:.6g}")
    print(f"A (sd)          = {res['A_sd']:.6g}")
    print(f"Ueff (mean, K)  = {res['Ueff_mean_K']:.6g}")
    print(f"Ueff (sd, K)    = {res['Ueff_sd_K']:.6g}")
    print(f"Corr(A, Ueff)   = {res['Corr_AU']:.6g}")
    print(f"R^2 (unweighted)= {res['R2_unweighted']:.6g}")
