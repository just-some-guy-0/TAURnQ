#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

LN10 = np.log(10)
tmin = 52
tmax = 97
PATH = "NSi2iPr3.tsv"

def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha)/(1 - alpha) 

def load_tsv(path, tmin, tmax):
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    df = df.iloc[:, :3]
    df.columns = ["T", "tau_mu", "alpha"]
    return df[(df["T"] >= tmin) & (df["T"] <= tmax)].reset_index(drop=True)


def model_moments(T, mu_R, mu_N, sd_R, sd_N, rho):
    t10 = np.log10(T)
    mu_L = mu_R + t10 * mu_N
    var_L = sd_R**2 + (t10**2)*(sd_N**2) + 2*t10*rho*sd_R*sd_N
    var_L = np.maximum(var_L, 1e-18)
    mu_ln = -LN10 * mu_L
    sd_ln = LN10 * np.sqrt(var_L)
    return mu_ln, sd_ln

def objective(theta, T, mu_ln_tgt, sd_ln_tgt, w_mu=1.0, w_sd=1.0):
    mu_R, mu_N, sd_R, sd_N, rho = theta
    mu_pred, sd_pred = model_moments(T, mu_R, mu_N, sd_R, sd_N, rho)

    s_mu = np.std(mu_ln_tgt) or 1.0
    s_sd = np.std(sd_ln_tgt) or 1.0
    r1 = (mu_pred - mu_ln_tgt) / s_mu
    r2 = (sd_pred - sd_ln_tgt) / s_sd
    return w_mu*np.sum(r1**2) + w_sd*np.sum(r2**2)

def initial_guess(T, mu_ln_tgt, sd_ln_tgt):
    t10 = np.log10(T)
    mu_L_tgt = -np.log10(np.exp(mu_ln_tgt))
    A = np.vstack([np.ones_like(t10), t10]).T
    mu_R0, mu_N0 = np.linalg.lstsq(A, mu_L_tgt, rcond=None)[0]

    gbar = float(np.mean(sd_ln_tgt))       
    sd_R0 = max(gbar / LN10 * 0.5, 1e-5)
    sd_N0 = max((gbar / LN10) / (np.mean(np.abs(t10)) + 1e-6) * 0.5, 1e-5)
    rho0  = -0.1
    return np.array([mu_R0, mu_N0, sd_R0, sd_N0, rho0])

# ---------- Driver ----------
def fit_raman_params(tsv_path, tmin, tmax, w_mu=1.0, w_sd=1.0):
    df = load_tsv(tsv_path, tmin, tmax)
    T = df["T"].to_numpy(float)
    mu_ln_tgt = np.log(df["tau_mu"].to_numpy(float))
    sd_ln_tgt = g_from_alpha(df["alpha"].to_numpy(float))

    th0 = initial_guess(T, mu_ln_tgt, sd_ln_tgt)
    bounds = [
        (None, None),    # mu_R
        (None, None),    # mu_N
        (0.0, None),     # sd_R
        (0.0, None),     # sd_N
        (-0.999, 0.999)    # rho
    ]
    res = minimize(objective, th0,
                   args=(T, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd),
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=10000, ftol=1e-12))

    mu_R, mu_N, sd_R, sd_N, rho = res.x
    mu_pred, sd_pred = model_moments(T, mu_R, mu_N, sd_R, sd_N, rho)

    out = {
        "success": res.success,
        "message": res.message,
        "theta": dict(mu_R=mu_R, mu_N=mu_N, sd_R=sd_R, sd_N=sd_N, rho=rho),
        "T": T, 
        "mu_ln_tgt": mu_ln_tgt, 
        "sd_ln_tgt": sd_ln_tgt,
        "mu_pred": mu_pred, 
        "sd_pred": sd_pred
    }
    return out

if __name__ == "__main__":
    path = PATH
    res = fit_raman_params(path, tmin, tmax, w_mu=1.0, w_sd=1.0)
    print("Success:", res["success"], "|", res["message"])
    print(f"Fitted parameters (Raman-only, {tmin}-{tmax} K):")
    for k, v in res["theta"].items():
        print(f"  {k:6s} = {v: .6f}")

    # # Diagnostics
    # T = res["T"]
    # plt.figure(figsize=(8,5))
    # plt.scatter(T, res["mu_ln_tgt"], label="target μ_ln", marker="o")
    # plt.plot(T, res["mu_pred"], label="model μ_ln", linewidth=2)
    # plt.scatter(T, res["sd_ln_tgt"], label="target σ_ln", marker="s")
    # plt.plot(T, res["sd_pred"], label="model σ_ln", linestyle="--", linewidth=2)
    # plt.xlabel("Temperature (K)")
    # plt.ylabel("μ_ln or σ_ln")
    # plt.title("Joint fit of ln(τ) moments (Raman window)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ==========================
    # Robustness check: refit with rho fixed to 0 (SSE scale)
    # ==========================
    print("\n=== Robustness check (Raman): refit with rho fixed to 0 ===")
    T_rc   = res["T"]
    mu_tgt = res["mu_ln_tgt"]
    sd_tgt = res["sd_ln_tgt"]

    theta_free = np.array([
        res["theta"]["mu_R"],
        res["theta"]["mu_N"],
        res["theta"]["sd_R"],
        res["theta"]["sd_N"],
        res["theta"]["rho"]
    ], dtype=float)

    loss_free = float(objective(theta_free, T_rc, mu_tgt, sd_tgt, 1.0, 1.0))  # SSE-style, matches objective

    th0_rho0 = theta_free.copy(); th0_rho0[-1] = 0.0
    bounds_rho0 = [
        (None, None),   # mu_R
        (None, None),   # mu_N
        (0.0, None),    # sd_R
        (0.0, None),    # sd_N
        (0.0, 0.0)      # rho fixed at 0
    ]
    res_rho0 = minimize(
        objective, th0_rho0,
        args=(T_rc, mu_tgt, sd_tgt, 1.0, 1.0),
        method="L-BFGS-B", bounds=bounds_rho0,
        options=dict(maxiter=10000, ftol=1e-12)
    )
    loss_rho0 = float(res_rho0.fun)
    dloss = loss_rho0 - loss_free
    print(f"Δloss (ρ=0 − free ρ): {dloss:.6g}  |  %Δ = {100*dloss/max(loss_free,1e-12):.2f}%")

    # ==========================
    # Profile over rho (expected negative) to get sd_R / sd_N ranges
    # SSE objective ⇒ χ²(1,95%) threshold = 3.84 (no scaling)
    # ==========================
    print("\n=== Profile over rho (negative) to get sd_R/sd_N ranges (95% χ²) ===")
    thr = 3.84  # SSE scale

    def fit_with_fixed_rho(rho_fixed):
        th0 = theta_free.copy(); th0[-1] = rho_fixed
        bnds = [(None,None),(None,None),(0.0,None),(0.0,None),(rho_fixed, rho_fixed)]
        res_fix = minimize(objective, th0, args=(T_rc, mu_tgt, sd_tgt, 1.0, 1.0),
                           method="L-BFGS-B", bounds=bnds,
                           options=dict(maxiter=5000, ftol=1e-12))
        return res_fix

    # If you want to allow slight positives too, change to np.linspace(-0.9, 0.2, 23)
    grid = np.linspace(-0.999, 0, 50)  # Raman: negative correlation expected
    keep = []
    for r in grid:
        rr = fit_with_fixed_rho(r)
        L = float(rr.fun)
        if (L - loss_free) <= thr and rr.success:
            mu_Ri, mu_Ni, sd_Ri, sd_Ni, _ = rr.x
            keep.append((r, L, sd_Ri, sd_Ni))

    if keep:
        sdR_vals = [k[2] for k in keep]
        sdN_vals = [k[3] for k in keep]
        r_min = min(k[0] for k in keep); r_max = max(k[0] for k in keep)
        print(f"Accepted rho values (ΔSSE ≤ {thr:.2f}): {r_min:+.2f} .. {r_max:+.2f}")
        print(f"sd_R range ~ [{min(sdR_vals):.3g}, {max(sdR_vals):.3g}]")
        print(f"sd_N range ~ [{min(sdN_vals):.3g}, {max(sdN_vals):.3g}]")
    else:
        print("No rho in the grid satisfies the 95% threshold.")
