#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

LN10 = np.log(10)
tmin = 82
tmax = 120
PATH = "NSiPr3.tsv"

def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha)/(1 - alpha)

def load_tsv(path, tmin, tmax):
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    df = df.iloc[:, :3]
    df.columns = ["T", "tau_mu", "alpha"]
    return df[(df["T"] >= tmin) & (df["T"] <= tmax)].reset_index(drop=True)

def model_moments(T, mu_A, mu_U, sd_A, sd_U, rho):
    mu_ln = LN10*mu_A + mu_U/T
    var_ln = (LN10**2)*(sd_A**2) + (sd_U**2)/(T**2) + 2*(LN10/T)*rho*sd_A*sd_U
    sd_ln = np.sqrt(np.maximum(var_ln, 1e-18))
    return mu_ln, sd_ln

def objective(theta, T, mu_ln_tgt, sd_ln_tgt, w_mu=1.0, w_sd=1.0):
    mu_A, mu_U, sd_A, sd_U, rho = theta
    mu_pred, sd_pred = model_moments(T, mu_A, mu_U, sd_A, sd_U, rho)
    s_mu = np.std(mu_ln_tgt) or 1.0
    s_sd = np.std(sd_ln_tgt) or 1.0
    r1 = (mu_pred - mu_ln_tgt)/s_mu
    r2 = (sd_pred - sd_ln_tgt)/s_sd
    return w_mu*np.mean(r1**2) + w_sd*np.mean(r2**2)  # use mean, not sum


def initial_guess(T, mu_ln_tgt, sd_ln_tgt):
    x = 1.0/T
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, mu_ln_tgt, rcond=None)[0]
    mu_A0 = a / LN10
    mu_U0 = b
    gbar = float(np.mean(sd_ln_tgt))
    sd_A0 = max(gbar / LN10 * 0.5, 1e-5)
    sd_U0 = max(gbar * float(np.mean(T)) * 0.1, 1e-3)
    rho0  = 0.0
    return np.array([mu_A0, mu_U0, sd_A0, sd_U0, rho0])

def fit_orbach_params(tsv_path, tmin, tmax, w_mu=1.0, w_sd=1.0):
    df = load_tsv(tsv_path, tmin, tmax)
    T = df["T"].to_numpy(float)
    mu_ln_tgt = np.log(df["tau_mu"].to_numpy(float))
    sd_ln_tgt = g_from_alpha(df["alpha"].to_numpy(float))

    th0 = initial_guess(T, mu_ln_tgt, sd_ln_tgt)
    bounds = [
        (None, None),    # mu_A
        (None, None),    # mu_U (Kelvin)
        (0.0, None),     # sd_A
        (0.0, None),     # sd_U (Kelvin)
        (-0.999, 0.999)  # rho
    ]
    res = minimize(objective, th0,
                   args=(T, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd),
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=10000, ftol=1e-12))

    mu_A, mu_U, sd_A, sd_U, rho = res.x
    mu_pred, sd_pred = model_moments(T, mu_A, mu_U, sd_A, sd_U, rho)

    out = {
        "success": res.success,
        "message": res.message,
        "theta": dict(mu_A=mu_A, mu_U=mu_U, sd_A=sd_A, sd_U=sd_U, rho=rho),
        "T": T,
        "mu_ln_tgt": mu_ln_tgt,
        "sd_ln_tgt": sd_ln_tgt,
        "mu_pred": mu_pred,
        "sd_pred": sd_pred
    }
    return out

if __name__ == "__main__":
    path = PATH
    res = fit_orbach_params(path, tmin, tmax, w_mu=1.0, w_sd=1.0)
    print("Success:", res["success"], "|", res["message"])
    print(f"Fitted parameters (Orbach-only, {tmin}-{tmax} K):")
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
    # plt.title("Joint fit of ln(τ) moments (Orbach window)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ==========================
    # Robustness check: refit with rho fixed to 0
    # ==========================
    print("\n=== Robustness check: refit with rho fixed to 0 ===")
    T_rc   = res["T"]
    mu_tgt = res["mu_ln_tgt"]
    sd_tgt = res["sd_ln_tgt"]

    # Start at the free-ρ solution but set rho=0 for the new run
    th0_rho0 = np.array([
        res["theta"]["mu_A"],
        res["theta"]["mu_U"],
        res["theta"]["sd_A"],
        res["theta"]["sd_U"],
        0.0
    ], dtype=float)

    bounds_rho0 = [
        (None, None),   # mu_A
        (None, None),   # mu_U
        (0.0, None),    # sd_A
        (0.0, None),    # sd_U
        (0.0, 0.0)      # rho fixed to 0
    ]

    res_rho0 = minimize(
        objective, th0_rho0,
        args=(T_rc, mu_tgt, sd_tgt, 1.0, 1.0),
        method="L-BFGS-B", bounds=bounds_rho0,
        options=dict(maxiter=10000, ftol=1e-12)
    )

    # Compute the free-ρ loss with the same objective
    theta_free = np.array([
        res["theta"]["mu_A"],
        res["theta"]["mu_U"],
        res["theta"]["sd_A"],
        res["theta"]["sd_U"],
        res["theta"]["rho"]
    ], dtype=float)
    loss_free = objective(theta_free, T_rc, mu_tgt, sd_tgt, 1.0, 1.0)     # MSE-style
    loss_rho0 = float(res_rho0.fun)
    dloss = loss_rho0 - loss_free
    print(f"Δloss (ρ=0 − free ρ): {dloss:.6g}  |  %Δ = {100*dloss/max(loss_free,1e-12):.2f}%")

    # === Profile ρ to get ranges for sd_A and sd_U ===
print("\n=== Profile over rho to get sd_A/sd_U ranges (95% χ²(1) threshold) ===")
T_rc, mu_tgt, sd_tgt = res["T"], res["mu_ln_tgt"], res["sd_ln_tgt"]
theta_free = np.array([res["theta"]["mu_A"], res["theta"]["mu_U"],
                       res["theta"]["sd_A"], res["theta"]["sd_U"], res["theta"]["rho"]], float)
loss_free = objective(theta_free, T_rc, mu_tgt, sd_tgt, 1.0, 1.0)
M = len(T_rc)
Nres = 2*M
thr = 3.84 / max(Nres, 1)  # convert χ² threshold to MSE-units

def fit_with_fixed_rho(rho_fixed):
    th0 = theta_free.copy(); th0[-1] = rho_fixed
    bnds = [(None,None),(None,None),(0.0,None),(0.0,None),(rho_fixed, rho_fixed)]
    res_fix = minimize(objective, th0, args=(T_rc, mu_tgt, sd_tgt, 1.0, 1.0),
                       method="L-BFGS-B", bounds=bnds,
                       options=dict(maxiter=5000, ftol=1e-12))
    return res_fix

grid = np.linspace(0.0, 0.9, 19) # positive correlation known so [0.0, 0.9]
keep = []
for r in grid:
    rr = fit_with_fixed_rho(r)
    L = float(rr.fun)
    if (L - loss_free) <= thr and rr.success:
        mu_Ai, mu_Ui, sd_Ai, sd_Ui, _ = rr.x
        keep.append((r, L, sd_Ai, sd_Ui))
if keep:
    sdA_vals = [k[2] for k in keep]
    sdU_vals = [k[3] for k in keep]
    print(f"Accepted rho values (Δloss ≤ {thr:.4g}):",
          f"{min(k[0] for k in keep):+.2f} .. {max(k[0] for k in keep):+.2f}")
    print(f"sd_A range ~ [{min(sdA_vals):.3g}, {max(sdA_vals):.3g}]")
    print(f"sd_U range ~ [{min(sdU_vals):.3g}, {max(sdU_vals):.3g}]")
else:
    print("No rho in the grid satisfies the 95% threshold.")
