#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

LN10 = np.log(10)
tmin = 9.7
tmax = 14
PATH = "Ni.tsv"

def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha)/(1 - alpha)

def load_tsv(path, tmin, tmax):
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    df = df.iloc[:, :3]
    df.columns = ["T", "tau_mu", "alpha"]
    return df[(df["T"] >= tmin) & (df["T"] <= tmax)].reset_index(drop=True)

def model_moments(T, mu_q, sd_q):
    nT = len(T)
    mu_ln = np.full(nT, LN10 * mu_q)
    sd_ln = np.full(nT, LN10 * sd_q)
    return mu_ln, sd_ln

def objective(theta, T, mu_ln_tgt, sd_ln_tgt, w_mu=1.0, w_sd=1.0):
    mu_q, sd_q = theta
    mu_pred, sd_pred = model_moments(T, mu_q, sd_q)

    s_mu = np.std(mu_ln_tgt) or 1.0
    s_sd = np.std(sd_ln_tgt) or 1.0
    r1 = (mu_pred - mu_ln_tgt) / s_mu
    r2 = (sd_pred - sd_ln_tgt) / s_sd
    return w_mu*np.sum(r1**2) + w_sd*np.sum(r2**2)

def initial_guess(T, mu_ln_tgt, sd_ln_tgt):
    mu_q0 = float(np.mean(mu_ln_tgt)) / LN10
    sd_q0 = max(float(np.mean(sd_ln_tgt)) / LN10, 1e-6)
    return np.array([mu_q0, sd_q0])

def fit_qtm_params(tsv_path, tmin, tmax, w_mu=1.0, w_sd=1.0):
    df = load_tsv(tsv_path, tmin, tmax)
    T = df["T"].to_numpy(float)
    mu_ln_tgt = np.log(df["tau_mu"].to_numpy(float))
    sd_ln_tgt = g_from_alpha(df["alpha"].to_numpy(float))

    th0 = initial_guess(T, mu_ln_tgt, sd_ln_tgt)
    bounds = [
        (None, None),  # mu_q
        (0.0,  None)   # sd_q >= 0
    ]

    res = minimize(objective, th0,
                   args=(T, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd),
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=10000, ftol=1e-12))

    mu_q, sd_q = res.x
    mu_pred, sd_pred = model_moments(T, mu_q, sd_q)

    out = {
        "success": res.success,
        "message": res.message,
        "theta": dict(mu_q=mu_q, sd_q=sd_q),
        "T": T,
        "mu_ln_tgt": mu_ln_tgt,
        "sd_ln_tgt": sd_ln_tgt,
        "mu_pred": mu_pred,
        "sd_pred": sd_pred
    }
    return out

if __name__ == "__main__":
    path = PATH
    res = fit_qtm_params(path, tmin, tmax, w_mu=1.0, w_sd=1.0)
    print("Success:", res["success"], "|", res["message"])
    print(f"Fitted parameters (QTM-only, {tmin}-{tmax} K):")
    for k, v in res["theta"].items():
        print(f"  {k:6s} = {v: .6f}")

    # # Diagnostics (kept commented to match your style)
    # T = res["T"]
    # plt.figure(figsize=(8,5))
    # plt.scatter(T, res["mu_ln_tgt"], label="target μ_ln", marker="o")
    # plt.plot(T, res["mu_pred"], label="model μ_ln (QTM)", linewidth=2)
    # plt.scatter(T, res["sd_ln_tgt"], label="target σ_ln", marker="s")
    # plt.plot(T, res["sd_pred"], label="model σ_ln (QTM)", linestyle="--", linewidth=2)
    # plt.xlabel("Temperature (K)")
    # plt.ylabel("μ_ln or σ_ln")
    # plt.title(f"QTM-only joint fit of ln(τ) moments ({tmin:.1f}–{tmax:.1f} K)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
