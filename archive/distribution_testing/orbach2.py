#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

LN10 = np.log(10)
tmin = 44
tmax = 58

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

    return w_mu*np.sum(r1**2) + w_sd*np.sum(r2**2)

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
    path = "tBuOCl.tsv"
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
