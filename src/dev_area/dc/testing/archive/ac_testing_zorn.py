#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LN10 = np.log(10)

# ==========================================================
# 1. Load TSV
# ==========================================================
def load_tsv(path, tmin, tmax):
    """
    Expected TSV columns:  T   tau_mu   alpha
    """
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    df = df.iloc[:, :3]
    df.columns = ["T", "tau_mu", "alpha"]
    return df[(df["T"] >= tmin) & (df["T"] <= tmax)].reset_index(drop=True)


# ==========================================================
# 2. Zorn (2002) exact Fuoss–Kirkwood log-moments
# ==========================================================
def zorn_fk_moments(tau_mu, alpha):
    """
    Zorn Table I (Fuoss–Kirkwood distribution):

        mu_ln_tau(T) = ln(tau_mu(T))
        sd_ln_tau(T) = pi / (sqrt(3) * alpha(T))

    Convert to Q where ln(tau) = Q * ln(10)
    => mu_Q = mu_ln_tau / ln(10)
       sd_Q = sd_ln_tau / ln(10)
    """
    mu_ln = np.log(tau_mu)
    sd_ln = np.pi / (np.sqrt(3) * alpha)

    mu_Q = mu_ln / LN10
    sd_Q = sd_ln / LN10

    return mu_ln, sd_ln, mu_Q, sd_Q


# ==========================================================
# 3. Replacement for your optimisation-based function
# ==========================================================
def fit_qtm_params_zorn(tsv_path, tmin, tmax):
    """
    Drop-in replacement for fit_qtm_params(),
    but uses exact analytical Fuoss–Kirkwood → Q distribution formulas.
    """
    df = load_tsv(tsv_path, tmin, tmax)

    T = df["T"].to_numpy(float)
    tau_mu = df["tau_mu"].to_numpy(float)
    alpha = df["alpha"].to_numpy(float)

    mu_ln, sd_ln, mu_Q, sd_Q = zorn_fk_moments(tau_mu, alpha)

    return {
        "T": T,
        "mu_ln_tau": mu_ln,
        "sd_ln_tau": sd_ln,
        "mu_Q": mu_Q,
        "sd_Q": sd_Q,
        "success": True,
        "message": "Analytical Zorn Fuoss–Kirkwood solution"
    }


# ==========================================================
# 4. Optional: demonstration / printout
# ==========================================================
if __name__ == "__main__":
    PATH = "tBuOCl.tsv"
    tmin = 9.7
    tmax = 9.7

    res = fit_qtm_params_zorn(PATH, tmin, tmax)

    print(res["message"])
    print(f"T range: {tmin}-{tmax} K (N={len(res['T'])})")
    print()

    for T, muQ, sdQ in zip(res["T"], res["mu_Q"], res["sd_Q"]):
        print(f"T={T:5.2f} K | mu_Q={muQ: .6f} | sd_Q={sdQ: .6f}")
