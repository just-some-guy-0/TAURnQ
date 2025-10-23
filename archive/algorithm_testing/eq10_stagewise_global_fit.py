#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stagewise and Global Fit of Eq.10 (Reta-Chilton style)
-----------------------------------------------------
Reads a TSV/CSV with columns: T(K), tau_mu, alpha.
Performs:
  1) Stagewise fitting to get initial guesses:
     - Orbach: linear fit of -log10(tau_mu) vs 1/T on high-T band
     - Raman: nonlinear fit with Orbach fixed on mid-T band (QTM negligible)
     - QTM:   nonlinear fit with Orbach+Raman fixed on low-T band
  2) Global nonlinear least squares fit of Eq.10 over all data
  3) Outputs parameter estimates with 1sigma (linearised covariance),
     plots data with +-1sigma error bars (from alpha) and fitted curve,
     and writes a CSV of fitted parameters.

Usage:
  python eq10_stagewise_global_fit.py --infile tBuOCl.tsv --outfile params.csv --plot out.png

Author: ChatGPT (generated)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

LN10 = np.log(10.0)

def g_from_alpha(alpha):
    """Natural-log standard deviation for the log-normal analogue (Reta-Chilton)."""
    alpha = np.clip(alpha, 1e-12, 0.999999)
    return 1.82*np.sqrt(alpha)/(1 - alpha)

def rate_model(T, A, Ueff, R, n, Q):
    """Eq. 10 total relaxation rate (in linear space)."""
    term_orb = (10.0**(-A)) * np.exp(-Ueff / T)  # Orbach
    term_ram = (10.0**(R))  * (T**n)             # Raman
    term_qtm = (10.0**(-Q))                      # QTM
    return term_orb + term_ram + term_qtm

def log10_rate_model(T, A, Ueff, R, n, Q):
    return np.log10(np.clip(rate_model(T, A, Ueff, R, n, Q), 1e-300, None))

def linearised_param_sd(ls_result):
    """Return (sd, cov, rss, s2) from linearised covariance (J^T J)^-1, with small ridge."""
    J = ls_result.jac
    N, P = J.shape
    rss = float(np.sum(ls_result.fun**2))
    dof = max(1, N - P)
    s2 = rss / dof
    JTJ = J.T @ J
    ridge = 1e-10 * np.eye(P)
    cov = np.linalg.pinv(JTJ + ridge) * s2
    sd = np.sqrt(np.maximum(0.0, np.diag(cov)))
    return sd, cov, rss, s2

def r2_score(y_true, y_pred):
    resid = y_true - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def stagewise_and_global_fit(T, tau_mu, alpha, T_orb_min=40.0, T_ram_lo=25.0, T_qtm_max=25.0,
                             weighted=False):
    """Run the stagewise initialisation and global fit."""
    # Observables
    rate_obs = 1.0 / tau_mu
    log10_rate_obs = np.log10(rate_obs)
    g_nat = g_from_alpha(alpha)
    sigma_log10 = g_nat / LN10  # std dev in log10(rate) from alpha

    # ----- Stage 1: Orbach high-T linear regression -----
    mask_orb = T >= T_orb_min
    x_orb = 1.0 / T[mask_orb]
    y_orb = -np.log10(tau_mu[mask_orb])  # log10(rate) in Orbach-only model
    A_mat = np.vstack([np.ones_like(x_orb), x_orb]).T
    c0, c1 = np.linalg.lstsq(A_mat, y_orb, rcond=None)[0]
    A0 = -c0
    U0 = -c1 * LN10

    # ----- Stage 2: Raman window (fix Orbach; ignore QTM) -----
    mask_ram = (T >= T_ram_lo) & (T < T_orb_min)
    T_ram = T[mask_ram]
    log10_rate_ram = log10_rate_obs[mask_ram]

    def resid_raman(p):
        R, n = p
        pred = log10_rate_model(T_ram, A0, U0, R, n, Q=20.0)  # Q large -> negligible QTM
        return (pred - log10_rate_ram)

    res_ram = least_squares(resid_raman, x0=np.array([-10.0, 7.0]),
                            bounds=([-40.0, 2.0],[10.0, 12.0]))
    R0, n0 = res_ram.x

    # ----- Stage 3: QTM window (fix Orbach & Raman) -----
    mask_qtm = (T < T_qtm_max)
    T_qtm = T[mask_qtm]
    log10_rate_qtm = log10_rate_obs[mask_qtm]

    def resid_qtm(q):
        Q = q[0]
        pred = log10_rate_model(T_qtm, A0, U0, R0, n0, Q)
        return (pred - log10_rate_qtm)

    res_qtm = least_squares(resid_qtm, x0=np.array([0.0]), bounds=([-5.0],[15.0]))
    Q0 = float(res_qtm.x[0])

    # ----- Global Eq.10 fit initialised by stagewise -----
    init = np.array([A0, U0, R0, n0, Q0])
    bounds_lo = np.array([-20.0,    0.0, -40.0, 2.0,  -5.0])
    bounds_hi = np.array([   5.0, 5000.0,  10.0,12.0, 15.0])

    def resid_global(params):
        A, Ueff, R, n, Q = params
        pred = log10_rate_model(T, A, Ueff, R, n, Q)
        if weighted:
            # weight by inverse variance from alpha-derived sigma_log10
            w = 1.0 / np.clip(sigma_log10, 1e-6, None)
            return (pred - log10_rate_obs) * w
        else:
            return pred - log10_rate_obs

    res_glob = least_squares(resid_global, init, bounds=(bounds_lo, bounds_hi),
                             method="trf", max_nfev=50000, x_scale='jac')

    A, Ueff, R, n, Q = res_glob.x
    param_sd, cov, rss, s2 = linearised_param_sd(res_glob)

    pred_global = log10_rate_model(T, A, Ueff, R, n, Q)
    RMSE = float(np.sqrt(np.mean((pred_global - log10_rate_obs)**2)))
    R2 = float(r2_score(log10_rate_obs, pred_global))

    # Collect outputs
    stage = {
        "A0": A0, "U0": U0, "R0": R0, "n0": n0, "Q0": Q0,
        "T_orb_min": T_orb_min, "T_ram_lo": T_ram_lo, "T_qtm_max": T_qtm_max
    }
    fit = {
        "A": A, "Ueff": Ueff, "R": R, "n": n, "Q": Q,
        "A_sd": param_sd[0], "U_sd": param_sd[1], "R_sd": param_sd[2],
        "n_sd": param_sd[3], "Q_sd": param_sd[4],
        "RMSE": RMSE, "R2": R2
    }
    arrays = {
        "T": T, "log10_rate_obs": log10_rate_obs, "sigma_log10": sigma_log10,
        "pred_global": pred_global
    }
    return stage, fit, arrays

def main():
    ap = argparse.ArgumentParser(description="Stagewise + Global Eq.10 fit with Reta-Chilton style plotting.")
    ap.add_argument("--infile", type=str, default="tBuOCl.tsv", help="Input TSV/CSV with columns: T, tau_mu, alpha")
    ap.add_argument("--outfile", type=str, default="eq10_fit_params.csv", help="Output CSV for fitted parameters")
    ap.add_argument("--plot", type=str, default="eq10_fit_plot.png", help="Output PNG for plot")
    ap.add_argument("--orbmin", type=float, default=40.0, help="Lower T (K) for Orbach linear band (default 40 K)")
    ap.add_argument("--ramlo", type=float, default=25.0, help="Lower T (K) for Raman band (default 25 K)")
    ap.add_argument("--qtmmax", type=float, default=25.0, help="Upper T (K) for QTM band (default 25 K)")
    ap.add_argument("--weighted", action="store_true", help="Use weighted least squares with alpha-derived sigmas")
    args = ap.parse_args()

    # Load
    sep = "\t" if args.infile.lower().endswith(".tsv") else ","
    df = pd.read_csv(args.infile, sep=sep, header=None)
    T = df[0].values.astype(float)
    tau_mu = df[1].values.astype(float)
    alpha = df[2].values.astype(float)

    stage, fit, arrays = stagewise_and_global_fit(
        T, tau_mu, alpha,
        T_orb_min=args.orbmin, T_ram_lo=args.ramlo, T_qtm_max=args.qtmmax,
        weighted=args.weighted
    )

    # Print results to console
    print("Stagewise initial guesses:")
    print(f"  Orbach (T >= {stage['T_orb_min']:.1f} K): A0={stage['A0']:.4f}, U0={stage['U0']:.1f} K")
    print(f"  Raman  ({stage['T_ram_lo']:.1f}-{stage['T_orb_min']:.1f} K): R0={stage['R0']:.4f}, n0={stage['n0']:.3f}")
    print(f"  QTM    (T < {stage['T_qtm_max']:.1f} K): Q0={stage['Q0']:.4f}")
    print("\nGlobal Eq. 10 fit (±1σ from linearised covariance):")
    print(f"  A   = {fit['A']:.4f} ± {fit['A_sd']:.4f}")
    print(f"  U   = {fit['Ueff']:.1f} ± {fit['U_sd']:.1f} K")
    print(f"  R   = {fit['R']:.4f} ± {fit['R_sd']:.4f}")
    print(f"  n   = {fit['n']:.3f} ± {fit['n_sd']:.3f}")
    print(f"  Q   = {fit['Q']:.4f} ± {fit['Q_sd']:.4f}")
    print(f"  RMSE(log10 rate) = {fit['RMSE']:.3f},   R^2 = {fit['R2']:.4f}")

    # Save parameters CSV
    out_df = pd.DataFrame({
        "param": ["A","Ueff_K","R","n","Q","RMSE_log10rate","R2"],
        "value": [fit["A"], fit["Ueff"], fit["R"], fit["n"], fit["Q"], fit["RMSE"], fit["R2"]],
        "sd":    [fit["A_sd"], fit["U_sd"], fit["R_sd"], fit["n_sd"], fit["Q_sd"], np.nan, np.nan],
    })
    out_df.to_csv(args.outfile, index=False)
    print(f"\nSaved parameters to: {os.path.abspath(args.outfile)}")

    # Plot Reta-Chilton style
    order = np.argsort(arrays["T"])
    T_sorted = arrays["T"][order]
    obs_sorted = arrays["log10_rate_obs"][order]
    err_sorted = arrays["sigma_log10"][order]
    pred_sorted = arrays["pred_global"][order]

    plt.figure(figsize=(8,5))
    plt.errorbar(T_sorted, obs_sorted, yerr=err_sorted, fmt='x', capsize=3, label="data: log10(rate) ± 1σ")
    plt.plot(T_sorted, pred_sorted, label="Eq. 10 (global fit)", linewidth=2)
    plt.gca().invert_xaxis()
    plt.xlabel("Temperature (K)")
    plt.ylabel("log10(rate)")
    plt.title(f"Global Eq.10 fit   R²={fit['R2']:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)
    print(f"Saved plot to: {os.path.abspath(args.plot)}")

if __name__ == "__main__":
    main()
