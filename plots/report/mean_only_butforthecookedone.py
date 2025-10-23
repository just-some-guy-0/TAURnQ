#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — Fit Eq. 10 to the MEAN relaxation times to obtain INITIAL GUESSES
==========================================================================
Input  : TSV/CSV with columns [T (K), tau_mu, alpha]. (alpha not used here)
Output : Prints fitted parameters (A, Ueff, R, n, Q) and R^2 to console,
         saves a CSV with the same, and optionally a plot.

Equation 10 (rate form):
    rate(T) = tau^{-1}(T) = 10^{-A} * exp(-Ueff/T) + 10^{R} * T^{n} + 10^{-Q}

We fit in log10-space to stabilise the dynamic range:
    y(T) = log10(rate_model(T; A,Ueff,R,n,Q))

This script produces sensible initial guesses from means (tau_mu) only.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

LN10 = np.log(10.0)
SRC = "NSiPr3"

# ---------------------------- Model ----------------------------
def rate_model(T, A, Ueff, R, n, Q):
    term_orb = (10.0**(-A)) * np.exp(-Ueff / T)  # Orbach
    term_ram = (10.0**(R))  * (T**n)             # Raman
    term_qtm = (10.0**(-Q))                      # QTM
    return term_orb + term_ram + term_qtm

def log10_rate_model(T, A, Ueff, R, n, Q):
    return np.log10(np.clip(rate_model(T, A, Ueff, R, n, Q), 1e-300, None))

# ------------------------ Objective & metrics ------------------------
def resid_means(params, T, tau_mu):
    A, Ueff, R, n, Q = params
    pred = log10_rate_model(T, A, Ueff, R, n, Q)
    obs  = np.log10(1.0/np.clip(tau_mu, 1e-300, None))
    return pred - obs

def r2_score(y_true, y_pred):
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

# --------------------- Heuristic initial guesses ---------------------
def guess_params_from_means(T, tau_mu, orbmin=40.5, qtmmax=0,
                            bounds_lo=np.array([-20.0,0.0,-40.0,2.0,-5.0]),
                            bounds_hi=np.array([5.0,5000.0,10.0,12.0,15.0])):
    rate_obs = 1.0 / np.clip(tau_mu, 1e-300, None)

    # ---- Orbach (high-T) ----
    mask_orb = T >= orbmin
    if mask_orb.sum() >= 2:
        x = 1.0 / T[mask_orb]
        y = np.log10(rate_obs[mask_orb])  # = log10(rate)
        A_mat = np.vstack([np.ones_like(x), x]).T
        c0, c1 = np.linalg.lstsq(A_mat, y, rcond=None)[0]
        A0 = -c0
        U0 = -c1 * LN10
    else:
        A0, U0 = -10.0, 800.0  # safe fallback

    # ---- QTM (low-T) ----
    mask_qtm = T <= qtmmax
    if mask_qtm.sum() >= 2:
        Q0 = -np.median(np.log10(rate_obs[mask_qtm]))
    else:
        Q0 = 0.0

    # ---- Raman (mid-T) ----
    ram_est = np.maximum(1e-300, rate_obs - ((10**(-A0))*np.exp(-U0/T) + 10**(-Q0)))
    T_mid_mask = (T > qtmmax) & (T < orbmin)
    xx = np.log10(T[T_mid_mask]) if T_mid_mask.sum() >= 2 else np.log10(T)
    yy = np.log10(np.maximum(1e-300, ram_est[T_mid_mask])) if T_mid_mask.sum() >= 2 else np.log10(np.maximum(1e-300, ram_est))

    m = np.isfinite(xx) & np.isfinite(yy)
    if m.sum() >= 2:
        X = np.vstack([np.ones_like(xx[m]), xx[m]]).T
        R0, n0 = np.linalg.lstsq(X, yy[m], rcond=None)[0]
    else:
        R0, n0 = -6.0, 4.0  # safe fallback

    # ---- Clip to bounds so x0 is feasible ----
    A0 = float(np.clip(A0, bounds_lo[0], bounds_hi[0]))
    U0 = float(np.clip(U0, bounds_lo[1], bounds_hi[1]))
    R0 = float(np.clip(R0, bounds_lo[2], bounds_hi[2]))
    n0 = float(np.clip(n0, bounds_lo[3], bounds_hi[3]))
    Q0 = float(np.clip(Q0, bounds_lo[4], bounds_hi[4]))
    return A0, U0, R0, n0, Q0

# ------------------------------- Main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fit Eq.10 to MEANS only (initial guesses)." )
    ap.add_argument("--infile", type=str, default=f"{SRC}.tsv",
                    help="Input TSV/CSV with columns: T, tau_mu, alpha (alpha unused)")
    ap.add_argument("--outfile", type=str, default="eq10_means_fit_params.csv",
                    help="Output CSV for initial-guess parameters")
    ap.add_argument("--plot", type=str, default="eq10_means_fit_plot.png",
                    help="Output PNG plot (data vs fit)")
    ap.add_argument("--orbmin", type=float, default=86.0,
                    help="Lower T (K) for Orbach guess")
    ap.add_argument("--qtmmax", type=float, default=9.0,
                    help="Upper T (K) for QTM guess")
    args = ap.parse_args()

    # Load data
    sep = "\t" if args.infile.lower().endswith(".tsv") else ","
    df = pd.read_csv(args.infile, sep=sep, header=None)
    T = df[0].values.astype(float)
    tau_mu = df[1].values.astype(float)

    # Heuristic initial guesses from means
    bounds_lo = np.array([-20.0,    0.0, -40.0, 2.0,  -5.0], dtype=float)
    bounds_hi = np.array([   5.0, 5000.0,  10.0,12.0, 15.0], dtype=float)
    A0, U0, R0, n0, Q0 = guess_params_from_means(
        T, tau_mu, orbmin=args.orbmin, qtmmax=args.qtmmax,
        bounds_lo=bounds_lo, bounds_hi=bounds_hi
    )
    init = np.array([A0, U0, R0, n0, Q0], dtype=float)

    # Nonlinear least squares on means
    res = least_squares(resid_means, init, bounds=(bounds_lo, bounds_hi),
                        args=(T, tau_mu), method="trf",
                        max_nfev=50000, x_scale='jac')

    A, Ueff, R, n, Q = res.x

    # Metrics
    y_obs = np.log10(1.0/np.clip(tau_mu, 1e-300, None))
    y_fit = log10_rate_model(T, A, Ueff, R, n, Q)
    RMSE  = float(np.sqrt(np.mean((y_fit - y_obs)**2)))
    R2    = float(r2_score(y_obs, y_fit))

    # Report
    print("Initial guesses (heuristics from means):")
    print(f"  A0={A0:.4f}, U0={U0:.1f} K, R0={R0:.4f}, n0={n0:.3f}, Q0={Q0:.4f}")
    print("\nEq.10 fit to MEANS (use these as initial guesses for later stages):")
    print(f"  A   = {A:.4f}")
    print(f"  U   = {Ueff:.1f} K")
    print(f"  R   = {R:.4f}")
    print(f"  n   = {n:.3f}")
    print(f"  Q   = {Q:.4f}")
    print(f"  RMSE(log10 rate) = {RMSE:.3f},   R^2 = {R2:.4f}")

    # Save CSV
    out = pd.DataFrame({
        "param": ["A","Ueff_K","R","n","Q","RMSE_log10rate","R2"],
        "value": [A, Ueff, R, n, Q, RMSE, R2]
    })
    out.to_csv(args.outfile, index=False)
    print(f"\nSaved initial-guess parameters to: {os.path.abspath(args.outfile)}")

    # Plot
    order = np.argsort(T)
    T_sorted = T[order]
    y_obs_s  = y_obs[order]
    y_fit_s  = y_fit[order]

    plt.figure(figsize=(8,5))
    plt.scatter(T_sorted, y_obs_s, s=28, label="data: log10(rate)")
    plt.plot(T_sorted, y_fit_s, label="Eq.10 fit to means", linewidth=2)
    plt.gca().invert_xaxis()
    plt.xlabel("Temperature (K)")
    plt.ylabel("log10(rate)")
    plt.title(f"Means-only fit (initial guesses)   R^2={R2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)
    print(f"Saved plot to: {os.path.abspath(args.plot)}")

if __name__ == "__main__":
    main()
