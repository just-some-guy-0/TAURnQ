#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute fractional contributions of Orbach, Raman, and QTM terms
using initial-guess parameters (or fitted params if you swap them in).

Input : TSV/CSV with columns like ["temperature (K)", "tau (s)", "alpha"]
        or ["T", "tau_mu", "alpha"] — loader is robust to either.
Output: CSV with terms and fractions, plus a stacked-area plot vs T.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

LN10 = np.log(10.0)
V_orbmin = 82
V_qtmmax = 7.5
PATH = "NSiPr3.tsv"  # you can override with --infile

def rate_model(T, A, Ueff, R, N, Q):
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1e-300)
    orbach = (10.0**(-A)) * np.exp(-Ueff / T_safe)
    raman  = (10.0**(R))  * (T_safe**N)
    qtm    = (10.0**(-Q))
    return orbach, raman, qtm

def rate_total(T, A, Ueff, R, N, Q):
    o, r, q = rate_model(T, A, Ueff, R, N, Q)
    return o + r + q

# --------- PARAMETER GUESSES FROM MEANS ----------
def guess_params_from_means(T, tau_mu, orbmin=V_orbmin, qtmmax=V_qtmmax, no_qtm=False, qtm_eps=1e-4):
    T = np.asarray(T, dtype=float)
    tau_mu = np.asarray(tau_mu, dtype=float)
    rate_obs = 1.0 / np.maximum(tau_mu, 1e-300)
    y_all    = np.log10(rate_obs)

    # ----- Orbach guess from high T -----
    mask_orb = T >= orbmin
    if mask_orb.sum() >= 2:
        x = 1.0 / T[mask_orb]
        y = y_all[mask_orb]
        A_mat = np.vstack([np.ones_like(x), x]).T
        c0, c1 = np.linalg.lstsq(A_mat, y, rcond=None)[0]
        A = -float(c0)
        Ueff = -float(c1) * LN10
    else:
        A, Ueff = -10.0, 100.0  # conservative default

    # ----- QTM guess -----
    if no_qtm:
        Q = 20.0  # effectively zero QTM: 10^{-20} s^-1
    else:
        mask_qtm = T <= qtmmax
        if mask_qtm.sum() >= 2:
            Q = -float(np.median(y_all[mask_qtm]))
        else:
            # Make QTM negligible vs Orbach at all T
            orbach_base = (10.0**(-A)) * np.exp(-Ueff / np.maximum(T, 1e-300))
            r_scale = np.maximum(np.min(orbach_base), 1e-300)
            qtm_target = max(qtm_eps * r_scale, 1e-300)
            Q = -np.log10(qtm_target)

    # ----- Raman guess from mid T (after stripping QTM & Orbach) -----
    orbach_base = (10.0**(-A)) * np.exp(-Ueff / np.maximum(T, 1e-300))
    qtm_base    = (10.0**(-Q))
    ram_est     = np.maximum(1e-300, rate_obs - (orbach_base + qtm_base))

    T_mid_mask = (T > qtmmax) & (T < orbmin)
    if T_mid_mask.sum() >= 2:
        xx = np.log10(T[T_mid_mask])
        yy = np.log10(ram_est[T_mid_mask])
    else:
        xx = np.log10(T)
        yy = np.log10(ram_est)

    X = np.vstack([np.ones_like(xx), xx]).T
    R, N = np.linalg.lstsq(X, yy, rcond=None)[0]

    return float(A), float(Ueff), float(R), float(N), float(Q)

# --------- ROBUST LOADER ----------
def load_t_tau_alpha(path):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    # Try reading with header first
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    # Normalize column names for matching
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df.columns = [cols_lower[c] for c in df.columns]

    # Map common variants
    renames = {}
    if "temperature (k)" in df.columns: renames["temperature (k)"] = "T"
    if "t (k)" in df.columns:           renames["t (k)"]           = "T"
    if "t" in df.columns:               renames["t"]               = "T"
    if "tau (s)" in df.columns:         renames["tau (s)"]         = "tau_mu"
    if "tau" in df.columns:             renames["tau"]             = "tau_mu"
    if "alpha" in df.columns:           renames["alpha"]           = "alpha"
    df = df.rename(columns=renames)

    # If required columns still missing, fall back to no-header read
    if not {"T", "tau_mu", "alpha"}.issubset(df.columns):
        df = pd.read_csv(path, sep=sep, header=None, names=["T", "tau_mu", "alpha"])

    # Coerce to numeric & drop any junk rows
    for c in ["T", "tau_mu", "alpha"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["T", "tau_mu", "alpha"]).reset_index(drop=True)
    return df

# ------------------------------- MAIN -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fractional contributions of Orbach/Raman/QTM vs T using initial guesses.")
    ap.add_argument("--infile", type=str, default=PATH,
                    help="Input .tsv/.csv; columns like: 'temperature (K)', 'tau (s)', 'alpha' or 'T, tau_mu, alpha'.")
    ap.add_argument("--orbmin", type=float, default=V_orbmin,
                    help="Lower T (K) for Orbach guess (default 40 K).")
    ap.add_argument("--qtmmax", type=float, default=V_qtmmax,
                    help="Upper T (K) for QTM guess (default 18 K).")
    ap.add_argument("--outfile", type=str, default=None,
                    help="Output CSV (default: <infile>_term_fractions.csv).")
    ap.add_argument("--plot", type=str, default=None,
                    help="Output plot PNG (default: <infile>_term_fractions.png).")
    ap.add_argument("--no_qtm", action="store_true",
                    help="Force QTM term to zero (negligible) across all T.")
    ap.add_argument("--qtm_eps", type=float, default=1e-4,
                    help="If no low-T points, set Q so qtm <= eps * min(non-QTM rate). Default 1e-4.")
    args = ap.parse_args()

    # Derive default out names from infile if not provided
    base = os.path.splitext(os.path.basename(args.infile))[0]
    outfile = args.outfile or f"{base}_term_fractions.csv"
    plotfile = args.plot or f"{base}_term_fractions.png"

    # Load data robustly
    df = load_t_tau_alpha(args.infile)
    T = df["T"].to_numpy(float)
    tau_mu = df["tau_mu"].to_numpy(float)

    # Guess parameters (pass through flags)
    A, Ueff, R, N, Q = guess_params_from_means(
        T, tau_mu,
        orbmin=args.orbmin,
        qtmmax=args.qtmmax,
        no_qtm=args.no_qtm,
        qtm_eps=args.qtm_eps
    )

    # Evaluate terms & fractions
    orbach, raman, qtm = rate_model(T, A, Ueff, R, N, Q)
    total = np.maximum(orbach + raman + qtm, 1e-300)

    f_orb = orbach / total
    f_ram = raman / total
    f_qtm = qtm    / total

    out = pd.DataFrame({
        "T_K": T,
        "frac_orbach": f_orb,
        "frac_raman": f_ram,
        "frac_qtm": f_qtm,
    }).sort_values("T_K", ascending=True)
    out.to_csv(outfile, index=False)

    # ------------------------------- OUTPUT -------------------------------
    print(f"Saved fractions to: {os.path.abspath(outfile)}\n")
    print("Mean-fitted parameters:")
    print(f"A={A:.3f}")
    print(f"Ueff={Ueff:.3f} K")
    print(f"R={R:.3f}")
    print(f"N={N:.3f}")
    print(f"Q={Q:.3f}\n")

    # Plot
    order = np.argsort(T)
    T_sorted = T[order]
    f_orb_s  = f_orb[order]
    f_ram_s  = f_ram[order]
    f_qtm_s  = f_qtm[order]

    plt.figure(figsize=(8,5))
    plt.stackplot(T_sorted, f_orb_s, f_ram_s, f_qtm_s, labels=["Orbach", "Raman", "QTM"])
    plt.ylim(0, 1)
    plt.ylabel("Fraction of total rate")
    plt.xlabel("Temperature (K)")
    plt.title("Fractional contributions to rate")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plotfile, dpi=200)
    print(f"Saved plot to: {os.path.abspath(plotfile)}")

if __name__ == "__main__":
    main()
