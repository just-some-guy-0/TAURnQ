#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot relaxation rate (1/tau) vs temperature (K).
Accepts TSV/CSV with columns like:
  - 'temperature (K)', 'tau (s)', 'alpha'
  - or 'T', 'tau_mu', 'alpha'
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_data(path):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")

    def norm(s): 
        return str(s).strip().lower().replace("\u00a0", " ")
    df.columns = [norm(c) for c in df.columns]

    temp_candidates = ["t","t (k)","temperature","temperature (k)","temp","temp (k)","T","Temperature (K)"]
    tau_candidates  = ["tau","tau (s)","tau_mu","τ","τ (s)"]

    def pick(colnames, candidates):
        cols_set = set(colnames)
        for c in candidates:
            cc = norm(c)
            if cc in cols_set:
                return cc
        return None

    t_col  = pick(df.columns, temp_candidates)
    tau_col = pick(df.columns, tau_candidates)

    if (t_col is None) or (tau_col is None):
        df = pd.read_csv(path, sep=sep, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"File has {df.shape[1]} column(s); need at least 2 for T and tau. Path: {path}")
        df = df.iloc[:, :2].copy()
        df.columns = ["T", "tau"]
    else:
        df = df[[t_col, tau_col]].copy()
        df.columns = ["T", "tau"]

    df["T"]   = pd.to_numeric(df["T"], errors="coerce")
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")
    df = df.dropna(subset=["T", "tau"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Plot relaxation rate vs temperature.")
    ap.add_argument("--infile", default = "6FPh.tsv", help="Input TSV or CSV with T and tau columns.")
    ap.add_argument("--outfile", default=None, help="Optional output PNG (default: <infile>_rate_vs_T.png).")
    ap.add_argument("--logy", action="store_true", help="Plot rate on log10 scale.")
    args = ap.parse_args()

    df = load_data(args.infile)

    # Compute rate safely
    tau_safe = np.maximum(df["tau"].to_numpy(float), 1e-300)
    rate = 1.0 / tau_safe

    base = os.path.splitext(os.path.basename(args.infile))[0]
    outplot = args.outfile or f"{base}_rate_vs_T.png"

    plt.figure(figsize=(7,5))
    plt.plot(df["T"], rate, "o-", label="rate = 1/τ")
    plt.xlabel("Temperature (K)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Rate (s$^{-1}$, log scale)")
    plt.title("Relaxation Rate vs Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig(outplot, dpi=200)
    print(f"Saved plot to {os.path.abspath(outplot)}")

    plt.show()

if __name__ == "__main__":
    main()
