#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Model ----------
def distribution_from_params(T, A, Ueff, R, n, Q):
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    term2 = 10.0**(R)  * (T_safe**n)
    term3 = 10.0**(Q)
    return term1 + term2 + term3

# ---------- Utilities ----------
def parse_floats_comma(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",") if x.strip() != ""]
    return np.array(vals, dtype=float)

def summarize(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    stats = {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p2.5": float(np.percentile(x, 2.5)),
        "p16":  float(np.percentile(x, 16)),
        "p84":  float(np.percentile(x, 84)),
        "p97.5": float(np.percentile(x, 97.5)),
    }
    return stats

def print_table(d: dict, title: str = ""):
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    width = max(len(k) for k in d.keys()) + 2
    for k, v in d.items():
        print(f"{k:<{width}} {v:.6g}")

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Monte Carlo distribution of Γ(T) at a single temperature.")
    p.add_argument("--means", type=parse_floats_comma,
                   default=np.array([10.6, 850.0, -5.3, 4.0, -0.35]),
                   help="Comma-separated means for [A,Ueff,R,n,Q].")
    p.add_argument("--sds", type=parse_floats_comma,
                   default=np.array([0.2, 50.0, 0.3, 0.4, 0.2]),
                   help="Comma-separated SDs for [A,Ueff,R,n,Q].")
    p.add_argument("--T", type=float, default=40.0, help="Temperature (K).")
    p.add_argument("--N", type=int, default=50000, help="Number of parameter samples.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--out-prefix", type=str, default="", help="If set, save figures with this prefix.")
    args = p.parse_args()

    means = np.asarray(args.means, dtype=float)
    sds   = np.asarray(args.sds, dtype=float)

    if means.shape != (5,) or sds.shape != (5,):
        raise ValueError("Provide exactly 5 means and 5 SDs for [A, Ueff, R, n, Q].")
    if np.any(sds < 0):
        raise ValueError("All SDs must be non-negative.")

    rng = np.random.default_rng(args.seed)

    # Independent normal sampling; swap to a full covariance if you have one.
    A     = rng.normal(means[0], sds[0], args.N)
    Ueff  = rng.normal(means[1], sds[1], args.N)
    R     = rng.normal(means[2], sds[2], args.N)
    n     = rng.normal(means[3], sds[3], args.N)
    Q     = rng.normal(means[4], sds[4], args.N)

    # Evaluate Γ at the chosen temperature
    T_star = float(args.T)
    gamma = distribution_from_params(T_star, A, Ueff, R, n, Q)

    # --- Summary stats ---
    stats_lin = summarize(gamma)
    # log10 stats: filter any non-positive (shouldn't occur with this model, but be safe)
    mask_pos = gamma > 0
    log10_gamma = np.log10(gamma[mask_pos])
    stats_log = {
        "mean_log10": float(np.mean(log10_gamma)),
        "sd_log10": float(np.std(log10_gamma, ddof=1)),
        "median_log10": float(np.median(log10_gamma)),
        "p2.5_log10": float(np.percentile(log10_gamma, 2.5)),
        "p16_log10": float(np.percentile(log10_gamma, 16)),
        "p84_log10": float(np.percentile(log10_gamma, 84)),
        "p97.5_log10": float(np.percentile(log10_gamma, 97.5)),
    }

    print(f"\nT* = {T_star} K, N = {args.N}")
    print_table(stats_lin, title="Γ(T*) summary (linear space, s^-1)")
    print_table(stats_log, title="log10 Γ(T*) summary")

    # Log10 histogram
    plt.figure(figsize=(7, 4.5))
    plt.hist(log10_gamma, bins=80, density=True)
    plt.xlabel(r"$\log_{10}\,\Gamma(T^*)$")
    plt.ylabel("Density")
    plt.title(f"Distribution of log10 Γ at T = {T_star} K")
    plt.tight_layout()
    if args.out_prefix:
        plt.savefig(f"{args.out_prefix}_log10_hist.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
