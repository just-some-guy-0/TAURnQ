#!/usr/bin/env python3
"""
dc_phase2.py  –  TAURnQ phase-2 analogue for DC magnetometry data.

Input TSV columns:  T (K)  |  tau_star (s)  |  beta
  - tau_star:  characteristic relaxation time from SEF fit
  - beta:      stretch parameter from SEF fit (0 < beta <= 1)

For each temperature row this script:
  1. Numerically evaluates the SEF log-domain distribution rho_log(tau*, x, beta)
  2. Fits a two-piece log-normal (mu, sigma1, sigma2) to it via least-squares
  3. Computes e^<ln tau> (= Blackmore et al.'s recommended central rate)
  4. Reports mu_ln, sd1_ln, sd2_ln as priors for dc_montecarlo.py

Output CSV columns:
    T, tau_star, beta, mu_log, sigma1_log, sigma2_log, elnTau_log
where _log quantities are in natural-log units (consistent with montecarlo.py).

Usage:
    python dc_phase2.py --infile mydata.tsv [--outfile dc_phase2_out.csv] [--plot]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
import argparse, os, sys

EULER_GAMMA = 0.5772156649015329

# ---------------------------------------------------------------------------
# SEF distribution in log10(tau) domain (Eq. 3/5 of Blackmore et al.)
# ---------------------------------------------------------------------------

def sef_rlog(log10_tau_star, x, beta, n_quad=2000):
    """
    rho_log(tau*, x, beta) evaluated at log10(tau) = x.
    Uses the substitution u -> t^(1/beta) to reduce oscillations,
    then truncated quadrature (Johnston 2006, Blackmore 2023 Eq. 3/5).
    Returns the probability density in log10(tau) space.
    """
    tau = 10.0 ** x
    tau_star = 10.0 ** log10_tau_star
    s = tau_star / tau

    phase = np.pi * beta / 2.0

    # Adaptive upper limit: damp when exp(-u^beta * cos(phase)) becomes negligible
    # u_max such that u_max^beta * cos(phase) > 50 (contribution < e^-50)
    cos_phase = np.cos(phase)
    if cos_phase > 1e-6:
        u_max = (50.0 / cos_phase) ** (1.0 / beta)
    else:
        u_max = 200.0  # near beta=1, cos(pi/2) -> 0
    u_max = min(u_max, 500.0)

    u = np.linspace(0, u_max, n_quad + 1)[1:]
    du = u_max / n_quad

    integrand = (np.exp(-u**beta * cos_phase)
                 * np.cos(s * u - u**beta * np.sin(phase)))
    rho_tau = np.sum(integrand) * du / np.pi

    return max(0.0, rho_tau * np.log(10.0) * tau)


def build_sef_curve(log10_tau_star, beta, xs, n_quad=600):
    """Evaluate SEF rho_log over an array of log10(tau) values."""
    return np.array([max(0.0, sef_rlog(log10_tau_star, x, beta, n_quad)) for x in xs])


# ---------------------------------------------------------------------------
# Two-piece log-normal fit to an SEF curve
# ---------------------------------------------------------------------------

def fit_two_piece_lognormal(log10_tau_star, beta, n_grid=600, x_half_range=None):
    """
    Fit a two-piece log-normal N_tp(mu, sigma1, sigma2) to the SEF distribution.

    The two-piece log-normal has density (in log10-tau space):
        f(x) = A * exp(-0.5 * ((x - mu)/sigma1)^2)   for x <= mu
        f(x) = A * exp(-0.5 * ((x - mu)/sigma2)^2)   for x >= mu
    where A is chosen for continuity at x=mu.

    Returns dict with keys:
        mu_log10, sigma1_log10, sigma2_log10   (all in log10 units)
        mu_ln, sigma1_ln, sigma2_ln            (converted to ln units for MC)
        elnTau_ln                              (e^<ln tau> in natural log)
        elnTau_log10                           (log10 of e^<ln tau>)
    """
    # Use Blackmore Eq. 7 analytical approximation for the mode location.
    # This centres the evaluation grid correctly for all beta.
    xmode_approx = -0.828 * (beta - 0.375)**2 + (207.0 / 640.0) + log10_tau_star

    # Grid half-width: wider for small beta (broader distribution)
    hw = max(4.0, 2.5 / beta)
    xs = np.linspace(xmode_approx - hw, xmode_approx + hw, n_grid)
    ys = build_sef_curve(log10_tau_star, beta, xs)

    # normalise to probability density (in log10 space)
    dx = xs[1] - xs[0]
    total = np.sum(ys) * dx
    if total < 1e-30:
        raise ValueError(f"SEF curve integrates to ~0 for tau*={10**log10_tau_star:.3g}, beta={beta:.3f}")
    yn = ys / total

    # --- 1. locate mode ---
    mu_idx = int(np.argmax(yn))
    mu = xs[mu_idx]
    peak = yn[mu_idx]

    # --- 2. fit sigma1 (left half) and sigma2 (right half) independently ---
    # Objective: minimise sum of squared residuals against half-Gaussian
    def fit_half(side_xs, side_ys, mu_val, peak_val):
        def obj(sig):
            if sig <= 0:
                return 1e30
            predicted = peak_val * np.exp(-0.5 * ((side_xs - mu_val) / sig) ** 2)
            return np.sum((side_ys - predicted) ** 2)
        res = minimize_scalar(obj, bounds=(1e-4, 10.0), method="bounded",
                              options={"xatol": 1e-6})
        return res.x

    left_mask = xs <= mu
    right_mask = xs >= mu

    sigma1 = fit_half(xs[left_mask], yn[left_mask], mu, peak)
    sigma2 = fit_half(xs[right_mask], yn[right_mask], mu, peak)

    # --- 3. Blackmore e^<ln tau> (Eq. 9): exact analytical result ---
    # <ln tau> = (1 - 1/beta)*Euler_gamma + ln(tau_star)
    ln_tau_star = log10_tau_star * np.log(10.0)
    elnTau_ln = np.exp((1.0 - 1.0 / beta) * EULER_GAMMA + ln_tau_star)
    elnTau_log10 = np.log10(elnTau_ln)

    # --- 4. convert log10 sigmas → ln sigmas (multiply by ln10) ---
    LN10 = np.log(10.0)
    return {
        "mu_log10":     mu,
        "sigma1_log10": sigma1,
        "sigma2_log10": sigma2,
        "mu_ln":        mu * LN10,
        "sigma1_ln":    sigma1 * LN10,
        "sigma2_ln":    sigma2 * LN10,
        "elnTau_ln":    np.log(elnTau_ln),
        "elnTau_log10": elnTau_log10,
    }


# ---------------------------------------------------------------------------
# Analytical approximation for e^<ln tau> ESD  (Blackmore Eq. 12/13)
# ---------------------------------------------------------------------------

def elnTau_esd_ln(beta):
    """
    Standard deviation of <ln tau> for the SEF (Blackmore Eq. 12):
        sigma_<ln_tau>^2 = (1/beta^2 - 1) * pi^2/6
    Returns sigma in natural-log units.
    """
    var = (1.0 / beta**2 - 1.0) * (np.pi**2 / 6.0)
    return np.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# Process a full dataset
# ---------------------------------------------------------------------------

def process_file(infile, outfile, make_plot=False):
    df = pd.read_csv(infile, sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["T", "tau_star", "beta"]
    df = df.astype(float)

    records = []
    for idx, row in df.iterrows():
        T, tau_star, beta = row["T"], row["tau_star"], row["beta"]
        log10_ts = np.log10(tau_star)

        try:
            fit = fit_two_piece_lognormal(log10_ts, beta)
        except Exception as e:
            print(f"[idx={idx}] T={T:.2f} K  ERROR: {e}", file=sys.stderr)
            continue

        esd = elnTau_esd_ln(beta)

        rec = {
            "T":           T,
            "tau_star":    tau_star,
            "beta":        beta,
            # log10-space quantities (for diagnostics / plotting)
            "mu_log10":    fit["mu_log10"],
            "sigma1_log10": fit["sigma1_log10"],
            "sigma2_log10": fit["sigma2_log10"],
            "elnTau_log10": fit["elnTau_log10"],
            # ln-space quantities (fed directly into dc_montecarlo.py)
            "mu_ln":        fit["mu_ln"],
            "sigma1_ln":    fit["sigma1_ln"],
            "sigma2_ln":    fit["sigma2_ln"],
            "elnTau_ln":    fit["elnTau_ln"],
            "esd_elnTau_ln": esd,
        }
        records.append(rec)

        print(f"[idx={idx:>3}] T={T:6.2f} K | "
              f"tau*={tau_star:.3e}  beta={beta:.3f} | "
              f"mu(log10)={fit['mu_log10']:+.3f}  "
              f"s1={fit['sigma1_log10']:.3f}  s2={fit['sigma2_log10']:.3f} | "
              f"e^<ln tau>={np.exp(fit['elnTau_ln']):.3e}  "
              f"ESD={esd:.3f} (ln)")

    if not records:
        print("No rows processed successfully.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(records)
    out_df.to_csv(outfile, index=False)
    print(f"\nWrote {len(records)} rows → {os.path.abspath(outfile)}")

    if make_plot:
        _diagnostic_plot(df, out_df)

    return out_df


def _diagnostic_plot(raw_df, out_df):
    """Overlay the two-piece log-normal fits on the SEF curves."""
    n = len(out_df)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    LN10 = np.log(10.0)
    for i, row in out_df.iterrows():
        ax = axes[i]
        log10_ts = np.log10(row["tau_star"])
        beta = row["beta"]
        mu = row["mu_log10"]
        s1, s2 = row["sigma1_log10"], row["sigma2_log10"]
        elnT = row["elnTau_log10"]

        xs = np.linspace(log10_ts - 7, log10_ts + 7, 400)
        ys_sef = build_sef_curve(log10_ts, beta, xs, n_quad=400)
        dx = xs[1] - xs[0]
        ys_sef /= (np.sum(ys_sef) * dx + 1e-30)

        peak = np.max(ys_sef)
        ys_tp = peak * np.where(
            xs <= mu,
            np.exp(-0.5 * ((xs - mu) / s1) ** 2),
            np.exp(-0.5 * ((xs - mu) / s2) ** 2)
        )

        ax.plot(xs, ys_sef, color="#378ADD", linewidth=1.8, label="SEF")
        ax.plot(xs, ys_tp, color="#D85A30", linewidth=1.6,
                linestyle="--", label="2-piece LN fit")
        ax.axvline(mu, color="#639922", linewidth=1.0, linestyle=":", label="μ (mode)")
        ax.axvline(elnT, color="#BA7517", linewidth=1.0, linestyle="-.", label="e^⟨ln τ⟩")
        ax.set_title(f"T = {row['T']:.1f} K, β = {beta:.2f}", fontsize=9)
        ax.set_xlabel("log₁₀(τ / s)", fontsize=8)
        ax.set_ylabel("ρ_log", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Two-piece log-normal fits to SEF distributions", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fit two-piece log-normal to SEF distributions (DC magnetometry phase 2)"
    )
    ap.add_argument("--infile",  default="dc_data.tsv",
                    help="TSV/CSV with columns: T, tau_star, beta")
    ap.add_argument("--outfile", default="dc_phase2_out.csv",
                    help="Output CSV with fitted two-piece log-normal parameters")
    ap.add_argument("--plot",    action="store_true",
                    help="Show diagnostic overlay plots")
    args = ap.parse_args()

    process_file(args.infile, args.outfile, make_plot=args.plot)


if __name__ == "__main__":
    main()