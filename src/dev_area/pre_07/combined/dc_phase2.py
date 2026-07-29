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

def _load_input(infile):
    """
    Flexibly load the input TSV/CSV.

    Accepted column orderings (with or without a header row):
      - T  tau_star  beta        (original script convention)
      - T  beta      tau_star    (Blackmore Table 1 order)
      - Any order if a header row is present with recognised names

    Recognised header names (case-insensitive):
      T / temp / temperature
      tau_star / tau* / taustar / tau / t_star
      beta / b / stretch
    """
    # ── try to read with automatic header detection ──────────────────────────
    raw = pd.read_csv(infile, sep=None, engine="python", header="infer")

    # normalise column names to lower-case, strip whitespace / asterisks
    raw.columns = [str(c).lower().strip().replace("*", "").replace(" ", "_")
                   for c in raw.columns]

    T_NAMES    = {"t", "temp", "temperature"}
    TAU_NAMES  = {"tau_star", "taustar", "tau", "t_star", "tau_s", "tau_mean"}
    BETA_NAMES = {"beta", "b", "stretch", "beta_stretch"}

    def find_col(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    t_col   = find_col(raw, T_NAMES)
    tau_col = find_col(raw, TAU_NAMES)
    b_col   = find_col(raw, BETA_NAMES)

    if t_col and tau_col and b_col:
        # header row was present and recognised
        df = raw[[t_col, tau_col, b_col]].copy()
        df.columns = ["T", "tau_star", "beta"]
        df = df.astype(float)
        print(f"  [loader] header detected: T='{t_col}', tau*='{tau_col}', beta='{b_col}'")
        return df

    # ── no recognised header — treat as headerless, infer column order ───────
    raw_nh = pd.read_csv(infile, sep=None, engine="python", header=None).iloc[:, :3]
    raw_nh.columns = ["c0", "c1", "c2"]
    raw_nh = raw_nh.astype(float, errors="ignore")

    # drop any non-numeric rows (e.g. an unrecognised header line)
    raw_nh = raw_nh[pd.to_numeric(raw_nh["c0"], errors="coerce").notna()]
    raw_nh = raw_nh[pd.to_numeric(raw_nh["c1"], errors="coerce").notna()]
    raw_nh = raw_nh[pd.to_numeric(raw_nh["c2"], errors="coerce").notna()]
    raw_nh = raw_nh.astype(float).reset_index(drop=True)

    # β is always in (0, 1]; τ* is always > 0 and typically >> 1 for SMMs.
    # Identify which column is β by finding the one whose values are all in (0,1].
    beta_candidates = [c for c in ["c0", "c1", "c2"]
                       if (raw_nh[c] > 0).all() and (raw_nh[c] <= 1.0).all()]

    if len(beta_candidates) == 1:
        b_col_nh = beta_candidates[0]
        remaining = [c for c in ["c0", "c1", "c2"] if c != b_col_nh]
        # of the remaining two, T is the smaller (typically 2-300 K vs tau* up to 1e5 s)
        if raw_nh[remaining[0]].mean() < raw_nh[remaining[1]].mean():
            t_col_nh, tau_col_nh = remaining[0], remaining[1]
        else:
            t_col_nh, tau_col_nh = remaining[1], remaining[0]
        df = raw_nh[[t_col_nh, tau_col_nh, b_col_nh]].copy()
        df.columns = ["T", "tau_star", "beta"]
        print(f"  [loader] no header — auto-detected columns: "
              f"T=col{t_col_nh[1]}, tau*=col{tau_col_nh[1]}, beta=col{b_col_nh[1]}")
        return df

    # ── last resort: assume T | tau_star | beta order ─────────────────────────
    print("  [loader] WARNING: could not auto-detect column order, "
          "assuming T | tau_star | beta", file=sys.stderr)
    df = raw_nh[["c0", "c1", "c2"]].copy()
    df.columns = ["T", "tau_star", "beta"]
    return df


def _in_window(T, window):
    """Return True if T is within (lo, hi) inclusive. None means no window (always False)."""
    if window is None:
        return False
    lo, hi = window
    return lo <= T <= hi


def process_file(infile, outfile, make_plot=False,
                 qtm_window=None, raman_window=None):
    """
    Parameters
    ----------
    qtm_window   : (Tlo, Thi) or None
        Temperature range where QTM dominates.  Rows inside this window
        produce valid Q priors.  Rows outside produce NaN for Q columns.
    raman_window : (Tlo, Thi) or None
        Temperature range where Raman has meaningful contribution.  Rows
        inside produce valid R/n priors.  Rows outside produce NaN for
        Raman columns.

    A and Ueff are NEVER estimated from DC data — they are always NaN.
    The Orbach term is negligible across the entire accessible DC range
    (contribution < 0.1% for typical SMMs at low T).
    """
    df = _load_input(infile)
    df = df.astype(float)

    # Summarise window coverage before processing
    T_vals = df["T"].values
    print(f"\n  Input temperatures: {sorted(T_vals)}")
    if qtm_window:
        n_qtm = sum(_in_window(T, qtm_window) for T in T_vals)
        print(f"  QTM   window {qtm_window[0]:.1f}–{qtm_window[1]:.1f} K : "
              f"{n_qtm}/{len(T_vals)} rows  → valid Q priors")
    else:
        print("  QTM   window: not set  → Q columns NaN for all rows")

    if raman_window:
        n_ram = sum(_in_window(T, raman_window) for T in T_vals)
        print(f"  Raman window {raman_window[0]:.1f}–{raman_window[1]:.1f} K : "
              f"{n_ram}/{len(T_vals)} rows  → valid R/n priors")
    else:
        print("  Raman window: not set  → Raman columns NaN for all rows")

    print(f"  Orbach (A, Ueff): always NaN — not estimable from DC data\n")

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

        in_qtm   = _in_window(T, qtm_window)
        in_raman = _in_window(T, raman_window)

        rec = {
            "T":        T,
            "tau_star": tau_star,
            "beta":     beta,

            # ── distribution shape (always computed) ──────────────────────
            "mu_log10":     fit["mu_log10"],
            "sigma1_log10": fit["sigma1_log10"],
            "sigma2_log10": fit["sigma2_log10"],
            "elnTau_log10": fit["elnTau_log10"],
            "mu_ln":        fit["mu_ln"],
            "sigma1_ln":    fit["sigma1_ln"],
            "sigma2_ln":    fit["sigma2_ln"],
            "elnTau_ln":    fit["elnTau_ln"],
            "esd_elnTau_ln": esd,

            # ── window flags (for downstream use) ─────────────────────────
            "in_qtm_window":   in_qtm,
            "in_raman_window": in_raman,

            # ── Orbach: never estimable from DC data ──────────────────────
            "Au_dc": np.nan,
            "Uu_dc": np.nan,
            "As_dc": np.nan,
            "Us_dc": np.nan,

            # ── QTM: valid inside qtm_window only ─────────────────────────
            # Q = log10(tau_QTM) = elnTau_ln / ln(10)
            # sQ proxy = sigma1_ln / ln(10)  (left-side spread of TPN)
            "Qu_dc": fit["elnTau_ln"] / np.log(10) if in_qtm else np.nan,
            "Qs_dc": fit["sigma1_ln"] / np.log(10) if in_qtm else np.nan,

            # ── Raman: only valid inside raman_window ─────────────────────
            # Ru_dc and Nu_dc cannot be computed per-row — R and n are
            # determined by the temperature SLOPE across Raman-window rows,
            # not by a single temperature. These are set to NaN here and
            # computed in the post-processing summary below after all rows
            # are available. The compiled R/n estimates are printed and
            # written as metadata but not per-row.
            "Ru_dc": np.nan,
            "Nu_dc": np.nan,
            "Rs_dc": fit["sigma1_ln"] / np.log(10) if in_raman else np.nan,
        }
        records.append(rec)

        # concise per-row summary
        term_flags = []
        if in_qtm:   term_flags.append("QTM✓")
        if in_raman: term_flags.append("Raman✓")
        if not term_flags: term_flags.append("no valid window")

        print(f"[idx={idx:>3}] T={T:6.2f} K | "
              f"tau*={tau_star:.3e}  beta={beta:.3f} | "
              f"mu(log10)={fit['mu_log10']:+.3f}  "
              f"s1={fit['sigma1_log10']:.3f}  s2={fit['sigma2_log10']:.3f} | "
              f"e^<ln tau>={np.exp(fit['elnTau_ln']):.3e}  "
              f"ESD={esd:.3f} (ln) | {', '.join(term_flags)}")

    if not records:
        print("No rows processed successfully.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(records)
    out_df.to_csv(outfile, index=False)

    # Summary of what was produced
    n_qtm_valid   = out_df["in_qtm_window"].sum()
    n_raman_valid = out_df["in_raman_window"].sum()
    print(f"\nWrote {len(records)} rows → {os.path.abspath(outfile)}")
    print(f"  Valid Q priors (QTM window):     {n_qtm_valid} rows")
    print(f"  Valid R/n priors (Raman window): {n_raman_valid} rows")
    print(f"  Valid A/Ueff priors (Orbach):    0 rows  (NaN — use AC data)")

    # ── Compiled parameter estimates for dc_montecarlo ──────────────────────
    # IVW-average the per-row estimates within each window to give starting
    # point values ready to pass as --pr_Q / --pr_R / --pr_n.
    print("\n=== Compiled prior estimates for dc_montecarlo ===")
    print("    (pass these directly as --pr_* arguments)\n")

    def _ivw(vals, sds):
        v = np.asarray(vals, float); s = np.asarray(sds, float)
        ok = np.isfinite(v) & np.isfinite(s) & (s > 0)
        if not ok.any(): return np.nan, np.nan
        w = 1.0 / s[ok] ** 2
        mean = float(np.sum(w * v[ok]) / np.sum(w))
        sem  = float(1.0 / np.sqrt(np.sum(w)))
        return mean, sem

    qtm_rows = out_df[out_df["in_qtm_window"]]
    if not qtm_rows.empty:
        Q_mean, Q_sem = _ivw(qtm_rows["Qu_dc"], qtm_rows["Qs_dc"])
        sQ_med = float(np.nanmedian(qtm_rows["Qs_dc"]))
        print(f"  Q  (IVW mean ± SEM): {Q_mean:.4f} ± {Q_sem:.4f}")
        print(f"  sQ (median spread):  {sQ_med:.4f}")
        print(f"\n  → --pr_Q {Q_mean:.6f} --pr_sQ {sQ_med:.6f}")
    else:
        print("  Q: no QTM-window rows")

    raman_rows = out_df[out_df["in_raman_window"]]
    if not raman_rows.empty and not qtm_rows.empty:
        # R and n are determined by the temperature slope of the Raman rate
        # across the Raman-window rows. We subtract the QTM contribution
        # (using the compiled Q) to isolate the Raman rate at each T.
        Q_compiled = Q_mean if np.isfinite(Q_mean) else np.nanmedian(qtm_rows["Qu_dc"])
        r_qtm = 10.0 ** (-Q_compiled)

        T_raman     = raman_rows["T"].values
        eln_raman   = raman_rows["elnTau_ln"].values
        r_total     = np.exp(-eln_raman)          # 1/e^<lntau> at each T
        r_raman     = r_total - r_qtm             # subtract QTM

        # only use rows where Raman rate is positive (QTM must not dominate fully)
        ok = r_raman > 0.05 * r_total             # Raman > 5% of total
        if ok.sum() >= 2:
            log10_T   = np.log10(T_raman[ok])
            log10_r   = np.log10(r_raman[ok])
            coeffs    = np.polyfit(log10_T, log10_r, 1)
            n_est     = float(coeffs[0])
            R_est     = float(coeffs[1])
            # residual spread as a proxy for uncertainty
            resid     = log10_r - np.polyval(coeffs, log10_T)
            sR_est    = float(np.std(resid)) if len(resid) > 2 else 0.5
            sN_est    = sR_est  # rough proxy — slope uncertainty

            print(f"\n  R  (slope fit, {ok.sum()} rows): {R_est:.4f}")
            print(f"  n  (slope fit, {ok.sum()} rows): {n_est:.4f}")
            print(f"  sR (residual spread):  {sR_est:.4f}")
            print(f"  sN (residual spread):  {sN_est:.4f}")
            print(f"\n  → --pr_R {R_est:.6f} --pr_sR {max(sR_est, 0.3):.6f} "
                  f"--pr_n {n_est:.6f} --pr_sN {max(sN_est, 0.2):.6f}")
            if ok.sum() < len(T_raman):
                skipped = T_raman[~ok]
                print(f"\n  NOTE: {(~ok).sum()} row(s) excluded (QTM-dominated): "
                      f"T={skipped} K")

            # write compiled R/n back into the Raman rows of the output CSV
            # so downstream scripts can use them
            out_df.loc[out_df["in_raman_window"], "Ru_dc"] = R_est
            out_df.loc[out_df["in_raman_window"], "Nu_dc"] = n_est
            out_df.loc[out_df["in_raman_window"], "Rs_dc"] = max(sR_est, 0.3)
            out_df.to_csv(outfile, index=False)   # rewrite with R/n filled in
            print(f"\n  Wrote R/n estimates back to {os.path.abspath(outfile)}")
        else:
            print(f"\n  Raman window: {len(raman_rows)} rows but fewer than 2 have "
                  f"resolvable Raman signal — R and n not estimable from DC data.")
            print(f"  All Raman-window rows are QTM-dominated. "
                  f"Use AC data for R and n.")
    elif not raman_rows.empty:
        print(f"\n  Raman window: {len(raman_rows)} rows — "
              f"no QTM window data to subtract, cannot estimate R/n.")
        print(f"  Run with --qtm_window as well to enable R/n estimation.")

    print()
    print("  NOTE: A and Ueff are not estimable from DC data.")
    print("  If you have AC data, pass --pr_A and --pr_Ueff from ac_phase2/ac_montecarlo.")

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

def _parse_window_arg(s):
    """Parse 'Tlo,Thi' string into (float, float) tuple, or None if empty."""
    if not s or str(s).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must be 'Tlo,Thi'")
    return (min(parts), max(parts))


def main():
    ap = argparse.ArgumentParser(
        description="Fit two-piece log-normal to SEF distributions (DC magnetometry phase 2)"
    )
    ap.add_argument("--infile",  default="dc_data.tsv",
                    help="TSV/CSV with columns: T, tau_star (or beta), beta (or tau_star)")
    ap.add_argument("--outfile", default="dc_phase2_out.csv",
                    help="Output CSV with fitted two-piece log-normal parameters")
    ap.add_argument("--plot",    action="store_true",
                    help="Show diagnostic overlay plots")

    ap.add_argument("--qtm_window", default="",
                    metavar="Tlo,Thi",
                    help="Temperature window (K) where QTM dominates, e.g. '2,9'. "
                         "Rows inside this window produce valid Q priors. "
                         "Rows outside get NaN Q columns. "
                         "Leave empty to mark all Q columns NaN.")

    ap.add_argument("--raman_window", default="",
                    metavar="Tlo,Thi",
                    help="Temperature window (K) where Raman contributes meaningfully, "
                         "e.g. '13,23'. Rows inside produce valid R/n priors. "
                         "Leave empty to mark all Raman columns NaN. "
                         "Note: Raman is rarely cleanly separable from DC data alone — "
                         "use cautiously and prefer AC data for R/n.")

    args = ap.parse_args()

    qtm_window   = _parse_window_arg(args.qtm_window)
    raman_window = _parse_window_arg(args.raman_window)

    process_file(args.infile, args.outfile,
                 make_plot=args.plot,
                 qtm_window=qtm_window,
                 raman_window=raman_window)


if __name__ == "__main__":
    main()