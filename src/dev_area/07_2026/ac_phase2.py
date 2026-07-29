#!/usr/bin/env python3
"""
ac_phase2.py  –  TAURnQ phase 2 for AC susceptibility data.

Combines orbach.py, raman2.py and qtm.py into a single script.
Each process is fitted only within its temperature window, set via
command-line arguments.  Any window left unset means that process is
skipped entirely.

For each active window the script:
  1. Fits the analytical moment model to (T, tau_mu, alpha) data
  2. Runs a robustness check (ρ=0 refit)
  3. Profiles ρ to get sd ranges at 95% χ²(1) threshold
  4. Writes fitted parameters to a CSV for use by montecarlo.py

Input TSV columns (no header):   T   tau_mu   alpha

Output CSV columns:
    process, mu_A, mu_U, sd_A, sd_U, rho_AU,   ← Orbach
             mu_R, mu_N, sd_R, sd_N, rho_RN,   ← Raman
             mu_Q, sd_Q                         ← QTM
    (columns for unused processes are NaN)

Usage:
    python ac_phase2.py --infile mydata.tsv \\
        --orbach_window 45,58 \\
        --raman_window  25,35 \\
        --qtm_window    2,10

    # Only Orbach and Raman (no QTM data):
    python ac_phase2.py --infile mydata.tsv \\
        --orbach_window 45,58 \\
        --raman_window  25,35
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse, os, sys

LN10 = np.log(10)


# ── Input loading ──────────────────────────────────────────────────────────

def load_data(path):
    """Load TSV/CSV with columns T, tau_mu, alpha (no header required)."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", header="infer")
        df.columns = [str(c).lower().strip() for c in df.columns]
        if {"t", "tau_mu", "alpha"}.issubset(df.columns):
            df = df[["t", "tau_mu", "alpha"]].rename(columns={"t": "T"})
        elif {"t", "tau_mean", "alpha"}.issubset(df.columns):
            df = df[["t", "tau_mean", "alpha"]].rename(
                columns={"t": "T", "tau_mean": "tau_mu"})
        else:
            raise ValueError("unrecognised header")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, :3]
        df.columns = ["T", "tau_mu", "alpha"]
    return df.astype(float)


def slice_window(df, window):
    lo, hi = window
    return df[(df["T"] >= lo) & (df["T"] <= hi)].reset_index(drop=True)


CL68 = 0.6826894921370859   # erf(1/sqrt(2)) — 68.27% confidence level


def g_from_alpha(alpha):
    """Exact 68.27% half-width of ln(tau/tau*) for the Generalised Debye
    distribution (Reta & Chilton style ±1σ bound), from the closed-form CDF:

        rho(s)  = (1/2π) sin(απ) / [cosh((1-α)s) - cos(απ)],  s = ln(τ/τ*)
        CDF(s)  = 1/2 + arctan[cot(απ/2) tanh((1-α)s/2)] / (π(1-α))
        s(q)    = (2/(1-α)) artanh[ tan(απ/2) tan(π(1-α)(q-1/2)) ]

    Replaces the former heuristic 1.82*sqrt(α)/(1-α), which matched neither
    the exact SD nor the exact 68% quantile (it overestimated the ±1σ band
    by 24% at α=0.1, 58% at α=0.05, 134% at α=0.02).  Returned in natural-log
    units, as before."""
    a = np.asarray(alpha, dtype=float)
    return (2.0 / (1.0 - a)) * np.arctanh(
        np.tan(a * np.pi / 2.0) * np.tan(np.pi * (1.0 - a) * CL68 / 2.0))


def parse_window(s):
    if not s or str(s).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must be 'Tlo,Thi'")
    return (min(parts), max(parts))


# ── Shared objective helper ────────────────────────────────────────────────

def _mse_objective(mu_pred, sd_pred, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd):
    """Normalised MSE used by Orbach (consistent with original orbach.py).

    The normalisers are floored: np.std of near-constant targets in a
    narrow window otherwise inflates that residual block without bound
    (`or 1.0` only catches an exact zero)."""
    s_mu = max(float(np.std(mu_ln_tgt)), 0.05)
    s_sd = max(float(np.std(sd_ln_tgt)),
               0.10 * float(np.mean(np.abs(sd_ln_tgt))), 1e-3)
    r1 = (mu_pred - mu_ln_tgt) / s_mu
    r2 = (sd_pred - sd_ln_tgt) / s_sd
    return w_mu * np.mean(r1 ** 2) + w_sd * np.mean(r2 ** 2)


def _sse_objective(mu_pred, sd_pred, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd):
    """Raw SSE used by Raman and QTM (consistent with originals).
    Normalisers floored — see _mse_objective."""
    s_mu = max(float(np.std(mu_ln_tgt)), 0.05)
    s_sd = max(float(np.std(sd_ln_tgt)),
               0.10 * float(np.mean(np.abs(sd_ln_tgt))), 1e-3)
    r1 = (mu_pred - mu_ln_tgt) / s_mu
    r2 = (sd_pred - sd_ln_tgt) / s_sd
    return w_mu * np.sum(r1 ** 2) + w_sd * np.sum(r2 ** 2)


def _profile_rho(objective_fn, theta_free, T, mu_tgt, sd_tgt,
                 rho_grid, loss_free, threshold, param_names):
    """
    Profile ρ over rho_grid, keeping fits within threshold of loss_free.
    Returns list of (rho, loss, *fitted_sd_params).
    param_names : names of the sd parameters (for reporting).
    """
    keep = []
    n_sd = len(param_names)
    for rho_fixed in rho_grid:
        th0 = theta_free.copy()
        th0[-1] = rho_fixed
        n = len(th0)
        # bounds: free for means, non-negative for sds, fixed for rho
        bnds = [(None, None)] * (n - n_sd - 1) + \
               [(0.0, None)]  * n_sd + \
               [(rho_fixed, rho_fixed)]
        res = minimize(objective_fn, th0,
                       args=(T, mu_tgt, sd_tgt, 1.0, 1.0),
                       method="L-BFGS-B", bounds=bnds,
                       options=dict(maxiter=5000, ftol=1e-12))
        L = float(res.fun)
        if (L - loss_free) <= threshold and res.success:
            sd_vals = res.x[-(n_sd + 1):-1]   # the sd params before rho
            keep.append((rho_fixed, L, *sd_vals))
    return keep


def _window_identifiability(x_pred, label, verbose=True):
    """Warn when the predictor span is too narrow to separate (sd_a, sd_b, ρ).

    The width model  var(x) = sd_a² + x²·sd_b² + 2x·ρ·sd_a·sd_b  evaluated
    over a narrow span of x is effectively ONE number constraining THREE
    parameters — a 2-D degenerate ridge.  The (sd_a, sd_b, ρ) point estimate
    is then an arbitrary point on that ridge (typically with ρ pinned at a
    bound); only the effective width near the window centre is identifiable.
    Returns the relative predictor span."""
    x = np.asarray(x_pred, float)
    span_rel = (x.max() - x.min()) / max(abs(float(x.mean())), 1e-12)
    if verbose and span_rel < 0.15:
        print(f"  {label} WARNING: predictor span is only {100*span_rel:.1f}% of "
              f"its mean — (sd, sd, ρ) are NOT separately identifiable in this "
              f"window. Trust sd_eff (window-centre width), not the individual "
              f"sd/ρ values; decompose only in the global stage.")
    return span_rel


# ═══════════════════════════════════════════════════════════════════════════
# ORBACH
# ═══════════════════════════════════════════════════════════════════════════

def _orbach_moments(T, mu_A, mu_U, sd_A, sd_U, rho):
    mu_ln  = LN10 * mu_A + mu_U / T
    var_ln = ((LN10 ** 2) * sd_A ** 2
              + sd_U ** 2 / T ** 2
              + 2 * (LN10 / T) * rho * sd_A * sd_U)
    return mu_ln, np.sqrt(np.maximum(var_ln, 1e-18))


def _orbach_objective(theta, T, mu_ln_tgt, sd_ln_tgt, w_mu=1.0, w_sd=1.0):
    mu_A, mu_U, sd_A, sd_U, rho = theta
    mu_p, sd_p = _orbach_moments(T, mu_A, mu_U, sd_A, sd_U, rho)
    return _mse_objective(mu_p, sd_p, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd)


def fit_orbach(df_window, w_mu=1.0, w_sd=1.0, verbose=True):
    T         = df_window["T"].to_numpy(float)
    mu_ln_tgt = np.log(df_window["tau_mu"].to_numpy(float))
    sd_ln_tgt = g_from_alpha(df_window["alpha"].to_numpy(float))

    # Initial guess: linear regression of mu_ln on 1/T
    x = 1.0 / T
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, mu_ln_tgt, rcond=None)[0]
    gbar  = float(np.mean(sd_ln_tgt))
    th0   = np.array([a / LN10, b,
                      max(gbar / LN10 * 0.5, 1e-5),
                      max(gbar * float(np.mean(T)) * 0.1, 1e-3),
                      0.5])    # warm-start rho at +0.5 (Orbach must be positive)
    # rho_AU must be >= 0: higher barrier compensated by larger prefactor
    bnds  = [(None,None),(None,None),(0.0,None),(0.0,None),(0.0,0.999)]

    res = minimize(_orbach_objective, th0,
                   args=(T, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd),
                   method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=10000, ftol=1e-12))
    mu_A, mu_U, sd_A, sd_U, rho = res.x

    if verbose:
        print(f"  Orbach  success={res.success} | A={mu_A:.5f} U={mu_U:.3f} "
              f"sdA={sd_A:.4f} sdU={sd_U:.3f} ρ={rho:.3f}")

    # robustness: ρ=0
    th_rho0       = res.x.copy(); th_rho0[-1] = 0.0
    bnds_rho0     = [(None,None),(None,None),(0.0,None),(0.0,None),(0.0,0.0)]
    res0          = minimize(_orbach_objective, th_rho0,
                             args=(T, mu_ln_tgt, sd_ln_tgt, 1.0, 1.0),
                             method="L-BFGS-B", bounds=bnds_rho0,
                             options=dict(maxiter=10000, ftol=1e-12))
    loss_free = _orbach_objective(res.x, T, mu_ln_tgt, sd_ln_tgt, 1.0, 1.0)
    dloss     = float(res0.fun) - loss_free
    if verbose:
        print(f"  Orbach  Δloss(ρ=0) = {dloss:.4g}  "
              f"({100*dloss/max(loss_free,1e-12):.1f}%)")

    # ρ profile  (positive correlation expected for Orbach)
    M         = len(T)
    thr       = 3.84 / max(2 * M, 1)        # MSE-scale χ²(1) threshold
    rho_grid  = np.linspace(0.0, 0.9, 19)
    keep      = _profile_rho(_orbach_objective, res.x,
                              T, mu_ln_tgt, sd_ln_tgt,
                              rho_grid, loss_free, thr,
                              param_names=["sd_A", "sd_U"])
    if verbose and keep:
        sdA_r = [k[2] for k in keep]; sdU_r = [k[3] for k in keep]
        print(f"  Orbach  ρ range [{min(k[0] for k in keep):+.2f}, "
              f"{max(k[0] for k in keep):+.2f}]  "
              f"sdA [{min(sdA_r):.3g},{max(sdA_r):.3g}]  "
              f"sdU [{min(sdU_r):.3g},{max(sdU_r):.3g}]")

    _window_identifiability(1.0 / T, "Orbach ", verbose=verbose)
    sd_eff = float(_orbach_moments(np.array([float(np.mean(T))]),
                                   mu_A, mu_U, sd_A, sd_U, rho)[1][0])

    return dict(mu_A=mu_A, mu_U=mu_U, sd_A=sd_A, sd_U=sd_U, rho_AU=rho,
                sd_eff=sd_eff,
                loss=float(res.fun), n_rows=len(T),
                sd_A_lo=min([k[2] for k in keep], default=np.nan),
                sd_A_hi=max([k[2] for k in keep], default=np.nan),
                sd_U_lo=min([k[3] for k in keep], default=np.nan),
                sd_U_hi=max([k[3] for k in keep], default=np.nan))


# ═══════════════════════════════════════════════════════════════════════════
# RAMAN
# ═══════════════════════════════════════════════════════════════════════════

def _raman_moments(T, mu_R, mu_N, sd_R, sd_N, rho):
    t10   = np.log10(T)
    mu_L  = mu_R + t10 * mu_N
    var_L = (sd_R ** 2
             + (t10 ** 2) * sd_N ** 2
             + 2 * t10 * rho * sd_R * sd_N)
    mu_ln  = -LN10 * mu_L
    sd_ln  = LN10 * np.sqrt(np.maximum(var_L, 1e-18))
    return mu_ln, sd_ln


def _raman_objective(theta, T, mu_ln_tgt, sd_ln_tgt, w_mu=1.0, w_sd=1.0):
    mu_R, mu_N, sd_R, sd_N, rho = theta
    mu_p, sd_p = _raman_moments(T, mu_R, mu_N, sd_R, sd_N, rho)
    return _sse_objective(mu_p, sd_p, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd)


def fit_raman(df_window, w_mu=1.0, w_sd=1.0, verbose=True):
    T         = df_window["T"].to_numpy(float)
    mu_ln_tgt = np.log(df_window["tau_mu"].to_numpy(float))
    sd_ln_tgt = g_from_alpha(df_window["alpha"].to_numpy(float))

    t10     = np.log10(T)
    mu_L_tgt = -np.log10(np.exp(mu_ln_tgt))
    A        = np.vstack([np.ones_like(t10), t10]).T
    mu_R0, mu_N0 = np.linalg.lstsq(A, mu_L_tgt, rcond=None)[0]
    gbar     = float(np.mean(sd_ln_tgt))
    th0      = np.array([mu_R0, mu_N0,
                         max(gbar / LN10 * 0.5, 1e-5),
                         max(gbar / LN10 / (np.mean(np.abs(t10)) + 1e-6) * 0.5, 1e-5),
                         -0.1])
    bnds     = [(None,None),(None,None),(0.0,None),(0.0,None),(-0.999,0.999)]

    res = minimize(_raman_objective, th0,
                   args=(T, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd),
                   method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=10000, ftol=1e-12))
    mu_R, mu_N, sd_R, sd_N, rho = res.x

    if verbose:
        print(f"  Raman   success={res.success} | R={mu_R:.5f} N={mu_N:.4f} "
              f"sdR={sd_R:.4f} sdN={sd_N:.4f} ρ={rho:.3f}")

    # robustness: ρ=0
    th_rho0   = res.x.copy(); th_rho0[-1] = 0.0
    bnds_rho0 = [(None,None),(None,None),(0.0,None),(0.0,None),(0.0,0.0)]
    res0      = minimize(_raman_objective, th_rho0,
                         args=(T, mu_ln_tgt, sd_ln_tgt, 1.0, 1.0),
                         method="L-BFGS-B", bounds=bnds_rho0,
                         options=dict(maxiter=10000, ftol=1e-12))
    loss_free = _raman_objective(res.x, T, mu_ln_tgt, sd_ln_tgt, 1.0, 1.0)
    dloss     = float(res0.fun) - loss_free
    if verbose:
        print(f"  Raman   Δloss(ρ=0) = {dloss:.4g}  "
              f"({100*dloss/max(loss_free,1e-12):.1f}%)")

    # ρ profile  (negative correlation expected for Raman)
    thr      = 3.84                           # SSE-scale χ²(1) threshold
    rho_grid = np.linspace(-0.999, 0.0, 50)
    keep     = _profile_rho(_raman_objective, res.x,
                             T, mu_ln_tgt, sd_ln_tgt,
                             rho_grid, loss_free, thr,
                             param_names=["sd_R", "sd_N"])
    if verbose and keep:
        sdR_r = [k[2] for k in keep]; sdN_r = [k[3] for k in keep]
        print(f"  Raman   ρ range [{min(k[0] for k in keep):+.2f}, "
              f"{max(k[0] for k in keep):+.2f}]  "
              f"sdR [{min(sdR_r):.3g},{max(sdR_r):.3g}]  "
              f"sdN [{min(sdN_r):.3g},{max(sdN_r):.3g}]")

    _window_identifiability(np.log10(T), "Raman  ", verbose=verbose)
    sd_eff = float(_raman_moments(np.array([float(np.mean(T))]),
                                  mu_R, mu_N, sd_R, sd_N, rho)[1][0])

    return dict(mu_R=mu_R, mu_N=mu_N, sd_R=sd_R, sd_N=sd_N, rho_RN=rho,
                sd_eff=sd_eff,
                loss=float(res.fun), n_rows=len(T),
                sd_R_lo=min([k[2] for k in keep], default=np.nan),
                sd_R_hi=max([k[2] for k in keep], default=np.nan),
                sd_N_lo=min([k[3] for k in keep], default=np.nan),
                sd_N_hi=max([k[3] for k in keep], default=np.nan))


# ═══════════════════════════════════════════════════════════════════════════
# QTM
# ═══════════════════════════════════════════════════════════════════════════

def _qtm_moments(T, mu_Q, sd_Q):
    mu_ln = np.full(len(T), LN10 * mu_Q)
    sd_ln = np.full(len(T), LN10 * sd_Q)
    return mu_ln, sd_ln


def _qtm_objective(theta, T, mu_ln_tgt, sd_ln_tgt, w_mu=1.0, w_sd=1.0):
    mu_Q, sd_Q = theta
    mu_p, sd_p = _qtm_moments(T, mu_Q, sd_Q)
    return _sse_objective(mu_p, sd_p, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd)


def fit_qtm(df_window, w_mu=1.0, w_sd=1.0, verbose=True):
    T         = df_window["T"].to_numpy(float)
    mu_ln_tgt = np.log(df_window["tau_mu"].to_numpy(float))
    sd_ln_tgt = g_from_alpha(df_window["alpha"].to_numpy(float))

    th0  = np.array([float(np.mean(mu_ln_tgt)) / LN10,
                     max(float(np.mean(sd_ln_tgt)) / LN10, 1e-6)])
    bnds = [(None, None), (0.0, None)]

    res = minimize(_qtm_objective, th0,
                   args=(T, mu_ln_tgt, sd_ln_tgt, w_mu, w_sd),
                   method="L-BFGS-B", bounds=bnds,
                   options=dict(maxiter=10000, ftol=1e-12))
    mu_Q, sd_Q = res.x

    if verbose:
        print(f"  QTM     success={res.success} | Q={mu_Q:.5f} sdQ={sd_Q:.5f}")

    # QTM has no ρ — just a 2-param model, no profile needed
    return dict(mu_Q=mu_Q, sd_Q=sd_Q,
                loss=float(res.fun), n_rows=len(T))


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic plot
# ═══════════════════════════════════════════════════════════════════════════

def _plot_process(T, mu_ln_tgt, sd_ln_tgt, mu_pred, sd_pred, label, window):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (tgt, pred, title) in zip(axes, [
        (mu_ln_tgt, mu_pred, "μ_ln(τ)"),
        (sd_ln_tgt, sd_pred, "σ_ln(τ)")
    ]):
        ax.scatter(T, tgt,  color="#378ADD", zorder=3, label="target")
        ax.plot(T,    pred, color="#D85A30", linewidth=2, label="model")
        ax.set_xlabel("T (K)"); ax.set_ylabel(title)
        ax.legend(fontsize=9)
    fig.suptitle(f"{label}  ({window[0]:.1f}–{window[1]:.1f} K)", fontsize=11)
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ AC phase 2 — fit Orbach / Raman / QTM moments "
                    "within user-defined temperature windows"
    )
    ap.add_argument("--infile",  required=True,
                    help="TSV/CSV  (T, tau_mu, alpha)")
    ap.add_argument("--outfile", default="ac_phase2_out.csv",
                    help="Output CSV with per-process fitted parameters")
    ap.add_argument("--plot",    action="store_true",
                    help="Show diagnostic μ/σ plots for each active process")

    ap.add_argument("--orbach_window", default="", metavar="Tlo,Thi",
                    help="Temperature window for Orbach (A, Ueff)")
    ap.add_argument("--raman_window",  default="", metavar="Tlo,Thi",
                    help="Temperature window for Raman  (R, n)")
    ap.add_argument("--qtm_window",    default="", metavar="Tlo,Thi",
                    help="Temperature window for QTM    (Q)")

    ap.add_argument("--w_mu", type=float, default=1.0,
                    help="Weight on μ residuals (default 1.0)")
    ap.add_argument("--w_sd", type=float, default=1.0,
                    help="Weight on σ residuals (default 1.0)")

    args = ap.parse_args()

    orbach_w = parse_window(args.orbach_window)
    raman_w  = parse_window(args.raman_window)
    qtm_w    = parse_window(args.qtm_window)

    if not any([orbach_w, raman_w, qtm_w]):
        print("ERROR: at least one of --orbach_window, --raman_window, "
              "--qtm_window must be set.", file=sys.stderr)
        sys.exit(1)

    df = load_data(args.infile)
    print(f"Loaded {len(df)} rows from {args.infile}  "
          f"(T range {df['T'].min():.1f}–{df['T'].max():.1f} K)\n")

    results = {}

    # ── Orbach ──────────────────────────────────────────────────────────────
    if orbach_w:
        dfw = slice_window(df, orbach_w)
        if len(dfw) < 2:
            print(f"WARNING: Orbach window {orbach_w} contains "
                  f"{len(dfw)} rows — skipping.", file=sys.stderr)
        else:
            print(f"Orbach window {orbach_w[0]:.1f}–{orbach_w[1]:.1f} K  "
                  f"({len(dfw)} rows)")
            r = fit_orbach(dfw, args.w_mu, args.w_sd, verbose=True)
            results["orbach"] = r
            if args.plot:
                mu_p, sd_p = _orbach_moments(
                    dfw["T"].values, r["mu_A"], r["mu_U"],
                    r["sd_A"], r["sd_U"], r["rho_AU"])
                _plot_process(dfw["T"].values,
                              np.log(dfw["tau_mu"].values),
                              g_from_alpha(dfw["alpha"].values),
                              mu_p, sd_p, "Orbach", orbach_w)
        print()

    # ── Raman ───────────────────────────────────────────────────────────────
    if raman_w:
        dfw = slice_window(df, raman_w)
        if len(dfw) < 2:
            print(f"WARNING: Raman window {raman_w} contains "
                  f"{len(dfw)} rows — skipping.", file=sys.stderr)
        else:
            print(f"Raman window {raman_w[0]:.1f}–{raman_w[1]:.1f} K  "
                  f"({len(dfw)} rows)")
            r = fit_raman(dfw, args.w_mu, args.w_sd, verbose=True)
            results["raman"] = r
            if args.plot:
                mu_p, sd_p = _raman_moments(
                    dfw["T"].values, r["mu_R"], r["mu_N"],
                    r["sd_R"], r["sd_N"], r["rho_RN"])
                _plot_process(dfw["T"].values,
                              np.log(dfw["tau_mu"].values),
                              g_from_alpha(dfw["alpha"].values),
                              mu_p, sd_p, "Raman", raman_w)
        print()

    # ── QTM ─────────────────────────────────────────────────────────────────
    if qtm_w:
        dfw = slice_window(df, qtm_w)
        if len(dfw) < 2:
            print(f"WARNING: QTM window {qtm_w} contains "
                  f"{len(dfw)} rows — skipping.", file=sys.stderr)
        else:
            print(f"QTM window {qtm_w[0]:.1f}–{qtm_w[1]:.1f} K  "
                  f"({len(dfw)} rows)")
            r = fit_qtm(dfw, args.w_mu, args.w_sd, verbose=True)
            results["qtm"] = r
        print()

    if not results:
        print("No processes were successfully fitted.", file=sys.stderr)
        sys.exit(1)

    # ── Build output row ────────────────────────────────────────────────────
    orb = results.get("orbach", {})
    ram = results.get("raman",  {})
    qtm = results.get("qtm",    {})

    row = {
        # Orbach
        "mu_A":    orb.get("mu_A",   np.nan),
        "mu_U":    orb.get("mu_U",   np.nan),
        "sd_A":    orb.get("sd_A",   np.nan),
        "sd_U":    orb.get("sd_U",   np.nan),
        "rho_AU":  orb.get("rho_AU", np.nan),
        "sd_A_lo": orb.get("sd_A_lo",np.nan),
        "sd_A_hi": orb.get("sd_A_hi",np.nan),
        "sd_U_lo": orb.get("sd_U_lo",np.nan),
        "sd_U_hi": orb.get("sd_U_hi",np.nan),
        "sd_eff_orbach": orb.get("sd_eff", np.nan),   # identifiable width (ln units)
        # Raman
        "mu_R":    ram.get("mu_R",   np.nan),
        "mu_N":    ram.get("mu_N",   np.nan),
        "sd_R":    ram.get("sd_R",   np.nan),
        "sd_N":    ram.get("sd_N",   np.nan),
        "rho_RN":  ram.get("rho_RN", np.nan),
        "sd_R_lo": ram.get("sd_R_lo",np.nan),
        "sd_R_hi": ram.get("sd_R_hi",np.nan),
        "sd_N_lo": ram.get("sd_N_lo",np.nan),
        "sd_N_hi": ram.get("sd_N_hi",np.nan),
        "sd_eff_raman": ram.get("sd_eff", np.nan),    # identifiable width (ln units)
        # QTM
        "mu_Q":    qtm.get("mu_Q",  np.nan),
        "sd_Q":    qtm.get("sd_Q",  np.nan),
        # metadata — use dash separator to avoid unquoted commas in CSV
        "orbach_window": f"{orbach_w[0]}-{orbach_w[1]}" if orbach_w else "",
        "raman_window":  f"{raman_w[0]}-{raman_w[1]}"  if raman_w  else "",
        "qtm_window":    f"{qtm_w[0]}-{qtm_w[1]}"      if qtm_w    else "",
        "infile":        args.infile,
    }

    out_df = pd.DataFrame([row])
    out_df.to_csv(args.outfile, index=False)
    print(f"Wrote results → {os.path.abspath(args.outfile)}")
    print()
    print("=== Summary ===")
    if orb: print(f"  Orbach  A={orb['mu_A']:.5f}  Ueff={orb['mu_U']:.2f}  "
                  f"sdA={orb['sd_A']:.4f}  sdU={orb['sd_U']:.3f}  ρ={orb['rho_AU']:.3f}")
    if ram: print(f"  Raman   R={ram['mu_R']:.5f}  n={ram['mu_N']:.4f}  "
                  f"sdR={ram['sd_R']:.4f}  sdN={ram['sd_N']:.4f}  ρ={ram['rho_RN']:.3f}")
    if qtm: print(f"  QTM     Q={qtm['mu_Q']:.5f}  sdQ={qtm['sd_Q']:.5f}")


if __name__ == "__main__":
    main()