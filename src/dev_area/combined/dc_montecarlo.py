#!/usr/bin/env python3
"""
dc_montecarlo.py  –  TAURnQ Monte Carlo phase for DC magnetometry data.

Window-aware version with optional AC priors and reduced rate models.

BEHAVIOUR
---------
Each row is dispatched based on (in_qtm_window, in_raman_window) flags
from dc_phase2.py, and on which AC priors the user has provided.

  QTM window rows:
    - AC Orbach + Raman priors given  → fit Q, sQ only (full model, 4 terms fixed)
    - AC Orbach prior only            → fit Q, sQ only (Orbach + QTM model)
    - AC Raman prior only             → fit Q, sQ only (Raman + QTM model)
    - No AC priors                    → fit Q, sQ only (QTM-only model: τ⁻¹ = 10^-Q)

  Raman window rows:
    - AC Orbach + Q priors given      → fit R, n, sR, sN (full model)
    - AC Orbach prior only            → fit R, n, sR, sN (Orbach + Raman model)
    - AC Q prior only                 → fit R, n, sR, sN (Raman + QTM model)
    - No AC priors                    → fit R, n, sR, sN (Raman-only model)

  Rows outside both windows are skipped.
  A and Ueff are NEVER fitted from DC data regardless of what is provided.

AC priors are all optional. Missing ones simply drop that term from the
rate model for rows where it would have been fixed.

Input:  dc_phase2.py output CSV
Output: same schema as montecarlo.py (Au/Uu/As/Us always NaN)
        plus window_type, fitted_params, active_terms columns

Usage (with AC priors):
    python dc_montecarlo.py --infile dc_phase2_out.csv \\
        --ac_A -12.10 --ac_Ueff 1842 --ac_sA 0.11 --ac_sU 14 \\
        --ac_R -7.34  --ac_n 3.84    --ac_sR 0.34 --ac_sN 0.22 \\
        --ac_Q  2.424 --ac_sQ 0.045

Usage (DC-only, no AC data):
    python dc_montecarlo.py --infile dc_phase2_out.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os, sys

PATH = "dc_phase2_out.csv"


# ---------------------------------------------------------------------------
# Rate model — full and reduced versions
# ---------------------------------------------------------------------------

def rate_model_full(T, A, Ueff, R, n, Q):
    """Standard three-term model."""
    T_safe = np.maximum(T, 1e-300)
    return (10.0 ** (-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
            + 10.0 ** R * T_safe ** n
            + 10.0 ** (-Q))


def rate_model_active(T, A=None, Ueff=None, R=None, n=None, Q=None):
    """
    Reduced rate model — only includes terms whose parameters are not None.
    This prevents insensitive parameters from polluting the fit when no
    AC priors are available to anchor them.
    """
    T_safe = np.maximum(T, 1e-300)
    rate = np.zeros_like(np.asarray(T_safe, dtype=float))
    if A is not None and Ueff is not None:
        rate = rate + 10.0 ** (-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    if R is not None and n is not None:
        rate = rate + 10.0 ** R * T_safe ** n
    if Q is not None:
        rate = rate + 10.0 ** (-Q)
    return rate


# ---------------------------------------------------------------------------
# Two-piece log-normal quantile function
# ---------------------------------------------------------------------------

def tplognormal_quantiles(mu_ln, sigma1_ln, sigma2_ln, qs):
    """Closed-form quantiles of a two-piece normal in ln(tau) space."""
    s1, s2 = sigma1_ln, sigma2_ln
    p_left = s1 / (s1 + s2)
    qs = np.asarray(qs, dtype=float)
    result = np.empty_like(qs)
    for i, q in enumerate(qs):
        if q <= p_left:
            z = norm.ppf(q * (s1 + s2) / (2.0 * s1))
            result[i] = mu_ln + z * s1
        else:
            z = norm.ppf(1.0 - (1.0 - q) * (s1 + s2) / (2.0 * s2))
            result[i] = mu_ln + z * s2
    return result


# ---------------------------------------------------------------------------
# Correlated draws — used only when AC priors are present (full model)
# Scalar version — used for reduced model (no cross-param correlations)
# ---------------------------------------------------------------------------

def draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z):
    A, U, R, n, Q = mu
    sA, sU, sR, sN, sQ = sigmas
    L_AU = np.array([[sA, 0.0],
                     [rho_AU * sU, sU * np.sqrt(max(1.0 - rho_AU ** 2, 1e-12))]])
    L_RN = np.array([[sR, 0.0],
                     [rho_RN * sN, sN * np.sqrt(max(1.0 - rho_RN ** 2, 1e-12))]])
    eps_AU = Z[:, :2] @ L_AU.T
    eps_RN = Z[:, 2:4] @ L_RN.T
    eps_Q  = Z[:, 4] * sQ
    return np.column_stack([A + eps_AU[:, 0], U + eps_AU[:, 1],
                             R + eps_RN[:, 0], n + eps_RN[:, 1],
                             Q + eps_Q])


def simulate_quantiles_reduced(T, free_param_draws, fixed_params, qs):
    """
    Simulate ln(tau) quantiles for a reduced model.

    free_param_draws : dict mapping param name to 1-D array of K samples
                       keys from {'Q', 'R', 'n'}
    fixed_params     : dict mapping param name to scalar fixed value
                       keys from {'A', 'Ueff', 'R', 'n', 'Q'}
    """
    K = next(iter(free_param_draws.values())).shape[0]

    def _get(name):
        if name in free_param_draws:
            return free_param_draws[name]
        if name in fixed_params:
            return np.full(K, fixed_params[name])
        return None

    A_s    = _get('A')
    Ueff_s = _get('Ueff')
    R_s    = _get('R')
    n_s    = _get('n')
    Q_s    = _get('Q')

    r = rate_model_active(T, A=A_s, Ueff=Ueff_s, R=R_s, n=n_s, Q=Q_s)
    tau = 1.0 / np.maximum(r, 1e-300)
    return np.quantile(np.log(tau), qs)


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

def penalty_Q(Q, sQ):
    pen = 0.0
    if Q < -20.0: pen += 1e-3 * (-20.0 - Q) ** 2
    if Q >  10.0: pen += 1e-3 * (Q - 10.0) ** 2
    pen += 1e-5 * sQ ** 2
    return pen


def penalty_RN(R, n, sR, sN):
    pen = 0.0
    if R < -20.0: pen += 1e-3 * (-20.0 - R) ** 2
    if R >  10.0: pen += 1e-3 * (R - 10.0) ** 2
    if n <   0.0: pen += 1e-3 * (0.0 - n) ** 2
    if n >  12.0: pen += 1e-3 * (n - 12.0) ** 2
    pen += 1e-5 * (sR ** 2 + sN ** 2)
    return pen


# ---------------------------------------------------------------------------
# QTM-window fit
# ---------------------------------------------------------------------------

def fit_qtm_row(T, mu_ln, sigma1_ln, sigma2_ln, ac, Z, qs):
    """
    Fit Q and sQ.  All other terms are either fixed from AC priors or dropped.

    ac : dict with optional keys A, Ueff, sA, sU, R, n, sR, sN, Q, sQ
         Missing keys mean that term is absent from the rate model.
    """
    target_lnq = tplognormal_quantiles(mu_ln, sigma1_ln, sigma2_ln, qs)

    # decide which terms appear in the model
    has_orbach = ('A' in ac and 'Ueff' in ac)
    has_raman  = ('R' in ac and 'n'    in ac)
    has_Q_init = 'Q' in ac

    Q_init  = ac.get('Q',  2.0)
    sQ_init = ac.get('sQ', 0.3)

    fixed = {}
    if has_orbach: fixed.update({'A': ac['A'], 'Ueff': ac['Ueff']})
    if has_raman:  fixed.update({'R': ac['R'], 'n':    ac['n']})

    active_terms = (['orbach'] if has_orbach else []) + \
                   (['raman']  if has_raman  else []) + ['qtm']

    # build free-param draw function — Q is the only free parameter
    sQ_arr = np.ones(Z.shape[0])  # placeholder, scaled in objective

    x0 = np.array([Q_init, np.log(max(sQ_init, 1e-4))], dtype=float)

    def objective(x):
        Q_try  = x[0]
        sQ_try = np.exp(x[1])
        Q_draws = Q_try + Z[:, 4] * sQ_try
        lnq_hat = simulate_quantiles_reduced(
            T,
            free_param_draws={'Q': Q_draws},
            fixed_params=fixed,
            qs=qs
        )
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid)) + penalty_Q(Q_try, sQ_try)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    Q_hat  = res.x[0]
    sQ_hat = np.exp(res.x[1])

    return {
        'Q': Q_hat, 'sQ': sQ_hat,
        'R': ac.get('R', np.nan), 'sR': ac.get('sR', np.nan),
        'n': ac.get('n', np.nan), 'sN': ac.get('sN', np.nan),
        'loss': float(res.fun),
        'wtype': 'qtm',
        'active_terms': '+'.join(active_terms),
        'fitted_params': 'Q,sQ',
    }


# ---------------------------------------------------------------------------
# Raman-window fit
# ---------------------------------------------------------------------------

def fit_raman_row(T, mu_ln, sigma1_ln, sigma2_ln, ac, Z, qs):
    """
    Fit R, n, sR, sN.  All other terms fixed from AC or dropped.
    """
    target_lnq = tplognormal_quantiles(mu_ln, sigma1_ln, sigma2_ln, qs)

    has_orbach = ('A' in ac and 'Ueff' in ac)
    has_Q      = ('Q' in ac)

    R_init  = ac.get('R',  -6.0)
    n_init  = ac.get('n',   4.0)
    sR_init = ac.get('sR',  0.5)
    sN_init = ac.get('sN',  0.5)

    fixed = {}
    if has_orbach: fixed.update({'A': ac['A'], 'Ueff': ac['Ueff']})
    if has_Q:      fixed.update({'Q': ac['Q']})

    active_terms = (['orbach'] if has_orbach else []) + \
                   ['raman'] + (['qtm'] if has_Q else [])

    x0 = np.array([R_init, n_init,
                   np.log(max(sR_init, 1e-4)),
                   np.log(max(sN_init, 1e-4))], dtype=float)

    def objective(x):
        R_try  = x[0];  n_try  = x[1]
        sR_try = np.exp(x[2]); sN_try = np.exp(x[3])
        R_draws = R_try + Z[:, 2] * sR_try
        n_draws = n_try + Z[:, 3] * sN_try
        lnq_hat = simulate_quantiles_reduced(
            T,
            free_param_draws={'R': R_draws, 'n': n_draws},
            fixed_params=fixed,
            qs=qs
        )
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid)) + penalty_RN(R_try, n_try, sR_try, sN_try)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    R_hat  = res.x[0];  n_hat  = res.x[1]
    sR_hat = np.exp(res.x[2]); sN_hat = np.exp(res.x[3])

    return {
        'Q': ac.get('Q', np.nan), 'sQ': ac.get('sQ', np.nan),
        'R': R_hat, 'sR': sR_hat,
        'n': n_hat, 'sN': sN_hat,
        'loss': float(res.fun),
        'wtype': 'raman',
        'active_terms': '+'.join(active_terms),
        'fitted_params': 'R,n,sR,sN',
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def fit_one_row(df, idx, out_path, ac, seed_base=12345, make_plot=False):
    """
    ac : dict with any subset of keys:
         A, Ueff, sA, sU, R, n, sR, sN, Q, sQ
         Missing keys = that prior is unavailable = drop term from model.
    """
    row       = df.loc[idx]
    T         = float(row["T"])
    mu_ln     = float(row["mu_ln"])
    sigma1_ln = float(row["sigma1_ln"])
    sigma2_ln = float(row["sigma2_ln"])
    in_qtm    = bool(row.get("in_qtm_window",   False))
    in_raman  = bool(row.get("in_raman_window", False))

    if not in_qtm and not in_raman:
        print(f"[idx={idx:>3}] T={T:6.2f} K  SKIP — outside all windows")
        return

    qs  = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98], dtype=float)
    rng = np.random.default_rng(int(seed_base) + int(idx))
    Z   = rng.standard_normal(size=(25000, 5))

    if in_qtm:
        result = fit_qtm_row(T, mu_ln, sigma1_ln, sigma2_ln, ac, Z, qs)
    else:
        result = fit_raman_row(T, mu_ln, sigma1_ln, sigma2_ln, ac, Z, qs)

    print(f"[idx={idx:>3}] T={T:6.2f} K [{result['wtype']:>5}] "
          f"active={result['active_terms']} | loss={result['loss']:.5g} | "
          f"R={result['R']:.4f} n={result['n']:.4f} Q={result['Q']:.4f} | "
          f"sR={result['sR']:.3f} sN={result['sN']:.3f} sQ={result['sQ']:.3f}")

    row_out = pd.DataFrame([{
        "T":   T,
        "Au":  np.nan,        "Uu":  np.nan,    # never from DC
        "Ru":  result['R'],   "Nu":  result['n'],   "Qu":  result['Q'],
        "As":  np.nan,        "Us":  np.nan,    # never from DC
        "Rs":  result['sR'],  "Ns":  result['sN'],  "Qs":  result['sQ'],
        "window_type":   result['wtype'],
        "fitted_params": result['fitted_params'],
        "active_terms":  result['active_terms'],
    }])
    header_needed = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    row_out.to_csv(out_path, index=False, mode="a", header=header_needed)

    if make_plot:
        _plot_fit(T, mu_ln, sigma1_ln, sigma2_ln, result, idx)


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def _plot_fit(T, mu_ln, s1, s2, result, idx):
    rng    = np.random.default_rng(54321 + int(idx))
    Z_plot = rng.standard_normal((60000, 5))

    Q_draws = result['Q'] + Z_plot[:, 4] * result['sQ']
    R_draws = (result['R'] + Z_plot[:, 2] * result['sR']
               if not np.isnan(result['R']) else None)
    n_draws = (result['n'] + Z_plot[:, 3] * result['sN']
               if not np.isnan(result['n']) else None)

    fp = {'Q': Q_draws}
    if R_draws is not None: fp['R'] = R_draws
    if n_draws is not None: fp['n'] = n_draws

    r      = simulate_quantiles_reduced.__wrapped__ if hasattr(simulate_quantiles_reduced, '__wrapped__') else None
    # just compute the rate directly for plotting
    rate = rate_model_active(T, Q=Q_draws,
                             R=R_draws, n=n_draws)
    tau_mc = 1.0 / np.maximum(rate, 1e-300)

    ln_grid = np.linspace(mu_ln - 8, mu_ln + 8, 800)
    norm_f  = (s1 + s2) / 2.0
    target  = np.where(
        ln_grid <= mu_ln,
        np.exp(-0.5 * ((ln_grid - mu_ln) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi)),
        np.exp(-0.5 * ((ln_grid - mu_ln) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
    ) * norm_f

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log(tau_mc), bins=np.linspace(ln_grid[0], ln_grid[-1], 120),
            density=True, alpha=0.45, color="#378ADD", edgecolor="none",
            label="MC ln(τ) samples")
    ax.plot(ln_grid, target, color="#D85A30", linewidth=2, label="TPN target")
    ax.axvline(mu_ln, color="#639922", linewidth=1.2, linestyle=":")
    ax.set_xlabel("ln(τ)"); ax.set_ylabel("density")
    ax.set_title(f"T = {T:.1f} K  [{result['wtype']}]  active: {result['active_terms']}")
    ax.legend(fontsize=9); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_range(spec, nmax):
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError("--range expects start:end or start:end:step")
    start = int(parts[0]) if parts[0] else 0
    end   = int(parts[1]) if parts[1] else nmax
    step  = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    return range(start, min(end, nmax), step)


def _parse_window(s):
    """Parse 'Tlo,Thi' string into (float, float) or None."""
    if not s or str(s).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must be 'Tlo,Thi'")
    return (min(parts), max(parts))


def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ DC Monte Carlo — window-aware, optional AC priors"
    )
    ap.add_argument("--infile",    default=PATH)
    ap.add_argument("--idx",       type=int, default=None)
    ap.add_argument("--idxs",      type=str, default=None)
    ap.add_argument("--range",     dest="range_spec", type=str, default=None)
    ap.add_argument("--all",       default=True, action="store_true")
    ap.add_argument("--out",       default="dc_mc_params.csv")
    ap.add_argument("--clear",     action="store_true")
    ap.add_argument("--seed_base", type=int, default=14322)
    ap.add_argument("--plot",      action="store_true")

    win = ap.add_argument_group(
        "Temperature windows (optional)",
        "Override or supply window assignments for rows. "
        "If the input CSV already has in_qtm_window / in_raman_window columns "
        "(written by dc_phase2.py), those are used by default. "
        "Supplying these arguments overrides the CSV flags entirely, "
        "letting you refine Q over a narrower or different range without "
        "re-running dc_phase2."
    )
    win.add_argument("--qtm_window",   default="", metavar="Tlo,Thi",
                     help="Temperature range to fit Q/sQ, e.g. '2,9'. "
                          "Rows inside → QTM fit. Overrides in_qtm_window column.")
    win.add_argument("--raman_window", default="", metavar="Tlo,Thi",
                     help="Temperature range to fit R/n/sR/sN, e.g. '13,23'. "
                          "Rows inside → Raman fit. Overrides in_raman_window column.")

    pr = ap.add_argument_group(
        "Priors for fixed terms (all optional)",
        "Values for terms not being fitted in a given window. "
        "Can come from ac_phase2/ac_montecarlo (AC experiment) or from "
        "dc_phase2 compiled estimates (DC-only experiment). "
        "Missing priors drop that term from the rate model entirely. "
        "A and Ueff are NEVER fitted from DC data."
    )
    # New --pr_* names (preferred)
    pr.add_argument("--pr_A",    type=float, default=None, help="A mean (Orbach prefactor, from AC)")
    pr.add_argument("--pr_Ueff", type=float, default=None, help="Ueff mean in K (from AC)")
    pr.add_argument("--pr_sA",   type=float, default=0.15)
    pr.add_argument("--pr_sU",   type=float, default=20.0)
    pr.add_argument("--pr_R",    type=float, default=None, help="R mean (Raman prefactor)")
    pr.add_argument("--pr_n",    type=float, default=None, help="n mean (Raman exponent)")
    pr.add_argument("--pr_sR",   type=float, default=0.35)
    pr.add_argument("--pr_sN",   type=float, default=0.25)
    pr.add_argument("--pr_Q",    type=float, default=None,
                    help="Q prior — from dc_phase2 compiled estimate or AC fit")
    pr.add_argument("--pr_sQ",   type=float, default=0.21)
    # Old --ac_* names kept as silent aliases for backward compatibility
    pr.add_argument("--ac_A",    type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_Ueff", type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_sA",   type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_sU",   type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_R",    type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_n",    type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_sR",   type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_sN",   type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_Q",    type=float, default=None, help=argparse.SUPPRESS)
    pr.add_argument("--ac_sQ",   type=float, default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    # --pr_* takes precedence; fall back to --ac_* alias if --pr_* not set
    def _resolve(pr_val, ac_val): return pr_val if pr_val is not None else ac_val

    A_val    = _resolve(args.pr_A,    args.ac_A)
    Ueff_val = _resolve(args.pr_Ueff, args.ac_Ueff)
    sA_val   = _resolve(args.pr_sA,   args.ac_sA)   or 0.15
    sU_val   = _resolve(args.pr_sU,   args.ac_sU)   or 20.0
    R_val    = _resolve(args.pr_R,    args.ac_R)
    n_val    = _resolve(args.pr_n,    args.ac_n)
    sR_val   = _resolve(args.pr_sR,   args.ac_sR)   or 0.35
    sN_val   = _resolve(args.pr_sN,   args.ac_sN)   or 0.25
    Q_val    = _resolve(args.pr_Q,    args.ac_Q)
    sQ_val   = _resolve(args.pr_sQ,   args.ac_sQ)   or 0.21

    # build priors dict — only include keys where the value was actually provided
    ac_priors = {}
    if A_val    is not None: ac_priors.update({'A':    A_val,    'sA': sA_val})
    if Ueff_val is not None: ac_priors.update({'Ueff': Ueff_val, 'sU': sU_val})
    if R_val    is not None: ac_priors.update({'R':    R_val,    'sR': sR_val})
    if n_val    is not None: ac_priors.update({'n':    n_val,    'sN': sN_val})
    if Q_val    is not None: ac_priors.update({'Q':    Q_val,    'sQ': sQ_val})

    # summarise what mode we are in
    has_orbach = 'A' in ac_priors and 'Ueff' in ac_priors
    has_raman  = 'R' in ac_priors and 'n'    in ac_priors
    has_Q      = 'Q' in ac_priors

    print("Priors provided:")
    if has_orbach:
        print(f"  A    = {ac_priors['A']:.4f} ± {ac_priors['sA']:.4f}  "
              f"Ueff = {ac_priors['Ueff']:.2f} ± {ac_priors['sU']:.2f}  "
              f"(fixed in all DC rows — from AC data)")
    else:
        print("  Orbach (A, Ueff): not provided — Orbach term dropped from model")
    if has_raman:
        print(f"  R    = {ac_priors['R']:.4f} ± {ac_priors['sR']:.4f}  "
              f"n    = {ac_priors['n']:.4f} ± {ac_priors['sN']:.4f}  "
              f"(fixed in QTM rows)")
    else:
        print("  Raman  (R, n):    not provided — Raman term dropped from QTM rows")
    if has_Q:
        print(f"  Q    = {ac_priors['Q']:.4f} ± {ac_priors['sQ']:.4f}  "
              f"(start for QTM rows, fixed in Raman rows)")
    else:
        print("  Q prior:          not provided — Q term dropped from Raman rows")

    import csv as _csv
    with open(args.infile, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
    try:
        sep = _csv.Sniffer().sniff(sample).delimiter
    except Exception:
        sep = ","

    # try with header first, then without
    df = pd.read_csv(args.infile, sep=sep, engine="python")
    cols_lower = [str(c).lower().strip() for c in df.columns]

    # detect headerless raw DC file: 3 numeric columns, first row all floats
    def _looks_headerless(df):
        try:
            df.iloc[0].astype(float)
            return True
        except (ValueError, TypeError):
            return False

    # detect headerless: column names are integers or non-descriptive
    is_headerless = all(str(c).strip().lstrip('-').replace('.','').isdigit()
                        for c in df.columns)
    if is_headerless:
        df = pd.read_csv(args.infile, sep=sep, engine="python", header=None)
        df.columns = [str(i) for i in range(len(df.columns))]
        cols_lower = list(df.columns)

    # map to lowercase for detection
    df_detect = df.copy()
    df_detect.columns = cols_lower

    # ── detect if the user passed the raw DC data file instead of dc_phase2 output ──
    # Raw DC file: either has tau_star/beta columns, or is headerless with 3 numeric cols
    phase2_cols = {"mu_ln", "sigma1_ln", "sigma2_ln"}
    has_phase2  = phase2_cols.issubset(set(cols_lower))
    has_raw_header = {"tau_star", "beta"}.issubset(set(cols_lower))
    # headerless with 3 cols = almost certainly T | tau_star | beta
    has_raw_headerless = is_headerless and len(df.columns) <= 4

    if (has_raw_header or has_raw_headerless) and not has_phase2:
        stem       = os.path.splitext(os.path.basename(args.infile))[0]
        phase2_out = f"{stem}_phase2.csv"
        qtm_hint   = f" --qtm_window {args.qtm_window}"    if args.qtm_window   else ""
        raman_hint = f" --raman_window {args.raman_window}" if args.raman_window else ""
        prior_flags = ""
        if Q_val    is not None: prior_flags += f" --pr_Q {Q_val} --pr_sQ {sQ_val}"
        if A_val    is not None: prior_flags += f" --pr_A {A_val} --pr_sA {sA_val}"
        if Ueff_val is not None: prior_flags += f" --pr_Ueff {Ueff_val} --pr_sU {sU_val}"
        if R_val    is not None: prior_flags += f" --pr_R {R_val} --pr_n {n_val}"

        print("ERROR: --infile appears to be a raw DC data file (T, tau_star, beta).",
              file=sys.stderr)
        print("       dc_montecarlo requires the dc_phase2.py output, not the raw data.",
              file=sys.stderr)
        print(f"\n  Step 1 — run dc_phase2.py first:", file=sys.stderr)
        print(f"    python dc_phase2.py --infile {args.infile}"
              f"{qtm_hint}{raman_hint} --outfile {phase2_out}", file=sys.stderr)
        print(f"\n  Step 2 — then run dc_montecarlo.py:", file=sys.stderr)
        print(f"    python dc_montecarlo.py --infile {phase2_out}"
              f"{qtm_hint}{raman_hint}{prior_flags}", file=sys.stderr)
        sys.exit(1)

    # restore proper column names for phase2 output
    df.columns = cols_lower
    if "t" in df.columns and "T" not in df.columns:
        df = df.rename(columns={"t": "T"})

    required = {"T", "mu_ln", "sigma1_ln", "sigma2_ln"}
    if missing := required - set(df.columns):
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        print("       Expected dc_phase2.py output with columns: "
              "T, mu_ln, sigma1_ln, sigma2_ln", file=sys.stderr)
        sys.exit(1)

    # ── resolve window flags ─────────────────────────────────────────────────
    qtm_w   = _parse_window(args.qtm_window)
    raman_w = _parse_window(args.raman_window)

    if qtm_w or raman_w:
        # CLI windows provided — override any existing columns
        if "in_qtm_window" in df.columns or "in_raman_window" in df.columns:
            print("  NOTE: --qtm_window/--raman_window provided — "
                  "overriding window flag columns from input CSV.")
        df["in_qtm_window"]   = (df["T"].between(qtm_w[0],   qtm_w[1])
                                 if qtm_w   else pd.Series(False, index=df.index))
        df["in_raman_window"] = (df["T"].between(raman_w[0], raman_w[1])
                                 if raman_w else pd.Series(False, index=df.index))
        if qtm_w:
            print(f"  QTM   window {qtm_w[0]:.1f}–{qtm_w[1]:.1f} K: "
                  f"{int(df['in_qtm_window'].sum())} rows → fit Q, sQ")
        if raman_w:
            print(f"  Raman window {raman_w[0]:.1f}–{raman_w[1]:.1f} K: "
                  f"{int(df['in_raman_window'].sum())} rows → fit R, n, sR, sN")
    elif "in_qtm_window" not in df.columns or "in_raman_window" not in df.columns:
        print("WARNING: no window flags found and no --qtm_window/--raman_window "
              "supplied.", file=sys.stderr)
        print("  Re-run dc_phase2.py with --qtm_window / --raman_window, or "
              "pass --qtm_window here.", file=sys.stderr)
        print("  Defaulting: all rows treated as QTM window.", file=sys.stderr)
        df["in_qtm_window"]   = True
        df["in_raman_window"] = False
    else:
        # use existing columns from dc_phase2 output — report coverage
        n_qtm_csv   = int(df["in_qtm_window"].sum())
        n_raman_csv = int(df["in_raman_window"].sum())
        print(f"  Using window flags from input CSV: "
              f"QTM={n_qtm_csv} rows  Raman={n_raman_csv} rows")

    df = df.reset_index(drop=True)
    nrows   = len(df)
    n_qtm   = int(df["in_qtm_window"].sum())
    n_raman = int(df["in_raman_window"].sum())
    n_skip  = nrows - int((df["in_qtm_window"] | df["in_raman_window"]).sum())
    print(f"\nRows: {nrows} total  |  QTM: {n_qtm}  Raman: {n_raman}  Skip: {n_skip}\n")

    if args.clear and os.path.exists(args.out):
        open(args.out, "w").close()

    if args.all:
        idx_list = list(range(nrows))
    elif args.range_spec:
        idx_list = list(parse_range(args.range_spec, nrows))
    elif args.idxs:
        idx_list = [int(s.strip()) for s in args.idxs.split(",") if s.strip()]
    elif args.idx is not None:
        idx_list = [int(args.idx)]
    else:
        print("Provide one of --idx, --idxs, --range, or --all.", file=sys.stderr)
        sys.exit(2)

    make_plot = args.plot and (len(idx_list) == 1)

    for idx in idx_list:
        if not (0 <= idx < nrows):
            print(f"Skip idx {idx}: out of range", file=sys.stderr); continue
        try:
            fit_one_row(df, idx, args.out, ac_priors, args.seed_base, make_plot)
        except Exception as e:
            print(f"[idx={idx}] ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Results → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()