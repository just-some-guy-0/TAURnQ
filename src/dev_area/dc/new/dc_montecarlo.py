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

    ac = ap.add_argument_group(
        "AC priors (all optional)",
        "If omitted, the corresponding term is dropped from the rate model "
        "for rows where it would have been fixed.  A and Ueff are NEVER "
        "fitted from DC data — if you have AC data, always pass them here."
    )
    ac.add_argument("--ac_A",    type=float, default=None, help="A mean (Orbach prefactor)")
    ac.add_argument("--ac_Ueff", type=float, default=None, help="Ueff mean (K)")
    ac.add_argument("--ac_sA",   type=float, default=0.15)
    ac.add_argument("--ac_sU",   type=float, default=20.0)
    ac.add_argument("--ac_R",    type=float, default=None, help="R mean (Raman prefactor)")
    ac.add_argument("--ac_n",    type=float, default=None, help="n mean (Raman exponent)")
    ac.add_argument("--ac_sR",   type=float, default=0.35)
    ac.add_argument("--ac_sN",   type=float, default=0.25)
    ac.add_argument("--ac_Q",    type=float, default=None,
                    help="Q prior (starting guess for QTM rows, fixed for Raman rows)")
    ac.add_argument("--ac_sQ",   type=float, default=0.21)

    args = ap.parse_args()

    # build ac dict — only include keys where the value was actually provided
    ac_priors = {}
    if args.ac_A    is not None: ac_priors.update({'A':    args.ac_A,    'sA': args.ac_sA})
    if args.ac_Ueff is not None: ac_priors.update({'Ueff': args.ac_Ueff, 'sU': args.ac_sU})
    if args.ac_R    is not None: ac_priors.update({'R':    args.ac_R,    'sR': args.ac_sR})
    if args.ac_n    is not None: ac_priors.update({'n':    args.ac_n,    'sN': args.ac_sN})
    if args.ac_Q    is not None: ac_priors.update({'Q':    args.ac_Q,    'sQ': args.ac_sQ})

    # summarise what mode we are in
    has_orbach = 'A' in ac_priors and 'Ueff' in ac_priors
    has_raman  = 'R' in ac_priors and 'n'    in ac_priors
    has_Q      = 'Q' in ac_priors

    print("AC priors provided:")
    print(f"  Orbach (A, Ueff): {'YES — fixed in all DC rows' if has_orbach else 'NO  — Orbach term dropped from model'}")
    print(f"  Raman  (R, n):    {'YES — fixed in QTM rows'    if has_raman  else 'NO  — Raman term dropped from QTM rows'}")
    print(f"  Q prior:          {'YES — start/fix for Raman rows' if has_Q   else 'NO  — Q term dropped from Raman rows'}")
    if not has_orbach:
        print("  NOTE: without A/Ueff, QTM rows fit tau^-1 = 10^R*T^n + 10^-Q  (or just 10^-Q if no Raman either)")

    df = pd.read_csv(args.infile)
    required = {"T", "mu_ln", "sigma1_ln", "sigma2_ln"}
    if missing := required - set(df.columns):
        print(f"ERROR: missing columns: {missing}", file=sys.stderr); sys.exit(1)

    if "in_qtm_window" not in df.columns or "in_raman_window" not in df.columns:
        print("WARNING: window flags missing — re-run dc_phase2.py with "
              "--qtm_window and --raman_window", file=sys.stderr)
        print("Defaulting: all rows treated as QTM window.", file=sys.stderr)
        df["in_qtm_window"]   = True
        df["in_raman_window"] = False

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