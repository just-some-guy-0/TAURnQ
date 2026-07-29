#!/usr/bin/env python3
"""
ac_montecarlo.py  –  TAURnQ Monte Carlo phase for AC susceptibility data.

Window-aware version with optional cross-term priors and reduced rate models.
Mirrors the design of dc_montecarlo.py exactly.

BEHAVIOUR
---------
Each row is dispatched based on window flags set by ac_phase2.py
(in_orbach_window, in_raman_window, in_qtm_window), and on which priors
the user has provided for the non-dominant terms.

  Orbach window rows:
    - All priors given        → fit A, U, sA, sU  (full model, R/n/Q fixed)
    - Raman prior only        → fit A, U, sA, sU  (Orbach + Raman model)
    - Q prior only            → fit A, U, sA, sU  (Orbach + QTM model)
    - No other priors         → fit A, U, sA, sU  (Orbach-only model)

  Raman window rows:
    - All priors given        → fit R, n, sR, sN  (full model, A/U/Q fixed)
    - Orbach prior only       → fit R, n, sR, sN  (Orbach + Raman model)
    - Q prior only            → fit R, n, sR, sN  (Raman + QTM model)
    - No other priors         → fit R, n, sR, sN  (Raman-only model)

  QTM window rows:
    - All priors given        → fit Q, sQ  (full model, A/U/R/n fixed)
    - Orbach prior only       → fit Q, sQ  (Orbach + QTM model)
    - Raman prior only        → fit Q, sQ  (Raman + QTM model)
    - No other priors         → fit Q, sQ  (QTM-only model: τ⁻¹ = 10^-Q)

  Rows outside all windows are skipped.

All priors are optional. Missing ones drop that term from the rate model.

Input:  ac_phase2.py output CSV (or original ccfit2 TSV with T, tau_mean/tau_mu, alpha)
        Required columns: T, tau_mean (or tau_mu), alpha
        Window flag columns (from ac_phase2.py):
            in_orbach_window, in_raman_window, in_qtm_window
        If window flags are absent, all rows are treated as all-windows active.

Output: same schema as original montecarlo.py
        T, Au, Uu, Ru, Nu, Qu, As, Us, Rs, Ns, Qs
        Columns for unfitted terms are NaN.
        Plus: window_type, fitted_params, active_terms

Usage (full AC dataset with priors from another window or experiment):
    python ac_montecarlo.py --infile ac_data.tsv \\
        --orbach_window 45,58 --raman_window 25,35 --qtm_window 2,10 \\
        --pr_R -7.34 --pr_n 3.84 --pr_sR 0.34 --pr_sN 0.22 \\
        --pr_Q 2.424 --pr_sQ 0.045

Usage (Orbach-only AC data, no other priors):
    python ac_montecarlo.py --infile ac_data.tsv --orbach_window 45,58
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os, sys

PATH = "ac_data.tsv"


# ---------------------------------------------------------------------------
# FK log-normal helpers
# ---------------------------------------------------------------------------

def fk_ln_quantiles(tau_m, alpha, qs):
    g = 1.82 * np.sqrt(alpha) / (1.0 - alpha)
    return np.log(tau_m) + norm.ppf(qs) * g


def rho_tau(tau, tau_m, alpha):
    g = 1.82 * np.sqrt(alpha) / (1.0 - alpha)
    return (np.exp(-0.5 * ((np.log(tau) - np.log(tau_m)) / g) ** 2)
            / (tau * g * np.sqrt(2 * np.pi)))


# ---------------------------------------------------------------------------
# Reduced rate model — only includes terms whose parameters are not None
# ---------------------------------------------------------------------------

def rate_model_active(T, A=None, Ueff=None, R=None, n=None, Q=None):
    T_safe = np.maximum(T, 1e-300)
    rate = np.zeros_like(np.asarray(T_safe, dtype=float))
    if A is not None and Ueff is not None:
        rate = rate + 10.0 ** (-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    if R is not None and n is not None:
        rate = rate + 10.0 ** R * T_safe ** n
    if Q is not None:
        rate = rate + 10.0 ** (-Q)
    return rate


def rate_model_full(T, A, Ueff, R, n, Q):
    T_safe = np.maximum(T, 1e-300)
    return (10.0 ** (-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
            + 10.0 ** R * T_safe ** n
            + 10.0 ** (-Q))


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_quantiles_reduced(T, free_draws, fixed, qs):
    """
    free_draws : dict param_name -> 1-D array of K samples
    fixed      : dict param_name -> scalar
    """
    K = next(iter(free_draws.values())).shape[0]

    def _get(name):
        if name in free_draws: return free_draws[name]
        if name in fixed:      return np.full(K, fixed[name])
        return None

    r = rate_model_active(T,
                          A=_get('A'), Ueff=_get('Ueff'),
                          R=_get('R'), n=_get('n'),
                          Q=_get('Q'))
    tau = 1.0 / np.maximum(r, 1e-300)
    return np.quantile(np.log(tau), qs)


# ---------------------------------------------------------------------------
# Penalties — scoped to free parameters only
# ---------------------------------------------------------------------------

def penalty_AU(A, U, sA, sU):
    pen = 0.0
    if -A <  0.0:   pen += 1e-3 * (0.0  - (-A))  ** 2
    if -A > 30.0:   pen += 1e-3 * ((-A) - 30.0)  ** 2
    if U  <  0.0:   pen += 1e-6 * (0.0  - U)      ** 2
    if U  > 3000.0: pen += 1e-6 * (U    - 3000.0) ** 2
    pen += 1e-5 * (sA ** 2 + sU ** 2)
    return pen


def penalty_RN(R, n, sR, sN):
    pen = 0.0
    if R < -20.0: pen += 1e-3 * (-20.0 - R) ** 2
    if R >  10.0: pen += 1e-3 * (R - 10.0) ** 2
    if n <   0.0: pen += 1e-3 * (0.0 - n) ** 2
    if n >  12.0: pen += 1e-3 * (n - 12.0) ** 2
    pen += 1e-5 * (sR ** 2 + sN ** 2)
    return pen


def penalty_Q(Q, sQ):
    pen = 0.0
    if Q < -20.0: pen += 1e-3 * (-20.0 - Q) ** 2
    if Q >  10.0: pen += 1e-3 * (Q - 10.0) ** 2
    pen += 1e-5 * sQ ** 2
    return pen


# ---------------------------------------------------------------------------
# Orbach-window fit: fit A, U, sA, sU
# ---------------------------------------------------------------------------

def fit_orbach_row(T, tau_m, alpha, pr, Z, qs):
    """
    Fit A, U, sA, sU. Fix R, n, Q from priors if available, else drop term.

    pr : dict with optional keys R, n, sR, sN, Q, sQ
    """
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    has_raman = ('R' in pr and 'n' in pr)
    has_Q     = ('Q' in pr)

    fixed = {}
    if has_raman: fixed.update({'R': pr['R'], 'n': pr['n']})
    if has_Q:     fixed.update({'Q': pr['Q']})

    active_terms = ['orbach'] + (['raman'] if has_raman else []) + (['qtm'] if has_Q else [])

    # initial guess: log10(tau_m) ≈ LN10*A + U/T  →  rough A/U from mean
    A_init   = pr.get('A',   -12.0)
    U_init   = pr.get('Ueff', 1000.0)
    sA_init  = pr.get('sA',    0.2)
    sU_init  = pr.get('sU',   15.0)

    x0 = np.array([A_init, U_init,
                   np.log(max(sA_init, 1e-4)),
                   np.log(max(sU_init, 1e-4))], dtype=float)

    def objective(x):
        A_try  = x[0];  U_try  = x[1]
        sA_try = np.exp(x[2]); sU_try = np.exp(x[3])
        A_draws = A_try  + Z[:, 0] * sA_try
        U_draws = U_try  + Z[:, 1] * sU_try
        lnq_hat = simulate_quantiles_reduced(
            T,
            free_draws={'A': A_draws, 'Ueff': U_draws},
            fixed=fixed, qs=qs)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid)) + penalty_AU(A_try, U_try, sA_try, sU_try)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 250, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    A_hat  = res.x[0];  U_hat  = res.x[1]
    sA_hat = np.exp(res.x[2]); sU_hat = np.exp(res.x[3])

    return {
        'A': A_hat, 'sA': sA_hat,
        'U': U_hat, 'sU': sU_hat,
        'R': pr.get('R', np.nan), 'sR': pr.get('sR', np.nan),
        'n': pr.get('n', np.nan), 'sN': pr.get('sN', np.nan),
        'Q': pr.get('Q', np.nan), 'sQ': pr.get('sQ', np.nan),
        'loss': float(res.fun),
        'wtype': 'orbach',
        'active_terms': '+'.join(active_terms),
        'fitted_params': 'A,U,sA,sU',
    }


# ---------------------------------------------------------------------------
# Raman-window fit: fit R, n, sR, sN
# ---------------------------------------------------------------------------

def fit_raman_row(T, tau_m, alpha, pr, Z, qs):
    """
    Fit R, n, sR, sN. Fix A, U, Q from priors if available, else drop term.
    """
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    has_orbach = ('A' in pr and 'Ueff' in pr)
    has_Q      = ('Q' in pr)

    fixed = {}
    if has_orbach: fixed.update({'A': pr['A'], 'Ueff': pr['Ueff']})
    if has_Q:      fixed.update({'Q': pr['Q']})

    active_terms = (['orbach'] if has_orbach else []) + ['raman'] + (['qtm'] if has_Q else [])

    R_init  = pr.get('R',  -6.0)
    n_init  = pr.get('n',   4.0)
    sR_init = pr.get('sR',  0.5)
    sN_init = pr.get('sN',  0.5)

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
            free_draws={'R': R_draws, 'n': n_draws},
            fixed=fixed, qs=qs)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid)) + penalty_RN(R_try, n_try, sR_try, sN_try)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    R_hat  = res.x[0];  n_hat  = res.x[1]
    sR_hat = np.exp(res.x[2]); sN_hat = np.exp(res.x[3])

    return {
        'A': pr.get('A',   np.nan), 'sA': pr.get('sA', np.nan),
        'U': pr.get('Ueff',np.nan), 'sU': pr.get('sU', np.nan),
        'R': R_hat, 'sR': sR_hat,
        'n': n_hat, 'sN': sN_hat,
        'Q': pr.get('Q', np.nan),  'sQ': pr.get('sQ', np.nan),
        'loss': float(res.fun),
        'wtype': 'raman',
        'active_terms': '+'.join(active_terms),
        'fitted_params': 'R,n,sR,sN',
    }


# ---------------------------------------------------------------------------
# QTM-window fit: fit Q, sQ
# ---------------------------------------------------------------------------

def fit_qtm_row(T, tau_m, alpha, pr, Z, qs):
    """
    Fit Q and sQ. Fix A, U, R, n from priors if available, else drop term.
    """
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    has_orbach = ('A' in pr and 'Ueff' in pr)
    has_raman  = ('R' in pr and 'n' in pr)

    fixed = {}
    if has_orbach: fixed.update({'A': pr['A'], 'Ueff': pr['Ueff']})
    if has_raman:  fixed.update({'R': pr['R'], 'n':    pr['n']})

    active_terms = (['orbach'] if has_orbach else []) + \
                   (['raman']  if has_raman  else []) + ['qtm']

    Q_init  = pr.get('Q',  2.0)
    sQ_init = pr.get('sQ', 0.3)

    x0 = np.array([Q_init, np.log(max(sQ_init, 1e-4))], dtype=float)

    def objective(x):
        Q_try  = x[0]
        sQ_try = np.exp(x[1])
        Q_draws = Q_try + Z[:, 4] * sQ_try
        lnq_hat = simulate_quantiles_reduced(
            T,
            free_draws={'Q': Q_draws},
            fixed=fixed, qs=qs)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid)) + penalty_Q(Q_try, sQ_try)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    Q_hat  = res.x[0]
    sQ_hat = np.exp(res.x[1])

    return {
        'A': pr.get('A',   np.nan), 'sA': pr.get('sA', np.nan),
        'U': pr.get('Ueff',np.nan), 'sU': pr.get('sU', np.nan),
        'R': pr.get('R',   np.nan), 'sR': pr.get('sR', np.nan),
        'n': pr.get('n',   np.nan), 'sN': pr.get('sN', np.nan),
        'Q': Q_hat, 'sQ': sQ_hat,
        'loss': float(res.fun),
        'wtype': 'qtm',
        'active_terms': '+'.join(active_terms),
        'fitted_params': 'Q,sQ',
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def fit_one_row(df, idx, out_path, priors, windows, seed_base=12345, make_plot=False):
    """
    priors : dict with any subset of keys:
             A, Ueff, sA, sU, R, n, sR, sN, Q, sQ
    windows: dict with keys orbach_w, raman_w, qtm_w — each (lo,hi) or None
    """
    row     = df.loc[idx]
    T       = float(row["T"])
    tau_m   = float(row.get("tau_mean", row.get("tau_mu", np.nan)))
    alpha   = float(row["alpha"])

    # determine which window(s) this row belongs to
    def _in(w):
        return w is not None and w[0] <= T <= w[1]

    in_orbach = bool(row.get("in_orbach_window", _in(windows['orbach_w'])))
    in_raman  = bool(row.get("in_raman_window",  _in(windows['raman_w'])))
    in_qtm    = bool(row.get("in_qtm_window",    _in(windows['qtm_w'])))

    # priority: if row falls in multiple windows use the most dominant
    # (orbach > raman > qtm by convention — user should avoid overlaps)
    if not in_orbach and not in_raman and not in_qtm:
        print(f"[idx={idx:>3}] T={T:6.2f} K  SKIP — outside all windows")
        return

    qs  = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98], dtype=float)
    rng = np.random.default_rng(int(seed_base) + int(idx))
    Z   = rng.standard_normal(size=(25000, 5))

    if in_orbach:
        result = fit_orbach_row(T, tau_m, alpha, priors, Z, qs)
    elif in_raman:
        result = fit_raman_row(T, tau_m, alpha, priors, Z, qs)
    else:
        result = fit_qtm_row(T, tau_m, alpha, priors, Z, qs)

    print(f"[idx={idx:>3}] T={T:6.2f} K [{result['wtype']:>6}] "
          f"active={result['active_terms']} | loss={result['loss']:.5g} | "
          f"A={result['A']:.4f} U={result['U']:.2f} "
          f"R={result['R']:.4f} n={result['n']:.4f} Q={result['Q']:.4f} | "
          f"sA={result['sA']:.3f} sU={result['sU']:.2f} "
          f"sR={result['sR']:.3f} sN={result['sN']:.3f} sQ={result['sQ']:.3f}")

    row_out = pd.DataFrame([{
        "T":   T,
        "Au":  result['A'],   "Uu":  result['U'],   "Ru":  result['R'],
        "Nu":  result['n'],   "Qu":  result['Q'],
        "As":  result['sA'],  "Us":  result['sU'],  "Rs":  result['sR'],
        "Ns":  result['sN'],  "Qs":  result['sQ'],
        "window_type":   result['wtype'],
        "fitted_params": result['fitted_params'],
        "active_terms":  result['active_terms'],
    }])
    header_needed = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    row_out.to_csv(out_path, index=False, mode="a", header=header_needed)

    if make_plot:
        _plot_fit(T, tau_m, alpha, result, idx)


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def _plot_fit(T, tau_m, alpha, result, idx):
    rng    = np.random.default_rng(54321 + int(idx))
    Z_plot = rng.standard_normal((60000, 5))

    A_draws  = (result['A'] + Z_plot[:,0]*result['sA']
                if not np.isnan(result['A'])  else None)
    U_draws  = (result['U'] + Z_plot[:,1]*result['sU']
                if not np.isnan(result['U'])  else None)
    R_draws  = (result['R'] + Z_plot[:,2]*result['sR']
                if not np.isnan(result['R'])  else None)
    n_draws  = (result['n'] + Z_plot[:,3]*result['sN']
                if not np.isnan(result['n'])  else None)
    Q_draws  = (result['Q'] + Z_plot[:,4]*result['sQ']
                if not np.isnan(result['Q'])  else None)

    rate   = rate_model_active(T, A=A_draws, Ueff=U_draws,
                               R=R_draws, n=n_draws, Q=Q_draws)
    tau_mc = 1.0 / np.maximum(rate, 1e-300)

    taus         = np.logspace(np.log10(tau_mc.min()*0.1),
                               np.log10(tau_mc.max()*10), 5000)
    rho_fk       = rho_tau(taus, tau_m, alpha)
    bins         = np.logspace(np.log10(tau_mc.min()*0.5),
                               np.log10(tau_mc.max()*2), 120)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(tau_mc, bins=bins, density=True, alpha=0.45,
            color="#378ADD", edgecolor="none", label="MC τ samples")
    ax.plot(taus, rho_fk, color="#D85A30", linewidth=2, label="FK log-normal target")
    ax.set_xscale("log")
    ax.set_xlabel("τ (s)"); ax.set_ylabel("ρ(τ)")
    ax.set_title(f"T = {T:.1f} K  [{result['wtype']}]  "
                 f"active: {result['active_terms']}")
    ax.legend(fontsize=9); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_input(path):
    """
    Load raw AC susceptibility data: T, tau_mean (or tau_mu), alpha.

    Accepts:
      - TSV/CSV with header row containing recognised column names
      - Headerless TSV/CSV with three columns in order T, tau_mean, alpha

    NOTE: --infile must be the raw per-temperature data file, NOT the
    ac_phase2.py output CSV.  The ac_phase2 output (columns mu_A, mu_U, ...)
    is a single-row parameter summary — pass its values via --pr_A, --pr_Ueff
    etc. instead.
    """
    # read with header detection — use csv module first to parse correctly,
    # then hand off to pandas. This avoids pandas misaligning columns when
    # fields contain commas (e.g. the window string "(64.0, 96.0)").
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
    lines = sample.splitlines()
    try:
        dialect = _csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ","
    df = pd.read_csv(path, sep=sep, engine="python",
                     header="infer", quoting=_csv.QUOTE_MINIMAL)
    cols_lower = [str(c).lower().strip() for c in df.columns]

    # detect if this looks like an ac_phase2 output (parameter summary, not data)
    ac_phase2_cols = {"mu_a", "mu_u", "sd_a", "sd_u", "rho_au"}
    if ac_phase2_cols.issubset(set(cols_lower)):
        print("ERROR: --infile appears to be an ac_phase2.py output CSV "
              "(columns: mu_A, mu_U, ...).", file=sys.stderr)
        print("       --infile should be your raw AC data file "
              "(columns: T, tau_mean/tau_mu, alpha).", file=sys.stderr)
        print("       Pass the ac_phase2 values as priors instead:", file=sys.stderr)
        # read the single-row and suggest the correct flags
        row_series = df.iloc[0]
        row_cols = {str(c).lower().strip(): v
                    for c, v in row_series.items()}
        suggestions = []
        if "mu_a"  in row_cols: suggestions.append(f"  --pr_A {row_cols['mu_a']}")
        if "mu_u"  in row_cols: suggestions.append(f"  --pr_Ueff {row_cols['mu_u']}")
        if "sd_a"  in row_cols: suggestions.append(f"  --pr_sA {row_cols['sd_a']}")
        if "sd_u"  in row_cols: suggestions.append(f"  --pr_sU {row_cols['sd_u']}")
        if "mu_r"  in row_cols: suggestions.append(f"  --pr_R {row_cols['mu_r']}")
        if "mu_n"  in row_cols: suggestions.append(f"  --pr_n {row_cols['mu_n']}")
        if "sd_r"  in row_cols: suggestions.append(f"  --pr_sR {row_cols['sd_r']}")
        if "sd_n"  in row_cols: suggestions.append(f"  --pr_sN {row_cols['sd_n']}")
        if "mu_q"  in row_cols: suggestions.append(f"  --pr_Q {row_cols['mu_q']}")
        if "sd_q"  in row_cols: suggestions.append(f"  --pr_sQ {row_cols['sd_q']}")
        if suggestions:
            print("       Suggested prior flags from this file:", file=sys.stderr)
            for s in suggestions:
                print(s, file=sys.stderr)
        sys.exit(1)

    df.columns = cols_lower

    # normalise column names
    rename = {}
    if "tau_mean" not in df.columns and "tau_mu" in df.columns:
        rename["tau_mu"] = "tau_mean"
    if "t" in df.columns and "T" not in df.columns:
        rename["t"] = "T"
    if rename:
        df = df.rename(columns=rename)

    required = {"T", "tau_mean", "alpha"}
    if required.issubset(df.columns):
        return df[["T", "tau_mean", "alpha"]].astype(float)

    # last resort: headerless — assume T | tau_mean | alpha column order
    df2 = pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, :3]
    # check first row is not a string header
    try:
        df2 = df2.astype(float)
    except (ValueError, TypeError):
        df2 = df2.iloc[1:].astype(float)
    df2.columns = ["T", "tau_mean", "alpha"]
    return df2.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_window_arg(s):
    if not s or str(s).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must be 'Tlo,Thi'")
    return (min(parts), max(parts))


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
        description="TAURnQ AC Monte Carlo — window-aware, optional cross-term priors"
    )
    ap.add_argument("--infile",    default=PATH,
                    help="TSV/CSV (T, tau_mean/tau_mu, alpha) — "
                         "window flag columns optional")
    ap.add_argument("--idx",       type=int, default=None)
    ap.add_argument("--idxs",      type=str, default=None)
    ap.add_argument("--range",     dest="range_spec", type=str, default=None)
    ap.add_argument("--all",       default=True, action="store_true")
    ap.add_argument("--out",       default="ac_mc_params.csv")
    ap.add_argument("--clear",     action="store_true")
    ap.add_argument("--seed_base", type=int, default=14322)
    ap.add_argument("--plot",      action="store_true")

    win = ap.add_argument_group(
        "Temperature windows",
        "Each row is assigned to its dominant process window. "
        "Rows outside all windows are skipped. "
        "If window flag columns are present in --infile (from ac_phase2.py), "
        "these arguments are ignored for those rows."
    )
    win.add_argument("--orbach_window", default="", metavar="Tlo,Thi")
    win.add_argument("--raman_window",  default="", metavar="Tlo,Thi")
    win.add_argument("--qtm_window",    default="", metavar="Tlo,Thi")

    pr = ap.add_argument_group(
        "Cross-term priors (all optional)",
        "Values for terms not being fitted in a given window. "
        "Missing priors drop that term from the rate model entirely."
    )
    # Orbach priors (used when fitting Raman or QTM rows)
    pr.add_argument("--pr_A",    type=float, default=None, help="A mean")
    pr.add_argument("--pr_Ueff", type=float, default=None, help="Ueff mean (K)")
    pr.add_argument("--pr_sA",   type=float, default=0.15)
    pr.add_argument("--pr_sU",   type=float, default=20.0)
    # Raman priors (used when fitting Orbach or QTM rows)
    pr.add_argument("--pr_R",    type=float, default=None, help="R mean")
    pr.add_argument("--pr_n",    type=float, default=None, help="n mean")
    pr.add_argument("--pr_sR",   type=float, default=0.35)
    pr.add_argument("--pr_sN",   type=float, default=0.25)
    # QTM prior (used when fitting Orbach or Raman rows)
    pr.add_argument("--pr_Q",    type=float, default=None, help="Q mean")
    pr.add_argument("--pr_sQ",   type=float, default=0.21)

    args = ap.parse_args()

    # build priors dict — only include keys actually provided
    priors = {}
    if args.pr_A    is not None: priors.update({'A':    args.pr_A,    'sA': args.pr_sA})
    if args.pr_Ueff is not None: priors.update({'Ueff': args.pr_Ueff, 'sU': args.pr_sU})
    if args.pr_R    is not None: priors.update({'R':    args.pr_R,    'sR': args.pr_sR})
    if args.pr_n    is not None: priors.update({'n':    args.pr_n,    'sN': args.pr_sN})
    if args.pr_Q    is not None: priors.update({'Q':    args.pr_Q,    'sQ': args.pr_sQ})

    windows = {
        'orbach_w': parse_window_arg(args.orbach_window),
        'raman_w':  parse_window_arg(args.raman_window),
        'qtm_w':    parse_window_arg(args.qtm_window),
    }

    # summarise mode
    print("Cross-term priors provided:")
    print(f"  Orbach (A, Ueff): {'YES — fixed in Raman/QTM rows' if ('A' in priors and 'Ueff' in priors) else 'NO  — Orbach term dropped from Raman/QTM rows'}")
    print(f"  Raman  (R, n):    {'YES — fixed in Orbach/QTM rows' if ('R' in priors and 'n' in priors) else 'NO  — Raman term dropped from Orbach/QTM rows'}")
    print(f"  Q prior:          {'YES — fixed in Orbach/Raman rows' if 'Q' in priors else 'NO  — QTM term dropped from Orbach/Raman rows'}")

    df = load_input(args.infile)
    nrows = len(df)
    print(f"\nLoaded {nrows} rows  (T {df['T'].min():.1f}–{df['T'].max():.1f} K)\n")

    # inject window flag columns if not present
    for col, wkey in [("in_orbach_window", "orbach_w"),
                      ("in_raman_window",  "raman_w"),
                      ("in_qtm_window",    "qtm_w")]:
        if col not in df.columns:
            w = windows[wkey]
            df[col] = (df["T"].between(w[0], w[1]) if w is not None
                       else pd.Series(False, index=df.index))

    n_orbach = int(df["in_orbach_window"].sum())
    n_raman  = int(df["in_raman_window"].sum())
    n_qtm    = int(df["in_qtm_window"].sum())
    any_window = df["in_orbach_window"] | df["in_raman_window"] | df["in_qtm_window"]
    n_skip   = nrows - int(any_window.sum())
    print(f"Rows: {nrows} total  |  Orbach: {n_orbach}  Raman: {n_raman}  QTM: {n_qtm}  Skip: {n_skip}\n")

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
            fit_one_row(df, idx, args.out, priors, windows,
                        seed_base=args.seed_base, make_plot=make_plot)
        except Exception as e:
            print(f"[idx={idx}] ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Results → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()