#!/usr/bin/env python3
"""
global_fit_combined.py  –  TAURnQ phase 4, combined AC + DC global fit.

Reads directly from the output CSVs of the updated ac_montecarlo.py and
dc_montecarlo.py.  No raw TSV or dc_phase2 CSV required.

CORRECTIONS (2026-07-27)
--------------------------------------
a. fk_ln_quantiles now uses the EXACT Generalised Debye quantiles (closed
   form) instead of a normal ppf with the heuristic width 1.82*sqrt(a)/(1-a).
b. DC targets now use EXACT stretched-exponential quantiles computed from
   (tau_star, beta) via the one-sided stable distribution whenever those
   columns are present in the dc_phase2 CSV; the two-piece-normal target is
   only a fallback.  When the TPN fallback is used, the TPN location passed
   is mu_ln (the MODE) — the previous code passed e^<ln tau> (the mean),
   which for a TPN differs from the mode by sqrt(2/pi)(s1 - s2).
c. Quantile targets are precomputed once in build_rows (they do not depend
   on the optimisation variables) instead of being rebuilt every objective
   call — required now that DC targets involve stable-distribution isf.
d. rho reparametrised: sigmoid map R -> (0, 0.999) with nonzero gradient
   everywhere, replacing tanh(x)**2 whose gradient vanished at rho = 0 and
   which saturated near +/-1 (Nelder-Mead parked on the 0.999 plateau).
   Sign conventions unchanged (rho_AU >= 0, rho_RN <= 0).
e. CLI-supplied fixed rho values are clipped to the valid range; the old
   pipeline had emitted |rho| > 1, which draw_params silently laundered
   through max(1 - rho**2, 1e-12).

CORRECTIONS (2026-07-29)
--------------------------------------
f. compile_window: the RN and Q branches now use a robust MEDIAN over the
   windowed rows, not an inverse-variance-weighted mean — matching what the
   AU branch already did, and for the same reason. A single temperature
   cannot separate R from N (or A from Ueff, or resolve Q); those per-row
   values are unidentified, and their reported per-row spread (Rs/Ns/Qs)
   reflects optimiser flatness, not physical uncertainty. IVW weights by
   1/spread**2, so a handful of rows that happened to clamp a parameter to
   its bound with a spuriously tiny spread dominate the compiled target and
   drag it to the penalty ceiling (this is what produced N ~= 12, sitting on
   the qout(N,0,12) bound, with R dragged very negative along the C-n ridge
   to compensate). The median reports the bulk of the window instead.
g. --show_compile prints the raw windowed (T, param, spread) rows that feed
   each compiled target, so you can see whether the compiled value reflects
   the bulk of the window or a couple of outliers.

KEY CHANGES from the previous version
--------------------------------------
1. Single params CSV per experiment type.
   The updated phase 3 scripts embed all needed information in their output:
     - ac_montecarlo CSV: T, Au,Uu,Ru,Nu,Qu, As,Us,Rs,Ns,Qs,
                          window_type, fitted_params, active_terms
     - dc_montecarlo CSV: same schema, Au/Uu/As/Us always NaN

2. Quantile targets built from params, not raw data.
   AC rows use FK targets derived from (tau_mean, alpha) — but these values
   are back-computed from the per-row MC fit rather than re-read from raw data.
   Alternatively, if --ac_tsv is supplied the raw (tau_mean, alpha) values are
   used directly (more accurate); otherwise the compiled AC params are used as
   a proxy via the domain regulariser only.

   DC rows use TPN targets from (mu_ln, sigma1_ln, sigma2_ln) stored in the
   dc_montecarlo output or a separately supplied --dc_phase2 CSV.

3. Per-row weighting by window type AND experiment source.
   --w_ac_orbach  : extra weight for AC rows in their Orbach window (A/U signal)
   --w_ac_raman   : extra weight for AC rows in their Raman window  (R/n signal)
   --w_dc_qtm     : extra weight for DC rows in their QTM window    (Q signal)
   --w_dc_raman   : extra weight for DC rows in their Raman window  (R/n signal)
   Default 1.0 = equal weighting; increase to up-weight that experiment/window.

4. Domain compilation windows are now explicit per source.
   --AU_window    : which temperatures to compile A/Ueff from (prefers AC)
   --RN_window    : which temperatures to compile R/n from    (prefers AC; DC fallback)
   --Q_window     : which temperatures to compile Q from      (prefers DC; AC fallback)

5. Skewness output is computed from the fitted distribution marginals,
   not re-read from dc_phase2.

Usage (full AC + DC):
    python global_fit_combined.py \\
        --ac_params ac_mc_params.csv  --ac_tsv ac_data.tsv \\
        --dc_params dc_mc_params.csv  --dc_phase2 dc_phase2_out.csv \\
        --AU_window 45,58  --RN_window 25,35  --Q_window 2,9

Usage (AC only — no DC):
    python global_fit_combined.py \\
        --ac_params ac_mc_params.csv  --ac_tsv ac_data.tsv \\
        --AU_window 45,58  --RN_window 25,35

Usage (DC only — no AC):
    python global_fit_combined.py \\
        --dc_params dc_mc_params.csv  --dc_phase2 dc_phase2_out.csv \\
        --Q_window 2,9  --RN_window 13,23
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, levy_stable, skew as scipy_skew
import argparse, os, sys

SQRT2PI = np.sqrt(2.0 / np.pi)


# ---------------------------------------------------------------------------
# Quantile target functions
# ---------------------------------------------------------------------------

def fk_ln_quantiles(tau_mean, alpha, qs):
    """Exact Generalised Debye quantiles of ln(tau), from the closed-form CDF

        F(s) = 1/2 + arctan[cot(a*pi/2) tanh((1-a)s/2)] / (pi(1-a)),
        s(q) = (2/(1-a)) artanh[ tan(a*pi/2) tan(pi(1-a)(q - 1/2)) ].

    Replaces the normal approximation with sigma = 1.82*sqrt(a)/(1-a): the
    GD distribution has exponential tails, so a normal ppf misses the outer
    quantiles badly (q = 0.98 at alpha = 0.1: exact 2.05 vs normal 1.31,
    ln units), and the heuristic sigma matched neither the exact SD nor the
    exact 68% half-width."""
    a  = float(alpha)
    qs = np.asarray(qs, dtype=float)
    return np.log(tau_mean) + (2.0 / (1.0 - a)) * np.arctanh(
        np.tan(a * np.pi / 2.0) * np.tan(np.pi * (1.0 - a) * (qs - 0.5)))


def sef_ln_quantiles(tau_star, beta, qs):
    """Exact stretched-exponential quantiles of ln(tau).

    exp[-(t/tau*)^beta] = E[exp(-t·S/tau*)] with S one-sided stable(beta),
    so s = tau*/tau ~ S and, since s is decreasing in tau,
        P(ln tau <= x) = SF_S(s)  =>  ln tau_q = ln tau* - ln isf(q)."""
    dist = levy_stable(float(beta), 1.0, loc=0.0,
                       scale=np.cos(np.pi * float(beta) / 2.0) ** (1.0 / float(beta)))
    qs = np.asarray(qs, dtype=float)
    return np.log(tau_star) - np.log(dist.isf(qs))


def tpn_ln_quantiles(mu_ln, s1, s2, qs):
    p_left = s1 / (s1 + s2)
    qs = np.asarray(qs, dtype=float)
    result = np.empty_like(qs)
    for i, q in enumerate(qs):
        if q <= p_left:
            result[i] = mu_ln + norm.ppf(q * (s1 + s2) / (2.0 * s1)) * s1
        else:
            result[i] = mu_ln + norm.ppf(1.0 - (1.0 - q) * (s1 + s2) / (2.0 * s2)) * s2
    return result


# ---------------------------------------------------------------------------
# Rate model + MC
# ---------------------------------------------------------------------------

def rate_model(T, A, Ueff, R, N, Q):
    Ts = np.maximum(T, 1e-300)
    return (10.0 ** (-A) * np.exp(-Ueff / np.maximum(Ts, 1e-12))
            + 10.0 ** R * Ts ** N
            + 10.0 ** (-Q))


def draw_params(mu, sigmas, rho_AU, rho_RN, Z):
    A, Ueff, R, N, Q = mu
    sA, sU, sR, sN, sQ = sigmas
    L_AU = np.array([[sA, 0.0],
                     [rho_AU * sU, sU * np.sqrt(max(1 - rho_AU ** 2, 1e-12))]])
    L_RN = np.array([[sR, 0.0],
                     [rho_RN * sN, sN * np.sqrt(max(1 - rho_RN ** 2, 1e-12))]])
    eps_AU = Z[:, :2] @ L_AU.T
    eps_RN = Z[:, 2:4] @ L_RN.T
    return np.column_stack([A + eps_AU[:, 0], Ueff + eps_AU[:, 1],
                             R + eps_RN[:, 0], N + eps_RN[:, 1],
                             Q + Z[:, 4] * sQ])


def mc_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, Z):
    th = draw_params(mu, sigmas, rho_AU, rho_RN, Z)
    r = rate_model(T, *th.T)
    return np.quantile(np.log(1.0 / np.maximum(r, 1e-300)), qs)


RHO_MAX = 0.999


def _rho_from_x(x):
    """Monotone map R -> (0, RHO_MAX) with nonzero gradient everywhere.

    Replaces tanh(x)**2, whose gradient vanished at rho = 0 (the optimiser
    could not move off zero) and which saturated to ~1 by |x| ~ 3 (the
    optimiser parked on the 0.999 plateau — cf. rho pinned at +/-0.999 in
    earlier outputs)."""
    return RHO_MAX / (1.0 + np.exp(-x))


def _x_from_rho(rho):
    """Inverse of _rho_from_x for |rho|.  Encode moderate values only —
    encoding rho ~ 0 or ~ RHO_MAX would start the optimiser in a saturated
    region of the sigmoid."""
    p = np.clip(abs(float(rho)) / RHO_MAX, 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def unpack(x, fixed_rho_AU=None, fixed_rho_RN=None):
    mu     = x[0:5]
    sigmas = 1e-3 + np.exp(x[5:10])
    # sign conventions retained: rho_AU >= 0 (with tau = 10^A e^{U/T}, a
    # higher barrier is compensated by a larger prefactor), rho_RN <= 0
    # (intercept/slope anticorrelation of the Raman power law for T > 1 K)
    rho_AU = fixed_rho_AU if fixed_rho_AU is not None else _rho_from_x(x[10])
    rho_RN = fixed_rho_RN if fixed_rho_RN is not None else -_rho_from_x(x[11])
    return mu, sigmas, rho_AU, rho_RN


# ---------------------------------------------------------------------------
# Window / compilation helpers
# ---------------------------------------------------------------------------

# When True, compile_window prints the raw windowed rows that feed each
# compiled target (set from main() by --show_compile).
_VERBOSE_COMPILE = False


def parse_window(s):
    if not s or str(s).strip() == "":
        return None
    lo, hi = [float(v.strip()) for v in str(s).split(",")]
    return (min(lo, hi), max(lo, hi))


def in_window(T_arr, window):
    if window is None:
        return np.ones(len(T_arr), dtype=bool)
    lo, hi = window
    return (np.asarray(T_arr) >= lo) & (np.asarray(T_arr) <= hi)


def ivw_mean(vals, sds):
    v, s = np.asarray(vals, float), np.asarray(sds, float)
    w = 1.0 / np.maximum(s ** 2, 1e-12)
    return float(np.nansum(w * v) / np.maximum(np.nansum(w), 1e-12))


def robust_med(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return float(np.nanmedian(v)) if v.size > 0 else np.nan


def _sigma_floor(vals, abs_floor=0.02, quantile_floor=0.25):
    """
    Compute a sensible lower bound for spread values.
    Uses max(abs_floor, quantile_floor-th quantile of finite positive values).
    Prevents near-zero sigmas from dominating IVW or domain regularisation.
    """
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return abs_floor
    return max(abs_floor, float(np.quantile(v, quantile_floor)))


def _dump_window(sub, which):
    """Print the raw windowed rows feeding a compiled target."""
    cols_by = {"AU": ["T", "Au", "As", "Uu", "Us"],
               "RN": ["T", "Ru", "Rs", "Nu", "Ns"],
               "Q":  ["T", "Qu", "Qs"]}
    cols = [c for c in cols_by[which] if c in sub.columns]
    if not cols:
        return
    print(f"\n  [compile {which}] {len(sub)} row(s) in window:")
    with pd.option_context("display.float_format", lambda v: f"{v:9.4f}"):
        print("    " + sub[cols].to_string(index=False).replace("\n", "\n    "))


def compile_window(df, mask, which):
    """Compile mean and spread for a parameter group from windowed rows."""
    sub = df[mask].copy()

    if _VERBOSE_COMPILE and not sub.empty:
        _dump_window(sub, which)

    if which == "AU":
        sub = sub.dropna(subset=["Au", "As", "Uu", "Us"])
        if sub.empty:
            return None
        # Use robust median for (A, Ueff) because per-row values are degenerate —
        # sU reflects optimizer flatness, not physical Ueff uncertainty, so IVW
        # would be dominated by rows where Nelder-Mead happened to converge tightly.
        floor_A = _sigma_floor(sub["As"].values)
        floor_U = _sigma_floor(sub["Us"].values)
        A_mean  = float(np.nanmedian(sub["Au"].values))
        U_mean  = float(np.nanmedian(sub["Uu"].values))
        sA_med  = max(robust_med(sub["As"].values), floor_A)
        sU_med  = max(robust_med(sub["Us"].values), floor_U)
        return ((A_mean, U_mean), (sA_med, sU_med))

    elif which == "RN":
        sub = sub.dropna(subset=["Ru", "Rs", "Nu", "Ns"])
        if sub.empty:
            return None
        # Robust median for (R, N), for exactly the same reason as (A, Ueff):
        # a single temperature only constrains the TOTAL rate there, so it has
        # no leverage to separate the Raman prefactor R from the exponent N.
        # The per-row (R, N) are therefore unidentified and Ns reflects
        # optimiser flatness, not physical spread. An inverse-variance-weighted
        # mean (weight = 1/Ns**2) lets a couple of rows that clamped N to its
        # qout() bound with a spuriously tiny Ns bulldoze every well-behaved
        # row — the mechanism that produced N ~= 12 with R dragged very
        # negative along the C-n ridge to compensate. The median reports the
        # bulk of the window. (Independent medians of R and N can land slightly
        # off the C-n ridge; that is acceptable for a starting point / domain
        # target — the global quantile loss re-couples them.)
        floor_R = _sigma_floor(sub["Rs"].values)
        floor_N = _sigma_floor(sub["Ns"].values)
        return ((robust_med(sub["Ru"].values), robust_med(sub["Nu"].values)),
                (max(robust_med(sub["Rs"].values), floor_R),
                 max(robust_med(sub["Ns"].values), floor_N)))

    elif which == "Q":
        sub = sub.dropna(subset=["Qu", "Qs"])
        if sub.empty:
            return None
        # Robust median for Q too — same identifiability argument, and with
        # often only a handful of QTM-window rows the IVW is especially fragile
        # (one tight row dominates the whole compiled value).
        floor_Q = _sigma_floor(sub["Qs"].values)
        return ((robust_med(sub["Qu"].values),),
                (max(robust_med(sub["Qs"].values), floor_Q),))


def build_initials(df_ac, df_dc, AU_w, RN_w, Q_w):
    """
    Build x0 and compiled domain targets.

    Priority:
      AU compilation  → AC rows in AU_window  (Orbach data lives here)
      RN compilation  → AC rows in RN_window, DC fallback
      Q  compilation  → DC rows in Q_window,  AC fallback
    """
    def _try(df, w, which):
        if df is None or df.empty:
            return None
        mask = in_window(df["T"].values, w) if w else np.ones(len(df), dtype=bool)
        return compile_window(df, mask, which)

    au = _try(df_ac, AU_w, "AU") or _try(df_dc, AU_w, "AU")
    rn = _try(df_ac, RN_w, "RN") or _try(df_dc, RN_w, "RN")
    q  = _try(df_dc, Q_w,  "Q")  or _try(df_ac, Q_w,  "Q")

    def safe(res, n):
        if res is None:
            return tuple([0.0] * n), tuple([1.0] * n)
        return res

    (A0,  U0),  (sA0, sU0) = safe(au, 2)
    (R0,  N0),  (sR0, sN0) = safe(rn, 2)
    (Q0,),      (sQ0,)     = safe(q,  1)

    clamp = lambda v: max(v if np.isfinite(v) else 1e-3, 1e-3)
    sA0, sU0, sR0, sN0, sQ0 = map(clamp, [sA0, sU0, sR0, sN0, sQ0])

    # estimate ρ from combined data
    frames = [d for d in [df_ac, df_dc] if d is not None and not d.empty]
    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def safe_corr(c1, c2, fb=0.0):
        if df_all.empty or c1 not in df_all or c2 not in df_all:
            return fb
        x, y = df_all[c1].values, df_all[c2].values
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 3 else fb

    rAU = np.clip(safe_corr("Au", "Uu"), 0.0, 0.95)
    rRN = np.clip(-abs(safe_corr("Ru", "Nu")), -0.95, 0.0)

    # start |rho| at >= 0.10 so the sigmoid gradient is healthy at x0
    x0 = np.array([A0, U0, R0, N0, Q0,
                   np.log(max(sA0, 1e-6)), np.log(max(sU0, 1e-6)),
                   np.log(max(sR0, 1e-6)), np.log(max(sN0, 1e-6)),
                   np.log(max(sQ0, 1e-6)),
                   _x_from_rho(np.clip(max(rAU, 0.10), 0.10, 0.95)),
                   _x_from_rho(np.clip(max(abs(rRN), 0.10), 0.10, 0.95))],
                  dtype=float)

    compiled = {
        "AU": {"mean": np.array([A0,  U0]),  "sd": np.array([sA0, sU0])},
        "RN": {"mean": np.array([R0,  N0]),  "sd": np.array([sR0, sN0])},
        "Q":  {"mean": np.array([Q0]),       "sd": np.array([sQ0])},
    }
    return x0, compiled


# ---------------------------------------------------------------------------
# Penalties + domain regulariser
# ---------------------------------------------------------------------------

def penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=0.0):
    A, Ueff, R, N, Q = mu
    pen = 0.0

    def qout(v, lo, hi, sc):
        if v < lo: return sc * (lo - v) ** 2
        if v > hi: return sc * (v - hi) ** 2
        return 0.0

    pen += qout(-A,   0,    30,   1e-3)
    pen += qout(Ueff, 0,  3000,   1e-6)
    pen += qout(R,  -20,    10,   1e-3)
    pen += qout(N,    0,    12,   1e-3)
    pen += qout(Q,  -20,    10,   1e-3)
    pen += 1e-5 * float(np.sum(sigmas ** 2))
    for r in (rho_AU, rho_RN):
        if abs(r) > 0.995:
            pen += 1e-4 * (abs(r) - 0.995) ** 2
    if ridge_Q > 0:
        pen += ridge_Q * Q ** 2
    return pen


def domain_reg(mu, compiled, lam_AU, lam_RN, lam_Q, eps=1e-6):
    A, Ueff, R, N, Q = mu
    pen = 0.0
    if lam_AU > 0:
        tgt, sd = compiled["AU"]["mean"], np.maximum(compiled["AU"]["sd"], eps)
        pen += lam_AU * ((A - tgt[0]) ** 2 / sd[0] ** 2
                         + (Ueff - tgt[1]) ** 2 / sd[1] ** 2)
    if lam_RN > 0:
        tgt, sd = compiled["RN"]["mean"], np.maximum(compiled["RN"]["sd"], eps)
        pen += lam_RN * ((R - tgt[0]) ** 2 / sd[0] ** 2
                         + (N - tgt[1]) ** 2 / sd[1] ** 2)
    if lam_Q > 0:
        tgt, sd = compiled["Q"]["mean"], np.maximum(compiled["Q"]["sd"], eps)
        pen += lam_Q * (Q - tgt[0]) ** 2 / sd[0] ** 2
    return float(pen)


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------

def objective(x, rows, qs, Z, compiled, lam_AU, lam_RN, lam_Q,
              ridge_Q=0.0, fixed_rho_AU=None, fixed_rho_RN=None):
    mu, sigmas, rho_AU, rho_RN = unpack(x, fixed_rho_AU, fixed_rho_RN)
    total, n_eff = 0.0, 0.0

    for row in rows:
        T   = row["T"]
        w   = row.get("weight", 1.0)
        lnq = mc_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, Z)

        tgt = row["tgt"]          # precomputed in build_rows

        resid  = lnq - tgt
        total += w * float(np.dot(resid, resid))
        n_eff += w * len(qs)

    loss  = total / max(n_eff, 1.0)
    loss += penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=ridge_Q)
    loss += domain_reg(mu, compiled, lam_AU, lam_RN, lam_Q)
    return loss


# ---------------------------------------------------------------------------
# Post-fit skewness from Q marginal samples
# ---------------------------------------------------------------------------

def compute_Q_skewness(mu, sigmas, rho_AU, rho_RN, K=200000, seed=999):
    rng  = np.random.default_rng(seed)
    Z    = rng.standard_normal((K, 5))
    th   = draw_params(mu, sigmas, rho_AU, rho_RN, Z)
    Q_s  = th[:, 4]
    return {
        "Q_mean":  float(np.mean(Q_s)),
        "Q_sigma": float(np.std(Q_s)),
        "Q_skew":  float(scipy_skew(Q_s)),
        "Q_p10":   float(np.percentile(Q_s, 10)),
        "Q_p50":   float(np.percentile(Q_s, 50)),
        "Q_p90":   float(np.percentile(Q_s, 90)),
    }


# ---------------------------------------------------------------------------
# Data loading — reads phase 3 output CSVs directly
# ---------------------------------------------------------------------------

def _ensure_cols(df, required, label):
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: {label} missing columns: {missing}", file=sys.stderr)
        sys.exit(1)


def load_params_csv(path, label):
    df = pd.read_csv(path)
    _ensure_cols(df, {"T", "Au", "Uu", "Ru", "Nu", "Qu",
                       "As", "Us", "Rs", "Ns", "Qs"}, label)
    return df.reset_index(drop=True)


def load_ac_tsv(path):
    """Raw ccfit2 output — T, tau_mean/tau_mu, alpha."""
    df = pd.read_csv(path, sep=None, engine="python", header="infer")
    df.columns = [c.lower().strip() for c in df.columns]
    if "tau_mu" in df.columns:
        df = df.rename(columns={"tau_mu": "tau_mean"})
    if "t" in df.columns:
        df = df.rename(columns={"t": "T"})
    if not {"T", "tau_mean", "alpha"}.issubset(df.columns):
        # headerless fallback
        df2 = pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, :3]
        df2.columns = ["T", "tau_mean", "alpha"]
        return df2.astype(float)
    return df[["T", "tau_mean", "alpha"]].astype(float)


def load_dc_phase2(path):
    df = pd.read_csv(path)
    _ensure_cols(df, {"T", "mu_ln", "sigma1_ln", "sigma2_ln"}, "dc_phase2")
    # NOTE: the previous version substituted e^<ln tau> (the mean) for the
    # TPN location.  The TPN quantile function requires the MODE; for a TPN
    # mean = mode + sqrt(2/pi)(s2 - s1), so that substitution shifted every
    # DC target systematically.  With the corrected dc_phase2.py, mu_ln is
    # the mode of the correct density and is used directly — and if
    # (tau_star, beta) are present we bypass the TPN entirely and use exact
    # stretched-exponential quantiles as targets.
    if {"tau_star", "beta"}.issubset(df.columns):
        print("  dc_phase2: (tau_star, beta) present — DC targets will use "
              "EXACT stretched-exponential quantiles.")
    else:
        print("  WARNING: dc_phase2 file lacks tau_star/beta columns — "
              "falling back to two-piece-normal targets anchored at mu_ln "
              "(the mode). Re-run the corrected dc_phase2.py for exact targets.",
              file=sys.stderr)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build the unified row list for the objective
# ---------------------------------------------------------------------------

def build_rows(df_ac_params, df_ac_tsv,
               df_dc_params, df_dc_phase2,
               windows, weights, qs):
    """
    Build a flat list of row dicts for the objective function.

    Targets are PRECOMPUTED here (they do not depend on the optimisation
    variables) and stored per row as "tgt"; the objective just reads them.

    For AC rows:
      - If ac_tsv is provided: exact Generalised Debye quantile targets from
        raw (tau_mean, alpha)
      - If not: the AC params CSV doesn't store raw tau/alpha, so these rows
        contribute only through the domain regulariser, not the quantile loss.
        A warning is printed.

    For DC rows:
      - If dc_phase2 has (tau_star, beta): exact SEF quantile targets
      - Else: TPN targets anchored at mu_ln (the MODE — never the mean)
      - If dc_phase2 not provided: same caveat as AC above.

    Weights are assigned by (experiment_source, window_type):
      AC orbach rows  → weights['ac_orbach']
      AC raman  rows  → weights['ac_raman']
      AC qtm    rows  → weights['ac_qtm']
      DC qtm    rows  → weights['dc_qtm']
      DC raman  rows  → weights['dc_raman']
    """
    rows = []

    # ── AC rows ─────────────────────────────────────────────────────────────
    if df_ac_params is not None:
        if df_ac_tsv is None:
            print("  WARNING: --ac_tsv not provided. AC rows contribute only "
                  "via domain regulariser, not quantile loss.", file=sys.stderr)
        else:
            # align on T
            T_ac = set(df_ac_params["T"].values)
            dft  = df_ac_tsv[df_ac_tsv["T"].isin(T_ac)].sort_values("T").reset_index(drop=True)
            for _, r in dft.iterrows():
                T      = float(r["T"])
                wtype  = _window_type(T, windows, "ac")
                weight = weights.get(f"ac_{wtype}", 1.0)
                rows.append({
                    "T":           T,
                    "target_type": "gd_exact",
                    "fk_tau_mean": float(r["tau_mean"]),
                    "fk_alpha":    float(r["alpha"]),
                    "tgt":         fk_ln_quantiles(float(r["tau_mean"]),
                                                   float(r["alpha"]), qs),
                    "weight":      weight,
                    "source":      "ac",
                    "wtype":       wtype,
                })

    # ── DC rows ─────────────────────────────────────────────────────────────
    if df_dc_params is not None:
        if df_dc_phase2 is None:
            print("  WARNING: --dc_phase2 not provided. DC rows contribute only "
                  "via domain regulariser, not quantile loss.", file=sys.stderr)
        else:
            T_dc = set(df_dc_params["T"].values)
            dfp2 = df_dc_phase2[df_dc_phase2["T"].isin(T_dc)].sort_values("T").reset_index(drop=True)
            dc_exact = {"tau_star", "beta"}.issubset(dfp2.columns)
            for _, r in dfp2.iterrows():
                T      = float(r["T"])
                wtype  = _window_type(T, windows, "dc")
                weight = weights.get(f"dc_{wtype}", 1.0)
                if dc_exact:
                    tgt   = sef_ln_quantiles(float(r["tau_star"]),
                                             float(r["beta"]), qs)
                    ttype = "sef_exact"
                else:
                    # fallback: TPN anchored at the MODE (mu_ln), never the mean
                    tgt   = tpn_ln_quantiles(float(r["mu_ln"]),
                                             float(r["sigma1_ln"]),
                                             float(r["sigma2_ln"]), qs)
                    ttype = "tpn"
                rows.append({
                    "T":           T,
                    "target_type": ttype,
                    "tpn_mu_ln":   float(r["mu_ln"]),
                    "tpn_s1":      float(r["sigma1_ln"]),
                    "tpn_s2":      float(r["sigma2_ln"]),
                    "tgt":         tgt,
                    "weight":      weight,
                    "source":      "dc",
                    "wtype":       wtype,
                })

    return rows


def _window_type(T, windows, source):
    """Return the dominant window name for temperature T."""
    # priority order: orbach > raman > qtm for AC; qtm > raman for DC
    if source == "ac":
        order = ["orbach", "raman", "qtm"]
    else:
        order = ["qtm", "raman", "orbach"]
    for name in order:
        w = windows.get(name)
        if w and w[0] <= T <= w[1]:
            return name
    return "other"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_range(spec, nmax):
    parts = spec.split(":")
    start = int(parts[0]) if parts[0] else 0
    end   = int(parts[1]) if parts[1] else nmax
    step  = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    return range(start, min(end, nmax), step)


def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ phase 4 — combined AC + DC global fit"
    )

    # ── inputs ──────────────────────────────────────────────────────────────
    inp = ap.add_argument_group("Inputs")
    inp.add_argument("--ac_params",  default=None,
                     help="ac_montecarlo.py output CSV")
    inp.add_argument("--ac_tsv",     default=None,
                     help="Raw AC data TSV (T, tau_mean/tau_mu, alpha) — "
                          "needed for FK quantile targets")
    inp.add_argument("--dc_params",  default=None,
                     help="dc_montecarlo.py output CSV")
    inp.add_argument("--dc_phase2",  default=None,
                     help="dc_phase2.py output CSV (T, mu_ln, sigma1_ln, sigma2_ln) — "
                          "needed for TPN quantile targets")

    # ── temperature windows ─────────────────────────────────────────────────
    win = ap.add_argument_group(
        "Temperature windows",
        "Used to (a) assign domain compilation sources and "
        "(b) weight rows by experiment type. "
        "Each window should correspond to the dominant process at those temperatures. "
        "AC data is expected to cover orbach/raman; DC data covers qtm/raman."
    )
    win.add_argument("--orbach_window", default="", metavar="Tlo,Thi",
                     help="Orbach-dominant temperature range (AC data)")
    win.add_argument("--raman_window",  default="", metavar="Tlo,Thi",
                     help="Raman-dominant temperature range (AC or DC data)")
    win.add_argument("--qtm_window",    default="", metavar="Tlo,Thi",
                     help="QTM-dominant temperature range (DC data)")
    # keep old names as aliases for backward compatibility
    win.add_argument("--AU_window",     default="",
                     help="Alias for --orbach_window")
    win.add_argument("--RN_window",     default="",
                     help="Alias for --raman_window")
    win.add_argument("--Q_window",      default="",
                     help="Alias for --qtm_window")

    # ── per-source per-window weights ────────────────────────────────────────
    wt = ap.add_argument_group(
        "Row weights",
        "Multiplicative weight applied to the squared-residual loss for each "
        "row by (source, window) combination. Default 1.0 = equal weighting. "
        "Increase to trust that experiment+window more in the global fit."
    )
    wt.add_argument("--w_ac_orbach", type=float, default=3.0,
                    help="Weight for AC rows in Orbach window  (A/Ueff signal)")
    wt.add_argument("--w_ac_raman",  type=float, default=1.5,
                    help="Weight for AC rows in Raman window   (R/n signal)")
    wt.add_argument("--w_ac_qtm",    type=float, default=1.0,
                    help="Weight for AC rows in QTM window     (mixed)")
    wt.add_argument("--w_dc_qtm",    type=float, default=3.0,
                    help="Weight for DC rows in QTM window     (Q signal)")
    wt.add_argument("--w_dc_raman",  type=float, default=1.5,
                    help="Weight for DC rows in Raman window   (R/n signal)")
    wt.add_argument("--w_dc_orbach", type=float, default=1.0,
                    help="Weight for DC rows in Orbach window  (not expected)")

    # ── domain regularisation ────────────────────────────────────────────────
    reg = ap.add_argument_group("Domain regularisation")
    reg.add_argument("--lambda_AU", type=float, default=0.5,
                     help="Regularisation strength toward compiled A/Ueff (default 0.5 — "
                          "kept low because per-row A/Ueff are degenerate)")
    reg.add_argument("--lambda_RN", type=float, default=1.0)
    reg.add_argument("--lambda_Q",  type=float, default=1.0)

    # ── initial guess overrides ──────────────────────────────────────────────
    x0g = ap.add_argument_group(
        "Initial guess overrides",
        "Seed the global fit with known-good starting values. "
        "Strongly recommended for A and Ueff, since the per-row AC montecarlo "
        "values are degenerate and the compiled median may be far from the true value. "
        "Use values from ac_phase2.py (mu_A, mu_U) or literature."
    )
    x0g.add_argument("--x0_A",    type=float, default=None,
                     help="Starting guess for A (e.g. from ac_phase2 mu_A)")
    x0g.add_argument("--x0_Ueff", type=float, default=None,
                     help="Starting guess for Ueff in K (e.g. from ac_phase2 mu_U)")
    x0g.add_argument("--x0_R",    type=float, default=None,
                     help="Starting guess for R")
    x0g.add_argument("--x0_n",    type=float, default=None,
                     help="Starting guess for n (Raman exponent)")
    x0g.add_argument("--x0_Q",    type=float, default=None,
                     help="Starting guess for Q")

    # ── optimiser ────────────────────────────────────────────────────────────
    opt = ap.add_argument_group("Optimiser")
    opt.add_argument("--K",       type=int,   default=30000)
    opt.add_argument("--seed",    type=int,   default=202)
    opt.add_argument("--maxiter", type=int,   default=1200)
    opt.add_argument("--ridgeQ",  type=float, default=0.0)
    opt.add_argument("--qs",      default="0.02,0.10,0.25,0.50,0.75,0.90,0.98")
    opt.add_argument("--fix_rho", action="store_true",
                     help="Fix rho_AU and rho_RN throughout the fit. "
                          "Recommended for stage 1 — prevents the optimizer "
                          "from using correlations as a free variable to absorb "
                          "spread (parameter trading). Use rho=0 for the most "
                          "conservative assumption; release in stage 2 to refine.")
    opt.add_argument("--rho_AU",  type=float, default=0.0,
                     help="Value to fix rho_AU at when --fix_rho is set "
                          "(default 0.0 — maximally non-committal; use 0.9 only "
                          "if you have independent evidence of strong compensation)")
    opt.add_argument("--rho_RN",  type=float, default=0.0,
                     help="Value to fix rho_RN at when --fix_rho is set "
                          "(default 0.0 — maximally non-committal; use -0.9 only "
                          "if you have independent evidence of strong anti-correlation)")

    opt.add_argument("--show_compile", action="store_true",
                     help="Print the raw windowed (T, param, spread) rows that "
                          "feed each compiled domain target, so you can see "
                          "whether the compiled value reflects the bulk of the "
                          "window or a couple of outliers.")

    opt.add_argument("--out",     default="global_combined.csv")
    opt.add_argument("--clear",   action="store_true")

    args = ap.parse_args()

    global _VERBOSE_COMPILE
    _VERBOSE_COMPILE = bool(args.show_compile)

    has_ac = args.ac_params is not None
    has_dc = args.dc_params is not None
    if not has_ac and not has_dc:
        print("ERROR: supply at least one of --ac_params or --dc_params.",
              file=sys.stderr)
        sys.exit(1)

    # resolve window aliases (new names take precedence)
    orbach_w = parse_window(args.orbach_window or args.AU_window)
    raman_w  = parse_window(args.raman_window  or args.RN_window)
    qtm_w    = parse_window(args.qtm_window    or args.Q_window)
    windows  = {"orbach": orbach_w, "raman": raman_w, "qtm": qtm_w}

    weights = {
        "ac_orbach": args.w_ac_orbach,
        "ac_raman":  args.w_ac_raman,
        "ac_qtm":    args.w_ac_qtm,
        "dc_qtm":    args.w_dc_qtm,
        "dc_raman":  args.w_dc_raman,
        "dc_orbach": args.w_dc_orbach,
        "ac_other":  1.0,
        "dc_other":  1.0,
    }

    # guard the CLI-supplied fixed correlations (the old pipeline once
    # emitted |rho| > 1, which draw_params silently laundered)
    args.rho_AU = float(np.clip(args.rho_AU, 0.0, RHO_MAX))
    args.rho_RN = float(np.clip(args.rho_RN, -RHO_MAX, 0.0))

    # quantile grid must exist before build_rows (targets precomputed there)
    qs = np.array([float(q.strip()) for q in args.qs.split(",")], dtype=float)

    # load params CSVs
    df_ac_p  = load_params_csv(args.ac_params, "AC params") if has_ac else None
    df_dc_p  = load_params_csv(args.dc_params, "DC params") if has_dc else None
    df_ac_t  = load_ac_tsv(args.ac_tsv)       if args.ac_tsv    else None
    df_dc_p2 = load_dc_phase2(args.dc_phase2) if args.dc_phase2 else None

    # build objective rows (targets precomputed once here)
    rows = build_rows(df_ac_p, df_ac_t, df_dc_p, df_dc_p2, windows, weights, qs)

    # report
    src_wtype = [(r["source"], r["wtype"]) for r in rows]
    for src in ["ac", "dc"]:
        for wt_name in ["orbach", "raman", "qtm", "other"]:
            n = sum(1 for s, w in src_wtype if s == src and w == wt_name)
            if n:
                key = f"{src}_{wt_name}"
                print(f"  {src.upper()} {wt_name:>7} rows: {n:>3}  "
                      f"(weight × {weights[key]:.1f})")

    if not rows:
        print("WARNING: no quantile-fit rows — fitting via domain regulariser only.")

    # build initials
    x0, compiled = build_initials(df_ac_p, df_dc_p, orbach_w, raman_w, qtm_w)
    mu0, sig0, _, _ = unpack(x0)

    print(f"\n  Compiled domain targets (initial guess):")
    print(f"    A    = {compiled['AU']['mean'][0]:.4f}  spread = {compiled['AU']['sd'][0]:.4f}")
    print(f"    Ueff = {compiled['AU']['mean'][1]:.2f}  spread = {compiled['AU']['sd'][1]:.2f}")
    print(f"    R    = {compiled['RN']['mean'][0]:.4f}  spread = {compiled['RN']['sd'][0]:.4f}")
    print(f"    N    = {compiled['RN']['mean'][1]:.4f}  spread = {compiled['RN']['sd'][1]:.4f}")
    print(f"    Q    = {compiled['Q']['mean'][0]:.4f}  spread = {compiled['Q']['sd'][0]:.4f}")

    # apply CLI overrides to the initial guess means
    override_map = [
        (args.x0_A,    0, "A"),
        (args.x0_Ueff, 1, "Ueff"),
        (args.x0_R,    2, "R"),
        (args.x0_n,    3, "n"),
        (args.x0_Q,    4, "Q"),
    ]
    overridden = []
    for val, idx, name in override_map:
        if val is not None:
            x0[idx] = val
            # also update the compiled domain target to match
            if name in ("A", "Ueff"):
                compiled["AU"]["mean"][{"A": 0, "Ueff": 1}[name]] = val
            elif name in ("R", "n"):
                compiled["RN"]["mean"][{"R": 0, "n": 1}[name]] = val
            elif name == "Q":
                compiled["Q"]["mean"][0] = val
            overridden.append(f"{name}={val}")

    if overridden:
        print(f"\n  x0 overrides applied: {', '.join(overridden)}")

    mu0, sig0, _, _ = unpack(x0)
    print(f"\n  Starting point:")
    print(f"    A={mu0[0]:.4f} Ueff={mu0[1]:.2f} R={mu0[2]:.4f} "
          f"N={mu0[3]:.4f} Q={mu0[4]:.4f}")
    print(f"    sA={sig0[0]:.4f} sU={sig0[1]:.3f} sR={sig0[2]:.4f} "
          f"sN={sig0[3]:.4f} sQ={sig0[4]:.4f}")

    # fixed rho values (None = free to optimise)
    fixed_rho_AU = args.rho_AU if args.fix_rho else None
    fixed_rho_RN = args.rho_RN if args.fix_rho else None

    if args.fix_rho:
        print(f"\n  Correlations fixed: rho_AU={args.rho_AU}  rho_RN={args.rho_RN}")
        print(f"  (10 free parameters instead of 12)")
        # when rho is fixed x[10] and x[11] are unused — trim x0 to 10 params
        x0 = x0[:10]
    else:
        print(f"\n  Correlations free (12 parameters)")

    Z  = np.random.default_rng(args.seed).standard_normal((args.K, 5))

    def obj(x):
        return objective(x, rows, qs, Z, compiled,
                         args.lambda_AU, args.lambda_RN, args.lambda_Q,
                         ridge_Q=args.ridgeQ,
                         fixed_rho_AU=fixed_rho_AU,
                         fixed_rho_RN=fixed_rho_RN)

    mu0, sig0, _, _ = unpack(x0, fixed_rho_AU, fixed_rho_RN)
    print(f"\n  Starting: A={mu0[0]:.4f}  Ueff={mu0[1]:.2f}  "
          f"R={mu0[2]:.4f}  N={mu0[3]:.4f}  Q={mu0[4]:.4f}")

    print(f"\n  Nelder-Mead  maxiter={args.maxiter}  K={args.K}  "
          f"rows={len(rows)}...")
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": args.maxiter,
                            "xatol": 2e-3, "fatol": 2e-4, "disp": True})

    # pad res.x back to 12 elements if rho was fixed
    res_x = res.x
    if args.fix_rho:
        # placeholders only — unpack() below receives the fixed values directly
        rho_AU_enc = _x_from_rho(max(args.rho_AU,  1e-3))
        rho_RN_enc = _x_from_rho(max(-args.rho_RN, 1e-3))
        res_x = np.concatenate([res_x, [rho_AU_enc, rho_RN_enc]])

    mu, sigmas, rho_AU, rho_RN = unpack(res_x, fixed_rho_AU, fixed_rho_RN)
    A, Ueff, R, N, Q = mu
    sA, sU, sR, sN, sQ = sigmas

    # post-fit Q skewness from sampling
    q_stats = compute_Q_skewness(mu, sigmas, rho_AU, rho_RN,
                                 K=200000, seed=args.seed + 1)

    # output
    if args.clear and os.path.exists(args.out):
        open(args.out, "w").close()

    out_row = pd.DataFrame([{
        "A": A, "Ueff": Ueff, "R": R, "N": N, "Q": Q,
        "sA": sA, "sU": sU, "sR": sR, "sN": sN, "sQ": sQ,
        "rho_AU": rho_AU, "rho_RN": rho_RN,
        "Q_mean":  q_stats["Q_mean"],
        "Q_sigma": q_stats["Q_sigma"],
        "Q_skew":  q_stats["Q_skew"],
        "Q_p10":   q_stats["Q_p10"],
        "Q_p50":   q_stats["Q_p50"],
        "Q_p90":   q_stats["Q_p90"],
        "loss":    float(res.fun),
        "success": bool(res.success),
        "nit":     int(getattr(res, "nit", -1)),
        "n_rows":  len(rows),
        "n_ac":    sum(1 for r in rows if r["source"] == "ac"),
        "n_dc":    sum(1 for r in rows if r["source"] == "dc"),
        "orbach_window": str(orbach_w) if orbach_w else "",
        "raman_window":  str(raman_w)  if raman_w  else "",
        "qtm_window":    str(qtm_w)    if qtm_w    else "",
        "lambda_AU": args.lambda_AU,
        "lambda_RN": args.lambda_RN,
        "lambda_Q":  args.lambda_Q,
        "w_ac_orbach": args.w_ac_orbach,
        "w_ac_raman":  args.w_ac_raman,
        "w_dc_qtm":    args.w_dc_qtm,
        "w_dc_raman":  args.w_dc_raman,
    }])
    header_needed = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    out_row.to_csv(args.out, index=False, mode="a", header=header_needed)

    # summary
    print("\n" + "=" * 62)
    print("  TAURnQ Phase 4 — Combined AC + DC global fit")
    print("=" * 62)
    n_ac = sum(1 for r in rows if r["source"] == "ac")
    n_dc = sum(1 for r in rows if r["source"] == "dc")
    print(f"  AC rows: {n_ac}   DC rows: {n_dc}")
    print(f"  Windows  orbach={str(orbach_w) if orbach_w else 'not set'}  "
          f"raman={str(raman_w) if raman_w else 'not set'}  "
          f"qtm={str(qtm_w) if qtm_w else 'not set'}")
    print()
    print(f"  {'Param':>6}  {'mean':>10}  {'sigma':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for name, val, sig in [("A", A, sA), ("Ueff", Ueff, sU),
                            ("R", R, sR), ("N", N, sN), ("Q", Q, sQ)]:
        print(f"  {name:>6}  {val:>10.5f}  {sig:>10.5f}")
    print()
    print(f"  Q distribution:  mean={q_stats['Q_mean']:.4f}  "
          f"sigma={q_stats['Q_sigma']:.4f}  skew={q_stats['Q_skew']:+.4f}")
    print(f"  Q percentiles:   p10={q_stats['Q_p10']:.3f}  "
          f"p50={q_stats['Q_p50']:.3f}  p90={q_stats['Q_p90']:.3f}")
    print(f"\n  rho_AU={rho_AU:.4f}  rho_RN={rho_RN:.4f}")
    print(f"  loss={float(res.fun):.5g}  success={res.success}  "
          f"nit={getattr(res, 'nit', '?')}")
    print(f"\n  Saved → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()