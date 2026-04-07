#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os, json

# ------------------ FK helpers ------------------
def fk_g(alpha):
    return 1.82 * np.sqrt(alpha) / (1 - alpha)

def fk_ln_quantiles(tau_mean, alpha, qs):
    g = fk_g(alpha)
    return np.log(tau_mean) + norm.ppf(qs) * g

# ------------------ Model ------------------
def rate_model(T, A, Ueff, R, N, Q):
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))  # Orbach
    term2 = 10.0**(R)  * (T_safe**N)                              # Raman
    term3 = 10.0**(-Q)                                             # QTM
    return term1 + term2 + term3

def draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z):
    A,Ueff,R,N,Q = mu
    sA,sU,sR,sN,sQ = sigmas

    L_AU = np.array([[sA,                     0.0],
                     [rho_AU * sU,  sU * np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    L_RN = np.array([[sR,                     0.0],
                     [rho_RN * sN,  sN * np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])

    Z_AU = Z[:, :2]
    Z_RN = Z[:, 2:4]
    zQ   = Z[:, 4]

    eps_AU = Z_AU @ L_AU.T
    eps_RN = Z_RN @ L_RN.T
    eps_Q  = zQ * sQ

    A_s = A + eps_AU[:, 0]
    U_s = Ueff + eps_AU[:, 1]
    R_s = R + eps_RN[:, 0]
    N_s = N + eps_RN[:, 1]
    Q_s = Q + eps_Q
    return np.column_stack([A_s, U_s, R_s, N_s, Q_s])

def quantiles_all_T(Ts, mu, sigmas, rho_AU, rho_RN, qs, Z):
    theta_draws = draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z)
    A_s, U_s, R_s, N_s, Q_s = theta_draws.T
    out = np.zeros((len(Ts), len(qs)), dtype=float)
    for i, T in enumerate(Ts):
        r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
        tau = 1.0 / np.maximum(r, 1e-300)
        ln_tau = np.log(tau)
        out[i, :] = np.quantile(ln_tau, qs)
    return out

# ------------------ Packing ------------------
def unpack(x):
    mu     = x[0:5]                      # A,Ueff,R,N,Q
    sigmas = 1e-3 + np.exp(x[5:10])      # log-positives -> positives (≥1e-3)
    eta_AU = x[10]
    eta_RN = x[11]
    rho_AU = np.tanh(eta_AU)**2          # ≥ 0
    rho_RN = - (np.tanh(eta_RN)**2)      # ≤ 0
    return mu, sigmas, rho_AU, rho_RN

# ------------------ Utilities for windows ------------------
def parse_window(s):
    if s is None or s.strip() == "":
        return None
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("Window must be 'Tmin,Tmax'")
    lo, hi = min(parts), max(parts)
    return lo, hi

def in_window(T_arr, window):
    if window is None:
        return np.ones_like(T_arr, dtype=bool)
    lo, hi = window
    return (T_arr >= lo) & (T_arr <= hi)

def ivw_mean(x, s):
    w = 1.0 / np.maximum(s**2, 1e-12)
    return float(np.nansum(w * x) / np.maximum(np.nansum(w), 1e-12))

def robust_med(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.nanmedian(x))

# ------------------ Domain-aware initials ------------------
def window_compilation(dfp, mask, which):
    # which in {'AU','RN','Q'}
    if which == 'AU':
        Abar = ivw_mean(dfp.loc[mask, "Au"].values, dfp.loc[mask, "As"].values)
        Ubar = ivw_mean(dfp.loc[mask, "Uu"].values, dfp.loc[mask, "Us"].values)
        sA   = robust_med(dfp.loc[mask, "As"].values)
        sU   = robust_med(dfp.loc[mask, "Us"].values)
        return (Abar, Ubar), (sA, sU)
    elif which == 'RN':
        Rbar = ivw_mean(dfp.loc[mask, "Ru"].values, dfp.loc[mask, "Rs"].values)
        Nbar = ivw_mean(dfp.loc[mask, "Nu"].values, dfp.loc[mask, "Ns"].values)
        sR   = robust_med(dfp.loc[mask, "Rs"].values)
        sN   = robust_med(dfp.loc[mask, "Ns"].values)
        return (Rbar, Nbar), (sR, sN)
    elif which == 'Q':
        Qbar = ivw_mean(dfp.loc[mask, "Qu"].values, dfp.loc[mask, "Qs"].values)
        sQ   = robust_med(dfp.loc[mask, "Qs"].values)
        return (Qbar,), (sQ,)
    else:
        raise ValueError("unknown domain key")

def initials_from_csv_domain_aware(dfp, AU_w, RN_w, Q_w):
    Tvals = dfp["T"].values
    m_AU = in_window(Tvals, AU_w)
    m_RN = in_window(Tvals, RN_w)
    m_Q  = in_window(Tvals, Q_w)

    # compile windowed means/SDs
    (A0, U0), (sA0, sU0) = window_compilation(dfp, m_AU, 'AU')
    (R0, N0), (sR0, sN0) = window_compilation(dfp, m_RN, 'RN')
    (Q0,),     (sQ0,)    = window_compilation(dfp, m_Q,  'Q')

    # safe fallbacks if any SD is NaN
    sA0 = 1e-3 if not np.isfinite(sA0) else sA0
    sU0 = 1e-3 if not np.isfinite(sU0) else sU0
    sR0 = 1e-3 if not np.isfinite(sR0) else sR0
    sN0 = 1e-3 if not np.isfinite(sN0) else sN0
    sQ0 = 1e-3 if not np.isfinite(sQ0) else sQ0

    # empirical sign-consistent rho seeds from whole set (stable)
    def safe_corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return 0.0
        return float(np.corrcoef(x[m], y[m])[0,1])

    rho_AU0 = np.clip(safe_corr(dfp["Au"].values, dfp["Uu"].values), 0.0, 0.95)
    rho_RN0 = -np.clip(abs(safe_corr(dfp["Ru"].values, dfp["Nu"].values)), 0.0, 0.95)

    x0 = np.array([A0,U0,R0,N0,Q0,
                   np.log(max(sA0,1e-6)), np.log(max(sU0,1e-6)),
                   np.log(max(sR0,1e-6)), np.log(max(sN0,1e-6)),
                   np.log(max(sQ0,1e-6)),
                   np.arctanh(np.sqrt(0.99)),
                   np.arctanh(np.sqrt(0.99))], dtype=float)
    # also return the compiled targets for domain regularizers
    compiled = {
        "AU": {"mean": np.array([A0, U0], dtype=float),
               "sd":   np.array([sA0, sU0], dtype=float)},
        "RN": {"mean": np.array([R0, N0], dtype=float),
               "sd":   np.array([sR0, sN0], dtype=float)},
        "Q":  {"mean": np.array([Q0], dtype=float),
               "sd":   np.array([sQ0], dtype=float)}
    }
    return x0, compiled

# ------------------ Penalties ------------------
def penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=0.0):
    A,Ueff,R,N,Q   = mu
    pen = 0.0

    def quad_out(val, lo, hi, scale):
        if val < lo:  return scale * (lo - val)**2
        if val > hi:  return scale * (val - hi)**2
        return 0.0

    pen += quad_out(-A, 0.0, 30.0, 1e-3)
    pen += quad_out(Ueff,  0.0, 3000.0, 1e-6)
    pen += quad_out(R, -20.0, 10.0, 1e-3)
    pen += quad_out(N,  0.0, 12.0, 1e-3)
    pen += quad_out(Q, -20.0, 10.0, 1e-3)

    pen += 1e-5 * float(np.sum(sigmas**2))

    for r in (rho_AU, rho_RN):
        if abs(r) > 0.995:
            pen += 1e-4 * (abs(r) - 0.995)**2

    if ridge_Q > 0.0:
        pen += float(ridge_Q) * (Q**2)
    return pen

def domain_regularizer(mu, compiled, lam_AU, lam_RN, lam_Q, eps=1e-6):
    """ Pull (A,Ueff) toward AU-window compilation, (R,N) toward RN, Q toward Q-window. """
    A,Ueff,R,N,Q = mu
    pen = 0.0
    if lam_AU > 0.0:
        tgt, sd = compiled["AU"]["mean"], np.maximum(compiled["AU"]["sd"], eps)
        pen += lam_AU * ( (A - tgt[0])**2 / (sd[0]**2) + (Ueff - tgt[1])**2 / (sd[1]**2) )
    if lam_RN > 0.0:
        tgt, sd = compiled["RN"]["mean"], np.maximum(compiled["RN"]["sd"], eps)
        pen += lam_RN * ( (R - tgt[0])**2 / (sd[0]**2) + (N - tgt[1])**2 / (sd[1]**2) )
    if lam_Q > 0.0:
        tgt, sd = compiled["Q"]["mean"],  np.maximum(compiled["Q"]["sd"],  eps)
        pen += lam_Q * ( (Q - tgt[0])**2 / (sd[0]**2) )
    return float(pen)

# ------------------ Objective ------------------
def objective_global(x, T_all, tau_mu_all, alpha_all, qs, Z,
                     compiled, lam_AU, lam_RN, lam_Q,
                     ridge_Q=0.0, FK_mask=None):
    mu, sigmas, rho_AU, rho_RN = unpack(x)

    # FK quantile match (optionally masked by FK_window)
    if FK_mask is None:
        T_use = T_all
        tau_use = tau_mu_all
        alpha_use = alpha_all
        mask_rows = slice(None)
    else:
        mask_rows = np.where(FK_mask)[0]
        T_use = T_all[FK_mask]
        tau_use = tau_mu_all[FK_mask]
        alpha_use = alpha_all[FK_mask]

    lnq_hat = quantiles_all_T(T_use, mu, sigmas, rho_AU, rho_RN, qs, Z)
    targets = np.vstack([fk_ln_quantiles(tau_use[i], alpha_use[i], qs)
                         for i in range(len(T_use))])
    resid = (lnq_hat - targets).ravel()
    loss  = float(np.dot(resid, resid)) / max(resid.size, 1)  # MSE over Ts×qs

    # Generic box constraints / ridge
    loss += penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=ridge_Q)

    # NEW: domain regularizers (compile by windows)
    loss += domain_regularizer(mu, compiled, lam_AU, lam_RN, lam_Q)

    return loss

# ------------------ CLI ------------------
def parse_list_or_json(s, expected_len=None):
    try:
        val = json.loads(s)
        if isinstance(val, list):
            arr = [float(x) for x in val]
        else:
            raise ValueError
    except Exception:
        arr = [float(x.strip()) for x in s.split(",")]
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(arr)}")
    return arr

def main():
    ap = argparse.ArgumentParser(description="Global joint refit with domain-based (windowed) compilation.")
    ap.add_argument("--params_csv", default="tBuO2.tsv_mc_params.csv", help="CSV with per-T MC outputs: T, Au,Uu,Ru,Nu,Qu, As,Us,Rs,Ns,Qs")
    ap.add_argument("--tsv", default = "tBuO2.tsv", help="TSV/CSV with columns [T, tau_mu, alpha]")
    ap.add_argument("--qs", default="0.02,0.10,0.25,0.50,0.75,0.90,0.98",
                    help="Comma-separated quantiles in [0,1].")
    ap.add_argument("--K", type=int, default=30000, help="MC draws per objective evaluation")
    ap.add_argument("--seed", type=int, default=202, help="RNG seed for CRNs")
    ap.add_argument("--ridgeQ", type=float, default=0.0, help="Optional ridge on Q")
    ap.add_argument("--maxiter", type=int, default=1200, help="Nelder-Mead maxiter")

    # -------- NEW: domain windows and strengths --------
    ap.add_argument("--AU_window", default="73.81,100", help="Tmin,Tmax for Orbach (A,Ueff) compilation, e.g. '45,58'")
    ap.add_argument("--RN_window", default="40,71.9", help="Tmin,Tmax for Raman (R,N) compilation")
    ap.add_argument("--Q_window",  default="0,0", help="Tmin,Tmax for QTM (Q) compilation")
    ap.add_argument("--lambda_AU", type=float, default=1.0, help="Strength of A–U domain regularizer")
    ap.add_argument("--lambda_RN", type=float, default=1.0, help="Strength of R–N domain regularizer")
    ap.add_argument("--lambda_Q",  type=float, default=1.0, help="Strength of Q   domain regularizer")

    # Optional: restrict FK quantile matching to a window
    ap.add_argument("--FK_window", default="", help="Optional FK fit window 'Tmin,Tmax' applied to quantile matching")

    # (kept) regime seeding overrides if you still want them
    ap.add_argument("--use_regime_means", action="store_true",
                    help="Overwrite initial means (A,Ueff,R,N,Q) with regime seeds.")
    ap.add_argument("--use_regime_sds", action="store_true",
                    help="Overwrite initial SDs (sA,sU,sR,sN,sQ) with regime seeds.")
    ap.add_argument("--regime_means",
                    default="-11.093491,900.572427,-4.831255,3.963503,-0.182074",
                    help="Regime mean seeds A,Ueff,R,N,Q (CSV or JSON).")
    ap.add_argument("--regime_sds",
                    default="0.224577,14.365442,2.171177,1.392262,0.565583",
                    help="Regime SD seeds sA,sU,sR,sN,sQ (CSV or JSON).")

    ap.add_argument("--out", default="global_fit2.csv", help="Output CSV (one row with global params)")
    args = ap.parse_args()

    # read per-T params (for Ts and domain compilation)
    dfp = pd.read_csv(args.params_csv)
    if "T" not in dfp.columns:
        raise ValueError("params_csv must contain column 'T'.")
    T_list = np.array(sorted(np.unique(dfp["T"].values)), dtype=float)

    # read FK targets and align to those Ts
    dft = pd.read_csv(args.tsv, sep=None, engine="python", header=None).iloc[:, :3]
    dft.columns = ["T", "tau_mu", "alpha"]
    dft = dft.astype(float)
    dft = dft[dft["T"].isin(T_list)].sort_values("T")
    if len(dft) == 0:
        raise ValueError("No overlapping temperatures between params_csv and tsv.")
    T_all    = dft["T"].to_numpy(float)
    tau_mu   = dft["tau_mu"].to_numpy(float)
    alpha    = dft["alpha"].to_numpy(float)

    # quantiles
    qs = np.array([float(q.strip()) for q in args.qs.split(",") if q.strip() != ""], dtype=float)

    # parse windows
    AU_w = parse_window(args.AU_window)
    RN_w = parse_window(args.RN_window)
    Q_w  = parse_window(args.Q_window)
    FK_w = parse_window(args.FK_window)
    FK_mask = None if FK_w is None else in_window(T_all, FK_w)

    # domain-aware initials + compiled targets
    x0, compiled = initials_from_csv_domain_aware(dfp, AU_w, RN_w, Q_w)

    # optional overrides by regime seeds
    if args.use_regime_means or args.use_regime_sds:
        reg_means = parse_list_or_json(args.regime_means, expected_len=5)
        reg_sds   = parse_list_or_json(args.regime_sds,   expected_len=5)
        if args.use_regime_means:
            x0[0:5] = np.array(reg_means, dtype=float)
        if args.use_regime_sds:
            x0[5:10] = np.log(np.maximum(reg_sds, np.full(5, 1e-3)))

    # common random numbers for stability
    rng = np.random.default_rng(args.seed)
    Z = rng.standard_normal(size=(args.K, 5))

    def obj(x):
        return objective_global(
            x, T_all, tau_mu, alpha, qs, Z,
            compiled=compiled,
            lam_AU=args.lambda_AU, lam_RN=args.lambda_RN, lam_Q=args.lambda_Q,
            ridge_Q=args.ridgeQ, FK_mask=FK_mask
        )

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": args.maxiter, "xatol": 2e-3, "fatol": 2e-4, "disp": True})

    mu, sigmas, rho_AU, rho_RN = unpack(res.x)
    A,Ueff,R,N,Q = mu
    sA,sU,sR,sN,sQ = sigmas

    out_row = pd.DataFrame([{
        "A":A, "Ueff":Ueff, "R":R, "N":N, "Q":Q,
        "sA":sA, "sU":sU, "sR":sR, "sN":sN, "sQ":sQ,
        "rho_AU":rho_AU, "rho_RN":rho_RN,
        "loss": float(res.fun),
        "success": bool(res.success),
        "nit": int(getattr(res, "nit", -1)),
        "AU_window": args.AU_window,
        "RN_window": args.RN_window,
        "Q_window":  args.Q_window,
        "lambda_AU": args.lambda_AU,
        "lambda_RN": args.lambda_RN,
        "lambda_Q":  args.lambda_Q,
        "FK_window": args.FK_window
    }])
    header_needed = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    out_row.to_csv(args.out, index=False, mode="a", header=header_needed)

    print("\n=== Global joint fit (Domain-weighted) ===")
    print(f"T points: {len(T_all)} | draws K={args.K} | ridgeQ={args.ridgeQ}")
    print(f"Windows  AU={args.AU_window or 'ALL'}  RN={args.RN_window or 'ALL'}  Q={args.Q_window or 'ALL'}  FK={args.FK_window or 'ALL'}")
    print(f"λ_AU={args.lambda_AU}  λ_RN={args.lambda_RN}  λ_Q={args.lambda_Q}")
    print(f"A={A:.6f}  Ueff={Ueff:.6f}  R={R:.6f}  N={N:.6f}  Q={Q:.6f}")
    print(f"sA={sA:.6f} sU={sU:.6f} sR={sR:.6f} sN={sN:.6f} sQ={sQ:.6f}")
    print(f"rho_AU={rho_AU:.6f} (≥0),  rho_RN={rho_RN:.6f} (≤0)")
    print(f"loss={float(res.fun):.6g}  success={res.success}  nit={getattr(res,'nit','?')}")

if __name__ == "__main__":
    main()
