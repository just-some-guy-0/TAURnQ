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
def rate_model(T, A, U, R, N, Q):
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0**(-A) * np.exp(-U / np.maximum(T_safe, 1e-12))
    term2 = 10.0**(R)  * (T_safe**N)
    term3 = 10.0**(Q)
    return term1 + term2 + term3

def draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z):
    A,U,R,N,Q = mu
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
    U_s = U + eps_AU[:, 1]
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

# ------------------ Objective ------------------
def unpack(x):
    mu     = x[0:5]                      # A,U,R,N,Q
    # tiny floor to avoid SD collapse
    sigmas = 1e-3 + np.exp(x[5:10])      # log-positives -> positives (≥1e-3)
    eta_AU = x[10]
    eta_RN = x[11]
    # enforce signs by construction
    rho_AU = np.tanh(eta_AU)**2          # ≥ 0
    rho_RN = - (np.tanh(eta_RN)**2)      # ≤ 0
    return mu, sigmas, rho_AU, rho_RN

def penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=0.0):
    A,U,R,N,Q   = mu
    pen = 0.0

    def quad_out(val, lo, hi, scale):
        if val < lo:  return scale * (lo - val)**2
        if val > hi:  return scale * (val - hi)**2
        return 0.0

    pen += quad_out(-A, 0.0, 30.0, 1e-3)
    pen += quad_out(U,  0.0, 3000.0, 1e-6)
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

def objective_global(x, T_all, tau_mu_all, alpha_all, qs, Z, w_q=1.0, ridge_Q=0.0):
    mu, sigmas, rho_AU, rho_RN = unpack(x)
    lnq_hat = quantiles_all_T(T_all, mu, sigmas, rho_AU, rho_RN, qs, Z)
    targets = np.vstack([fk_ln_quantiles(tau_mu_all[i], alpha_all[i], qs)
                         for i in range(len(T_all))])
    resid = (lnq_hat - targets).ravel()
    loss  = float(np.dot(resid, resid)) / max(resid.size, 1)  # MSE over all Ts×qs
    loss += penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=ridge_Q)
    return loss

# ------------------ Initials from per-T CSV ------------------
def initials_from_csv(df_params):
    cols = ["Au","Uu","Ru","Nu","Qu","As","Us","Rs","Ns","Qs"]
    for c in cols:
        if c not in df_params.columns:
            raise ValueError(f"Missing column '{c}' in params CSV.")

    def ivw(x, s):
        w = 1.0 / np.maximum(s**2, 1e-12)
        return float(np.nansum(w * x) / np.maximum(np.nansum(w), 1e-12))

    A0 = ivw(df_params["Au"].values, df_params["As"].values)
    U0 = ivw(df_params["Uu"].values, df_params["Us"].values)
    R0 = ivw(df_params["Ru"].values, df_params["Rs"].values)
    N0 = ivw(df_params["Nu"].values, df_params["Ns"].values)
    Q0 = ivw(df_params["Qu"].values, df_params["Qs"].values)

    sA0 = np.nanmedian(df_params["As"].values)
    sU0 = np.nanmedian(df_params["Us"].values)
    sR0 = np.nanmedian(df_params["Rs"].values)
    sN0 = np.nanmedian(df_params["Ns"].values)
    sQ0 = np.nanmedian(df_params["Qs"].values)

    Au, Uu = df_params["Au"].values, df_params["Uu"].values
    Ru, Nu = df_params["Ru"].values, df_params["Nu"].values

    def safe_corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return 0.0
        return float(np.corrcoef(x[m], y[m])[0,1])

    rho_AU0 = np.clip(safe_corr(Au, Uu), 0.0, 0.95)
    rho_RN0 = -np.clip(abs(safe_corr(Ru, Nu)), 0.0, 0.95)

    x0 = np.array([A0,U0,R0,N0,Q0,
                   np.log(max(sA0,1e-6)), np.log(max(sU0,1e-6)),
                   np.log(max(sR0,1e-6)), np.log(max(sN0,1e-6)),
                   np.log(max(sQ0,1e-6)),
                   np.arctanh(np.clip(rho_AU0, 0.0, 0.95)),
                   np.arctanh(np.clip(-rho_RN0, 0.0, 0.95))], dtype=float)
    return x0

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
    ap = argparse.ArgumentParser(description="Global joint refit across all T using FK quantile matching.")
    ap.add_argument("--params_csv", required=True, help="CSV with per-T MC outputs: T, Au,Uu,Ru,Nu,Qu, As,Us,Rs,Ns,Qs")
    ap.add_argument("--tsv", required=True, help="TSV/CSV with columns [T, tau_mu, alpha]")
    ap.add_argument("--qs", default="0.02,0.10,0.25,0.50,0.75,0.90,0.98",
                    help="Comma-separated quantiles in [0,1].")
    ap.add_argument("--K", type=int, default=30000, help="MC draws per objective evaluation")
    ap.add_argument("--seed", type=int, default=202, help="RNG seed for CRNs")
    ap.add_argument("--ridgeQ", type=float, default=0.0, help="Optional ridge on Q")
    ap.add_argument("--maxiter", type=int, default=1200, help="Nelder-Mead maxiter")

    # --------- NEW: regime-wise seeding controls (Correction 3) ---------
    ap.add_argument("--use_regime_means", action="store_true",
                    help="Overwrite initial means (A,U,R,N,Q) with regime seeds.")
    ap.add_argument("--use_regime_sds", action="store_true",
                    help="Overwrite initial SDs (sA,sU,sR,sN,sQ) with regime seeds.")
    ap.add_argument("--regime_means",
                    default="-11.093491,900.572427,-4.831255,3.963503,-0.182074",
                    help="Regime mean seeds A,U,R,N,Q (CSV or JSON).")
    ap.add_argument("--regime_sds",
                    default="0.224577,14.365442,2.171177,1.392262,0.565583",
                    help="Regime SD seeds sA,sU,sR,sN,sQ (CSV or JSON).")
    # -------------------------------------------------------------------

    ap.add_argument("--out", default="global_fit.csv", help="Output CSV (one row with global params)")
    args = ap.parse_args()

    # read per-T params (for Ts and smart initials)
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

    # initials from per-T CSV
    x0 = initials_from_csv(dfp)

    # --------- APPLY CORRECTION 3: overwrite initials with regime seeds ---------
    if args.use_regime_means or args.use_regime_sds:
        # parse seeds
        reg_means = parse_list_or_json(args.regime_means, expected_len=5)
        reg_sds   = parse_list_or_json(args.regime_sds,   expected_len=5)
        if args.use_regime_means:
            x0[0:5] = np.array(reg_means, dtype=float)
        if args.use_regime_sds:
            # store as logs (respect 1e-3 floor downstream)
            x0[5:10] = np.log(np.maximum(reg_sds, np.full(5, 1e-3)))
    # ---------------------------------------------------------------------------

    # common random numbers for stability
    rng = np.random.default_rng(args.seed)
    Z = rng.standard_normal(size=(args.K, 5))

    def obj(x):
        return objective_global(x, T_all, tau_mu, alpha, qs, Z, w_q=1.0, ridge_Q=args.ridgeQ)

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": args.maxiter, "xatol": 2e-3, "fatol": 2e-4, "disp": True})

    mu, sigmas, rho_AU, rho_RN = unpack(res.x)
    A,U,R,N,Q = mu
    sA,sU,sR,sN,sQ = sigmas

    out_row = pd.DataFrame([{
        "A":A, "U":U, "R":R, "N":N, "Q":Q,
        "sA":sA, "sU":sU, "sR":sR, "sN":sN, "sQ":sQ,
        "rho_AU":rho_AU, "rho_RN":rho_RN,
        "loss": float(res.fun),
        "success": bool(res.success),
        "nit": int(getattr(res, "nit", -1))
    }])
    header_needed = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    out_row.to_csv(args.out, index=False, mode="a", header=header_needed)

    print("\n=== Global joint fit (Method 3) ===")
    print(f"T points: {len(T_all)} | draws K={args.K} | ridgeQ={args.ridgeQ}")
    print(f"A={A:.6f}  U={U:.6f}  R={R:.6f}  N={N:.6f}  Q={Q:.6f}")
    print(f"sA={sA:.6f} sU={sU:.6f} sR={sR:.6f} sN={sN:.6f} sQ={sQ:.6f}")
    print(f"rho_AU={rho_AU:.6f} (≥0),  rho_RN={rho_RN:.6f} (≤0)")
    print(f"loss={float(res.fun):.6g}  success={res.success}  nit={getattr(res,'nit','?')}")
    print(f"Saved: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
