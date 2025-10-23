#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse, sys

LN10 = np.log(10.0)

# ---------- FK helper ----------
def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha)/(1 - alpha)

# ---------- Model terms ----------
def rate_terms(T, A, U, R, N, Q, use_orbach=True, use_raman=True, use_qtm=True):
    T = np.maximum(T, 1e-300)
    r = 0.0
    if use_orbach:
        r += 10.0**(-A) * np.exp(-U / T)
    if use_raman:
        r += 10.0**(R) * (T**N)
    if use_qtm:
        r += 10.0**(Q)
    return np.maximum(r, 1e-300)

# ---------- Draws ----------
def draw_full(A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN, Z):
    # Blocks: (A,U), (R,N), Q
    L_AU = np.array([[sA,                     0.0],
                     [rho_AU * sU,  sU*np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    L_RN = np.array([[sR,                     0.0],
                     [rho_RN * sN,  sN*np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])

    eps_AU = Z[:, :2] @ L_AU.T
    eps_RN = Z[:, 2:4] @ L_RN.T
    eps_Q  = Z[:, 4] * sQ

    A_s = A + eps_AU[:,0]
    U_s = U + eps_AU[:,1]
    R_s = R + eps_RN[:,0]
    N_s = N + eps_RN[:,1]
    Q_s = Q + eps_Q
    return A_s, U_s, R_s, N_s, Q_s

def draw_rn_only(A,U,R,N,Q, sR,sN, rho_RN, Z):
    L_RN = np.array([[sR,                     0.0],
                     [rho_RN * sN,  sN*np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])
    eps_RN = Z[:, :2] @ L_RN.T
    R_s = R + eps_RN[:,0]
    N_s = N + eps_RN[:,1]
    return np.full(len(R_s), A), np.full(len(R_s), U), R_s, N_s, np.full(len(R_s), Q)

# ---------- Width calculators ----------
def sigma_ln_tau_full(T, pars, Z, use_orbach=True, use_raman=True, use_qtm=True):
    A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN = pars
    A_s,U_s,R_s,N_s,Q_s = draw_full(A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN, Z)
    r = rate_terms(T, A_s, U_s, R_s, N_s, Q_s, use_orbach, use_raman, use_qtm)
    tau = 1.0/np.maximum(r, 1e-300)
    return float(np.std(np.log(tau)))

def sigma_ln_tau_rn_only(T, pars, Z, rho_override=None):
    A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN = pars
    rho_use = rho_RN if rho_override is None else float(rho_override)
    A_s,U_s,R_s,N_s,Q_s = draw_rn_only(A,U,R,N,Q, sR,sN, rho_use, Z)
    r = rate_terms(T, A_s, U_s, R_s, N_s, Q_s, use_orbach=False, use_raman=True, use_qtm=False)
    tau = 1.0/np.maximum(r, 1e-300)
    return float(np.std(np.log(tau)))

# ---------- IO helpers ----------
def read_global_tail(path):
    df = pd.read_csv(path)
    if len(df)==0:
        raise ValueError("global_csv is empty.")
    row = df.iloc[-1]
    need = ["A","U","R","N","Q","sA","sU","sR","sN","sQ","rho_AU","rho_RN"]
    for k in need:
        if k not in row:
            raise ValueError(f"Missing '{k}' in {path}")
    vals = [float(row[k]) for k in need]
    return vals  # in the order above

def read_fk(tsv_path):
    dft = pd.read_csv(tsv_path, sep=None, engine="python", header=None).iloc[:, :3]
    dft.columns = ["T","tau_mu","alpha"]
    dft = dft.astype(float)
    return dft

def parse_window(s):
    s = s.strip()
    lo, hi = [float(x) for x in s.split(",")]
    if lo>hi: lo,hi = hi,lo
    return lo,hi

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Sanity checks for Raman width under global fit.")
    ap.add_argument("--global_csv", required=True, help="CSV written by global fitter (last row used)")
    ap.add_argument("--tsv", required=True, help="FK targets TSV/CSV with columns [T, tau_mu, alpha]")
    ap.add_argument("--RN_window", required=True, help="Raman window, e.g. '26,36'")
    ap.add_argument("--K", type=int, default=50000, help="MC draws per T")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pars = read_global_tail(args.global_csv)
    dft  = read_fk(args.tsv)
    lo,hi = parse_window(args.RN_window)
    dfw = dft[(dft["T"]>=lo) & (dft["T"]<=hi)].copy()
    if len(dfw)==0:
        print("No FK rows inside RN_window.", file=sys.stderr); sys.exit(1)

    rng = np.random.default_rng(args.seed)
    # For FULL: need 5 normals; for RN-only: we’ll reuse the first 2 columns
    Z_full = rng.standard_normal(size=(args.K,5))
    Z_rn   = rng.standard_normal(size=(args.K,2))

    # For convenience, keep pars in exact order expected by calculators
    A,U,R,N,Q,sA,sU,sR,sN,sQ,rho_AU,rho_RN = pars
    pars_tuple = (A,U,R,N,Q,sA,sU,sR,sN,sQ,rho_AU,rho_RN)

    rows = []
    for _, r in dfw.iterrows():
        T = float(r["T"])
        galpha = g_from_alpha(float(r["alpha"]))
        # (1) full jitter, all terms on
        s_full = sigma_ln_tau_full(T, pars_tuple, Z_full, True, True, True)
        # (2) RN-only jitter
        s_rn   = sigma_ln_tau_rn_only(T, pars_tuple, Z_rn, rho_override=None)
        # (3) RN-only jitter with rho=-0.99
        s_rn99 = sigma_ln_tau_rn_only(T, pars_tuple, Z_rn, rho_override=-0.99)

        rows.append(dict(
            T=T,
            g_alpha=galpha,
            s_full=s_full,
            s_rn=s_rn,
            s_rn_rho_m099=s_rn99,
            d_full= s_full - galpha,
            d_rn  = s_rn   - galpha,
            d_rn99= s_rn99 - galpha
        ))

    out = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    def rmse(a,b): 
        a,b = np.asarray(a), np.asarray(b)
        return float(np.sqrt(np.mean((a-b)**2)))

    rmse_full = rmse(out["s_full"], out["g_alpha"])
    rmse_rn   = rmse(out["s_rn"], out["g_alpha"])
    rmse_rn99 = rmse(out["s_rn_rho_m099"], out["g_alpha"])

    # Print concise report
    print("\n=== Raman-window width sanity check ===")
    print(f"Window: {lo}..{hi} K   |   K={args.K} draws   |   seed={args.seed}")
    print(f"Global means: R={R:.6g}  N={N:.6g}   SDs: sR={sR:.6g}  sN={sN:.6g}   rho_RN={rho_RN:.6g}")
    print(f"RMSE vs g(alpha):  FULL={rmse_full:.4g}   RN-only={rmse_rn:.4g}   RN-only (rho=-0.99)={rmse_rn99:.4g}")
    print("\nT [K]   g(alpha)   s_full   s_rn   s_rn(rho=-0.99)   d_full   d_rn   d_rn99")

    for row in out.itertuples(index=False):
        print(f"{row.T:6.1f}  {row.g_alpha:8.4f}  {row.s_full:7.4f} {row.s_rn:7.4f} {row.s_rn_rho_m099:14.4f}   {row.d_full:7.4f} {row.d_rn:7.4f} {row.d_rn99:7.4f}")

    # Save CSV for inspection
    out.to_csv("rn_window_width_check.csv", index=False)
    print("\nSaved: rn_window_width_check.csv")

if __name__ == "__main__":
    main()
