#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse, sys

# ---------- FK helper ----------
def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha)/(1 - alpha)

# ---------- Model ----------
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
    L_AU = np.array([[sA,                     0.0],
                     [rho_AU * sU,  sU*np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    L_RN = np.array([[sR,                     0.0],
                     [rho_RN * sN,  sN*np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])
    eps_AU = Z[:, :2] @ L_AU.T
    eps_RN = Z[:, 2:4] @ L_RN.T
    eps_Q  = Z[:, 4] * sQ
    A_s = A + eps_AU[:,0]; U_s = U + eps_AU[:,1]
    R_s = R + eps_RN[:,0]; N_s = N + eps_RN[:,1]
    Q_s = Q + eps_Q
    return A_s, U_s, R_s, N_s, Q_s

def draw_au_only(A,U,R,N,Q, sA,sU, rho_AU, Z):
    L_AU = np.array([[sA,                     0.0],
                     [rho_AU * sU,  sU*np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    eps_AU = Z[:, :2] @ L_AU.T
    A_s = A + eps_AU[:,0]
    U_s = U + eps_AU[:,1]
    return A_s, U_s, np.full(len(A_s), R), np.full(len(A_s), N), np.full(len(A_s), Q)

# ---------- Width calculators ----------
def sigma_ln_tau_full(T, pars, Z):
    A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN = pars
    A_s,U_s,R_s,N_s,Q_s = draw_full(A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN, Z)
    r = rate_terms(T, A_s, U_s, R_s, N_s, Q_s, True, True, True)
    tau = 1.0/np.maximum(r, 1e-300)
    return float(np.std(np.log(tau)))

def sigma_ln_tau_au_only(T, pars, Z, rho_override=None):
    A,U,R,N,Q, sA,sU,sR,sN,sQ, rho_AU, rho_RN = pars
    rho_use = rho_AU if rho_override is None else float(rho_override)
    A_s,U_s,R_s,N_s,Q_s = draw_au_only(A,U,R,N,Q, sA,sU, rho_use, Z)
    r = rate_terms(T, A_s, U_s, R_s, N_s, Q_s, use_orbach=True, use_raman=False, use_qtm=False)
    tau = 1.0/np.maximum(r, 1e-300)
    return float(np.std(np.log(tau)))

# ---------- IO ----------
def read_global_tail(path):
    df = pd.read_csv(path)
    if len(df)==0:
        raise ValueError("global_csv is empty.")
    row = df.iloc[-1]
    need = ["A","U","R","N","Q","sA","sU","sR","sN","sQ","rho_AU","rho_RN"]
    for k in need:
        if k not in row:
            raise ValueError(f"Missing '{k}' in {path}")
    return [float(row[k]) for k in need]

def parse_cli_vec(s, n):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()!=""]
    if len(vals)!=n: raise ValueError(f"Expected {n} values, got {len(vals)}")
    return vals

def read_params(args):
    if args.global_csv:
        return read_global_tail(args.global_csv)
    if not (args.params and args.sds and args.rhos):
        raise ValueError("Provide --global_csv OR all of --params, --sds, --rhos.")
    A,U,R,N,Q = parse_cli_vec(args.params, 5)
    sA,sU,sR,sN,sQ = parse_cli_vec(args.sds, 5)
    rho_AU,rho_RN  = parse_cli_vec(args.rhos, 2)
    return [A,U,R,N,Q,sA,sU,sR,sN,sQ,rho_AU,rho_RN]

def read_fk(tsv_path):
    dft = pd.read_csv(tsv_path, sep=None, engine="python", header=None).iloc[:, :3]
    dft.columns = ["T","tau_mu","alpha"]
    dft = dft.astype(float)
    return dft

def parse_window(s):
    lo, hi = [float(x) for x in s.split(",")]
    return (min(lo,hi), max(lo,hi))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Sanity checks for Orbach width under global fit.")
    ap.add_argument("--global_csv", default="", help="CSV with global fit (last row used)")
    ap.add_argument("--params", default="", help="A,U,R,N,Q (CSV) if not using --global_csv")
    ap.add_argument("--sds",    default="", help="sA,sU,sR,sN,sQ (CSV)")
    ap.add_argument("--rhos",   default="", help="rho_AU,rho_RN (CSV)")
    ap.add_argument("--tsv", required=True, help="FK targets TSV/CSV with columns [T, tau_mu, alpha]")
    ap.add_argument("--AU_window", required=True, help="Orbach window, e.g. '45,58'")
    ap.add_argument("--K", type=int, default=50000, help="MC draws per T")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    pars = read_params(args)
    dft  = read_fk(args.tsv)
    (lo,hi) = parse_window(args.AU_window)
    dfw = dft[(dft["T"]>=lo) & (dft["T"]<=hi)].copy()
    if len(dfw)==0:
        print("No FK rows inside AU_window.", file=sys.stderr); sys.exit(1)

    rng = np.random.default_rng(args.seed)
    Z_full = rng.standard_normal(size=(args.K,5))
    Z_au   = rng.standard_normal(size=(args.K,2))

    A,U,R,N,Q,sA,sU,sR,sN,sQ,rho_AU,rho_RN = pars
    pars_tuple = (A,U,R,N,Q,sA,sU,sR,sN,sQ,rho_AU,rho_RN)

    rows = []
    for _, r in dfw.iterrows():
        T = float(r["T"])
        galpha = g_from_alpha(float(r["alpha"]))
        s_full = sigma_ln_tau_full(T, pars_tuple, Z_full)                 # all jitter
        s_au   = sigma_ln_tau_au_only(T, pars_tuple, Z_au)                # AU-only
        s_au99 = sigma_ln_tau_au_only(T, pars_tuple, Z_au, rho_override=+0.99)  # AU-only, rho=+0.99
        rows.append(dict(
            T=T, g_alpha=galpha,
            s_full=s_full, s_au=s_au, s_au_rho_p099=s_au99,
            d_full=s_full-galpha, d_au=s_au-galpha, d_au99=s_au99-galpha
        ))

    out = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    def rmse(a,b):
        a,b = np.asarray(a), np.asarray(b)
        return float(np.sqrt(np.mean((a-b)**2)))
    rmse_full = rmse(out["s_full"], out["g_alpha"])
    rmse_au   = rmse(out["s_au"],   out["g_alpha"])
    rmse_au99 = rmse(out["s_au_rho_p099"], out["g_alpha"])

    print("\n=== Orbach-window width sanity check ===")
    print(f"Window: {lo}..{hi} K   |   K={args.K} draws   |   seed={args.seed}")
    print(f"Global means: A={A:.6g}  U={U:.6g}   SDs: sA={sA:.6g}  sU={sU:.6g}   rho_AU={rho_AU:.6g}")
    print(f"RMSE vs g(alpha):  FULL={rmse_full:.4g}   AU-only={rmse_au:.4g}   AU-only (rho=+0.99)={rmse_au99:.4g}")
    print("\nT [K]   g(alpha)   s_full   s_AU   s_AU(rho=+0.99)   d_full   d_AU   d_AU99")
    for row in out.itertuples(index=False):
        print(f"{row.T:6.1f}  {row.g_alpha:8.4f}  {row.s_full:7.4f} {row.s_au:7.4f} {row.s_au_rho_p099:16.4f}   {row.d_full:7.4f} {row.d_au:7.4f} {row.d_au99:7.4f}")

    out.to_csv("au_window_width_check.csv", index=False)
    print("\nSaved: au_window_width_check.csv")

if __name__ == "__main__":
    main()
