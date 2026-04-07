#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os, sys

PATH = "NSi2iPr3.tsv"

# --- fk log-normal approx ---
def rho_tau(tau, tau_m, alpha):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.exp(-0.5*((np.log(tau) - np.log(tau_m))/g)**2) / (tau * g * np.sqrt(2*np.pi))

def fk_ln_quantiles(tau_m, alpha, qs):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.log(tau_m) + norm.ppf(qs)*g

# eqn(10): tau^{-1} = 10^{-A} * exp(-Ueff/T) + 10^{R} * T^{n} + 10^{Q}
def rate_model(T, A, Ueff, R, n, Q):
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    term2 = 10.0**(R)  * (T_safe**n)
    term3 = 10.0**(-Q)
    return term1 + term2 + term3

# ---- correlated draws for (A,U) and (R,n), Q independent ----
def draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z):
    A,U,R,n,Q = mu
    sA,sU,sR,sN,sQ = sigmas
    L_AU = np.array([[sA,                    0.0],
                     [rho_AU*sU,  sU*np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    L_RN = np.array([[sR,                    0.0],
                     [rho_RN*sN,  sN*np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])
    Z_AU = Z[:, :2]
    Z_RN = Z[:, 2:4]
    zQ   = Z[:, 4]
    eps_AU = Z_AU @ L_AU.T
    eps_RN = Z_RN @ L_RN.T
    eps_Q  = zQ * sQ
    A_s = A + eps_AU[:, 0]
    U_s = U + eps_AU[:, 1]
    R_s = R + eps_RN[:, 0]
    N_s = n + eps_RN[:, 1]
    Q_s = Q + eps_Q
    return np.column_stack([A_s, U_s, R_s, N_s, Q_s])

def simulate_ln_tau_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, Z):
    theta = draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z)
    A_s, U_s, R_s, N_s, Q_s = theta.T
    r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
    tau = 1.0/np.maximum(r, 1e-300)
    ln_tau = np.log(tau)
    return np.quantile(ln_tau, qs)

def fit_one_row(df, idx, out_path, seed_base=12345, make_plot=False):
    # --- pull row ---
    T     = float(df.loc[idx, "T"])
    tau_m = float(df.loc[idx, "tau_mean"])
    alpha = float(df.loc[idx, "alpha"])

    # targets
    qs = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98], dtype=float)
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    # initial guesses (tune if you like)
    A    = -11.527074
    Ueff =   2643.356232
    R    = -5.602306
    N    = 3.061309
    Q    = 1.826901
    sA, sUeff, sR, sN, sQ = 0, 109.001269, 2.681844, 1.436694,  0.713267

    # corr params
    eta_AU = 0.9
    eta_RN = np.arctanh(0.99)   # negative rho_RN = -tanh(eta_RN)

    x0 = np.array([A,Ueff,R,N,Q,
                   np.log(sA), np.log(sUeff), np.log(sR), np.log(sN), np.log(sQ),
                   eta_AU, eta_RN], dtype=float)

    # common random numbers (idx-specific seed avoids coupling across rows)
    K = 25000
    rng = np.random.default_rng(int(seed_base) + int(idx))
    random_draws = rng.standard_normal(size=(K,5))

    def unpack(x):
        mu = x[:5]
        sigmas = np.exp(x[5:10])
        eta_AU, eta_RN = x[10], x[11]
        rho_AU = np.tanh(eta_AU)
        rho_RN = -np.tanh(eta_RN)
        return mu, sigmas, rho_AU, rho_RN

    def penalty(mu, sigmas, rho_AU, rho_RN):
        A,Ueff,R,n,Q = mu
        pen = 0.0
        def quad_out(val, lo, hi, scale):
            if val < lo:  return scale*(lo - val)**2
            if val > hi:  return scale*(val - hi)**2
            return 0.0
        pen += quad_out(-A,   0.0, 30.0, 1e-3)
        pen += quad_out(Ueff, 0.0, 3000.0, 1e-6)
        pen += quad_out(R,   -20.0, 10.0, 1e-3)
        pen += quad_out(n,     0.0, 12.0, 1e-3)
        pen += quad_out(Q,   -20.0, 10.0, 1e-3)
        pen += 1e-5 * float(np.sum(sigmas**2))
        pen += 1e-4 * ((abs(rho_AU) > 0.995)*(abs(rho_AU)-0.995)**2
                       + (abs(rho_RN) > 0.995)*(abs(rho_RN)-0.995)**2)
        return pen

    def objective(x):
        mu, sigmas, rho_AU, rho_RN = unpack(x)
        lnq_hat = simulate_ln_tau_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, random_draws)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid) + penalty(mu, sigmas, rho_AU, rho_RN))

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 150, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    mu_hat, sig_hat, rho_AU_hat, rho_RN_hat = unpack(res.x)
    A,Ueff,R,N,Q = mu_hat
    sA,sUeff,sR,sN,sQ = sig_hat

    print(f"[idx={idx:>3}] T={T:6.2f} K | loss={float(res.fun):.5g} | "
          f"A={A:.4f} U={Ueff:.2f} R={R:.4f} n={N:.4f} Q={Q:.4f} | "
          f"sA={sA:.3f} sU={sUeff:.2f} sR={sR:.2f} sN={sN:.2f} sQ={sQ:.2f} | "
          f"rho_AU={rho_AU_hat:.3f} rho_RN={rho_RN_hat:.3f}")

    # append one row
    row = pd.DataFrame([{
        "T": T,
        "Au": A,        "Uu": Ueff,  "Ru": R,       "Nu": N,       "Qu": Q,
        "As": sA,       "Us": sUeff, "Rs": sR,      "Ns": sN,      "Qs": sQ,
    }])
    header_needed = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    row.to_csv(out_path, index=False, mode="a", header=header_needed)

    if make_plot:
        Kplot = 60000
        Zplot = np.random.default_rng(54321 + int(idx)).standard_normal(size=(Kplot,5))
        theta_plot = draw_params_correlated(mu_hat, sig_hat, rho_AU_hat, rho_RN_hat, Zplot)
        A_s, U_s, R_s, N_s, Q_s = theta_plot.T
        r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
        tau_mc = 1.0/np.maximum(r, 1e-300)
        taus = np.logspace(-6, 3, 5000)
        rho_fk_lognorm = rho_tau(taus, tau_m=np.float64(tau_m), alpha=np.float64(alpha))
        plt.figure(figsize=(7,5))
        bins = np.logspace(np.log10(1e-6), np.log10(1e3), 150)
        plt.hist(tau_mc, bins=bins, density=True, alpha=0.45, edgecolor="none",
                 label="MC eqn (10) τ (fitted, corr)")
        plt.plot(taus, rho_fk_lognorm, linewidth=2, label="FK log-normal approx")
        plt.xscale("log")
        plt.xlabel("τ (s)")
        plt.ylabel("ρ(τ)")
        plt.title(f"Fitted MC vs FK log-normal at T={T:.1f} K")
        plt.legend()
        plt.tight_layout()
        plt.show()

def parse_range(spec, nmax):
    # "start:end[:step]" (end is exclusive, like Python slices)
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError("--range expects start:end or start:end:step")
    start = int(parts[0]) if parts[0] != "" else 0
    end   = int(parts[1]) if parts[1] != "" else nmax
    step  = int(parts[2]) if len(parts) == 3 and parts[2] != "" else 1
    return range(start, min(end, nmax), step)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=PATH, help="TSV/CSV with cols: T, tau_mean, alpha")
    ap.add_argument("--idx", type=int, default=None, help="Single row index to fit")
    ap.add_argument("--idxs", type=str, default=None, help="Comma-separated indices, e.g. '0,2,5'")
    ap.add_argument("--range", dest="range_spec", type=str, default=None,
                    help="Slice-like range 'start:end[:step]' (end exclusive)")
    ap.add_argument("--all", default = True, action="store_true", help="Process all rows")
    ap.add_argument("--out", default=f"{PATH}_mc_params.csv",
                    help="Output CSV (T,Au,Uu,Ru,Nu,Qu,As,Us,Rs,Ns,Qs)")
    ap.add_argument("--clear", action="store_true", help="Overwrite output file before writing")
    ap.add_argument("--seed_base", type=int, default=14322, help="Base RNG seed")
    ap.add_argument("--plot", action="store_true", help="Show plot (only sensible for single idx)")
    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.infile, sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["T", "tau_mean", "alpha"]
    df = df.astype(float)
    nrows = len(df)

    # clear output if requested
    if args.clear and os.path.exists(args.out):
        open(args.out, "w").close()

    # pick indices to run
    idx_list = []
    if args.all:
        idx_list = list(range(nrows))
    elif args.range_spec:
        idx_list = list(parse_range(args.range_spec, nrows))
    elif args.idxs:
        idx_list = [int(s.strip()) for s in args.idxs.split(",") if s.strip() != ""]
    elif args.idx is not None:
        idx_list = [int(args.idx)]
    else:
        print("Provide one of --idx, --idxs, --range, or --all.", file=sys.stderr)
        sys.exit(2)

    # when looping, suppress plots unless explicitly asked for a single idx
    make_plot = args.plot and (len(idx_list) == 1)

    for i, idx in enumerate(idx_list, 1):
        if not (0 <= idx < nrows):
            print(f"Skip idx {idx}: out of range [0,{nrows-1}]", file=sys.stderr)
            continue
        try:
            fit_one_row(df, idx, args.out, seed_base=args.seed_base, make_plot=make_plot)
        except Exception as e:
            print(f"[idx={idx}] ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Appended {len(idx_list)} row(s) to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
