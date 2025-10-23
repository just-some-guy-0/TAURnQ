#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os

PATH = "tBuOCl.tsv"
idx = 33  # row index to fit (change or loop externally)

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
    term3 = 10.0**(Q)
    return term1 + term2 + term3

# ---- correlated draws for (A,U) and (R,n), Q independent ----
def draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z):
    """
    mu=(A,U,R,n,Q), sigmas=(sA,sU,sR,sN,sQ), Z~N(0,I) shape (K,5)
    Returns Theta samples, shape (K,5)
    """
    A,U,R,n,Q = mu
    sA,sU,sR,sN,sQ = sigmas
    # 2x2 Cholesky for AU block
    L_AU = np.array([[sA,                    0.0],
                     [rho_AU*sU,  sU*np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    # 2x2 Cholesky for RN block
    L_RN = np.array([[sR,                    0.0],
                     [rho_RN*sN,  sN*np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])

    Z_AU = Z[:, :2]     # columns 0,1
    Z_RN = Z[:, 2:4]    # columns 2,3
    zQ   = Z[:, 4]      # column 4

    eps_AU = Z_AU @ L_AU.T         # (K,2)
    eps_RN = Z_RN @ L_RN.T         # (K,2)
    eps_Q  = zQ * sQ               # (K,)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=PATH, help="TSV/CSV with cols: T, tau_mean, alpha")
    ap.add_argument("--idx", type=int, default=idx, help="Row index to fit")
    ap.add_argument("--out", default="mc_params.csv", help="Output CSV to append (T,Au,Uu,Ru,Nu,Qu,As,Us,Rs,Ns,Qs)")
    ap.add_argument("--noplot", action="store_true", help="Skip histogram/overlay plot")
    args = ap.parse_args()

    # ---load data ---
    df = pd.read_csv(args.infile, sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["T", "tau_mean", "alpha"]
    df = df.astype(float)

    if not (0 <= args.idx < len(df)):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(df)-1}]")

    # single-T test row
    T     = float(df.loc[args.idx, "T"])
    tau_m = float(df.loc[args.idx, "tau_mean"])
    alpha = float(df.loc[args.idx, "alpha"])

    # --- target ln(tau) quantiles ---  (WIDER RANGE)
    qs = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98], dtype=float)
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    # --- initial guesses (means & SDs) ---
    A    = -11.093491
    Ueff =  900.572427
    R    =  -4.831255
    N    =   3.963503
    Q    =  -0.182074
    sA, sUeff, sR, sN, sQ = 0.224577, 14.365442, 2.171177, 1.392262, 0.565583

    # correlations: rho_AU free via tanh(eta_AU); rho_RN constrained NEGATIVE via -tanh(eta_RN)
    eta_AU = 0.9
    eta_RN = np.arctanh(0.9)   # -> rho_RN ≈ -0.5 as a starting point

    # pack params: [5 means][5 log-SDs][2 corr-etas]
    x0 = np.array([A,Ueff,R,N,Q,
                   np.log(sA), np.log(sUeff), np.log(sR), np.log(sN), np.log(sQ),
                   eta_AU, eta_RN], dtype=float)

    # --- common random numbers ----
    K = 25000
    rng = np.random.default_rng(12345)
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
        pen += quad_out(-A,   0.0, 30.0, 1e-3)    # -A in [0,30]
        pen += quad_out(Ueff, 0.0, 3000.0, 1e-6)  # U in [0,3000]
        pen += quad_out(R,   -20.0, 10.0, 1e-3)   # R in [-20,10]
        pen += quad_out(n,     0.0, 12.0, 1e-3)   # n in [0,12]
        pen += quad_out(Q,   -20.0, 10.0, 1e-3)   # Q in [-20,10]
        pen += 1e-5 * float(np.sum(sigmas**2))    # discourage huge SDs
        # Mild penalty near |rho| -> 1 to avoid numerical issues
        pen += 1e-4 * ((abs(rho_AU) > 0.995)*(abs(rho_AU)-0.995)**2
                       + (abs(rho_RN) > 0.995)*(abs(rho_RN)-0.995)**2)
        return pen

    def objective(x):
        mu, sigmas, rho_AU, rho_RN = unpack(x)
        lnq_hat = simulate_ln_tau_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, random_draws)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid) + penalty(mu, sigmas, rho_AU, rho_RN))

    # --- optimize ---
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 150, "xatol": 1e-3, "fatol": 2e-4, "disp": True})

    mu_hat, sig_hat, rho_AU_hat, rho_RN_hat = unpack(res.x)
    A,Ueff,R,N,Q = mu_hat
    sA,sUeff,sR,sN,sQ = sig_hat

    print("Fitted parameter values (with correlations):")
    print(f"  T = {T:.1f} K ")
    print(f"  A  = {A:.6f}")
    print(f"  Ueff  = {Ueff:.6f} K")
    print(f"  R  = {R:.6f}")
    print(f"  n  = {N:.6f}")
    print(f"  Q  = {Q:.6f}")
    print(f"  sA = {sA:.6f}")
    print(f"  sUeff = {sUeff:.6f}")
    print(f"  sR = {sR:.6f}")
    print(f"  sN = {sN:.6f}")
    print(f"  sQ = {sQ:.6f}")
    print(f"  rho_AU = {rho_AU_hat:.6f}")
    print(f"  rho_RN = {rho_RN_hat:.6f}  # forced ≤ 0")
    print("Final loss:", float(res.fun))

    # ---------- NEW: append to CSV in the requested format ----------
    row = pd.DataFrame([{
        "T": T,
        "Au": A,        "Uu": Ueff,  "Ru": R,       "Nu": N,       "Qu": Q,
        "As": sA,       "Us": sUeff, "Rs": sR,      "Ns": sN,      "Qs": sQ,
    }])
    header_needed = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    row.to_csv(args.out, index=False, mode="a", header=header_needed)
    print(f"\nAppended parameters to: {os.path.abspath(args.out)}")
    # ---------------------------------------------------------------

    if not args.noplot:
        # --- plot (use correlated draws) ---
        Kplot = 60000
        Zplot = np.random.default_rng(54321).standard_normal(size=(Kplot,5))
        theta_plot = draw_params_correlated(mu_hat, sig_hat, rho_AU_hat, rho_RN_hat, Zplot)
        A_s, U_s, R_s, N_s, Q_s = theta_plot.T
        r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
        tau_mc = 1.0/np.maximum(r, 1e-300)

        taus = np.logspace(-6, 3, 5000)
        rho_fk_lognorm = rho_tau(taus, tau_m, alpha)

        # plt.figure(figsize=(7,5))
        # bins = np.logspace(np.log10(1e-6), np.log10(1e3), 150)
        # plt.hist(tau_mc, bins=bins, density=True, alpha=0.45, edgecolor="none",
        #         label="MC eqn (10) τ (fitted, corr)")
        # plt.plot(taus, rho_fk_lognorm, linewidth=2, label="FK log-normal approx")
        # plt.xscale("log")
        # plt.xlabel("τ (s)")
        # plt.ylabel("ρ(τ)")
        # plt.title(f"Fitted MC vs FK log-normal at T={T:.1f} K")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

if __name__ == "__main__":
    main()
