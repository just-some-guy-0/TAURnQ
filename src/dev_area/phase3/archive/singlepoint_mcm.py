#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# --- fk log-normal approx ---
def rho_tau(tau, tau_m, alpha):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.exp(-0.5*((np.log(tau) - np.log(tau_m))/g)**2) / (tau * g * np.sqrt(2*np.pi))

def fk_ln_quantiles(tau_m, alpha, qs):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.log(tau_m) + norm.ppf(qs)*g

# eqn(10) ---> tau^{-1} = 10^{-A} * exp(-Ueff/T) + 10^{R} * T^{n} + 10^{Q}
def rate_model(T, A, Ueff, R, n, Q):
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    term2 = 10.0**(R)  * (T_safe**n)
    term3 = 10.0**(Q)
    return term1 + term2 + term3

def simulate_ln_tau_quantiles(T, mu, sigmas, qs, Z):
    A,U,R,n,Q = mu
    sA,sU,sR,sN,sQ = sigmas
    theta = np.array([A,U,R,n,Q]) + Z * np.array([sA,sU,sR,sN,sQ])
    A_s, U_s, R_s, N_s, Q_s = theta.T
    r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
    tau = 1.0/np.maximum(r, 1e-300)
    ln_tau = np.log(tau)
    return np.quantile(ln_tau, qs)

def main():
    # ---load data ---
    df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["T", "tau_mean", "alpha"]
    df = df.astype(float)
    idx = 22
    T     = float(df.loc[idx, "T"])
    tau_m = float(df.loc[idx, "tau_mean"])
    alpha = float(df.loc[idx, "alpha"])

    # --- target ln(tau) quantiles ---
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float) # extend to improve fit
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    # --- initial guesses ---
    A    = -11.160788
    Ueff    =  908.817531
    R    =  -4.831860
    N    =   3.963907
    Q    =  -0.182074
    A_sd, Ueff_sd, R_sd, N_sd, Q_sd = 0.2, 17.376, 2.171111, 1.392216, 0.565583

    # 10D vector
    x0 = np.array([A,Ueff,R,N,Q, np.log(A_sd), np.log(Ueff_sd), np.log(R_sd), np.log(N_sd), np.log(Q_sd)], dtype=float)

    # --- pseudo-rando nums ---
    K = 25000 #
    rng = np.random.default_rng(12345)
    random_draws = rng.standard_normal(size=(K,5))

    def penalty(mu, sigmas):
        # soft plausibility penalties
        A,Ueff,R,N,Q = mu
        pen = 0.0
        def quad_out(val, lo, hi, scale):
            if val < lo:  return scale*(lo - val)**2
            if val > hi:  return scale*(val - hi)**2
            return 0.0
        pen += quad_out(-A, 0.0, 30.0, 1e-3)   # -A in [0,30]
        pen += quad_out(Ueff,  0.0, 3000.0, 1e-6) # U in [0,3000]
        pen += quad_out(R, -20.0, 10.0, 1e-3)  # R in [-20,10]
        pen += quad_out(N,  0.0, 12.0, 1e-3)   # n in [0,12]
        pen += quad_out(Q, -20.0, 10.0, 1e-3)  # Q in [-20,10]
        pen += 1e-5 * float(np.sum(sigmas**2)) # discourage huge SDs
        return pen

    def least_squares_error(x):
        mu, sigmas = x[:5], np.exp(x[5:])
        lnq_hat = simulate_ln_tau_quantiles (T, mu, sigmas, qs, random_draws)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid) + penalty(mu, sigmas))

    # --- optimize ---
    res = minimize(least_squares_error, x0, method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 2e-3, "fatol": 2e-4, "disp": True})

    mu_hat, sig_hat = res.x[:5], np.exp(res.x[5:])
    A,Ueff,R,N,Q = mu_hat
    sA,sU,sR,sN,sQ = sig_hat

    print("Fitted parameter values:")
    print(f"  T = {T:.1f} K ")
    print(f"  A = {A:.6f}")
    print(f"  U = {Ueff:.6f} K")
    print(f"  R = {R:.6f}")
    print(f"  n = {N:.6f}")
    print(f"  Q = {Q:.6f}")
    print(f"  sA = {sA:.6f}")
    print(f"  sU = {sU:.6f}")
    print(f"  sR = {sR:.6f}")
    print(f"  sN = {sN:.6f}")
    print(f"  sQ = {sQ:.6f}")
    print("Final loss:", float(res.fun))

    # --- plot ---
    Kplot = 60000
    Zplot = np.random.default_rng(54321).standard_normal(size=(Kplot,5))
    param_draws = mu_hat + Zplot * sig_hat
    A_s, U_s, R_s, N_s, Q_s = param_draws.T
    r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
    tau_mc = 1.0/np.maximum(r, 1e-300)

    taus = np.logspace(-6, 3, 5000)
    rho_fk_lognorm = rho_tau(taus, tau_m, alpha)

    plt.figure(figsize=(7,5))
    bins = np.logspace(np.log10(1e-6), np.log10(1e3), 150)
    plt.hist(tau_mc, bins=bins, density=True, alpha=0.45, edgecolor="none", label="MC eqn (10) τ (fitted)")
    plt.plot(taus, rho_fk_lognorm, linewidth=2, label="FK log-normal approx")
    plt.xscale("log")
    plt.xlabel("τ (s)")
    plt.ylabel("ρ(τ)")
    plt.title(f"Fitted MC vs FK log-normal at T={T:.1f} K")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # --- output csv ---
    # out = {
    #     "A_mean": float(A), "Ueff_mean": float(U), "R_mean": float(R), "n_mean": float(N), "Q_mean": float(Q),
    #     "A_sd": float(sA), "Ueff_sd": float(sU), "R_sd": float(sR), "n_sd": float(sN), "Q_sd": float(sQ),
    #     "loss": float(res.fun), "success": bool(res.success), "iters": int(res.nit)
    # }
    # pd.Series(out).to_csv("fit_10var_singleT_params.csv")

if __name__ == "__main__":
    main()
