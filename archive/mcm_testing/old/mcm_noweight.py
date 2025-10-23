#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ---------- FK log-normal ----------
def fk_ln_quantiles(tau_m, alpha, qs):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.log(tau_m) + norm.ppf(qs)*g

# ---------- eqn (10) ----------
def rate_from_params(T, A, Ueff, R, n, Q):
    # tau^{-1} = 10^{-A} * exp(-Ueff/T) + 10^{R} * T^{n} + 10^{Q}
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
    r = rate_from_params(T, A_s, U_s, R_s, N_s, Q_s)
    tau = 1.0/np.maximum(r, 1e-300)
    ln_tau = np.log(tau)
    return np.quantile(ln_tau, qs)

def fit_single_T(T, tau_m, alpha):
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    target_lnq = fk_ln_quantiles(tau_m, alpha, qs)

    # initial guess
    A0    = -11.537938241219702
    U0    =  956.6247933615898
    R0    =  -5.432741018066928
    n0    =   4.320107244859063
    Q0    =  -0.12029577025666016
    sA0, sU0, sR0, sN0, sQ0 = 0.15, 60.0, 0.15, 0.20, 0.15
    x0 = np.array([A0,U0,R0,n0,Q0,
                   np.log(sA0),np.log(sU0),np.log(sR0),np.log(sN0),np.log(sQ0)],dtype=float)

    # common random numbers
    K = 25000
    rng = np.random.default_rng(54321)
    Z = rng.standard_normal(size=(K,5))

    def unpack(x):
        mu = x[:5]; sigmas = np.exp(x[5:])
        return mu, sigmas

    def penalty(mu, sigmas):
        A,U,R,N,Q = mu
        pen = 0.0
        def quad_out(val, lo, hi, scale):
            if val < lo: return scale*(lo-val)**2
            if val > hi: return scale*(val-hi)**2
            return 0.0
        pen += quad_out(-A,0,30,1e-3)
        pen += quad_out(U,0,3000,1e-6)
        pen += quad_out(R,-20,10,1e-3)
        pen += quad_out(N,0,12,1e-3)
        pen += quad_out(Q,-20,10,1e-3)
        pen += 1e-5*np.sum(sigmas**2)
        return pen

    def loss(x):
        mu,sigmas = unpack(x)
        lnq_hat = simulate_ln_tau_quantiles(T, mu, sigmas, qs, Z)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid,resid)+penalty(mu,sigmas))

    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter":350,"xatol":2e-3,"fatol":2e-4,"disp":False})

    mu_hat,sig_hat = unpack(res.x)
    A,U,R,N,Q = mu_hat
    sA,sU,sR,sN,sQ = sig_hat
    return {
        "A_mean":A,"Ueff_mean":U,"R_mean":R,"n_mean":N,"Q_mean":Q,
        "A_sd":sA,"Ueff_sd":sU,"R_sd":sR,"n_sd":sN,"Q_sd":sQ,
        "loss":res.fun,"success":res.success,"iters":res.nit
    }

def main():
    df = pd.read_csv("tBuOCl.tsv",sep=None,engine="python",header=None).iloc[:,:3]
    df.columns = ["T","tau_mean","alpha"]
    df = df.astype(float)

    rows=[]
    for _,row in df.iterrows():
        res = fit_single_T(row["T"],row["tau_mean"],row["alpha"])
        rows.append({"T":row["T"],**res})

    out_df = pd.DataFrame(rows)
    out_df.to_csv("t_all_params.csv",index=False)
    out_df[["T","A_mean","A_sd"]].rename(columns={"T":"temp","A_mean":"a_mu","A_sd":"a_sd"}) \
         .to_csv("t_all_A_only.csv",index=False)

if __name__=="__main__":
    main()
