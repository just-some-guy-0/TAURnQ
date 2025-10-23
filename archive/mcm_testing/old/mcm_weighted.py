#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from scipy.stats import norm

# --- fk log-normal approx ---
def fk_ln_quantiles(tau_m, alpha, qs):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.log(tau_m) + norm.ppf(qs) * g

def rho_tau_lognormal(tau, tau_m, alpha):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.exp(-0.5*((np.log(tau) - np.log(tau_m))/g)**2) / (tau * g * np.sqrt(2*np.pi))

# eqn(10) ---> tau^{-1} = 10^{-A} * exp(-Ueff/T) + 10^{R} * T^{n} + 10^{Q}
def rate_terms(T, A, Ueff, R, n, Q):
    T = np.asarray(T)
    T_safe = np.maximum(T, 1e-300)
    r_orb  = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    r_ram  = 10.0**(R)  * (T_safe**n)
    r_qtm  = 10.0**(Q)
    return r_orb, r_ram, r_qtm

def total_rate(T, A, Ueff, R, n, Q):
    r1, r2, r3 = rate_terms(T, A, Ueff, R, n, Q)
    return r1 + r2 + r3

# ---------------- mcm helpers ----------------
def run_mcm_alltemps(T_vec, mu, sigmas, qs, Z):
    A,U,R,N,Qp = mu
    sA,sU,sR,sN,sQ = sigmas
    theta = np.array([A,U,R,N,Qp]) + Z * np.array([sA,sU,sR,sN,sQ])  # (K,5)
    A_s, U_s, R_s, N_s, Q_s = theta.T

    out = []
    for T in T_vec:
        r = total_rate(T, A_s, U_s, R_s, N_s, Q_s)
        tau = 1.0 / np.maximum(r, 1e-300)
        out.append(np.quantile(np.log(tau), qs))
    return np.vstack(out)  # (M,Q)

def mc_term_fractions_by_T(T_vec, mu, sigmas, Z): ######
    A,U,R,N,Qp = mu
    sA,sU,sR,sN,sQ = sigmas
    theta = np.array([A,U,R,N,Qp]) + Z * np.array([sA,sU,sR,sN,sQ])  # (K,5)
    A_s, U_s, R_s, N_s, Q_s = theta.T

    fq, fr, fo = [], [], []
    for T in T_vec:
        r_orb, r_ram, r_qtm = rate_terms(T, A_s, U_s, R_s, N_s, Q_s)
        r_tot = np.maximum(r_orb + r_ram + r_qtm, 1e-300)
        fq.append(np.mean(r_qtm / r_tot))
        fr.append(np.mean(r_ram / r_tot))
        fo.append(np.mean(r_orb / r_tot))
    return np.array(fq), np.array(fr), np.array(fo)

# ---------------- smooth weighting ----------------
def smooth_weights_over_T(T_vec, w, width_K=3.0):
    T = np.asarray(T_vec, float)
    w = np.asarray(w, float)
    out = np.zeros_like(w)
    for i in range(len(T)):
        d = T - T[i]
        k = np.exp(-0.5*(d/width_K)**2)
        k /= np.maximum(k.sum(), 1e-12)
        out[i] = float((k * w).sum())
    return np.clip(out, 0.0, 1.0)

def infer_domain_weights(T_vec, frac_qtm, frac_raman, frac_orb,
                         smooth_K=3.0, min_floor=0.15):
    # Hard labels by argmax
    stacks = np.vstack([frac_qtm, frac_raman, frac_orb])  # (3,M)
    labels = np.argmax(stacks, axis=0)  # 0=qtm, 1=raman, 2=orb

    w_qtm   = (labels == 0).astype(float)
    w_raman = (labels == 1).astype(float)
    w_orb   = (labels == 2).astype(float)

    # Smooth each across T
    w_qtm_s   = smooth_weights_over_T(T_vec, w_qtm,   width_K=smooth_K)
    w_raman_s = smooth_weights_over_T(T_vec, w_raman, width_K=smooth_K)
    w_orb_s   = smooth_weights_over_T(T_vec, w_orb,   width_K=smooth_K)

    # Add floor and renormalize
    W = np.vstack([w_qtm_s, w_raman_s, w_orb_s]) + min_floor
    W /= np.maximum(W.sum(axis=0, keepdims=True), 1e-12)
    return W[0], W[1], W[2]  # soft weights that sum to 1 per T

# ---------------- Loss pieces ----------------
def penalty(mu, sigmas):
    A,U,R,N,Qp = mu
    pen = 0.0
    def quad_out(val, lo, hi, scale):
        if val < lo:  return scale*(lo - val)**2
        if val > hi:  return scale*(val - hi)**2
        return 0.0
    pen += quad_out(-A, 0.0, 30.0, 1e-3)
    pen += quad_out(U,  0.0, 3000.0, 1e-6)
    pen += quad_out(R, -20.0, 10.0, 1e-3)
    pen += quad_out(N,  0.0, 12.0, 1e-3)
    pen += quad_out(Qp, -20.0, 10.0, 1e-3)
    pen += 1e-5 * float(np.sum(sigmas**2))
    return pen

def domain_dominance_penalty(T_vec, mu, sigmas, Z, w_qtm, w_raman, w_orb,
                             lam_qtm=15.0, lam_raman=15.0, lam_orb=15.0):
    fq, fr, fo = mc_term_fractions_by_T(T_vec, mu, sigmas, Z)
    p_qtm   = lam_qtm   * float(np.dot(w_qtm,   (1.0 - fq)**2) / max(np.sum(w_qtm), 1.0))
    p_raman = lam_raman * float(np.dot(w_raman, (1.0 - fr)**2) / max(np.sum(w_raman), 1.0))
    p_orb   = lam_orb   * float(np.dot(w_orb,   (1.0 - fo)**2) / max(np.sum(w_orb), 1.0))
    return p_qtm + p_raman + p_orb

# ---------------- Main: two-stage auto-domain joint fit ----------------
def main():
    # ---------- Load data ----------
    df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["T", "tau_mean", "alpha"]
    df = df.astype(float).reset_index(drop=True)

    T_vec    = df["T"].values
    tau_mvec = df["tau_mean"].values
    alphavec = df["alpha"].values

    # Quantiles and targets
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)
    target_lnq_all = np.vstack([fk_ln_quantiles(tm, a, qs) for tm, a in zip(tau_mvec, alphavec)])

    # Initial guess (same as before)
    A0    = -11.537938241219702
    U0    =  956.6247933615898
    R0    =  -5.432741018066928
    n0    =   4.320107244859063
    Q0    =  -0.12029577025666016
    sA0, sU0, sR0, sN0, sQ0 = 0.15, 60.0, 0.15, 0.20, 0.15
    x0 = np.array([A0,U0,R0,n0,Q0, np.log(sA0),np.log(sU0),np.log(sR0),np.log(sN0),np.log(sQ0)], float)

    # Global CRNs
    K = 25_000
    rng = np.random.default_rng(54321)   # set to 12345 to match earlier runs
    Z = rng.standard_normal(size=(K, 5))

    # ---------- Stage 1: unweighted joint fit ----------
    def unpack(x):
        mu = x[:5]; sigmas = np.exp(x[5:]); return mu, sigmas

    def loss_stage1(x):
        mu, sigmas = unpack(x)
        lnq_hat = run_mcm_alltemps(T_vec, mu, sigmas, qs, Z)
        resid = (lnq_hat - target_lnq_all).ravel()
        return float(np.dot(resid, resid) + penalty(mu, sigmas))

    res1 = minimize(
        loss_stage1, x0, method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-4, "disp": True}
    )
    mu1, sig1 = unpack(res1.x)

    # Infer domain weights from fractions (auto domains)
    fq, fr, fo = mc_term_fractions_by_T(T_vec, mu1, sig1, Z)
    wq, wr, wo = infer_domain_weights(T_vec, fq, fr, fo, smooth_K=3.0, min_floor=0.15)

    # Save diagnostics for the inferred domains
    dom_df = pd.DataFrame({
        "T": T_vec,
        "frac_qtm": fq, "frac_raman": fr, "frac_orbach": fo,
        "w_qtm": wq, "w_raman": wr, "w_orbach": wo
    })
    dom_df.to_csv("auto_domain_diagnostics.csv", index=False)

    # ---------- Stage 2: refit with domain dominance penalties ----------
    LAMBDA_QTM, LAMBDA_RAMAN, LAMBDA_ORB = 15.0, 15.0, 15.0  # tune

    def obj_func(x):
        mu, sigmas = unpack(x)
        lnq_hat = run_mcm_alltemps(T_vec, mu, sigmas, qs, Z)
        resid = (lnq_hat - target_lnq_all).ravel()
        return (float(np.dot(resid, resid))
                + penalty(mu, sigmas)
                + domain_dominance_penalty(T_vec, mu, sigmas, Z, wq, wr, wo,
                                           lam_qtm=LAMBDA_QTM, lam_raman=LAMBDA_RAMAN, lam_orb=LAMBDA_ORB))

    res2 = minimize(
        obj_func, res1.x, method="Nelder-Mead",
        options={"maxiter": 700, "xatol": 1e-3, "fatol": 1e-4, "disp": True}
    )
    mu_hat, sig_hat = unpack(res2.x)
    A,U,R,N,Qp = mu_hat
    sA,sU,sR,sN,sQ = sig_hat

    # ---------- Save fitted params ----------
    summary = pd.Series({
        "A_mean": A, "Ueff_mean": U, "R_mean": R, "n_mean": N, "Q_mean": Qp,
        "A_sd": sA, "Ueff_sd": sU, "R_sd": sR, "n_sd": sN, "Q_sd": sQ,
        "loss_stage1": float(res1.fun), "iters_stage1": int(res1.nit), "success1": bool(res1.success),
        "loss_stage2": float(res2.fun), "iters_stage2": int(res2.nit), "success2": bool(res2.success),
        "K": K, "seed": 54321,
        "lambda_qtm": LAMBDA_QTM, "lambda_raman": LAMBDA_RAMAN, "lambda_orb": LAMBDA_ORB
    })
    summary.to_csv("multiT_fitted_params.csv")

    # ---------- Per-T quantile comparison ----------
    lnq_hat_all = run_mcm_alltemps(T_vec, mu_hat, sig_hat, qs, Z)
    cols = []
    for q in qs:
        qlab = f"{int(100*q)}"
        cols += [f"target_lnq_{qlab}", f"model_lnq_{qlab}"]
    rows = []
    for i, T in enumerate(T_vec):
        row = [T, tau_mvec[i], alphavec[i]]
        for j in range(len(qs)):
            row += [target_lnq_all[i, j], lnq_hat_all[i, j]]
        rows.append(row)
    pd.DataFrame(rows, columns=["T","tau_mean","alpha"]+cols).to_csv("multiT_quantile_fit.csv", index=False)

    # ---------- Plots across all T ----------
    Kplot = 80_000
    rng_plot = np.random.default_rng(7)
    Zplot = rng_plot.standard_normal(size=(Kplot, 5))
    theta = mu_hat + Zplot * sig_hat
    A_s, U_s, R_s, N_s, Q_s = theta.T

    os.makedirs("plots", exist_ok=True)
    tau_min, tau_max = 1e-6, 1e3
    taus = np.logspace(np.log10(tau_min), np.log10(tau_max), 5000)
    bins = np.logspace(np.log10(tau_min), np.log10(tau_max), 150)

    with PdfPages("plots_all_temps.pdf") as pdf:
        for i, (T, tm, a) in enumerate(zip(T_vec, tau_mvec, alphavec), start=1):
            r = total_rate(T, A_s, U_s, R_s, N_s, Q_s)
            tau_mc = 1.0 / np.maximum(r, 1e-300)
            rho_fk = rho_tau_lognormal(taus, tm, a)

            plt.figure(figsize=(7,5))
            plt.hist(tau_mc, bins=bins, density=True, alpha=0.45, edgecolor="none",
                     label="MC eqn (10) τ (joint-fit, auto-domains)")
            plt.plot(taus, rho_fk, linewidth=2, label="FK log-normal approx")
            plt.xscale("log")
            plt.xlabel("τ (s)")
            plt.ylabel("ρ(τ)")
            plt.title(f"T = {T:.3f} K  (row {i}/{len(T_vec)})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("plots", f"tau_dist_T{T:.3f}K.png"), dpi=160)
            pdf.savefig()
            plt.close()

    print("\n=== Joint fit complete (auto-domain) ===")
    print(summary.to_string())
    print("Wrote: multiT_fitted_params.csv, multiT_quantile_fit.csv, auto_domain_diagnostics.csv, plots_all_temps.pdf, plots/*.png")

if __name__ == "__main__":
    main()