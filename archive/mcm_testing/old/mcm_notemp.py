#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from scipy.stats import norm

# ======================= FK log-normal target (on ln tau) =======================
def fk_ln_quantiles(tau_m, alpha, qs):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.log(tau_m) + norm.ppf(qs) * g

def rho_tau_lognormal(tau, tau_m, alpha):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.exp(-0.5*((np.log(tau) - np.log(tau_m))/g)**2) / (tau * g * np.sqrt(2*np.pi))

# ======================= Eqn (10): Orbach + Raman + QTM =======================
# tau^{-1}(T) = 10^{-A} * exp(-Ueff/T) + 10^{R} * T^{n} + 10^{Q}
def rate_terms(T, A, Ueff, R, n, Q):
    T_safe = np.maximum(T, 1e-300)
    r_orb  = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    r_ram  = 10.0**(R)  * (T_safe**n)
    r_qtm  = 10.0**(Q)
    return r_orb, r_ram, r_qtm

def total_rate(T, A, Ueff, R, n, Q):
    r1, r2, r3 = rate_terms(T, A, Ueff, R, n, Q)
    return r1 + r2 + r3

# ======================= Monte Carlo helpers =======================
def mc_ln_tau_quantiles_allT(T_vec, mu, sigmas, qs, Z):
    """Return (M,Q) ln(tau) quantiles at each T using global CRNs Z."""
    A,U,R,N,Qp = mu
    sA,sU,sR,sN,sQ = sigmas
    theta = np.array([A,U,R,N,Qp]) + Z * np.array([sA,sU,sR,sN,sQ])  # (K,5)
    A_s, U_s, R_s, N_s, Q_s = theta.T

    lnq_list = []
    for T in T_vec:
        r = total_rate(T, A_s, U_s, R_s, N_s, Q_s)
        tau = 1.0 / np.maximum(r, 1e-300)
        lnq_list.append(np.quantile(np.log(tau), qs))
    return np.vstack(lnq_list)

def mc_term_fractions_by_T(T_vec, mu, sigmas, Z):
    """
    For each T, return the average fractions E[r_term/r_total] for
    QTM, Raman, Orbach across the MC draws.
    Returns three (M,) arrays: frac_qtm, frac_raman, frac_orbach.
    """
    A,U,R,N,Qp = mu
    sA,sU,sR,sN,sQ = sigmas
    theta = np.array([A,U,R,N,Qp]) + Z * np.array([sA,sU,sR,sN,sQ])  # (K,5)
    A_s, U_s, R_s, N_s, Q_s = theta.T

    fq_list, fr_list, fo_list = [], [], []
    for T in T_vec:
        r_orb, r_ram, r_qtm = rate_terms(T, A_s, U_s, R_s, N_s, Q_s)
        r_tot = np.maximum(r_orb + r_ram + r_qtm, 1e-300)
        fq_list.append(np.mean(r_qtm / r_tot))
        fr_list.append(np.mean(r_ram / r_tot))
        fo_list.append(np.mean(r_orb / r_tot))
    return np.array(fq_list), np.array(fr_list), np.array(fo_list)

# ======================= Smooth domain weight functions =======================
def trapezoid_weight(T_vec, lo, hi, ramp_lo=3.0, ramp_hi=3.0):
    """
    Trapezoidal weight in [0,1]:
      - linearly rises from 0 at (lo - ramp_lo) to 1 at lo
      - equals 1 between lo and hi
      - linearly falls from 1 at hi to 0 at (hi + ramp_hi)
      - 0 outside those regions
    """
    T_vec = np.asarray(T_vec, dtype=float)
    w = np.zeros_like(T_vec)

    # rising edge
    if ramp_lo > 0:
        mask = (T_vec > lo - ramp_lo) & (T_vec < lo)
        w[mask] = (T_vec[mask] - (lo - ramp_lo)) / ramp_lo
    w[T_vec >= lo] = 1.0

    # falling edge
    if ramp_hi > 0:
        mask2 = (T_vec > hi) & (T_vec < hi + ramp_hi)
        w[mask2] = 1.0 - (T_vec[mask2] - hi) / ramp_hi
    w[T_vec >= hi + ramp_hi] = 0.0

    # clip for safety
    return np.clip(w, 0.0, 1.0)

# ======================= Main: multi-T joint fit with domain weights =======================
def main():
    # ----- Load data -----
    df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
    df.columns = ["T", "tau_mean", "alpha"]
    df = df.astype(float).reset_index(drop=True)

    T_vec    = df["T"].values
    tau_mvec = df["tau_mean"].values
    alphavec = df["alpha"].values
    M = len(T_vec)

    # Quantiles to match and targets per T
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)
    target_lnq_all = np.vstack([fk_ln_quantiles(tm, a, qs) for tm, a in zip(tau_mvec, alphavec)])  # (M,Q)

    # ----- Initial guess (your single-T start) -----
    A0    = -11.537938241219702
    U0    =  956.6247933615898
    R0    =  -5.432741018066928
    n0    =   4.320107244859063
    Q0    =  -0.12029577025666016
    sA0, sU0, sR0, sN0, sQ0 = 0.15, 60.0, 0.15, 0.20, 0.15
    x0 = np.array([A0, U0, R0, n0, Q0,
                   np.log(sA0), np.log(sU0), np.log(sR0), np.log(sN0), np.log(sQ0)], dtype=float)

    # ----- Global CRNs -----
    K = 25_000
    rng = np.random.default_rng(54321)   # set to 12345 if you want to match earlier runs
    Z = rng.standard_normal(size=(K, 5))

    # ----- Domain definitions (edit as desired) -----
    # From your note: QTM dominates ~9.7–20 K; Orbach > 40 K; Raman between.
    QTM_LO, QTM_HI   = 9.7, 20.0
    RAMAN_LO, RAMAN_HI = 20.0, 40.0
    ORBACH_LO, ORBACH_HI = 40.0, T_vec.max()  # open-ended upper bound

    # Smooth ramps (K) around domain edges
    RAMP_IN  = 3.0  # ramp width entering domain
    RAMP_OUT = 3.0  # ramp width leaving domain

    w_qtm   = trapezoid_weight(T_vec, QTM_LO,   QTM_HI,   ramp_lo=RAMP_IN, ramp_hi=RAMP_OUT)
    w_raman = trapezoid_weight(T_vec, RAMAN_LO, RAMAN_HI, ramp_lo=RAMP_IN, ramp_hi=RAMP_OUT)
    # For Orbach, we only ramp in at ~40 K; no falling edge
    w_orb   = trapezoid_weight(T_vec, ORBACH_LO, ORBACH_HI, ramp_lo=RAMP_IN, ramp_hi=0.0)

    # Penalty strengths (tune as needed)
    LAMBDA_QTM   = 15.0
    LAMBDA_RAMAN = 15.0
    LAMBDA_ORB   = 15.0

    # ----- Helpers -----
    def unpack(x):
        mu = x[:5]
        sigmas = np.exp(x[5:])
        return mu, sigmas

    def penalty_plausibility(mu, sigmas):
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

    def penalty_domain_dominance(mu, sigmas):
        """
        Encourage each term to dominate within its own domain by pushing the
        corresponding average fraction toward 1 (quadratic penalty).
        """
        frac_qtm, frac_raman, frac_orb = mc_term_fractions_by_T(T_vec, mu, sigmas, Z)  # (M,) each

        # weighted mean squared shortfall from 1 within each domain
        p_qtm   = LAMBDA_QTM   * float(np.dot(w_qtm,   (1.0 - frac_qtm)**2)   / max(np.sum(w_qtm), 1.0))
        p_raman = LAMBDA_RAMAN * float(np.dot(w_raman, (1.0 - frac_raman)**2) / max(np.sum(w_raman), 1.0))
        p_orb   = LAMBDA_ORB   * float(np.dot(w_orb,   (1.0 - frac_orb)**2)   / max(np.sum(w_orb), 1.0))
        return p_qtm + p_raman + p_orb

    def loss(x):
        mu, sigmas = unpack(x)
        lnq_hat = mc_ln_tau_quantiles_allT(T_vec, mu, sigmas, qs, Z)   # (M,Q)
        resid = (lnq_hat - target_lnq_all).ravel()
        return (float(np.dot(resid, resid))
                + penalty_plausibility(mu, sigmas)
                + penalty_domain_dominance(mu, sigmas))

    # ----- Optimize -----
    res = minimize(
        loss, x0, method="Nelder-Mead",
        options={"maxiter": 700, "xatol": 1e-3, "fatol": 1e-4, "disp": True}
    )
    mu_hat, sig_hat = unpack(res.x)
    A,U,R,N,Qp = mu_hat
    sA,sU,sR,sN,sQ = sig_hat

    # ----- Save fitted params -----
    summary = pd.Series({
        "A_mean": A, "Ueff_mean": U, "R_mean": R, "n_mean": N, "Q_mean": Qp,
        "A_sd": sA, "Ueff_sd": sU, "R_sd": sR, "n_sd": sN, "Q_sd": sQ,
        "loss": float(res.fun), "success": bool(res.success), "iters": int(res.nit),
        "K": K, "seed": 54321,
        "LAMBDA_QTM": LAMBDA_QTM, "LAMBDA_RAMAN": LAMBDA_RAMAN, "LAMBDA_ORB": LAMBDA_ORB,
        "QTM_range": f"[{QTM_LO},{QTM_HI}]", "RAMAN_range": f"[{RAMAN_LO},{RAMAN_HI}]",
        "ORBACH_lo": ORBACH_LO
    })
    summary.to_csv("multiT_fitted_params.csv")

    # ----- Per-T quantile comparison table -----
    lnq_hat_all = mc_ln_tau_quantiles_allT(T_vec, mu_hat, sig_hat, qs, Z)
    cols = []
    for q in qs:
        qlab = f"{int(100*q)}"
        cols += [f"target_lnq_{qlab}", f"model_lnq_{qlab}"]
    data_rows = []
    for i, T in enumerate(T_vec):
        row = [T, tau_mvec[i], alphavec[i]]
        for j in range(len(qs)):
            row += [target_lnq_all[i, j], lnq_hat_all[i, j]]
        data_rows.append(row)
    quant_df = pd.DataFrame(data_rows, columns=["T","tau_mean","alpha"] + cols)
    quant_df.to_csv("multiT_quantile_fit.csv", index=False)

    # ----- Plots across ALL temperatures -----
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
                     label="Monte Carlo eqn (10)")
            plt.plot(taus, rho_fk, linewidth=2, label="FK log-normal approx")
            plt.xscale("log")
            plt.xlabel("τ (s)")
            plt.ylabel("ρ(τ)")
            plt.title(f"T = {T:.3f} K")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("plots", f"tau_dist_T{T:.3f}K.png"), dpi=160)
            pdf.savefig()
            plt.close()

    print("\n=== Joint fit complete (domain-weighted dominance) ===")
    print(summary.to_string())
    print("Wrote: multiT_fitted_params.csv, multiT_quantile_fit.csv, plots_all_temps.pdf, and plots/*.png")

if __name__ == "__main__":
    main()
