#!/usr/bin/env python3
"""
dc_montecarlo.py  –  TAURnQ Monte Carlo phase for DC magnetometry data.

Reads the output of dc_phase2.py (columns: T, mu_ln, sigma1_ln, sigma2_ln, ...)
and runs the same correlated-parameter Monte Carlo as montecarlo.py, but with
two key differences:

  1. TARGET QUANTILES come from a two-piece log-normal (asymmetric) instead of
     the FK log-normal (symmetric).  This correctly represents the SEF distribution.

  2. The central value used is e^<ln tau> (Blackmore et al. Eq. 9), NOT tau_star.
     The asymmetric sigma1/sigma2 encode the left/right widths of the SEF.

Everything else – the rate model, parameter draw structure, Nelder-Mead
optimisation, output format – is kept identical to montecarlo.py so the output
CSV drops straight into your existing phase4/global_fit.py.

Input CSV columns (from dc_phase2.py):
    T, tau_star, beta, mu_ln, sigma1_ln, sigma2_ln, elnTau_ln, esd_elnTau_ln

Output CSV columns  (same schema as montecarlo.py):
    T, Au, Uu, Ru, Nu, Qu, As, Us, Rs, Ns, Qs

Usage:
    python dc_montecarlo.py --infile dc_phase2_out.csv [--out dc_mc_params.csv]
                            [--idx 3] [--all] [--plot] [--clear]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os, sys

PATH = "dc_phase2_out.csv"


# ---------------------------------------------------------------------------
# Rate model
# ---------------------------------------------------------------------------

def rate_model(T, A, Ueff, R, n, Q):
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0 ** (-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    term2 = 10.0 ** R    * (T_safe ** n)
    term3 = 10.0 ** (-Q)
    return term1 + term2 + term3


# ---------------------------------------------------------------------------
# Two-piece log-normal quantile function
# ---------------------------------------------------------------------------

def tplognormal_quantiles(mu_ln, sigma1_ln, sigma2_ln, qs):
    """
    Quantiles of a two-piece normal distribution in ln(tau) space.

    The two-piece normal has:
        left  piece: N(mu_ln, sigma1_ln^2)  for ln(tau) <= mu_ln
        right piece: N(mu_ln, sigma2_ln^2)  for ln(tau) >= mu_ln

    The CDF is (Wallis 2014, Eq. in Section 1):
        F(x) = sigma1/(sigma1+sigma2) * 2*Phi((x-mu)/sigma1)        for x <= mu
        F(x) = 1 - sigma2/(sigma1+sigma2) * 2*(1-Phi((x-mu)/sigma2)) for x >= mu

    Inverting this analytically gives closed-form quantiles.
    """
    s1, s2 = sigma1_ln, sigma2_ln
    p_left = s1 / (s1 + s2)     # probability mass on left piece

    qs = np.asarray(qs, dtype=float)
    result = np.empty_like(qs)

    for i, q in enumerate(qs):
        if q <= p_left:
            # in the left piece: q = s1/(s1+s2) * 2*Phi((x-mu)/s1)
            # => Phi((x-mu)/s1) = q*(s1+s2)/(2*s1)
            z = norm.ppf(q * (s1 + s2) / (2.0 * s1))
            result[i] = mu_ln + z * s1
        else:
            # in the right piece: 1-q = s2/(s1+s2) * 2*(1-Phi((x-mu)/s2))
            # => Phi((x-mu)/s2) = 1 - (1-q)*(s1+s2)/(2*s2)
            z = norm.ppf(1.0 - (1.0 - q) * (s1 + s2) / (2.0 * s2))
            result[i] = mu_ln + z * s2

    return result


# ---------------------------------------------------------------------------
# Correlated parameter draws  (identical to montecarlo.py)
# ---------------------------------------------------------------------------

def draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z):
    A, U, R, n, Q = mu
    sA, sU, sR, sN, sQ = sigmas
    L_AU = np.array([[sA, 0.0],
                     [rho_AU * sU, sU * np.sqrt(max(1.0 - rho_AU ** 2, 1e-12))]])
    L_RN = np.array([[sR, 0.0],
                     [rho_RN * sN, sN * np.sqrt(max(1.0 - rho_RN ** 2, 1e-12))]])
    eps_AU = Z[:, :2] @ L_AU.T
    eps_RN = Z[:, 2:4] @ L_RN.T
    eps_Q  = Z[:, 4] * sQ
    return np.column_stack([
        A + eps_AU[:, 0],
        U + eps_AU[:, 1],
        R + eps_RN[:, 0],
        n + eps_RN[:, 1],
        Q + eps_Q
    ])


def simulate_ln_tau_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, Z):
    theta = draw_params_correlated(mu, sigmas, rho_AU, rho_RN, Z)
    A_s, U_s, R_s, N_s, Q_s = theta.T
    r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
    tau = 1.0 / np.maximum(r, 1e-300)
    return np.quantile(np.log(tau), qs)


# ---------------------------------------------------------------------------
# Penalty  (identical to montecarlo.py)
# ---------------------------------------------------------------------------

def penalty(mu, sigmas, rho_AU, rho_RN):
    A, Ueff, R, n, Q = mu
    pen = 0.0

    def quad_out(val, lo, hi, scale):
        if val < lo: return scale * (lo - val) ** 2
        if val > hi: return scale * (val - hi) ** 2
        return 0.0

    pen += quad_out(-A,    0.0,    30.0,   1e-3)
    pen += quad_out(Ueff,  0.0,  3000.0,   1e-6)
    pen += quad_out(R,   -20.0,    10.0,   1e-3)
    pen += quad_out(n,     0.0,    12.0,   1e-3)
    pen += quad_out(Q,   -20.0,    10.0,   1e-3)
    pen += 1e-5 * float(np.sum(sigmas ** 2))
    pen += 1e-4 * ((abs(rho_AU) > 0.995) * (abs(rho_AU) - 0.995) ** 2
                   + (abs(rho_RN) > 0.995) * (abs(rho_RN) - 0.995) ** 2)
    return pen


# ---------------------------------------------------------------------------
# Fit a single temperature row
# ---------------------------------------------------------------------------

def fit_one_row(df, idx, out_path, seed_base=12345, make_plot=False):
    row = df.loc[idx]
    T        = float(row["T"])
    mu_ln    = float(row["mu_ln"])       # ln(tau) at mode of SEF
    sigma1   = float(row["sigma1_ln"])   # left-side ln width
    sigma2   = float(row["sigma2_ln"])   # right-side ln width

    # Quantile targets from the two-piece log-normal (asymmetric!)
    qs = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98], dtype=float)
    target_lnq = tplognormal_quantiles(mu_ln, sigma1, sigma2, qs)

    # Initial parameter guesses  (same defaults as montecarlo.py – adjust per molecule)
    A    = -11.093491
    Ueff =  900.572427
    R    =   -4.831255
    N    =    3.963503
    Q    =   -0.182074
    sA, sUeff, sR, sN, sQ = 0.224577, 14.365442, 2.171177, 1.392262, 0.565583
    eta_AU = 0.9
    eta_RN = np.arctanh(0.99)

    x0 = np.array([A, Ueff, R, N, Q,
                   np.log(sA), np.log(sUeff), np.log(sR), np.log(sN), np.log(sQ),
                   eta_AU, eta_RN], dtype=float)

    K = 25000
    rng = np.random.default_rng(int(seed_base) + int(idx))
    Z = rng.standard_normal(size=(K, 5))

    def unpack(x):
        mu_params = x[:5]
        sigmas    = np.exp(x[5:10])
        rho_AU    = np.tanh(x[10])
        rho_RN    = -np.tanh(x[11])
        return mu_params, sigmas, rho_AU, rho_RN

    def objective(x):
        mu_p, sigs, rho_AU, rho_RN = unpack(x)
        lnq_hat = simulate_ln_tau_quantiles(T, mu_p, sigs, rho_AU, rho_RN, qs, Z)
        resid = lnq_hat - target_lnq
        return float(np.dot(resid, resid) + penalty(mu_p, sigs, rho_AU, rho_RN))

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 150, "xatol": 1e-3, "fatol": 2e-4, "disp": False})

    mu_hat, sig_hat, rho_AU_hat, rho_RN_hat = unpack(res.x)
    A_h, U_h, R_h, N_h, Q_h = mu_hat
    sA_h, sU_h, sR_h, sN_h, sQ_h = sig_hat

    print(f"[idx={idx:>3}] T={T:6.2f} K | loss={float(res.fun):.5g} | "
          f"A={A_h:.4f} U={U_h:.2f} R={R_h:.4f} n={N_h:.4f} Q={Q_h:.4f} | "
          f"sA={sA_h:.3f} sU={sU_h:.2f} sR={sR_h:.2f} sN={sN_h:.2f} sQ={sQ_h:.2f} | "
          f"rho_AU={rho_AU_hat:.3f} rho_RN={rho_RN_hat:.3f}")

    row_out = pd.DataFrame([{
        "T":  T,
        "Au": A_h,   "Uu": U_h,   "Ru": R_h,   "Nu": N_h,   "Qu": Q_h,
        "As": sA_h,  "Us": sU_h,  "Rs": sR_h,  "Ns": sN_h,  "Qs": sQ_h,
    }])
    header_needed = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    row_out.to_csv(out_path, index=False, mode="a", header=header_needed)

    if make_plot:
        _plot_fit(T, mu_ln, sigma1, sigma2, mu_hat, sig_hat, rho_AU_hat, rho_RN_hat, idx)


def _plot_fit(T, mu_ln, s1, s2, mu_hat, sig_hat, rho_AU, rho_RN, idx):
    """Compare MC-sampled ln(tau) histogram vs the two-piece log-normal target."""
    Kplot = 60000
    Z_plot = np.random.default_rng(54321 + int(idx)).standard_normal((Kplot, 5))
    theta = draw_params_correlated(mu_hat, sig_hat, rho_AU, rho_RN, Z_plot)
    A_s, U_s, R_s, N_s, Q_s = theta.T
    r = rate_model(T, A_s, U_s, R_s, N_s, Q_s)
    tau_mc = 1.0 / np.maximum(r, 1e-300)

    # Target two-piece log-normal curve
    ln_grid = np.linspace(mu_ln - 8, mu_ln + 8, 800)
    target_density = np.where(
        ln_grid <= mu_ln,
        np.exp(-0.5 * ((ln_grid - mu_ln) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi)),
        np.exp(-0.5 * ((ln_grid - mu_ln) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
    )
    # re-normalise for the two-piece case
    norm_factor = (s1 + s2) / 2.0
    target_density = target_density * norm_factor

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ln_tau_mc = np.log(tau_mc)
    bins = np.linspace(ln_grid[0], ln_grid[-1], 120)
    ax.hist(ln_tau_mc, bins=bins, density=True, alpha=0.45,
            color="#378ADD", edgecolor="none", label="MC ln(τ) samples")
    ax.plot(ln_grid, target_density, color="#D85A30", linewidth=2,
            label="Two-piece log-normal target")
    ax.axvline(mu_ln, color="#639922", linewidth=1.2, linestyle=":",
               label=f"mode μ_ln = {mu_ln:.2f}")
    ax.set_xlabel("ln(τ)")
    ax.set_ylabel("density")
    ax.set_title(f"MC fit vs two-piece log-normal target  |  T = {T:.1f} K")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI  (identical argument structure to montecarlo.py)
# ---------------------------------------------------------------------------

def parse_range(spec, nmax):
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError("--range expects start:end or start:end:step")
    start = int(parts[0]) if parts[0] else 0
    end   = int(parts[1]) if parts[1] else nmax
    step  = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    return range(start, min(end, nmax), step)


def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ Monte Carlo for DC magnetometry (two-piece log-normal targets)"
    )
    ap.add_argument("--infile",     default=PATH,
                    help="CSV from dc_phase2.py (T, mu_ln, sigma1_ln, sigma2_ln, ...)")
    ap.add_argument("--idx",        type=int, default=None)
    ap.add_argument("--idxs",       type=str, default=None)
    ap.add_argument("--range",      dest="range_spec", type=str, default=None)
    ap.add_argument("--all",        default=True, action="store_true")
    ap.add_argument("--out",        default="dc_mc_params.csv")
    ap.add_argument("--clear",      action="store_true")
    ap.add_argument("--seed_base",  type=int, default=14322)
    ap.add_argument("--plot",       action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    required = {"T", "mu_ln", "sigma1_ln", "sigma2_ln"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: input CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.reset_index(drop=True)
    nrows = len(df)

    if args.clear and os.path.exists(args.out):
        open(args.out, "w").close()

    idx_list = []
    if args.all:
        idx_list = list(range(nrows))
    elif args.range_spec:
        idx_list = list(parse_range(args.range_spec, nrows))
    elif args.idxs:
        idx_list = [int(s.strip()) for s in args.idxs.split(",") if s.strip()]
    elif args.idx is not None:
        idx_list = [int(args.idx)]
    else:
        print("Provide one of --idx, --idxs, --range, or --all.", file=sys.stderr)
        sys.exit(2)

    make_plot = args.plot and (len(idx_list) == 1)

    for idx in idx_list:
        if not (0 <= idx < nrows):
            print(f"Skip idx {idx}: out of range [0,{nrows-1}]", file=sys.stderr)
            continue
        try:
            fit_one_row(df, idx, args.out,
                        seed_base=args.seed_base, make_plot=make_plot)
        except Exception as e:
            print(f"[idx={idx}] ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Appended {len(idx_list)} row(s) to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
