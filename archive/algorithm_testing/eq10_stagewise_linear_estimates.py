import argparse
import numpy as np
import pandas as pd
import os

LN10 = np.log(10.0)

def g_from_alpha(alpha):
    alpha = np.clip(alpha, 1e-12, 0.999999)
    return 1.82*np.sqrt(alpha)/(1 - alpha)  # natural-log std

def fit_linear(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def stage_orbach(T, tau_mu, g):
    # Means: ln tau = LN10 * mu_A + mu_U / T
    y = np.log(tau_mu)
    X_mean = np.column_stack([np.ones_like(T), 1.0/T])
    b0, b1 = fit_linear(y, X_mean)
    mu_A = b0 / LN10
    mu_U = b1

    # Variance: g^2(T) = (LN10)^2*sigma_A^2 + (sigma_U^2)/T^2 + 2*(LN10/T)*rho*sigma_A*sigma_U
    var = g**2
    X_var = np.column_stack([
        np.ones_like(T),        # -> (LN10)^2 * sigma_A^2
        (1.0/T)**2,             # -> sigma_U^2
        (1.0/T),                # -> 2 * LN10 * rho * sigma_A * sigma_U
    ])
    a0, a1, a2 = fit_linear(var, X_var)
    sigma_A = np.sqrt(max(0.0, a0)) / LN10
    sigma_U = np.sqrt(max(0.0, a1))
    denom = 2.0 * LN10 * max(1e-30, sigma_A * sigma_U)
    rho = np.clip(a2 / denom, -1.0, 1.0)

    return mu_A, mu_U, sigma_A, sigma_U, rho

def stage_raman(T, tau_mu, g):
    # Means: ln tau = -LN10*R - n*ln T
    y = np.log(tau_mu)
    x = np.log(T)
    X_mean = np.column_stack([np.ones_like(T), x])
    c0, c1 = fit_linear(y, X_mean)
    mu_R = -c0 / LN10
    mu_n = -c1

    # Variance: g^2(T) = (LN10)^2*sigma_R^2 + (ln T)^2*sigma_n^2 + 2*(LN10)(ln T)*Cov(R,n)
    var = g**2
    X_var = np.column_stack([
        np.ones_like(T),   # -> (LN10)^2 * sigma_R^2
        (np.log(T))**2,    # -> sigma_n^2
        (np.log(T)),       # -> 2 * LN10 * Cov(R,n) = 2*LN10*rho*sigma_R*sigma_n
    ])
    a0, a1, a2 = fit_linear(var, X_var)
    sigma_R = np.sqrt(max(0.0, a0)) / LN10
    sigma_n = np.sqrt(max(0.0, a1))
    denom = 2.0 * LN10 * max(1e-30, sigma_R * sigma_n)
    rho = np.clip(a2 / denom, -1.0, 1.0)

    return mu_R, mu_n, sigma_R, sigma_n, rho

def stage_qtm(T, tau_mu, g):
    y = np.log(tau_mu)
    mu_Q = float(np.mean(y) / LN10)         # ln tau = LN10 * Q -> Q = ln tau / LN10
    sigma_Q = float(np.mean(g) / LN10)      # Var(ln tau) ~ const -> sigma_Q = g/LN10
    return mu_Q, sigma_Q

def main():
    ap = argparse.ArgumentParser(description="Stagewise linear regressions to estimate parameter means & SDs in dominant windows.")
    ap.add_argument("--infile", type=str, default="tBuOCl.tsv", help="Input TSV/CSV with columns: T, tau_mu, alpha")
    ap.add_argument("--outfile", type=str, default="stagewise_param_estimates.csv", help="Output CSV for stagewise parameter estimates")
    ap.add_argument("--orbmin", type=float, default=40.0, help="Lower T (K) for Orbach window (T >= orbmin)")
    ap.add_argument("--ramlo", type=float, default=25.0, help="Lower T (K) for Raman window (ramlo <= T < orbmin)")
    ap.add_argument("--qtmmax", type=float, default=25.0, help="Upper T (K) for QTM window (T < qtmmax)")
    args = ap.parse_args()

    # Load data
    sep = "\t" if args.infile.lower().endswith(".tsv") else ","
    df = pd.read_csv(args.infile, sep=sep, header=None)
    T = df[0].values.astype(float)
    tau_mu = df[1].values.astype(float)
    alpha = df[2].values.astype(float)

    g = g_from_alpha(alpha)

    # Masks
    mask_orb = T >= args.orbmin
    mask_ram = (T >= args.ramlo) & (T < args.orbmin)
    mask_qtm = T < args.qtmmax

    rows = []

    # Orbach
    if mask_orb.sum() >= 3:
        mu_A, mu_U, sd_A, sd_U, rho_AU = stage_orbach(T[mask_orb], tau_mu[mask_orb], g[mask_orb])
        rows.append(dict(term="Orbach", mu_A=mu_A, mu_U=mu_U, sd_A=sd_A, sd_U=sd_U, rho=rho_AU))
    else:
        rows.append(dict(term="Orbach", mu_A=np.nan, mu_U=np.nan, sd_A=np.nan, sd_U=np.nan, rho=np.nan))

    # Raman
    if mask_ram.sum() >= 3:
        mu_R, mu_n, sd_R, sd_n, rho_Rn = stage_raman(T[mask_ram], tau_mu[mask_ram], g[mask_ram])
        rows.append(dict(term="Raman", mu_R=mu_R, mu_n=mu_n, sd_R=sd_R, sd_n=sd_n, rho=rho_Rn))
    else:
        rows.append(dict(term="Raman", mu_R=np.nan, mu_n=np.nan, sd_R=np.nan, sd_n=np.nan, rho=np.nan))

    # QTM
    if mask_qtm.sum() >= 2:
        mu_Q, sd_Q = stage_qtm(T[mask_qtm], tau_mu[mask_qtm], g[mask_qtm])
        rows.append(dict(term="QTM", mu_Q=mu_Q, sd_Q=sd_Q))
    else:
        rows.append(dict(term="QTM", mu_Q=np.nan, sd_Q=np.nan))

    out = pd.DataFrame(rows)
    out.to_csv(args.outfile, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved stagewise estimates to: {os.path.abspath(args.outfile)}")


if __name__ == "__main__":
    main()
