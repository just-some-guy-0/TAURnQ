#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

LN10 = np.log(10.0)
tmin, tmax = 44, 58
PATH = "tBuOCl.tsv"

# -------- FK helpers --------
def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha) / (1 - alpha)

def rho_tau_fk(tau, tau_mu, alpha):
    g = g_from_alpha(alpha)
    z = (np.log(tau) - np.log(tau_mu)) / g
    return np.exp(-0.5*z*z) / (tau * g * np.sqrt(2*np.pi))

# -------- Data loader --------
def load_tsv(path, tmin, tmax):
    df = pd.read_csv(path, delim_whitespace=True, header=None).iloc[:, :3]
    df.columns = ["T", "tau_mu", "alpha"]
    return df[(df["T"] >= tmin) & (df["T"] <= tmax)].reset_index(drop=True)

# -------- Orbach -> log-normal moments --------
def model_moments(T, mu_A, mu_U, sd_A, sd_U, rho):
    # ln(tau) = A*ln(10) + U/T  with (A,U) ~ correlated Gaussian
    mu_ln = LN10*mu_A + mu_U/T
    var_ln = (LN10**2)*(sd_A**2) + (sd_U**2)/(T**2) + 2*(LN10/T)*rho*sd_A*sd_U
    sd_ln = np.sqrt(np.maximum(var_ln, 1e-18))
    return mu_ln, sd_ln

# ===================== main =====================
df = load_tsv(PATH, tmin, tmax)

# Pick a temperature in your Orbach-dominant window (e.g. closest to 45 K)
idx = (df["T"] - 45).abs().idxmin()
T = float(df.loc[idx, "T"])
tau_mu = float(df.loc[idx, "tau_mu"])
alpha = float(df.loc[idx, "alpha"])

# ----- Target FK moments from data -----
mu_ln_fk = np.log(tau_mu)
sd_ln_fk = g_from_alpha(alpha)

# ----- Choose Orbach parameter moments to MATCH the FK moments exactly -----
# For a constructive match: take sd_A = 0 and rho = 0
sd_A = 0
rho = 0.0
sd_U = sd_ln_fk * T              # so that var_ln = (sd_U/T)^2 = sd_ln_fk^2

# Choose any mu_U0 (e.g., 1000 K) and solve mu_A so the mean matches ln(tau_mu)
mu_U = 1000.0                    # pick a convenient scale (you can swap your fitted value here)
mu_A = (mu_ln_fk - mu_U/T) / LN10

# Verify the moments line up
mu_ln_orb, sd_ln_orb = model_moments(T, mu_A, mu_U, sd_A, sd_U, rho)

# PDFs on a tau grid
taus = np.logspace(-4, -1.3, 400)

pdf_fk  = rho_tau_fk(taus, tau_mu, alpha)
pdf_orb = lognorm.pdf(taus, s=sd_ln_orb, scale=np.exp(mu_ln_orb))  # same moments

# ----- plot -----
plt.figure(figsize=(8, 5.2))
plt.plot(taus, pdf_fk,'k-', lw=2, label="Experimental rate distribution")
plt.plot(taus, pdf_orb+0,'#FFA500', lw=2, linestyle= "--", label="Orbach-only distribution")
plt.xscale("log")
plt.xlabel(r"Relaxation rate $(\tau^{-1})$", fontsize = 14)
plt.ylabel(r"Probability density", fontsize = 14)
plt.legend(fontsize = 12, loc = 'upper right')
plt.tight_layout()

# remove tick marks and tick labels, keep axis titles
plt.xticks([])
plt.yticks([])
plt.tick_params(which="both", length=0)

plt.show()

# Optional: print the parameters used
print(f"T = {T:.2f} K | tau_mu = {tau_mu:.3e} s | alpha = {alpha:.3f}")
print(f"Matched moments: mu_ln = {mu_ln_orb:.6f} vs {mu_ln_fk:.6f},  sd_ln = {sd_ln_orb:.6f} vs {sd_ln_fk:.6f}")
print(f"Orbach moments used: mu_A = {mu_A:.3f}, mu_U = {mu_U:.1f} K, sd_A = {sd_A:.3f}, sd_U = {sd_U:.3f}, rho = {rho:.2f}")
