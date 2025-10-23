import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- FK log-normal ----------
def rho_tau(tau, tau_m, alpha):
    g = 1.82*np.sqrt(alpha)/(1 - alpha)
    return np.exp(-0.5*((np.log(tau) - np.log(tau_m))/g)**2) / (tau * g * np.sqrt(2*np.pi))

# ---------- parameter means ----------
A_mean    = -11.537938241219702
Ueff_mean =  956.6247933615898  # Kelvin
R_mean    =  -5.432741018066928
n_mean    =   4.320107244859063
Q_mean    =  -0.12029577025666016

# ---------- MC settings ----------
K   = 50000
rng = np.random.default_rng(7)

# independent SDs (tune as needed)
sd_A, sd_Ueff, sd_R, sd_n, sd_Q = 0.15, 60.0, 0.15, 0.20, 0.15

# ---------- load data & pick row index=20 ----------
df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
df.columns = ["T", "tau_mean", "alpha"]
df = df.astype(float)

idx   = 20
T     = float(df.loc[idx, "T"])
tau_m = float(df.loc[idx, "tau_mean"])
alpha = float(df.loc[idx, "alpha"])

# ---------- eqn(10) model ----------
def rate_from_params(T, A, Ueff, R, n, Q):
    term1 = 10.0**(-A) * np.exp(-Ueff / np.maximum(T, 1e-12))
    term2 = 10.0**R * (np.maximum(T, 1e-300)**n)
    term3 = 10.0**Q
    return term1 + term2 + term3

# MC draw
A_s = rng.normal(A_mean, sd_A, size=K)
U_s = rng.normal(Ueff_mean, sd_Ueff, size=K)
R_s = rng.normal(R_mean, sd_R, size=K)
n_s = rng.normal(n_mean, sd_n, size=K)
Q_s = rng.normal(Q_mean, sd_Q, size=K)

rates  = rate_from_params(T, A_s, U_s, R_s, n_s, Q_s)
tau_mc = 1.0 / np.maximum(rates, 1e-300)

# FK log-normal curve
taus = np.logspace(-6, 3, 5000)
rho_fk_lognorm = rho_tau(taus, tau_m, alpha)

# ---------- plot overlay ----------
plt.figure(figsize=(7,5))
bins = np.logspace(np.log10(1e-6), np.log10(1e3), 120)
plt.hist(tau_mc, bins=bins, density=True, alpha=0.45, edgecolor="none", label="MC eqn (10) τ")
plt.plot(taus, rho_fk_lognorm, linewidth=2, label="FK log-normal approx")
plt.xscale("log")
plt.xlabel("τ (s)")
plt.ylabel("ρ(τ)")
plt.title(f"τ distribution at T = {T:.1f} K (index {idx})")
plt.legend()
plt.tight_layout()
plt.show()
