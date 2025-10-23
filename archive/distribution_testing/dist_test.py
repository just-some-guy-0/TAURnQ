import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load your TSV file (T, tau_mean, alpha) ---
df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
df.columns = ["T", "tau_mean", "alpha"]

T        = df["T"].astype(float).values
tau_mean = df["tau_mean"].astype(float).values
alpha    = df["alpha"].astype(float).values

# --- Map alpha -> log-normal width g (Reta & Chilton) ---
g = 1.82 * np.sqrt(alpha) / (1.0 - alpha)

# Mean rate and asymmetric 1σ in log10(rate)
rate      = 1.0 / tau_mean
rate_hi   = 1.0 / (tau_mean * np.exp(-g))  # +1σ in rate
rate_lo   = 1.0 / (tau_mean * np.exp(+g))  # -1σ in rate
y         = np.log10(rate)
y_plus    = np.log10(rate_hi) - y
y_minus   = y - np.log10(rate_lo)

# --- Eqn (10) model ---
def log10_rate(T, A, Ueff, R, n, Q):
    term_orb = (10.0**(-A)) * np.exp(-Ueff / T)
    term_ram = (10.0**(R))  * (T**n)
    term_qtm = (10.0**(-Q))
    r = term_orb + term_ram + term_qtm
    return np.log10(np.clip(r, 1e-300, None))

# Correct Table 2 parameters for [Dy(tBuO)Cl(THF)5][B(Ph)4]
# 0σ set
A_0s, U_0s, R_0s, n_0s, Q_0s = -11.54, 957, -5.39, 4.32, -0.12
# 1σ set
A_1s, U_1s, R_1s, n_1s, Q_1s = -12, 1000, -6, 4.0, -0.1


# Plot grid
Tg = np.linspace(T.min()*0.95, T.max()*1.05, 400)
y_1s = log10_rate(Tg, A_1s, U_1s, R_1s, n_1s, Q_1s)
y_0s = log10_rate(Tg, A_0s, U_0s, R_0s, n_0s, Q_0s)

# --- Plot ---
plt.figure(figsize=(8,5))
plt.errorbar(T, y, yerr=[y_minus, y_plus], fmt='o', ms=5, capsize=3,
             label="mean τ (with 1σ from α)")
plt.plot(Tg, y_1s, 'r-',  lw=2, label="Eq.10 with Table 2 (1σ set)")
plt.plot(Tg, y_0s, 'k--', lw=1.5, label="Eq.10 with Table 2 (0σ set)")
plt.xlabel("Temperature (K)")
plt.ylabel("log10(rate = 1/τ)")
plt.title("[tBuOCl")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
