import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rho_tau(tau, tau_m, alpha):
    g = 1.82*np.sqrt(alpha) / (1 - alpha)
    z = (np.log(tau) - np.log(tau_m)) / g
    return np.exp(-0.5*z*z) / (tau * g * np.sqrt(2*np.pi))

# --- load data ---
df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
df.columns = ["T", "tau_mean", "alpha"]

idx = 20  # T = 45 K in tBuOCl
tau_mu = float(df.loc[idx, "tau_mean"])
alpha  = float(df.loc[idx, "alpha"])

taus = np.logspace(-5, -0.2, 5000)
rho_lg = rho_tau(taus, tau_mu, alpha)

# --- plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(taus, rho_lg, 'tab:orange', label="FK distribution")

ax.set_xscale("log")
ax.set_xlabel(r"$\mu$", fontsize = 18, labelpad = 8)
ax.set_ylabel(r"Probability density",fontsize = 18,labelpad =8)

# remove tick marks and tick labels, keep axis titles
ax.set_xticks([])
ax.set_yticks([])
# (optional) also ensure no tick marks are drawn)
ax.tick_params(which="both", length=0)

# ax.legend()
fig.tight_layout()
plt.show()
