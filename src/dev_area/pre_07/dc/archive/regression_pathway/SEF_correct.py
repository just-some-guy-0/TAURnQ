import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

# -----------------------------
# SEF (stretched exponential) distribution in log10(tau)
# -----------------------------
# Eq (3):  ρ(s,β) with s = τ*/τ
# Eq (5):  ρ_log(τ*, x, β) = ρ(τ*/10^x,β) * ln(10) * (τ*/10^x)
#
# For 0 < β < 1, Eq (3) is a right-skewed (one-sided) Lévy-stable pdf (Nolan S0 form).
# The scaling below maps Eq (3) -> scipy.stats.levy_stable.pdf.

def rho_s(s: np.ndarray, beta: float) -> np.ndarray:
    """
    ρ(s,β) from Eq (3), evaluated via a Lévy-stable pdf.
    Valid for 0 < beta < 1.
    """
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)

    mask = s > 0
    if not np.any(mask):
        return out

    theta = np.pi * beta / 2.0
    c = np.cos(theta)

    # Change of variable u = t / c^(1/beta) gives:
    # ρ(s,β) = c^(-1/beta) * f_stable( s / c^(1/beta) ; alpha=beta, skew=+1, scale=1 )
    s_scaled = s[mask] / (c ** (1.0 / beta))
    out[mask] = (c ** (-1.0 / beta)) * levy_stable.pdf(
        s_scaled, beta, 1.0, loc=0.0, scale=1.0
    )
    return out

def rho_log_tau(tau: np.ndarray, beta: float, tau_star: float = 1.0) -> np.ndarray:
    """
    ρ_log(τ*, x, β) but returned as a function of τ (x = log10 τ).
    So the y-values integrate to 1 over x = log10(τ).
    """
    tau = np.asarray(tau, dtype=float)
    s = tau_star / tau
    return np.log(10.0) * s * rho_s(s, beta)

# -----------------------------
# Reproduce Fig 1a-style plot
# -----------------------------
tau_star = 106.4
tau = np.logspace(-4, 7, 2000)

beta = 0.857

### Single-Plot Code

plt.figure(figsize=(6.6, 4.8))
y = rho_log_tau(tau, beta, tau_star=tau_star)
plt.plot(tau, y, label=f"{beta:g}")

ZETA3 = 1.2020569031595942854
var_ln = (np.pi**2 / 6.0) * (1.0 / beta**2 - 1.0)
mu3_ln = 2.0 * ZETA3 * (1.0 - 1.0 / beta**3)

skew_q = mu3_ln / (var_ln ** 1.5)

print(skew_q)

### Multiple-Plot Code

# betas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# for b in betas:
#     y = rho_log_tau(tau, b, tau_star=tau_star)
#     plt.plot(tau, y, label=f"{b:g}")


# Convert to log-log plot:

# plt.xscale("log")
# plt.yscale("log")
# plt.xlim(1e-4, 1e7)
# plt.ylim(1e-2, 1e2)

### For linear plot:

plt.xlim(0, 300)
plt.ylim(0, 5)

plt.xlabel("Relaxation Rate (1/s)")
plt.ylabel(r"$\rho_{\log}(\tau^*=1\ \mathrm{s},\,\beta)$")
plt.legend(title=None, loc="upper right", frameon=True)
plt.tight_layout()
plt.show()
