import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
beta = 0.35
tau_star = 1.0  # seconds

def rho_s_beta(s, beta):
    c = np.cos(np.pi * beta / 2)
    d = np.sin(np.pi * beta / 2)

    def integrand(u):
        u_beta = u**beta
        return np.exp(-u_beta * c) * np.cos(s*u - u_beta * d)

    val, err = quad(integrand, 0, np.inf, limit=200)
    return val / np.pi

# τ range 10^-2 – 10^2 → s = τ*/τ = 1/x
tau_vals = np.logspace(-2, 2, 300)
s_vals = tau_star / tau_vals

# Evaluate ρ(s,β) for each s (IMPORTANT)
rho_vals = np.array([rho_s_beta(s, beta) for s in s_vals])

# ------------------------------------------------------------
# Plot on log scale
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(tau_vals, rho_vals)
plt.xscale("log")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\rho(\tau^*/\tau, \beta)$")
plt.title("SEF distribution")
plt.grid(True)
plt.show()
