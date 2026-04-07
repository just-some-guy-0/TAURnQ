import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import lognorm

BETA = 0.6
SIGMA=0.5
MU=-1

# --- Eqn (3) integrand ---
def rho_integrand(u, s, beta):
    c = np.cos(np.pi * beta / 2)
    s_term = np.sin(np.pi * beta / 2)
    return np.exp(-u**beta * c) * np.cos(s*u - u**beta * s_term)

# --- ρ(s, β) ---
def rho(s, beta, u_max=50):
    """Compute the SEF distribution ρ(s,β) via numerical integration."""
    integral, _ = quad(rho_integrand, 0, u_max, args=(s, beta))
    return integral / np.pi

# --- Lognormal distribution with skew parameter sigma ---
def lognormal_pdf(s, mu=MU, sigma=SIGMA):
    """Lognormal PDF with adjustable skew via sigma."""
    return lognorm.pdf(s, s=sigma, scale=np.exp(mu))

# --- Plotting ---
def plot_rho_with_lognormal(beta=BETA, s_min=0.03, s_max=5, n_points=400,
                            mu=MU, sigma=SIGMA):
    s_vals = np.linspace(s_min, s_max, n_points)
    rho_vals = [rho(s, beta) for s in s_vals]

    # lognormal PDF
    logn_vals = lognormal_pdf(s_vals, mu=MU, sigma=SIGMA)

    plt.figure(figsize=(9, 5))
    plt.plot(s_vals, rho_vals, lw=2, label=f"SEF ρ(s, β={beta})")
    plt.plot(s_vals, logn_vals, lw=2, linestyle="--",
             label=f"Lognormal (μ={mu}, σ={sigma})")

    plt.xscale("log")
    plt.xlabel(r"$s = \tau^*/\tau$", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.title("SEF vs Lognormal Distribution", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Example plot ---
plot_rho_with_lognormal(beta=BETA, sigma=SIGMA)
