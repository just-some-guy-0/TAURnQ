import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# ----- Target (Reta–Chilton) τ distribution -----
def g_from_alpha(alpha):
    return 1.82*np.sqrt(alpha)/(1 - alpha)  # natural-log std

def rho_tau_target(tau, tau_mu, alpha):
    g = g_from_alpha(alpha)
    return np.exp(-0.5*((np.log(tau) - np.log(tau_mu))/g)**2) / (tau * g * np.sqrt(2*np.pi))

# ----- τ distribution implied by Raman with R,N ~ Normal(base-10) -----
def tau_pdf_from_RN(tau, mu_R, sd_R, mu_N, sd_N, rho, T):
    t10 = np.log10(T)
    mu_L = mu_R + t10*mu_N
    var_L = sd_R**2 + (t10**2)*(sd_N**2) + 2*t10*rho*sd_R*sd_N
    # Natural-log parameters of τ:
    mu_ln_tau = - np.log(10) * mu_L
    sd_ln_tau = np.sqrt((np.log(10)**2) * var_L)
    return lognorm.pdf(tau, s=sd_ln_tau, scale=np.exp(mu_ln_tau))

# ----- Solve R params to exactly match the target τ distribution -----
def solve_R_params(tau_mu, alpha, T, mu_N, sd_N, rho=0.0):
    g = g_from_alpha(alpha)
    t10 = np.log10(T)
    mu_L_target = -np.log10(tau_mu)
    sd_L_target = g / np.log(10)

    mu_R = mu_L_target - t10*mu_N

    # Solve σ_R from: σ_R^2 + 2 t ρ σ_R σ_N + t^2 σ_N^2 = σ_L^2  (take non-negative root)
    A = 1.0
    B = 2*t10*rho*sd_N
    C = (t10**2)*(sd_N**2) - sd_L_target**2
    disc = B**2 - 4*A*C
    if disc < 0:
        raise ValueError("No real σ_R satisfies the variance constraint; reduce sd_N or |ρ|.")
    sd_R = max(0.0, (-B + np.sqrt(disc)) / (2*A))
    return mu_R, sd_R

# ---------------- Demo ----------------
T = 26.0
tau_mu  = 6.6337e-01     # from your example
alpha   = 2.7243e-01

# Choose what you believe about N (mean exponent, its spread, and correlation with R)
mu_N = 4.32               # e.g., typical Raman exponent guess
sd_N = 0.30              # choose a spread
rho  = 0.0               # assume independence first

# Solve R to match the target τ distribution exactly
mu_R, sd_R = solve_R_params(tau_mu, alpha, T, mu_N, sd_N, rho)

# Prepare grid and PDFs
tau = np.logspace(-4, 4, 2000)
pdf_target = rho_tau_target(tau, tau_mu, alpha)
pdf_RN     = tau_pdf_from_RN(tau, mu_R, sd_R, mu_N, sd_N, rho, T)

# Plot
plt.figure(figsize=(8,5))
plt.plot(tau, pdf_target, label="target ρ(τ) (τμ, α)", lw=2)
plt.plot(tau, pdf_RN,    label="ρ(τ) from Raman (R,N)", ls="--")
plt.xscale("log")
plt.xlabel("τ")
plt.ylabel("ρ(τ)")
plt.legend()
plt.tight_layout()
plt.show()

# Print the matched (R,N) parameters (base-10)
print(f"T = {T} K, t10 = {np.log10(T):.6f}")
print(f"Matched parameters (base-10):")
print(f"  mu_R = {mu_R:.4f},  sd_R = {sd_R:.4f}")
print(f"  mu_N = {mu_N:.4f},  sd_N = {sd_N:.4f},  rho = {rho:.2f}")
