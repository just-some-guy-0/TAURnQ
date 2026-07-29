import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize


# ======================================================================
# 1.  SEF distribution ρ(s, β)   (Eqn 3 in the paper)
# ======================================================================
def rho_s_beta(s, beta, u_max=80):
    """Compute ρ(s, β) using the integral definition."""
    c = np.cos(np.pi * beta / 2)
    s_term = np.sin(np.pi * beta / 2)

    def integrand(u):
        return np.exp(-u**beta * c) * np.cos(s*u - u**beta * s_term)

    val, _ = quad(integrand, 0, u_max, limit=400)
    return val / np.pi


# ======================================================================
# 2.  Convert SEF to log-space PDF ρ_log(x)   (Eqn 5)
#     x = log10(τ),  τ = 10^x
# ======================================================================
def rho_log_tau(x, tau_star, beta):
    tau = 10**x
    s = tau_star / tau
    
    rho_s = rho_s_beta(s, beta)

    # Correct Jacobian from Blackmore et al Eqn (5)
    return rho_s * tau_star * np.log(10) * 10**(-x)



# ======================================================================
# 3.  Two-piece normal PDF in ln-space
# ======================================================================
def two_piece_normal_pdf_x(x, mu, sigma_left, sigma_right):
    A = np.sqrt(2/np.pi) / (sigma_left + sigma_right)
    if x < mu:
        return A * np.exp(-(x - mu)**2 / (2*sigma_left**2))
    else:
        return A * np.exp(-(x - mu)**2 / (2*sigma_right**2))

tpn_vec = np.vectorize(two_piece_normal_pdf_x, otypes=[float])


# ======================================================================
# 4.  Fit μ so that the mode aligns correctly
# ======================================================================
def compute_mu_for_mode(x_mode, sigma_left, sigma_right):
    if sigma_left < sigma_right:
        return x_mode + sigma_left**2
    elif sigma_right < sigma_left:
        return x_mode - sigma_right**2
    else:
        return x_mode


# ======================================================================
# 5.  Two-piece lognormal PDF in x = log(τ)
# ======================================================================
def tpn_pdf_x(x, mu, sigma_left, sigma_right):
    return tpn_vec(x, mu, sigma_left, sigma_right)


# ======================================================================
# 6.  Split-lognormal moments (for Q, τgeom)
# ======================================================================
def split_lognormal_moments(mu, sigma_left, sigma_right):
    mu_X = mu + np.sqrt(2/np.pi) * (sigma_right - sigma_left)
    v_X  = (1 - 2/np.pi)*(sigma_right - sigma_left)**2 + sigma_left*sigma_right

    tau_geom_factor = np.exp(-mu_X)  # multiplying by τ* later
    return tau_geom_factor, mu_X, v_X


# ======================================================================
# INPUT PARAMETERS
# ======================================================================
beta     = 0.466       # example
tau_star = 814.3       # example
NPTS     = 400

x_vals = np.linspace(-6, 6, NPTS)
rho_log_vals = np.array([rho_log_tau(x, tau_star, beta) for x in x_vals])

# find mode location
x_mode_sef = x_vals[np.argmax(rho_log_vals)]


# ======================================================================
# 7.  Fit two-piece lognormal in log-space
# ======================================================================
def objective(params):
    sigma_left, sigma_right = params
    if sigma_left <= 0 or sigma_right <= 0:
        return 1e12

    mu = compute_mu_for_mode(x_mode_sef, sigma_left, sigma_right)
    model = tpn_pdf_x(x_vals, mu, sigma_left, sigma_right)

    # matching shapes only; PDFs are not normalized the same
    return np.sum((rho_log_vals - model)**2)


res = minimize(
    objective,
    x0=[0.3, 0.7],
    bounds=[(1e-4, 5), (1e-4, 5)],
    method="L-BFGS-B"
)

sigma_left, sigma_right = res.x
mu = compute_mu_for_mode(x_mode_sef, sigma_left, sigma_right)


# ======================================================================
# 8.  Compute τ statistics (Q parameter)
# ======================================================================
tau_geom_factor, mu_X, var_X = split_lognormal_moments(mu, sigma_left, sigma_right)

tau_geom = tau_star * tau_geom_factor
Q_value = np.log10(tau_geom)

print("\n=== FIT RESULTS ===")
print(f"σ_left  = {sigma_left:.4f}")
print(f"σ_right = {sigma_right:.4f}")
print(f"μ (adjusted) = {mu:.4f}")

print("\n=== τ STATISTICS ===")
print(f"exp(<ln τ>) = τ_geom = {tau_geom:.4f} s")
print(f"Q parameter = {Q_value:.4f}")


# ======================================================================
# 9.  Plot comparison
# ======================================================================
plt.figure(figsize=(8,5))

plt.plot(x_vals, rho_log_vals, lw=2, label="SEF log-PDF (exact)")
plt.plot(x_vals, tpn_pdf_x(x_vals, mu, sigma_left, sigma_right),
         lw=2, label="Two-piece lognormal fit")

plt.xlabel("x = log10(τ)")
plt.ylabel("PDF (log-domain)")
plt.title("SEF vs Two-piece Lognormal (log τ domain)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
