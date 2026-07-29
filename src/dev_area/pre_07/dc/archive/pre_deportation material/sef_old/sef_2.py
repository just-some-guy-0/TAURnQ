import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ============================================================
# 1. SEF distribution ρ(s, β) from eqn (3)
# ============================================================
def rho_s_beta(s, beta, u_max=50):
    c = np.cos(np.pi * beta / 2)
    s_term = np.sin(np.pi * beta / 2)

    def integrand(u):
        return np.exp(-u**beta * c) * np.cos(s*u - u**beta * s_term)

    val, _ = quad(integrand, 0, u_max, limit=500)
    return val / np.pi


# ============================================================
# 2. Mode-matching function for two-piece lognormal
# ============================================================
def compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right):
    # ensures the lognormal's mode = s_mode_sef
    if sigma_left < sigma_right:
        return np.log(s_mode_sef) + sigma_left**2
    elif sigma_right < sigma_left:
        return np.log(s_mode_sef) - sigma_right**2
    else:
        return np.log(s_mode_sef)


# ============================================================
# 3. Two-piece normal in log-space
# ============================================================
def two_piece_normal_pdf_x(x, mu, sigma_left, sigma_right):
    A = np.sqrt(2 / np.pi) / (sigma_left + sigma_right)
    if x < mu:
        return A * np.exp(-(x - mu)**2 / (2 * sigma_left**2))
    else:
        return A * np.exp(-(x - mu)**2 / (2 * sigma_right**2))


# vectorized version
tpn_vec = np.vectorize(two_piece_normal_pdf_x, otypes=[float])


# ============================================================
# 4. Two-piece lognormal PDF
# ============================================================
def two_piece_lognormal_pdf(s, mu, sigma_left, sigma_right):
    s = np.asarray(s)
    x = np.log(s)
    return tpn_vec(x, mu, sigma_left, sigma_right) / s


# ============================================================
# 5. Evaluate SEF distribution
# ============================================================
beta = 0.466
s_vals = np.logspace(-1.5, 0.5, 400)   # s = τ*/τ
rho_vals = np.array([rho_s_beta(s, beta) for s in s_vals])

# find the SEF mode numerically
idx_mode = np.argmax(rho_vals)
s_mode_sef = s_vals[idx_mode]


# ============================================================
# 6. Construct split lognormal with same mode as SEF
# ============================================================
from scipy.optimize import minimize

# -------------------------------------------------------------
# Objective function: fit sigma_left and sigma_right
# -------------------------------------------------------------
def objective(params):
    sigma_left, sigma_right = params

    # Keep them positive
    if sigma_left <= 0 or sigma_right <= 0:
        return 1e9

    # compute mu to match the mode
    mu = compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right)

    # compute model
    model = two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right)

    # least squares mismatch (avoid log scale bias)
    return np.sum((rho_vals - model)**2)


# Initial guesses
x0 = np.array([0.3, 0.7])

# Bounds to keep the solution physical
bounds = [(1e-3, 5.0), (1e-3, 5.0)]

res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

sigma_left_fit, sigma_right_fit = res.x
mu_fit = compute_mu_for_mode(s_mode_sef, sigma_left_fit, sigma_right_fit)

print("=== Fit results (two-piece lognormal) ===")
print(f" sigma_left  = {sigma_left_fit:.5f}")
print(f" sigma_right = {sigma_right_fit:.5f}")
print(f" mu (mode-aligned) = {mu_fit:.5f}")
print(f" Objective function = {res.fun:.6e}")

split_vals = two_piece_lognormal_pdf(s_vals, mu_fit, sigma_left_fit, sigma_right_fit)


# ============================================================
# 7. Plot both distributions
# ============================================================
plt.figure(figsize=(7,4))
plt.plot(s_vals, rho_vals, lw=2, label=f"SEF β={beta}")
plt.plot(s_vals, split_vals, lw=2, label="Two-piece lognormal (mode matched)")

plt.xscale("log")
plt.xlabel("s = τ*/τ")
plt.ylabel("PDF")
plt.title("SEF distribution vs. Two-piece lognormal")
plt.grid(True)
plt.legend()
plt.show()


# ============================================================
# 8. Verify SEF normalisation
# ============================================================

# Smax = 20
# def integrand_s(s):
#     return rho_s_beta(s, beta)

# norm_val, err = quad(integrand_s, -Smax, Smax, limit=300)
# print(f"β = {beta}")
# print(f"Integral of ρ(s,β) over s ∈ [{-Smax}, {Smax}] = {norm_val:.6f}")
# print(f"Estimated integration error = {err:.2e}")
# print(f"SEF mode located at s = {s_mode_sef:.5f}")
# print(f"lognormal μ used = {mu:.5f}")
