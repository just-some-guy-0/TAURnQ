import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma, gammaln

# ==========================================================
# 1. SEF distribution ρ(s, β)
# ==========================================================

import numpy as np
from scipy.integrate import quad

def sef_rho_stable(s, beta, t_max=20):
    """
    Stable evaluation of the SEF distribution for arbitrary β using the
    substitution u = exp(t). Works extremely well for β < 0.4.

    ρ(s,β) = 1/pi ∫ exp(t) * exp(-exp(beta*t)*cos(piβ/2)) *
                     cos(s*exp(t) - exp(beta*t)*sin(piβ/2)) dt
    where t ∈ (-∞, +∞)

    We truncate t to [-t_max, t_max] where t_max ~ 18–22 is enough.
    """

    c = np.cos(np.pi * beta / 2)
    d = np.sin(np.pi * beta / 2)

    def integrand(t):
        u = np.exp(t)
        u_beta = np.exp(beta * t)

        return (
            np.exp(t)
            * np.exp(-u_beta * c)
            * np.cos(s * u - u_beta * d)
        )

    # Integrate symmetric range [-t_max, t_max]
    val, _ = quad(integrand, -t_max, t_max, limit=500)
    return val / np.pi



# # ==========================================================
# # 2. Mode alignment for split lognormal
# # ==========================================================
# def compute_mu_for_mode(s_mode, sigma_left, sigma_right):
#     if sigma_left < sigma_right:
#         return np.log(s_mode) + sigma_left**2
#     elif sigma_right < sigma_left:
#         return np.log(s_mode) - sigma_right**2
#     else:
#         return np.log(s_mode)


# # ==========================================================
# # 3. Two-piece normal PDF in ln(s)
# # ==========================================================
# def two_piece_normal_pdf_x(x, mu, sigma_left, sigma_right):
#     A = np.sqrt(2/np.pi) / (sigma_left + sigma_right)
#     if x < mu:
#         return A * np.exp(-(x - mu)**2 / (2*sigma_left**2))
#     else:
#         return A * np.exp(-(x - mu)**2 / (2*sigma_right**2))

# tpn_vec = np.vectorize(two_piece_normal_pdf_x, otypes=[float])


# # ==========================================================
# # 4. Two-piece lognormal PDF
# # ==========================================================
# def two_piece_lognormal_pdf(s, mu, sigma_left, sigma_right):
#     x = np.log(s)
#     return tpn_vec(x, mu, sigma_left, sigma_right) / s


# # ==========================================================
# # 5. Moments of split-lognormal (needed for τ statistics)
# # ==========================================================
# def split_lognormal_moments(mu, sigma_left, sigma_right):
#     mu_X = mu + np.sqrt(2/np.pi) * (sigma_right - sigma_left)
#     v_X  = (1 - 2/np.pi)*(sigma_right - sigma_left)**2 + sigma_left*sigma_right

#     mean_S      = np.exp(mu_X + 0.5 * v_X)
#     mean_inv_S  = np.exp(-mu_X + 0.5 * v_X)

#     return mean_S, mean_inv_S, mu_X, v_X


# ==========================================================
# 6. Input SEF parameters
# ==========================================================
beta     = 0.3
tau_star = 1

s_vals  = np.logspace(-6, 3, 500)
rho_vals = np.array([sef_rho_stable(s, beta) for s in s_vals])


s_mode_sef = s_vals[np.argmax(rho_vals)]


# # ==========================================================
# # 7. Fit split-lognormal to SEF
# # ==========================================================
# def objective(params):
#     sigma_left, sigma_right = params
#     if sigma_left <= 0 or sigma_right <= 0:
#         return 1e9

#     mu = compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right)
#     model = two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right)

#     return np.sum((rho_vals - model)**2)


# res = minimize(objective, x0=[0.3, 0.7],
#                bounds=[(1e-3, 5), (1e-3, 5)],
#                method="L-BFGS-B")

# sigma_left, sigma_right = res.x
# mu = compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right)

# print("=== FIT RESULTS ===")
# print(f"sigma_left  = {sigma_left:.4f}")
# print(f"sigma_right = {sigma_right:.4f}")
# print(f"mu (aligned) = {mu:.4f}")


# # ==========================================================
# # 8. Compute τ statistics
# # ==========================================================
# mean_S, mean_inv_S, mu_X, v_X = split_lognormal_moments(mu, sigma_left, sigma_right)

# # geometric mean τ = τ* exp(-mu_X)
# tau_geom = tau_star * np.exp(-mu_X)

# # Q mean in log10 space
# Q_mean = np.log10(tau_geom)

# # Q standard deviation in log10 space
# Q_std = np.sqrt(v_X) / np.log(10)

# print("\n=== τ STATISTICS ===")
# print(f"exp(<ln τ>) (geometric) = {tau_geom:.6f} s")
# print(f"Q_mean                  = {Q_mean:.6f}")
# print(f"Q_std                   = {Q_std:.6f}")

# ==========================================================
# 9. Plot results
# ==========================================================
plt.figure(figsize=(7,4))
plt.plot(s_vals, rho_vals, lw=2, label="Exact SEF")
# plt.plot(s_vals,
#          two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right),
#          lw=2, label="Two-piece lognormal fit")

plt.xscale("log")
plt.xlabel("s = τ*/τ")
plt.ylabel("PDF")
plt.title("SEF vs Two-piece Lognormal Fit")
plt.grid(True)
plt.legend()
plt.show()
