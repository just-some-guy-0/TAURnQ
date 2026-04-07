import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# ---------------------------------------------
# SEF distribution
# ---------------------------------------------
def rho_integrand(u, s, beta):
    c = np.cos(np.pi * beta / 2)
    s_term = np.sin(np.pi * beta / 2)
    return np.exp(-u**beta * c) * np.cos(s*u - u**beta * s_term)

def rho(s, beta, u_max=60):
    val, _ = quad(rho_integrand, 0, u_max, args=(s, beta))
    return val / np.pi

# ---------------------------------------------
# Two-piece lognormal PDF
# ---------------------------------------------
def two_piece_normal_pdf_x(x, mu, sigma_left, sigma_right):
    A = np.sqrt(2/np.pi) / (sigma_left + sigma_right)
    if x < mu:
        return A * np.exp(-(x - mu)**2 / (2 * sigma_left**2))
    else:
        return A * np.exp(-(x - mu)**2 / (2 * sigma_right**2))

tpn_vec = np.vectorize(two_piece_normal_pdf_x)

def two_piece_lognormal_pdf(s, mu, sigma_left, sigma_right):
    x = np.log(s)
    return tpn_vec(x, mu, sigma_left, sigma_right) / s

# ---------------------------------------------
# SEF mode (numerical)
# ---------------------------------------------
def sef_mode(beta, s_min=1e-4, s_max=10, n_points=800):
    s_vals = np.logspace(np.log10(s_min), np.log10(s_max), n_points)
    rho_vals = np.array([rho(s, beta) for s in s_vals])
    return s_vals[np.argmax(rho_vals)]

# ---------------------------------------------
# μ calculation: match lognormal mode to SEF mode
# (correct σ² formula)
# ---------------------------------------------
def compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right):
    if sigma_left < sigma_right:
        return np.log(s_mode_sef) + sigma_left**2
    elif sigma_right < sigma_left:
        return np.log(s_mode_sef) - sigma_right**2
    else:
        return np.log(s_mode_sef)

# ---------------------------------------------
# LOSS FUNCTION FOR REGRESSION
# ---------------------------------------------
def loss_sigma(params, s_vals, sef_vals, s_mode_sef):
    sigma_left, sigma_right = params

    # keep sigmas positive
    if sigma_left <= 0 or sigma_right <= 0:
        return 1e9

    # compute μ that aligns mode properly
    mu = compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right)

    # compute two-piece lognormal PDF
    tpn_vals = two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right)

    # least-squares error (linear PDF space)
    return np.sum((tpn_vals - sef_vals)**2)

# ---------------------------------------------
# FITTING FUNCTION
# ---------------------------------------------
def fit_two_piece_to_sef(beta=0.6):
    # grid for comparison
    s_vals = np.logspace(-4, 2, 500)
    sef_vals = np.array([rho(s, beta) for s in s_vals])
    s_mode_sef = s_vals[np.argmax(sef_vals)]

    # initial guess for sigmas
    x0 = np.array([0.5, 1.0])

    # minimize loss
    res = minimize(
        loss_sigma,
        x0,
        args=(s_vals, sef_vals, s_mode_sef),
        method='Nelder-Mead',
        options={'maxiter': 200, 'disp': True}
    )

    sigma_left_opt, sigma_right_opt = res.x

    # compute optimal μ
    mu_opt = compute_mu_for_mode(s_mode_sef, sigma_left_opt, sigma_right_opt)

    return sigma_left_opt, sigma_right_opt, mu_opt

sigma_left_opt, sigma_right_opt, mu_opt = fit_two_piece_to_sef(beta=0.6)

print("Optimal σ_left:", sigma_left_opt)
print("Optimal σ_right:", sigma_right_opt)
print("Optimal μ:", mu_opt)