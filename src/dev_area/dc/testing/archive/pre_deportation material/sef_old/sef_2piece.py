import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.optimize import minimize

BETA = 0.466
TAUSTAR = 814.3

# ---------------------------------------------
# SEF distribution
# ---------------------------------------------
def rho_integrand(u, tau, beta, tau_star=TAUSTAR):
    s = tau_star/tau
    c = np.cos(np.pi * beta / 2)
    s_term = np.sin(np.pi * beta / 2)
    return np.exp(-u**beta * c) * np.cos(s*u - u**beta * s_term)

def rho(s, beta, u_max=60):
    val, _ = quad(rho_integrand, 0, u_max, args=(s, beta))
    return val / np.pi

def compute_mu_for_mode(s_mode_sef, sigma_left, sigma_right):
    if sigma_left < sigma_right:
        return np.log(s_mode_sef) + sigma_left**2
    elif sigma_right < sigma_left:
        return np.log(s_mode_sef) - sigma_right**2
    else:
        return np.log(s_mode_sef)

# ---------------------------------------------
# Two-piece normal in log-space
# ---------------------------------------------
def two_piece_normal_pdf_x(x, mu, sigma_left, sigma_right):
    A = np.sqrt(2/np.pi) / (sigma_left + sigma_right)
    if x < mu:
        return A * np.exp(-(x - mu)**2 / (2*sigma_left**2))
    else:
        return A * np.exp(-(x - mu)**2 / (2*sigma_right**2))

tpn_vec = np.vectorize(two_piece_normal_pdf_x)

def two_piece_lognormal_pdf(s, mu, sigma_left, sigma_right):
    x = np.log(s)
    return tpn_vec(x, mu, sigma_left, sigma_right) / s

# =====================================================
# CLOSED-FORM MEAN & VARIANCE for two-piece NORMAL (in log-space)
# =====================================================

def tpn_mean_x(mu, sigma_left, sigma_right):
    """
    Mean of the split normal distribution.
    Uses formula:
        E[X] = mu + sqrt(2/pi) * (sigma_right - sigma_left)
    """
    return mu + np.sqrt(2/np.pi) * (sigma_right - sigma_left)

def tpn_var_x(mu, sigma_left, sigma_right):
    """
    Variance of the split normal distribution.
    Uses formula:
        Var[X] = (1 - 2/pi)*(sigma_right - sigma_left)^2 + sigma_left * sigma_right
    """
    return (1 - 2/np.pi) * (sigma_right - sigma_left)**2 + sigma_left * sigma_right

# =====================================================
# NUMERICAL MEAN & VARIANCE of S (two-piece lognormal)
# =====================================================

def tpn_mean_s(mu, sigma_left, sigma_right):
    s_vals = np.logspace(-8, 4, 4000)
    pdf = two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right)
    return simpson(s_vals * pdf, s_vals)

def tpn_var_s(mu, sigma_left, sigma_right):
    s_vals = np.logspace(-8, 4, 4000)
    pdf = two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right)
    mean_s = simpson(s_vals * pdf, s_vals)
    mean_s2 = simpson((s_vals**2) * pdf, s_vals)
    return mean_s2 - mean_s**2


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
def fit_two_piece_to_sef(beta):
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

# ---------------------------------------------
# Calculate tau mean
# ---------------------------------------------

def sef_mean_tau(beta, tau_star):
    s_vals = np.logspace(-8, 4, 4000)
    pdf_vals = np.array([rho(s, beta) for s in s_vals])
    mean_inv_s = simpson((1/s_vals) * pdf_vals, s_vals)
    return tau_star * mean_inv_s


# ---------------------------------------------
# Plot SEF + two-piece lognormal (mode-aligned numerically)
# ---------------------------------------------
def plot_sef_and_two_piece(beta, sigma_left=0.4, sigma_right=0.8,
                           s_min=1e-3, s_max=5, n_points=800):

    # s grid
    s_vals = np.logspace(np.log10(s_min), np.log10(s_max), n_points)

    # compute SEF PDF
    rho_vals = np.array([rho(s, beta) for s in s_vals])

    # -----------------------------
    # NUMERIC MODE MATCHING
    # -----------------------------
    sigma_left, sigma_right, mu = fit_two_piece_to_sef(beta)

    # compute two-piece PDF
    tpn_vals = two_piece_lognormal_pdf(s_vals, mu, sigma_left, sigma_right)

    # print( sef_mean_tau(beta=0.466, tau_star=814))

    # plot
    plt.figure(figsize=(10,5))
    plt.plot(s_vals, rho_vals, lw=2, label=f"SEF ρ(s), β={beta}")
    plt.plot(s_vals, tpn_vals, "--", lw=2,
             label=f"Two-piece lognormal (σL={sigma_left}, σR={sigma_right})")

    plt.xscale("log")
    plt.xlabel(r"$s = \tau^*/\tau$")
    plt.ylabel("Probability Density")
    plt.title("SEF vs Two-piece Lognormal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example run

plot_sef_and_two_piece(beta=BETA, sigma_left=0, sigma_right=0)



