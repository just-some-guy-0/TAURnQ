import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

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

EULER_GAMMA = 0.5772156649015328606
ZETA3 = 1.2020569031595942854  # Apery’s constant ζ(3)

def sef_moments_q(tau_star: float, beta: float):
    """
    For KWW/SEF intrinsic distribution of relaxation times, with q = log10(tau):

    Zorn (JCP 2002) gives for Kohlrausch/KWW:
      E[ln tau]      = ln(tau*) + (1/beta - 1) * gamma_E
      Var(ln tau)    = (pi^2/6) * (1/beta^2 - 1)
      mu3_central(ln tau) = 2*zeta(3) * (1 - 1/beta^3)

    Skewness of ln(tau):
      g1_ln = mu3 / Var^(3/2)

    Since q = ln(tau)/ln(10) is a linear rescaling, skewness is invariant:
      g1_q = g1_ln
    """
    if not (0 < beta <= 1):
        raise ValueError("beta must be in (0, 1].")
    if tau_star <= 0:
        raise ValueError("tau_star must be positive.")

    # Mean/variance in ln-space
    mu_ln  = np.log(tau_star) + (1.0 / beta - 1.0) * EULER_GAMMA
    var_ln = (np.pi**2 / 6.0) * (1.0 / beta**2 - 1.0)

    # Third central moment in ln-space (Zorn, Table I, Kohlrausch row)
    mu3_ln = 2.0 * ZETA3 * (1.0 - 1.0 / beta**3)

    # Skewness in ln-space, and hence in q-space
    if var_ln <= 0:
        # beta=1 gives var=0; distribution collapses to a delta, skewness undefined.
        skew_q = np.nan
    else:
        skew_q = mu3_ln / (var_ln ** 1.5)

    # Convert mean/std to q=log10(tau)
    ln10 = np.log(10.0)
    mu_q = mu_ln / ln10
    sigma_q = np.sqrt(var_ln) / ln10

    return mu_q, sigma_q, skew_q


def plot_q_pdf(rho_log_tau_func, tau_star, beta,
               q_min=None, q_max=None, nq=5000):
    # Use your moment-based bounds (optional but convenient)
    mu_q, sigma_q, _ = sef_moments_q(tau_star, beta)
    if q_min is None:
        q_min = mu_q - 8*sigma_q
    if q_max is None:
        q_max = mu_q + 12*sigma_q

    q = np.linspace(q_min, q_max, nq)
    tau = 10.0**q

    # This is the pdf in q-space
    f_q = rho_log_tau_func(tau, beta, tau_star=tau_star)

    # Normalize numerically over q (good sanity check)
    Z = np.trapz(f_q, q)
    f_q = f_q / Z

    plt.figure()
    plt.plot(q, f_q)
    plt.xlabel(r"$q=\log_{10}(\tau/\mathrm{s})$")
    plt.ylabel(r"$f_q(q)$")
    plt.title(f"KWW/SEF intrinsic distribution in q (beta={beta:g})")
    plt.tight_layout()
    plt.show()

    return q, f_q

# Example call:
q, f_q = plot_q_pdf(rho_log_tau, tau_star=106.4, beta=0.457)
