import numpy as np

EULER_GAMMA = 0.5772156649015328606

def sef_moments_q(tau_star: float, beta: float):
    """
    For SEF/KWW intrinsic distribution of relaxation times, with q = log10(tau):

    E[ln tau] = ln(tau*) + (1/beta - 1) * gamma_E
    Var(ln tau) = (pi^2/6) * (1/beta^2 - 1)

    Then:
    E[q] = E[ln tau] / ln(10)
    Var(q) = Var(ln tau) / (ln(10))^2
    """
    if not (0 < beta <= 1):
        raise ValueError("beta must be in (0, 1].")

    mu_ln = np.log(tau_star) + (1.0 / beta - 1.0) * EULER_GAMMA
    var_ln = (np.pi**2 / 6.0) * (1.0 / beta**2 - 1.0)

    mu_q = mu_ln / np.log(10.0)
    sigma_q = np.sqrt(var_ln) / np.log(10.0)

    tau_esd = np.exp(mu_ln + np.sqrt(var_ln))
    # print(tau_esd)

    return mu_q, sigma_q


# ---- sampling q from your SEF pdf over x = log10(tau) ----
def sample_q_from_sef_pdf(rho_log_tau_func, tau_star: float, beta: float,
                          nsamples: int = 5000,
                          x_min: float = None, x_max: float = None,
                          nx: int = 200000,
                          seed: int = 0):
    """
    Samples q = log10(tau) from the SEF intrinsic distribution using the pdf in x-space:
      f(q) = rho_log_tau(tau=10^q, beta, tau_star)   which integrates to 1 over q.

    rho_log_tau_func: function(tau_array, beta, tau_star) -> pdf over q
                      (this is your rho_log_tau)
    """
    rng = np.random.default_rng(seed)

    # Choose a sensible x-range based on mean±K sigma in q-space
    mu_q, sigma_q = sef_moments_q(tau_star, beta)
    if x_min is None:
        x_min = mu_q - 8.0 * sigma_q
    if x_max is None:
        x_max = mu_q + 12.0 * sigma_q  # allow more right tail

    x = np.linspace(x_min, x_max, nx)
    tau = 10.0**x

    f = rho_log_tau_func(tau, beta, tau_star=tau_star)
    # normalize on this grid (protect against truncation)
    f = np.clip(f, 0.0, np.inf)
    Z = np.trapz(f, x)
    if Z <= 0:
        raise RuntimeError("PDF normalization failed; adjust x_min/x_max.")
    f = f / Z

    cdf = np.cumsum(f) * (x[1] - x[0])
    cdf[-1] = 1.0  # guard numerical drift

    u = rng.random(nsamples)
    q_samples = np.interp(u, cdf, x)
    return q_samples, mu_q, sigma_q


# -----------------------------
# Testing
# -----------------------------

tau_star = 106.4
beta = 0.857

mu_q, sigma_q = sef_moments_q(tau_star, beta)
print("Intrinsic q mean (E[q])  :", mu_q)
print("Intrinsic q sigma (Std)  :", sigma_q)
