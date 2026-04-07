import numpy as np
from scipy.stats import levy_stable

# ------------------------------------------------------------
# Your experimental lists (replace with your actual arrays)
# tau_star_exp = np.array([...])   # seconds
# beta_exp     = np.array([...])
# ------------------------------------------------------------

def q_distribution_from_experiment(tau_star, beta, nsamples=5000):
    """
    Given experimental τ* and β from DC decay,
    return samples from the intrinsic distribution of q = log10(tau).
    """

    # Euler–Mascheroni constant
    gamma = 0.5772156649015328606

    # 1. Compute μ and σ for ln(tau)
    mu_ln = np.log(tau_star) + (1 - 1/beta) * gamma
    sigma_ln = np.sqrt((1/beta**2 - 1) * (np.pi**2) / 6)

    # 2. Convert to scale/location for q = log10(tau)
    mu_q = mu_ln / np.log(10)
    sigma_q = sigma_ln / np.log(10)

    # 3. Sample from the one-sided Lévy stable law
    # alpha = beta,   beta(stability skewness parameter) = 1 (fully skewed)
    ln_tau_samples = levy_stable.rvs(alpha=beta, beta=1, loc=mu_ln, scale=sigma_ln, size=nsamples)

    # convert to q samples
    q_samples = ln_tau_samples / np.log(10)

    return q_samples, mu_q, sigma_q


# ------------------------------------------------------------
# Example of applying to a single measurement
# ------------------------------------------------------------

# Replace these with your actual experimental values:
tau_star = 814.3
beta = 0.466

q_samples, q_mean, q_sigma = q_distribution_from_experiment(tau_star, beta)

print("Intrinsic q mean:", q_mean)
print("Intrinsic q sigma:", q_sigma)
print("q 16th–50th–84th percentiles:", 
      np.percentile(q_samples, [16, 50, 84]))
