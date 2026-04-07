import numpy as np
from scipy.stats import levy_stable

# ------------------------------------------------------------
# 1. Function: intrinsic q-distribution from (tau*, beta)
# ------------------------------------------------------------

def q_distribution_from_experiment(tau_star, beta, nsamples=50000):
    """
    Given experimental τ* and β from DC decay,
    return samples from the intrinsic distribution of q = log10(tau),
    plus analytical mean and sigma of q.
    """

    gamma = 0.5772156649015328606  # Euler–Mascheroni constant

    # Intrinsic distribution parameters for ln(tau)
    mu_ln = np.log(tau_star) + (1 - 1/beta) * gamma
    sigma_ln = np.sqrt((1/beta**2 - 1) * np.pi**2 / 6)

    # Convert to q = log10(tau)
    mu_q = mu_ln / np.log(10)
    sigma_q = sigma_ln / np.log(10)

    # One-sided Lévy stable distribution sampling
    ln_tau_samples = levy_stable.rvs(
        alpha=beta, beta=1,
        loc=mu_ln, scale=sigma_ln, size=nsamples
    )

    # Convert to q samples
    q_samples = ln_tau_samples / np.log(10)

    return q_samples, mu_q, sigma_q


# ------------------------------------------------------------
# 2. Multi-temperature loop
# ------------------------------------------------------------

temperatures = np.array([2, 4, 6, 9, 13, 16, 20, 23])
betas        = np.array([0.466, 0.574, 0.627, 0.675, 0.737, 0.778, 0.827, 0.857])
tau_stars    = np.array([814.3, 534.2, 410.9, 302.8, 222.1, 177.3, 132.1, 106.4])

# Temperature selection
T_min = 2
T_max = 13
idx = np.where((temperatures >= T_min) & (temperatures <= T_max))[0]

temps_sel = temperatures[idx]
betas_sel = betas[idx]
taus_sel  = tau_stars[idx]

print(f"\n=== Selected temperatures: {temps_sel} K ===\n")

# Storage arrays
Q_means = []
Q_sigmas = []
Q_16 = []
Q_50 = []
Q_84 = []
ALL_Q_SAMPLES = []     # <-- used for final averaging

# ------------------------------------------------------------
# 3. Loop over selected temperatures and compute q distributions
# ------------------------------------------------------------

for T, beta, tau_star in zip(temps_sel, betas_sel, taus_sel):

    q_samples, q_mean, q_sigma = q_distribution_from_experiment(
        tau_star, beta, nsamples=50000
    )

    Q_means.append(q_mean)
    Q_sigmas.append(q_sigma)

    print(f"T = {T:2d} K | β = {beta:.3f} | τ* = {tau_star:.1f} s")
    print(f"  q_mean (analytic):  {q_mean:.4f}")
    print(f"  q_sigma (analytic): {q_sigma:.4f}")

# ------------------------------------------------------------
# 4. Final averaged q across all temperatures
# ------------------------------------------------------------

Q_final_mean = sum(Q_means)/len(Q_means)
Q_final_sds = sum(Q_sigmas)/len(Q_sigmas)

print("\n=== FINAL AVERAGED q (across selected temperatures) ===")
print(f"Mean q:                 {Q_final_mean:.4f}")
print(f"SD q:                   {Q_final_sds:.4f}")
print("========================================================\n")
