import numpy as np
import matplotlib.pyplot as plt

# --- your model ---
def distribution_from_params(T, A, Ueff, R, n, Q):
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1e-300)
    term1 = 10.0**(-A) * np.exp(-Ueff / np.maximum(T_safe, 1e-12))
    term2 = 10.0**(R)  * (T_safe**n)
    term3 = 10.0**(Q)
    return term1 + term2 + term3

# --- Monte-Carlo predictive wrapper ---
def predictive_curve_MC(
    T_grid,
    mu,                      # shape (5,)
    sd=None,                 # shape (5,), ignored if Sigma provided
    Sigma=None,              # shape (5,5) full covariance (optional)
    N=20000,                 # total parameter samples (will round to even if antithetic)
    antithetic=True,         # variance reduction
    seed=123
):
    rng = np.random.default_rng(seed)

    # Parameter names for reference: [A, Ueff, R, n, Q]
    mu = np.asarray(mu, dtype=float).reshape(5,)

    if Sigma is None:
        assert sd is not None, "Provide either sd (diag) or Sigma (full covariance)."
        sd = np.asarray(sd, dtype=float).reshape(5,)
        Sigma = np.diag(sd**2)

    # Cholesky for correlated normals
    L = np.linalg.cholesky(Sigma)

    # Antithetic pairing
    if antithetic:
        if N % 2: N += 1
        half = N // 2
        Z = rng.standard_normal((5, half))
        Z = np.concatenate([Z, -Z], axis=1)
    else:
        Z = rng.standard_normal((5, N))

    # Sample parameters: theta = mu + L @ Z
    Theta = (mu[:, None] + L @ Z).T  # shape (N,5)
    A, Ueff, R, n, Q = [Theta[:, i] for i in range(5)]

    # If you need constraints (e.g., n>0 or Ueff>0), enforce here:
    # Ueff = np.clip(Ueff, 0.0, None)
    # n    = np.clip(n, 0.0, None)

    T_grid = np.asarray(T_grid, dtype=float)
    N_T = T_grid.size

    # Evaluate model for all samples; vectorize over T
    # Result: values shape (N, N_T)
    vals = np.empty((Theta.shape[0], N_T), dtype=float)
    for j, T in enumerate(T_grid):
        vals[:, j] = distribution_from_params(T, A, Ueff, R, n, Q)

    # Predictive summary: MC expectation and quantiles at each T
    mean = vals.mean(axis=0)
    q16, q50, q84 = np.percentile(vals, [16, 50, 84], axis=0)   # ~68% interval
    q025, q975     = np.percentile(vals, [2.5, 97.5], axis=0)   # ~95% interval

    summary = {
        "T": T_grid,
        "mean": mean,
        "median": q50,
        "q16": q16,
        "q84": q84,
        "q025": q025,
        "q975": q975,
    }
    return summary

# --------- EXAMPLE USAGE (fill in your numbers) ----------
# Temperature grid to plot over (change as needed)
T = np.linspace(2.0, 300.0, 400)

# Your parameter beliefs (example placeholders):
mu = np.array([
    10.6,     # A (log10 scale in your model)
    850.0,    # Ueff (Kelvin-ish if your units are K)
    -5.3,     # R  (log10 coefficient for Raman-like term)
    4.0,      # n  (power for Raman-like term)
    -0.35,    # Q  (log10 coefficient for QTM-like term)
])

# Either independent SDs...
sd = np.array([0.2, 50.0, 0.3, 0.4, 0.2])

# ...or a full covariance (example: small correlations)
# corr = np.array([
#     [ 1.00,  0.10, -0.20,  0.05,  0.00],
#     [ 0.10,  1.00, -0.15,  0.10,  0.00],
#     [-0.20, -0.15,  1.00,  0.30,  0.05],
#     [ 0.05,  0.10,  0.30,  1.00,  0.00],
#     [ 0.00,  0.00,  0.05,  0.00,  1.00],
# ])
# Sigma = np.outer(sd, sd) * corr

Sigma = None  # set to the covariance matrix if using one

summary = predictive_curve_MC(T, mu, sd=sd, Sigma=Sigma, N=20000, antithetic=True, seed=42)

# --- Plot ---
plt.figure(figsize=(7,4.5))
plt.plot(summary["T"], summary["median"], label="Median")
plt.fill_between(summary["T"], summary["q16"], summary["q84"], alpha=0.3, label="68% band")
plt.fill_between(summary["T"], summary["q025"], summary["q975"], alpha=0.15, label="95% band")
plt.xlabel("T")
plt.ylabel("distribution_from_params(T, θ)")
plt.title("Predictive distribution with parameter uncertainty")
plt.legend()
plt.tight_layout()
plt.show()
