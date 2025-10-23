
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Stretched exponential target
# ----------------------------
def g_true(x, beta, tau=1.0):
    x = np.asarray(x, dtype=float) / float(tau)
    return np.exp(-np.power(x, beta))


# ---------------------------------------
# Prony fit (sum of exponentials) utility
# ---------------------------------------
def fit_prony(beta, N=12, x_max=40.0, ngrid=4000, tau=1.0, lam=1e3, kmin=0.04, kmax=400.0, seed=42):
    """
    Fit g_true(x, beta) on x in [0, x_max] with a Prony series:
        g_P(x) = sum_i w_i * exp(-K_i * (x/tau))
    using a simple least-squares with a soft constraint sum_i w_i = 1
    and clipping to enforce non-negativity on weights (then renormalizing).

    Returns:
        x (grid), y_true, y_prony, K (rates), w (weights), rmse
    """
    rng = np.random.default_rng(seed)

    # grid
    x = np.linspace(0.0, float(x_max), int(ngrid))
    y = g_true(x, beta=beta, tau=tau)

    # choose K on log grid covering short/long times
    K = np.geomspace(float(kmin), float(kmax), int(N))

    # design matrix A_{j,i} = exp(-K_i * x_j / tau)
    A = np.exp(-np.outer(x / float(tau), K))

    # soft constraint: sum w_i = 1 via a penalty row
    A_aug = np.vstack([A, np.sqrt(lam) * np.ones((1, N))])
    y_aug = np.concatenate([y, [np.sqrt(lam) * 1.0]])

    # LS solve
    w, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)

    # clip to non-negative and renormalize
    w = np.clip(w, 0.0, None)
    if w.sum() == 0:
        w[:] = 1.0 / N
    else:
        w /= w.sum()

    y_prony = A @ w
    rmse = np.sqrt(np.mean((y_prony - y)**2))

    return x, y, y_prony, K, w, rmse


# -------------------
# Helper: derivative
# -------------------
def numerical_derivative(x, f):
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    df[0] = (f[1] - f[0]) / (x[1] - x[0])
    df[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    return df


# -------------------
# Plotting pipeline
# -------------------
def make_plots(beta, N=12, x_max=40.0, ngrid=4000, tau=1.0, lam=1e3, kmin=0.04, kmax=400.0, seed=42):
    x, y, y_p, K, w, rmse = fit_prony(beta, N, x_max, ngrid, tau, lam, kmin, kmax, seed)

    # Plot 1: full range
    plt.figure(figsize=(7,5))
    plt.plot(x, y, label=f"Stretched exponential (β={beta:g})")
    plt.plot(x, y_p, label=f"Prony series (N={N}), RMSE={rmse:.2e}", ls="--")
    plt.xlabel("Normalized time  x = t/τ")
    plt.ylabel("g(x)")
    plt.title(f"Stretched Exponential vs Prony Series (β={beta:g})")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: derivative near 0
    dy_true = numerical_derivative(x, y)
    dy_prony = numerical_derivative(x, y_p)
    mask = x <= 0.2

    plt.figure(figsize=(7,5))
    plt.plot(x[mask], dy_true[mask], label="dg/dx (true) near x=0")
    plt.plot(x[mask], dy_prony[mask], label="dg/dx (Prony) near x=0")
    plt.xlabel("x (zoomed to 0 ≤ x ≤ 0.2)")
    plt.ylabel("dg/dx")
    plt.title(f"Short-time slope: divergent (true) vs finite (Prony), β={beta:g}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: absolute error up to x=1
    abs_err = np.abs(y_p - y)
    mask2 = x <= 1.0
    plt.figure(figsize=(7,5))
    plt.plot(x[mask2], abs_err[mask2], label="|g_P(x) - g(x)|")
    plt.xlabel("x (zoomed to 0 ≤ x ≤ 1)")
    plt.ylabel("Absolute error")
    plt.title(f"Short-time absolute error (β={beta:g}, N={N})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print coefficients for reproducibility
    print("Prony K_i (rate constants):")
    print(K)
    print("\nProny w_i (weights, sum=1):")
    print(w)
    print(f"\nGlobal RMSE over x in [0, {x_max}]: {rmse:.3e}")


if __name__ == "__main__":
    # Reproduce the two cases from the chat
    for b in (0.1, 0.5):
        make_plots(beta=b, N=5, x_max=40.0, ngrid=4000, tau=1.0, lam=1e3, kmin=0.04, kmax=400.0, seed=42)
