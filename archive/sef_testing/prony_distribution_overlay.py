import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Stretched exponential target
# -----------------------------
def g_true(x, beta, tau=1.0):
    x = np.asarray(x, dtype=float) / float(tau)
    return np.exp(-np.power(x, beta))

# -----------------------------------------------
# Prony fit: g_P(x) = sum_i w_i * exp(-K_i * x)
# -----------------------------------------------
def fit_prony(beta, N=100, x_max=40.0, ngrid=4000, tau=1.0,
              lam=1e3, kmin=1e-3, kmax=1e3):
    x = np.linspace(0.0, float(x_max), int(ngrid))
    y = g_true(x, beta=beta, tau=tau)

    # Log-spaced rate nodes covering a wide range
    K = np.geomspace(float(kmin), float(kmax), int(N))

    # Design matrix and soft constraint (sum w_i = 1)
    A = np.exp(-np.outer(x/float(tau), K))
    A_aug = np.vstack([A, np.sqrt(lam) * np.ones((1, N))])
    y_aug = np.concatenate([y, [np.sqrt(lam) * 1.0]])

    w, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)

    # Nonnegativity & renormalization
    w = np.clip(w, 0.0, None)
    if w.sum() == 0:
        w[:] = 1.0 / N
    else:
        w /= w.sum()

    y_p = A @ w
    rmse = np.sqrt(np.mean((y_p - y) ** 2))
    return x, y, y_p, K, w, rmse

# -------------------------------------------------
# Exact Laplace distribution ρ(s, β)
# -------------------------------------------------
def rho_exact(s, beta, u_max=60.0, n_t=4000):
    s = float(s)
    beta = float(beta)
    c = np.cos(np.pi * beta / 2.0)
    s2 = np.sin(np.pi * beta / 2.0)

    T = np.log1p(u_max)         # u = exp(t) - 1 mapping
    t = np.linspace(0.0, T, int(n_t))
    u = np.expm1(t)
    du_dt = np.exp(t)

    phase = s * u - (u ** beta) * s2
    damp = np.exp(- (u ** beta) * c)
    integrand = damp * np.cos(phase) * du_dt

    h = t[1] - t[0]
    S = integrand[0] + integrand[-1] + 4.0 * integrand[1:-1:2].sum() + 2.0 * integrand[2:-2:2].sum()
    integral = (h / 3.0) * S
    return integral / np.pi

def rho_exact_grid(s_grid, beta, u_max=60.0, n_t=4000):
    return np.array([rho_exact(s, beta, u_max=u_max, n_t=n_t) for s in s_grid])

# -----------------------------------------------
# Smoothed Prony distribution (visual aid only)
# -----------------------------------------------
def smooth_spikes_logx(K, w, sigma_log=0.08, s_grid=None):
    if s_grid is None:
        s_grid = np.logspace(np.log10(min(K)/3), np.log10(max(K)*3), 1500)
    log_s = np.log(s_grid)
    rho = np.zeros_like(s_grid, dtype=float)
    for Ki, wi in zip(K, w):
        rho += wi * np.exp(-0.5 * ((log_s - np.log(Ki))/sigma_log)**2) / (s_grid * sigma_log * np.sqrt(2*np.pi))
    return s_grid, rho

# -----------------------------------------------
# Driver: overlay plots
# -----------------------------------------------
def make_overlay(beta=0.6, N=100):
    # 1) Fit Prony
    x, y, y_p, K, w, rmse = fit_prony(beta=beta, N=N)

    # 2) Exact distribution
    s_grid = np.logspace(-3, 3, 300)  # wide range of s
    rho_ex = rho_exact_grid(s_grid, beta)

    # 3) Smoothed Prony distribution
    s_vis, rho_vis = smooth_spikes_logx(K, w, s_grid=s_grid)

    # ---- Plot 1: Distribution overlay ----
    plt.figure(figsize=(7,5))
    plt.plot(s_grid, rho_ex, label="Exact ρ(s,β)", linewidth=2)
    (ml, sl, bl) = plt.stem(K, w, linefmt='C1-', markerfmt='C1o', basefmt=" ", label="Prony spikes")
    plt.setp(sl, linewidth=0.7)
    plt.plot(s_vis, rho_vis, label="Smoothed Prony (visual)", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Relaxation rate s")
    plt.ylabel(r"$\rho(s,\beta)$")
    plt.title(f"Rate distribution (β={beta}, N={N})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: Time-domain fit ----
    plt.figure(figsize=(7,5))
    plt.plot(x, y, label="Exact g(x)")
    plt.plot(x, y_p, label=f"Prony fit (N={N}), RMSE={rmse:.2e}")
    plt.xlabel("x = t/τ")
    plt.ylabel("g(x)")
    plt.title("Time-domain stretched exponential vs Prony fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("=== Prony coefficients (first 10 shown) ===")
    print("K_i:", K[:10], "...")
    print("w_i:", w[:10], "...")
    print("RMSE:", rmse)

if __name__ == "__main__":
    make_overlay(beta=0.6, N=100)
