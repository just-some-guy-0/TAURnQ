import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy.optimize import minimize

# -----------------------------
# Your SEF: rho_log_tau(tau) = pdf over x = log10(tau)
# -----------------------------
def rho_s(s: np.ndarray, beta: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)

    mask = s > 0
    if not np.any(mask):
        return out

    theta = np.pi * beta / 2.0
    c = np.cos(theta)

    s_scaled = s[mask] / (c ** (1.0 / beta))
    out[mask] = (c ** (-1.0 / beta)) * levy_stable.pdf(
        s_scaled, beta, 1.0, loc=0.0, scale=1.0
    )
    return out

def rho_log_tau(tau: np.ndarray, beta: float, tau_star: float = 1.0) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    s = tau_star / tau
    return np.log(10.0) * s * rho_s(s, beta)

# -----------------------------
# Two-piece normal PDF in x = log10(tau)
# f(x) = A exp(-(x-mu)^2/(2*sigma1^2)) for x<=mu
#      = A exp(-(x-mu)^2/(2*sigma2^2)) for x>=mu
# with A = (sqrt(2*pi) * (sigma1+sigma2)/2)^(-1)
# -----------------------------
def two_piece_normal_pdf_x(x, mu, sigma1, sigma2):
    x = np.asarray(x, dtype=float)
    sigma1 = float(sigma1); sigma2 = float(sigma2)
    if sigma1 <= 0 or sigma2 <= 0:
        return np.zeros_like(x)

    A = 1.0 / (np.sqrt(2.0*np.pi) * (sigma1 + sigma2) / 2.0)
    y = np.empty_like(x)

    mL = x <= mu
    dx = x - mu
    y[mL]  = A * np.exp(-0.5 * (dx[mL]  / sigma1)**2)
    y[~mL] = A * np.exp(-0.5 * (dx[~mL] / sigma2)**2)
    return y

# Evaluate two-piece normal in tau-space while still being a pdf over x=log10(tau)
def two_piece_normal_pdf_tau(tau, mu_x, sigma1, sigma2):
    x = np.log10(np.asarray(tau, dtype=float))
    return two_piece_normal_pdf_x(x, mu_x, sigma1, sigma2)

# -----------------------------
# Fit two-piece normal to SEF in x-space (stable + tail-sensitive)
# - fit only where f is above a threshold
# - use KL(f||g) on the truncated domain
# - constrain sigmas to [s_min, s_max] using a sigmoid parameterization
# - optional extra weighting of the right tail
# -----------------------------
def fit_two_piece_to_sef(beta, tau_star=1.0,
                         x_min=-6, x_max=6, nx=8000,
                         fix_mu_to_mode=True,
                         thr_rel=1e-8,
                         s_min=1e-3, s_max=5.0,
                         right_tail_weight=0.0, right_tail_power=2.0):
    """
    Fits a two-piece normal f_TPN(x) to SEF pdf over x=log10(tau).

    Improvements vs the simple SSE:
      - Truncates domain to where f is non-negligible (prevents sigma blow-up)
      - Uses KL(f||g) on truncated domain (more tail-sensitive, stable)
      - Constrains sigma1,sigma2 to [s_min, s_max] via sigmoid reparam
      - Optionally emphasizes right tail with a smooth weight

    Returns (mu_x, sigma1, sigma2), info dict, and (x, f_full).
    """
    x = np.linspace(x_min, x_max, nx)
    tau = 10.0**x
    f_full = rho_log_tau(tau, beta, tau_star=tau_star)

    # mode (in x) of the SEF on this grid
    mu0 = float(x[np.argmax(f_full)])

    # --- truncate fit domain ---
    thr = float(thr_rel * f_full.max())
    mfit = f_full > thr
    if np.count_nonzero(mfit) < 50:
        # fallback: if threshold too strict, relax it
        mfit = f_full > (1e-12 * f_full.max())

    x_fit = x[mfit]
    f_fit = f_full[mfit]

    # renormalize f on truncated domain
    Zf = np.trapz(f_fit, x_fit)
    f_fit = f_fit / Zf

    eps = 1e-300

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def inv_sigmoid(y):
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y / (1 - y))

    def unpack(params):
        if fix_mu_to_mode:
            z1, z2 = params
            mu = mu0
        else:
            mu, z1, z2 = params
        sigma1 = s_min + (s_max - s_min) * sigmoid(z1)
        sigma2 = s_min + (s_max - s_min) * sigmoid(z2)
        return mu, sigma1, sigma2

    # smooth weight to emphasize right tail, if desired
    if right_tail_weight > 0:
        t = np.maximum(0.0, x_fit - mu0)
        # normalize t to [0,1] for stability
        denom = max(1e-12, float(x_fit.max() - mu0))
        tt = t / denom
        w_tail = 1.0 + right_tail_weight * (tt ** right_tail_power)
    else:
        w_tail = 1.0

    def loss(params):
        mu, s1, s2 = unpack(params)
        g = two_piece_normal_pdf_x(x_fit, mu, s1, s2)

        # renormalize g on truncated domain
        Zg = np.trapz(g, x_fit)
        g = g / (Zg + eps)

        # KL(f||g) with optional right-tail weighting
        integrand = w_tail * f_fit * (np.log(f_fit + eps) - np.log(g + eps))
        kl = np.trapz(integrand, x_fit)

        # very mild regularization to avoid pinning to edges
        reg = 1e-3 * ((s1 - 0.5)**2 + (s2 - 0.5)**2)
        return float(kl + reg)

    # --- initial guesses from HWHM as before, but mapped to constrained z-space ---
    i0 = np.argmax(f_full)
    half = 0.5 * f_full.max()

    il = np.where(f_full[:i0] <= half)[0]
    xl = x[il[-1]] if len(il) else x[0]
    ir = np.where(f_full[i0:] <= half)[0]
    xr = x[i0 + ir[0]] if len(ir) else x[-1]

    s1_0 = max((mu0 - xl) / np.sqrt(2*np.log(2)), 1e-2)
    s2_0 = max((xr - mu0) / np.sqrt(2*np.log(2)), 1e-2)

    # clamp into [s_min, s_max]
    s1_0 = float(np.clip(s1_0, s_min + 1e-6, s_max - 1e-6))
    s2_0 = float(np.clip(s2_0, s_min + 1e-6, s_max - 1e-6))

    y1 = (s1_0 - s_min) / (s_max - s_min)
    y2 = (s2_0 - s_min) / (s_max - s_min)
    z1_0 = inv_sigmoid(y1)
    z2_0 = inv_sigmoid(y2)

    x0 = np.array([z1_0, z2_0]) if fix_mu_to_mode else np.array([mu0, z1_0, z2_0])

    res = minimize(loss, x0, method="Nelder-Mead")

    mu_hat, s1_hat, s2_hat = unpack(res.x)

    # diagnostics: left mass in x for SEF vs implied by TPN (on full domain grid)
    F_left = float(np.trapz(f_full[x <= mu_hat], x[x <= mu_hat]))
    left_mass_tpn = s1_hat / (s1_hat + s2_hat)

    info = dict(
        success=bool(res.success),
        message=str(res.message),
        mu_mode_x=mu0,
        mu_hat_x=mu_hat,
        sef_left_mass=F_left,
        tpn_left_mass=left_mass_tpn,
        fun=float(res.fun),
        nfev=int(res.nfev),
        thr_rel=thr_rel,
        fit_fraction=float(np.trapz(f_full[mfit], x[mfit])),
        s_min=s_min,
        s_max=s_max,
        right_tail_weight=right_tail_weight,
        right_tail_power=right_tail_power,
    )

    return (mu_hat, s1_hat, s2_hat), info, (x, f_full)

# -----------------------------
# Example: your beta
# -----------------------------
if __name__ == "__main__":
    beta = 0.7
    tau_star = 1.0

    (mu_x, s1, s2), info, (x, f_sef) = fit_two_piece_to_sef(
        beta=beta, tau_star=tau_star,
        x_min=-6, x_max=6, nx=8000,
        fix_mu_to_mode=True,
        thr_rel=1e-8,
        s_min=1e-3, s_max=5.0,
        # Increase right_tail_weight (e.g. 2..20) if you want tighter right tail
        right_tail_weight=0.0, right_tail_power=2.0,
    )

    print("Fit info:", info)
    print(f"Two-piece normal params in x=log10(tau): mu={mu_x:.6f}, sigma1={s1:.6f}, sigma2={s2:.6f}")

    # Compare on a linear tau plot (like your current figure)
    tau_lin = np.linspace(0.0, 30.0, 400)
    tau_lin[0] = 1e-12  # avoid log10(0)
    y_sef = rho_log_tau(tau_lin, beta, tau_star=tau_star)
    y_tpn = two_piece_normal_pdf_tau(tau_lin, mu_x, s1, s2)

    plt.figure(figsize=(6.6, 4.8))
    plt.plot(tau_lin, y_sef, label=f"SEF beta={beta:g}")
    plt.plot(tau_lin, y_tpn, "--", label="Two-piece normal fit")
    plt.xlim(0, 30)
    plt.ylim(0, 2)
    plt.xlabel("Relaxation time (s)")
    plt.ylabel(r"$\rho_{\log}(\tau^*=1\ \mathrm{s},\,\beta)$  (pdf over $x=\log_{10}\tau$)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()
