import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy.special import gamma
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

def sef_mean_tau(beta: float, tau_star: float = 1.0) -> float:
    """
    Expectation value <tau> for stretched exponential distribution:
    <tau> = (tau_star / beta) * Gamma(1 / beta)
    """
    return (tau_star / beta) * gamma(1.0 / beta)

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
# Fit two-piece normal to SEF in x-space
# -----------------------------
def fit_two_piece_to_sef(beta, tau_star=1.0,
                         x_min=-6, x_max=6, nx=6000,
                         fix_mu_to_mode=True):
    x = np.linspace(x_min, x_max, nx)
    tau = 10.0**x
    f = rho_log_tau(tau, beta, tau_star=tau_star)

    mu0 = float(x[np.argmax(f)])
    dx = x[1] - x[0]

    # ---- only fit where SEF is significant ----
    thr = 1e-8 * f.max()
    mask = f > thr
    x_fit = x[mask]
    f_fit = f[mask]

    eps = 1e-300

    def loss(params):
        if fix_mu_to_mode:
            log_s1, log_s2 = params
            mu = mu0
        else:
            mu, log_s1, log_s2 = params

        s1 = np.exp(log_s1)
        s2 = np.exp(log_s2)
        g = two_piece_normal_pdf_x(x_fit, mu, s1, s2)

        # ---- log-space least squares (tail-sensitive) ----
        return np.mean((np.log(g + eps) - np.log(f_fit + eps))**2)

    # ---- initial guesses (same as yours) ----
    i0 = np.argmax(f)
    half = 0.5 * f.max()

    il = np.where(f[:i0] <= half)[0]
    xl = x[il[-1]] if len(il) else x[0]
    ir = np.where(f[i0:] <= half)[0]
    xr = x[i0 + ir[0]] if len(ir) else x[-1]

    s1_0 = max((mu0 - xl) / np.sqrt(2*np.log(2)), 1e-3)
    s2_0 = max((xr - mu0) / np.sqrt(2*np.log(2)), 1e-3)

    x0 = np.array([np.log(s1_0), np.log(s2_0)])

    res = minimize(loss, x0, method="Nelder-Mead")

    mu_hat = mu0
    s1_hat, s2_hat = np.exp(res.x[0]), np.exp(res.x[1])

    return (mu_hat, s1_hat, s2_hat), dict(fun=res.fun), (x, f)


# -----------------------------
# Example: your beta
# -----------------------------
if __name__ == "__main__":
    beta = 0.675
    tau_star = 302.8

    (mu_x, s1, s2), info, (x, f_sef) = fit_two_piece_to_sef(
        beta=beta, tau_star=tau_star,
        x_min=-6, x_max=6, nx=8000,
        fix_mu_to_mode=True
    )

    # ---- Goodness of fit ----
    f_fit = two_piece_normal_pdf_x(x, mu_x, s1, s2)

    # restrict to meaningful region
    mask = f_sef > 1e-10 * f_sef.max()
    f1 = f_sef[mask]
    f2 = f_fit[mask]
    dx = x[1] - x[0]

    # R^2 in log space
    logf1 = np.log(f1)
    logf2 = np.log(f2)
    ss_res = np.sum((logf1 - logf2)**2)
    ss_tot = np.sum((logf1 - logf1.mean())**2)
    R2_log = 1 - ss_res / ss_tot

    # KL divergence D_KL(SEF || TPN)
    KL = np.sum(f1 * np.log(f1 / f2)) * dx

    print(f"Goodness of fit:")
    print(f"  R^2 (log-density) = {R2_log:.6f}")
    print(f"  KL divergence     = {KL:.6e}")

    print("Fit info:", info)
    print(f"Two-piece normal params in x=log10(tau): mu={mu_x:.6f}, sigma1={s1:.6f}, sigma2={s2:.6f}")

    # Compare on a linear tau plot (like your current figure)
    tau_lin = np.logspace(-3, 3, 400)
    tau_lin[0] = 1e-12  # avoid log10(0)
    y_sef = rho_log_tau(tau_lin, beta, tau_star=tau_star)
    y_tpn = two_piece_normal_pdf_tau(tau_lin, mu_x, s1, s2)

    sef_mean_tau = sef_mean_tau(beta, tau_star=tau_star)
    f_sigma = (1 - (2/np.pi)) * ((s2-s1)**2) + (s1*s2)

    t_sigma = ((1/(beta**2))-1) * ((np.pi**2) / 6) 

    print(f"<tau> = {sef_mean_tau:.6g}")
    print(f"sigma  = {f_sigma:.6g}")
    print(f"t_sigma = {t_sigma:.6g}")

    print(f"Q_mu = {np.log10(sef_mean_tau):.6g}")
    print(f"Q_sigma = {np.log10(f_sigma):.6g}")


    plt.figure(figsize=(6.6, 4.8))
    plt.plot(tau_lin, y_sef, label=f"SEF beta={beta:g}")
    plt.plot(tau_lin, y_tpn, "--", label="Two-piece normal fit")
    plt.xscale("log")
    plt.xlim(0, 30)
    plt.ylim(0, 2)
    plt.xlabel("Relaxation time (s)")
    plt.ylabel(r"$\rho_{\log}(\tau^*=1\ \mathrm{s},\,\beta)$  (pdf over $x=\log_{10}\tau$)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()
