import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==================== EDIT THESE ====================
BETA     = 0.60        # 0 < β < 1 (change here)
TAU_STAR = 1.0         # τ* (s)
N_TAU    = 220         # number of τ samples to plot (reduce to speed up)
TAU_MIN, TAU_MAX = 1e-4, 1e4
# Integration controls (looser -> faster, tighter -> slower)
QUAD_LIMIT = 250
EPSABS, EPSREL = 1e-9, 1e-9
# ====================================================

# Eq. (3): ρ(s,β)  with s = τ*/τ
def rho_s(s, beta):
    c = np.cos(np.pi*beta/2.0)
    s_ = np.sin(np.pi*beta/2.0)
    def integrand(u):
        u_b = u**beta
        return np.exp(-u_b * c) * np.cos(s*u - u_b * s_)
    val, _ = quad(integrand, 0.0, np.inf, limit=QUAD_LIMIT, epsabs=EPSABS, epsrel=EPSREL)
    return val / np.pi

# Eq. (5): convert to log10-τ domain
def rho_log10_tau(tau, tau_star, beta):
    s = tau_star / tau
    return rho_s(s, beta) * np.log(10.0) * (tau_star / tau)

# ---- grid and evaluation (single β) ----
tau = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)
rho = np.array([rho_log10_tau(t, TAU_STAR, BETA) for t in tau])

# ---- plot ----
plt.figure(figsize=(8,6))
plt.loglog(tau, rho, label=fr"SEF (β={BETA:g},  τ*={TAU_STAR:g} s)")
plt.axvline(TAU_STAR, color="lightgray", lw=1)
plt.xlim(TAU_MIN, TAU_MAX); plt.ylim(1e-2, 1e2)
plt.xlabel("Relaxation time τ (s)")
plt.ylabel(r"$\rho(\tau,\beta)$ per $\log_{10}\tau$")
plt.title("SEF distribution (Eq. 3 → Eq. 5), single β")
plt.legend()
plt.show()

# ---- optional: normalization check  ∫ ρ d(log10 τ) ≈ 1 ----
x = np.log10(tau)
area = np.trapz(rho, x)
print(f"Normalization check: ∫ ρ dlog10τ ≈ {area:.6f}")
