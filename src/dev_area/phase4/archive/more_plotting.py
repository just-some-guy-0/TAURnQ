#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC = "NSi2iPr3"
PATH = SRC + ".tsv"

# ---------- model (QTM = 10**(-Q)) ----------
def log10_rate(T, A, U, R, n, Q):
    T = np.maximum(T, 1e-300)
    term_orb = (10.0**(-A)) * np.exp(-U / T)    # Orbach
    term_ram = (10.0**(R))  * (T**n)            # Raman
    term_qtm = (10.0**(-Q))                     # QTM
    r = term_orb + term_ram + term_qtm
    return np.log10(np.clip(r, 1e-300, None))

# y1 param set (means, SDs, correlations)
A,U,R,n,Q,sA,sU,sR,sn,sQ, rho_AU, rho_RN = (
-11.222366302894539,2594.6441566734748,-5.535445489286728,2.9828490014518296,1.859440556251219,0.0010005942218391593,85.78324893996029,2.515317899485107,1.4741114589658593,0.7142739711880814,0.9913592042210823,-0.983909409569086
)

# ---------- MC band config ----------
K = 50000
seed = 12345
Tg = np.linspace(18, 52.5, 400)

# Reference (original) curve to overlay ----> only from reta/chilton
A0, U0, R0, n0, Q0 = -11.54, 957, -5.4, 4.32, -0.12
y0 = log10_rate(Tg, A0, U0, R0, n0, Q0)

A2, U2, R2, n2, Q2 = -11.55, 2563.92, -5.5, 3.01, 1.83

y2 = log10_rate(Tg, A2, U2, R2, n2, Q2)

# ---------- correlated parameter draws ----------
def draw_params(K, seed):
    rng = np.random.default_rng(seed)
    # (A,U)
    L_AU = np.array([[sA, 0.0],
                     [rho_AU * sU, sU * np.sqrt(max(1.0 - rho_AU**2, 1e-12))]])
    Z_AU = rng.standard_normal(size=(K, 2))
    eps_AU = Z_AU @ L_AU.T
    A_s = A + eps_AU[:, 0]
    U_s = U + eps_AU[:, 1]
    # (R,n)
    L_RN = np.array([[sR, 0.0],
                     [rho_RN * sn, sn * np.sqrt(max(1.0 - rho_RN**2, 1e-12))]])
    Z_RN = rng.standard_normal(size=(K, 2))
    eps_RN = Z_RN @ L_RN.T
    R_s = R + eps_RN[:, 0]
    n_s = n + eps_RN[:, 1]
    # Q independent
    Q_s = Q + rng.standard_normal(size=K) * sQ
    return A_s, U_s, R_s, n_s, Q_s

A_s, U_s, R_s, n_s, Q_s = draw_params(K, seed)

# ---------- compute mean curve and ±1σ band via MC ----------
y1 = log10_rate(Tg, A, U, R, n, Q)
y_mean = np.empty_like(Tg)
y_std  = np.empty_like(Tg)
for i, T in enumerate(Tg):
    y_draws = log10_rate(T, A_s, U_s, R_s, n_s, Q_s)
    y_mean[i] = np.mean(y_draws)
    y_std[i]  = np.std(y_draws)

# ---------- FK 1σ bars, centered on a chosen reference curve ----------
df = pd.read_csv(PATH, sep=None, engine="python", header=None).iloc[:, :3]
df.columns = ["T", "tau_mean", "alpha"]
T_data   = df["T"].astype(float).values
tau_mean = df["tau_mean"].astype(float).values
alpha    = df["alpha"].astype(float).values

ln10 = np.log(10.0)
sigma_ln = 1.82 * np.sqrt(alpha) / (1.0 - alpha)    # FK width in ln(tau)
y_half   = sigma_ln / ln10                           # FK ±1σ width in log10(rate)

# --- choose what to center on ---
# Options:
#   1) Center on a specific model curve via parameters (recommended)
#   2) Center on FK mean itself (-log10(tau_mean)) for "physically correct" FK bars

# (1) e.g. center on y0 curve
ref_params = (A2, U2, R2, n2, Q2)  # swap to (A, U, R, n, Q) or any set you want
y_center = log10_rate(T_data, *ref_params)

# (2) OR: uncomment to center on FK mean instead
# y_center = -np.log10(tau_mean)

# Now plot FK ±1σ widths around the chosen center
plt.errorbar(T_data, y_center, yerr=y_half,
             fmt='none', ecolor='k', elinewidth=1.2, capsize=0,
             label='FK ±1σ (centered on ref)')


# ---------- plot ----------
plt.figure(figsize=(8, 5))

# band behind
plt.fill_between(Tg, y_mean - y_std, y_mean + y_std,
                 alpha=0.20, label="±1σ band")

# curves
plt.plot(Tg, y1, 'r-',  lw=1.8,   label="Analytically derived rate")
# plt.plot(Tg, y0, 'k--', lw=1.8, label="Purely mean-fitted rate")
plt.plot(Tg, y2, 'b-', lw=1.8, label="Uncertainty-fitted rate")


# FK error bars: black, symmetric in log space, no markers/caps
plt.errorbar(T_data, y_center, yerr=y_half,
             fmt='none', ecolor='k', elinewidth=1.2, capsize=0,
             label='±1σ uncertainties')

plt.xlabel("Temperature (K)")
plt.ylabel(r'$\log_{10}(\tau^{-1})$')
plt.title(SRC)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

out_png = f"{SRC}_compare_plot.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved figure: {out_png}")
# plt.show()
