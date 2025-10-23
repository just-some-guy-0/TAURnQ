import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- eqn(10) ---
def log10_rate(T, A, Ueff, R, n, Q):
    term_orb = (10.0**(-A)) * np.exp(-Ueff / T)  # Orbach
    term_ram = (10.0**(R))  * (T**n)             # Raman
    term_qtm = (10.0**(-Q))                      # QTM
    r = term_orb + term_ram + term_qtm
    return np.log10(np.clip(r, 1e-300, None))

# --- params ---
# analytic
A0, U0, R0, n0, Q0 = -11.54, 957, -5.4, 4.32, -0.12
label0,c0 = "Relaxation rate",'r-'

A1, U1, R1, n1, Q1 = -12,1000,-6,4,-0.1
label1,c1 = "1 sd",'r-'
A2, U2, R2, n2, Q2 = -11.088476492154259,902.4156007711586,-4.816441819237312,3.864241783310507,-0.1214270124793344
label2,c2 = "safe correlation",'g-'

# --- eval curves ---
Tg = np.linspace(9, 58, 400)
y0 = log10_rate(Tg, A0, U0, R0, n0, Q0)   # 0s curve
y1 = log10_rate(Tg, A1, U1, R1, n1, Q1)   # 1s curve
y2 = log10_rate(Tg, A2, U2, R2, n2, Q2)

# --- load df ---
df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
df.columns = ["T", "tau_mean", "alpha"]

T_data     = df["T"].astype(float).values
tau_mean   = df["tau_mean"].astype(float).values
alpha      = df["alpha"].astype(float).values
g = 1.82 * np.sqrt(alpha) / (1.0 - alpha)

# --- calc error bars ---
rate      = 1.0 / tau_mean
rate_hi   = 1.0 / (tau_mean * np.exp(-g))  # +1s
rate_lo   = 1.0 / (tau_mean * np.exp(+g))  # -1s
y_mean    = np.log10(rate)
y_plus    = np.log10(rate_hi) - y_mean
y_minus   = y_mean - np.log10(rate_lo)

# --- plot ---
plt.figure(figsize=(7.5, 4.8))
plt.plot(Tg, y0, c0, lw=2, label=label0)
# plt.plot(Tg, y1, c1,  lw=2,  label=label1)
# plt.plot(Tg, y2, c2,  lw=1.5,  label=label2)
plt.errorbar(T_data, y_mean, yerr=[y_minus, y_plus],
             fmt='none', ms=5, capsize=0, elinewidth=1,ecolor = 'k',
             label='±1σ uncertainties')

plt.xlabel("Temperature (K)", fontsize = 12)
plt.ylabel(r'Relaxation Rate $(\tau^{-1})$', fontsize = 12)
plt.title("")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.ylim(0,5)
plt.show()
