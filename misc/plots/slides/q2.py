import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- eqn(10) ---
def log10_rate(T, A, Ueff, R, n, Q):
    term_orb = (10.0**(-A)) * np.exp(-Ueff / T)  # Orbach
    term_ram = (10.0**(R))  * (T**n)             # Raman
    term_qtm = (10.0**(Q))                       # QTM
    r = term_orb + term_ram + term_qtm
    return np.log10(np.clip(r, 1e-300, None))

# ---- NEW: accent function to sharpen slope changes at specific T ----
def emphasize_kinks(T, y, kinks=(15, 40), amps=(0.45, 1.1), widths=(1.4, 1.8)):
    """
    Adds smooth slope 'kinks' centered at each temperature in `kinks`.
      amps   = vertical strengths (bigger -> stronger slope change)
      widths = transition widths in K (smaller -> sharper)
    """
    y = y.copy()
    for Tk, a, w in zip(kinks, amps, widths):
        y += a * np.tanh((T - Tk) / w)
    return y

# --- params ---
A0, U0, R0, n0, Q0 = -11.54, 957, -5.4, 4.32, -0.12
label0,c0 = "Relaxation rate",'k-'
A1, U1, R1, n1, Q1 = -12,1000,-6,4,-0.1
label1,c1 = "Analytically derived rate",'b-'
A2, U2, R2, n2, Q2 = -11.088476492154259,902.4156007711586,-4.816441819237312,3.864241783310507,-0.1214270124793344
label2,c2 = "safe correlation",'g-'

# --- eval curves ---
Tg = np.linspace(9, 58, 400)
y0 = log10_rate(Tg, A0, U0, R0, n0, Q0)
y0 = emphasize_kinks(Tg, y0, kinks=(15, 40), amps=(0.45, 1.1), widths=(1.4, 1.8))  # <- NEW

# y1 = log10_rate(Tg, A1, U1, R1, n1, Q1)
# y2 = log10_rate(Tg, A2, U2, R2, n2, Q2)

# --- load df ---
df = pd.read_csv("tBuOCl.tsv", sep=None, engine="python", header=None).iloc[:, :3]
df.columns = ["T", "tau_mean", "alpha"]

T_data   = df["T"].astype(float).values
tau_mean = df["tau_mean"].astype(float).values
alpha    = df["alpha"].astype(float).values
g        = 1.82 * np.sqrt(alpha) / (1.0 - alpha)

# --- calc error bars ---
MEANS    = tau_mean
rate     = 1.0 / MEANS
rate_hi  = 1.0 / (MEANS * np.exp(-g))  # +1σ
rate_lo  = 1.0 / (MEANS * np.exp(+g))  # -1σ
y_mean   = np.log10(rate)
y_plus   = np.log10(rate_hi) - y_mean
y_minus  = y_mean - np.log10(rate_lo)

# --- helper: piecewise-linear fit anchored at breakpoints ---
def piecewise_linear_from_model(T, y_model, breaks):
    """
    Return (Tx, yx) for a continuous piecewise-linear curve through given breakpoints.
    For each segment, slope is from a least-squares fit to y_model over that segment,
    and the intercept is chosen so the line passes exactly through the left breakpoint.
    """
    breaks = np.asarray(sorted(breaks))
    T = np.asarray(T); y_model = np.asarray(y_model)
    assert np.all((breaks[0] >= T.min()) & (breaks[-1] <= T.max()))

    # values of model at the breakpoints (for continuity)
    y_at_breaks = np.interp(breaks, T, y_model)

    Tx_list, yx_list = [], []
    seg_edges = np.r_[T.min(), breaks, T.max()]   # [Tmin, 15, 40, Tmax]

    for i in range(len(seg_edges) - 1):
        a, b = seg_edges[i], seg_edges[i+1]
        mask = (T >= a) & (T <= b)
        Ti = T[mask]; yi = y_model[mask]

        # slope from LS fit on this interval
        m, c = np.polyfit(Ti, yi, 1)

        # anchor intercept to pass exactly through left breakpoint
        left_x = a if i == 0 else breaks[i-1]
        left_y = np.interp(left_x, T, y_model) if i == 0 else y_at_breaks[i-1]
        c_adj = left_y - m*left_x

        # segment X and Y
        Tseg = Ti
        yseg = m*Tseg + c_adj

        # avoid duplicating the junction x except for the final point
        if i > 0:
            Tseg = Tseg[Tseg > left_x]
            yseg = yseg[Tseg > left_x]

        Tx_list.append(Tseg)
        yx_list.append(yseg)

    Tx = np.concatenate(Tx_list)
    yx = np.concatenate(yx_list)
    return Tx, yx

# --- build the piecewise-linear curve using your current model y0 ---
breaks = np.array([15.0, 40.0])
Tx, y_lin = piecewise_linear_from_model(Tg, y0, breaks)


# --- plot ---
fig, ax = plt.subplots(figsize=(7.5, 4.8))

# piecewise-linear (three straight lines joined at 15 K and 40 K)
ax.plot(Tx, y_lin, 'k-', lw=2.6, label="Piecewise linear (15 K, 40 K)")

# (optional) overlay original model faintly for reference
# ax.plot(Tg, y0, color='0.6', lw=1.5, label="Model", alpha=0.6)

ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r'$\log_{10}(\tau^{-1})$')
ax.legend(loc="upper left", frameon=False)

# hide tick marks and values
ax.set_xticks([]); ax.set_yticks([])
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.show()
