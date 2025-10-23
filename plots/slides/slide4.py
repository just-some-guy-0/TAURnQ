#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ----- knobs you can tweak -----
T1, T2, T3, T4 = 10, 20, 40, 58      # breakpoints (K)
m1, m2, m3 = 0.01, 0.10, 0.30        # slopes
y_T1 = 0.0

NUM_ERR_BARS = 9                     # how many error bars on the middle line
ERR_PLUS  = 1.2                      # +error (y units)
ERR_MINUS = 1.2                      # -error (y units)

FIT_COLOR = (0.80, 0.20, 0.16)       # red
FIT_LW = 1.6

# how close the shallow line's end sits to the top of the last bar (1.0 = at the top)
FRAC_TOP_LAST = 0
# --------------------------------

# Continuity intercepts
y_T2 = y_T1 + m1 * (T2 - T1)
y_T3 = y_T2 + m2 * (T3 - T2)

# Segments
T_seg1 = np.linspace(T1, T2, 50)
T_seg2 = np.linspace(T2, T3, 100)
T_seg3 = np.linspace(T3, T4, 80)

y_seg1 = y_T1 + m1 * (T_seg1 - T1)
y_seg2 = y_T2 + m2 * (T_seg2 - T2)
y_seg3 = y_T3 + m3 * (T_seg3 - T3)

# ---------------- schematic error bars on the middle line ----------------
T_err = np.linspace(T2, T3, NUM_ERR_BARS)
y_mid = y_T2 + m2 * (T_err - T2)      # centers are on the middle line
y_top = y_mid + ERR_PLUS
y_bot = y_mid - ERR_MINUS

# ---------------- two independent red straight lines within bars ---------
# Line A (steep): bottom of first bar -> top of last bar
ext = 5

T_a0, y_a0 = T_err[0], y_bot[0]
T_a1, y_a1 = T_err[-1], y_top[-1]
mA = (y_a1 - y_a0) / (T_a1 - T_a0) * 0.85
bA = y_a0 - mA*T_a0
T_A = np.linspace(T2-ext, T3+ext, 100)
y_A = mA*T_A + bA + 0.25

# Line B (shallow): top of first bar -> near-top of last bar
T_b0, y_b0 = T_err[0], y_top[0]
T_b1 = T_err[-1]
y_b1 = y_top[-1] - (1.0 - FRAC_TOP_LAST)*ERR_PLUS  # stay inside the top bar
mB = 0.01
bB = y_b0 - mB*T_b0
T_B = np.linspace(T2-ext, T3+ext, 100)
y_B = mB*T_B + bB-0.25

# (Optional safety: clip lines to always lie within bars at the discrete T_err points)
# for k, Tk in enumerate(T_err):
#     yA_k = mA*Tk + bA; yB_k = mB*Tk + bB
#     assert y_bot[k] - 1e-9 <= yA_k <= y_top[k]
#     assert y_bot[k] - 1e-9 <= yB_k <= y_top[k]

# ---------------- plot ----------------
plt.figure(figsize=(9, 5.5))
lw = 2

# base three black segments
plt.plot(T_seg1, y_seg1, 'k-', lw=lw)
plt.plot(T_seg2, y_seg2, 'k-', lw=lw)
plt.plot(T_seg3, y_seg3, 'k-', lw=lw)

# # red lines (different gradients), both within bars
plt.plot(T_A, y_A,'--', color='#ff0000', lw=2)
plt.plot(T_B, y_B,'--', color='#ff0000', lw=2)

# error bars on middle segment
plt.errorbar(T_err, y_mid, yerr=[y_mid - y_bot, y_top - y_mid],
             fmt='none', ecolor='k', elinewidth=1.4, capsize=0)

plt.xlabel("Temperature (K)", labelpad=10, fontsize=14)
plt.ylabel(r"Relaxation Rate $(\tau^{-1})$", labelpad=10, fontsize=14)
plt.grid(True, lw=0.5, alpha=0.3)
plt.tight_layout()

# minimalist axes: keep labels only
plt.xticks([])
plt.yticks([])
plt.tick_params(which="both", length=0)
plt.ylim(-2, 8)

plt.show()
# plt.savefig("two_independent_lines_within_errorbars.png", dpi=300, bbox_inches="tight")
