#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ----- knobs you can tweak -----
# Breakpoints (inclusive ranges): [T1,T2], [T2,T3], [T3,T4]
T1, T2, T3, T4 = 10, 20, 40, 58  # K

# Slopes (gradients) for the three linear regimes
m1, m2, m3 = 0.01, 0.1, 0.3

# Starting value at T1 (sets the vertical placement)
y_T1 = 0.0
# --------------------------------

# Compute intercepts so the segments are continuous
y_T2 = y_T1 + m1 * (T2 - T1)
y_T3 = y_T2 + m2 * (T3 - T2)

# Build the segments
T_seg1 = np.linspace(T1, T2, 50)
T_seg2 = np.linspace(T2, T3, 100)
T_seg3 = np.linspace(T3, T4, 80)

y_seg1 = y_T1 + m1 * (T_seg1 - T1)
y_seg2 = y_T2 + m2 * (T_seg2 - T2)
y_seg3 = y_T3 + m3 * (T_seg3 - T3)

# Plot
plt.figure(figsize=(9, 5.5))
lw = 2
col = 'k-'
plt.plot(T_seg1, y_seg1, col, lw=lw, label="")
plt.plot(T_seg2, y_seg2, col, lw=lw, label="")
plt.plot(T_seg3, y_seg3, col, lw=lw, label="")

plt.xlabel("Temperature (K)", labelpad= 10, fontsize = 13)
plt.ylabel(r'Relaxation rate (log$_{10}[\tau^{-1}]$)', labelpad= 10, fontsize = 13)
plt.grid(True, lw=0.5, alpha=0.3)
plt.tight_layout()

# remove tick marks and tick labels, keep axis titles
plt.xticks([])
plt.yticks([])
# (optional) also ensure no tick marks are drawn)
plt.tick_params(which="both", length=0)


plt.show()
# plt.savefig("three_linear_segments.png", dpi=300, bbox_inches="tight")
