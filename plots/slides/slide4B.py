#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ----- knobs you can tweak -----
T1, T2, T3, T4 = 10, 20, 40, 58      # breakpoints (K)
m1, m2, m3 = 0.01, 0.10, 0.30        # slopes
y_T1 = 0.0

NUM_ERR_BARS = 9                     # how many shapes on the middle line
ERR_PLUS  = 1.2                      # legacy ±error (y units)
ERR_MINUS = 1.2

FRAC_TOP_LAST = 0
# --------------------------------

# ---------- helper: one-sided sideways normal ("half-violin") ----------
def half_sideways_normal(ax, Ti, yi, sigma_y, *,
                         side="right", width=0.9, k=3.0, n=200,
                         ec="k", fc=None, alpha=0.18, lw=1.2,
                         draw_center_edge=False):
    """
    Draw a one-sided horizontal normal density centered at yi, located at x=Ti.
    sigma_y: vertical std dev of the normal.
    side: 'right' or 'left' (which side of x=Ti to draw).
    width: visual half-width scaling in x (purely aesthetic).
    k: extent in ±k*sigma_y vertically.
    """
    sgn = +1 if side.lower() == "right" else -1
    y = np.linspace(yi - k*sigma_y, yi + k*sigma_y, n)
    # normal pdf (only used up to a scale factor)
    pdf = np.exp(-0.5*((y - yi)/sigma_y)**2) / (sigma_y * np.sqrt(2*np.pi))
    pdf /= pdf.max()  # scale to [0,1] for consistent width

    x_edge = Ti + sgn * width * pdf

    # closed polygon: from the vertical seam at x=Ti out to the curved edge
    x_poly = np.concatenate([np.full_like(y, Ti), x_edge[::-1]])
    y_poly = np.concatenate([y,                  y[::-1]])

    ax.fill(x_poly, y_poly, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw)
    if draw_center_edge:
        ax.plot([Ti, Ti], [y.min(), y.max()], c=ec, lw=lw)
    ax.plot(x_edge, y, c=ec, lw=lw)

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

# ---------------- centers for the half-violins on the middle line --------
T_err = np.linspace(T2, T3, NUM_ERR_BARS)
y_mid = y_T2 + m2 * (T_err - T2)      # centers are on the middle line

# Map your old ±ERR to a vertical sigma:
# If ±ERR ~ 2σ, use ERR_PLUS/2; if ±ERR ~ 1σ, use ERR_PLUS.
SIGMA_Y = ERR_PLUS / 3.5
WIDTH   = 1.3     # visual thickness in x (purely aesthetic)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
lw = 2

# base three black segments
ax.plot(T_seg1, y_seg1, 'k-', lw=lw)
ax.plot(T_seg2, y_seg2, 'r--', lw=lw)
ax.plot(T_seg3, y_seg3, 'k-', lw=lw)

# replace error bars with half-violins (to the right of the backbone)
for Ti, yi in zip(T_err, y_mid):
    half_sideways_normal(ax, Ti, yi, SIGMA_Y,
                         side="left", width=WIDTH, k=3.0,
                         ec="k", fc='tab:blue', alpha=0.18, lw=1.1,
                         draw_center_edge=False)

ax.set_xlabel("Temperature (K)", labelpad=10, fontsize=14)
ax.set_ylabel(r"Relaxation Rate $(\tau^{-1})$", labelpad=10, fontsize=14)
ax.grid(True, lw=0.5, alpha=0.3)
plt.tight_layout()

# minimalist axes: keep labels only
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(which="both", length=0)
plt.ylim(-2, 8)

plt.show()
# plt.savefig("half_violins.png", dpi=300, bbox_inches="tight")
