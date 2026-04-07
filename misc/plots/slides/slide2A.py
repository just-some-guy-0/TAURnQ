import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -------- helpers --------
def inverted_parabola(mj, apex_height=20.0, curvature=0.9, apex_mj=0.0):
    mj = np.asarray(mj, dtype=float)
    return apex_height - curvature * (mj - apex_mj)**2

def draw_levels(ax, mj_vals, E_vals, seg_width=0.7, lw=2.2, c="k"):
    for m, e in zip(mj_vals, E_vals):
        ax.plot([m - seg_width/2, m + seg_width/2], [e, e], c=c, lw=lw, solid_capstyle="round")

def style_axes(ax, title):
    ax.set_title(title, pad=12, fontsize=13, weight="bold")

    # labels
    ax.set_xlabel("Magnetic moment", fontsize = 13)
    ax.set_ylabel("Energy", fontsize = 13)

    # keep only the central tick labeled as 0
    ax.set_xticks([0])
    ax.set_xticklabels(["0"], fontsize=12)
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)

    # keep bottom and left spines visible for schematic style
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)

    ax.set_xlim(-6.8, 6.8)
    ax.set_ylim(-2, 26)

def draw_orbach_discrete(ax, mJ_vals, E_vals, head=10, lw=2, color="tab:blue",
                         label=True, shorten=0.1):
    """
    Orbach arrows *between successive levels*, trimmed by `shorten` (0..0.45).
    shorten=0 → full-length; larger → shorter arrows.
    """
    def shorten_seg(p0, p1, frac):
        x0,y0 = p0; x1,y1 = p1
        return (x0 + frac*(x1-x0), y0 + frac*(y1-y0)), (x1 - frac*(x1-x0), y1 - frac*(y1-y0))

    # left: min mJ -> 0
    mL = mJ_vals[mJ_vals <= 0]
    EL = E_vals[:len(mL)]
    for x0,y0,x1,y1 in zip(mL[:-1], EL[:-1], mL[1:], EL[1:]):
        (sx,sy),(tx,ty) = shorten_seg((x0,y0),(x1,y1), shorten)
        ax.add_patch(FancyArrowPatch((sx,sy),(tx,ty),
            arrowstyle="-|>", mutation_scale=head, linewidth=lw,
            color=color, zorder=6, shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.0"))

    # right: 0 -> max mJ
    mR = mJ_vals[mJ_vals >= 0]
    ER = E_vals[-len(mR):]
    for x0,y0,x1,y1 in zip(mR[:-1], ER[:-1], mR[1:], ER[1:]):
        (sx,sy),(tx,ty) = shorten_seg((x0,y0),(x1,y1), shorten)
        ax.add_patch(FancyArrowPatch((sx,sy),(tx,ty),
            arrowstyle="-|>", mutation_scale=head, linewidth=lw,
            color=color, zorder=6, shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.0"))

    if label:
        x0 = 0.0
        ax.text(x0, np.interp(x0, mJ_vals, E_vals) + 0.08*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                "Orbach", color=color, ha="center", va="bottom", fontsize=15)


def draw_raman_ends(ax, mJ_vals, E_vals, mid_y=None, head=10, lw=1.5, color="tab:red", label=True, shorten=0.03):
    """
    Raman: two straight arrows from left ground (mJ_min,E_min) to a virtual mid near mJ=0,
    then to right ground. Endpoints are exactly on the ground-state levels.
    """
    xL, yL = mJ_vals.min(), E_vals[0]+0.3             # left ground
    xR, yR = mJ_vals.max(), E_vals[-1]+0.3            # right ground
    if mid_y is None:
        mid_y = min(yL,yR) + 0.35*(max(E_vals)-min(E_vals))
    xM = 0.0

    def shorten_seg(p0, p1, frac):
        x0,y0 = p0; x1,y1 = p1
        return (x0 + frac*(x1-x0), y0 + frac*(y1-y0)), (x1 - frac*(x1-x0), y1 - frac*(y1-y0))

    (s0,t0),(s1,t1) = shorten_seg((xL,yL), (xM,mid_y), shorten)
    ax.add_patch(FancyArrowPatch((s0,t0),(s1,t1), arrowstyle="-|>", mutation_scale=head,
                                 linewidth=lw, color=color, zorder=7, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad=0.0"))

    (s0,t0),(s1,t1) = shorten_seg((xM,mid_y), (xR,yR), shorten)
    ax.add_patch(FancyArrowPatch((s0,t0),(s1,t1), arrowstyle="-|>", mutation_scale=head,
                                 linewidth=lw, color=color, zorder=7, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad=0.0"))

    if label:
        ax.text(xM, mid_y + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), "Raman",
                color=color, ha="center", va="bottom", fontsize=15)

def draw_qtm(ax, mJ_vals, E_vals, head=11, lw=2, color="tab:purple",
             label=True, style="double", curve=0.0, shorten=0.10, y_offset=0.0):
    """
    Quantum tunneling of magnetization between the two ground states.
      style: "double" (<->) or "single" (-|>)
      curve: 0.0 (straight) … 0.4 (arched)
      shorten: trims a fraction from both ends so arrowheads don't cover the dots/bars
      y_offset: raise/lower the arrow relative to the ground level (in data units)
    """
    xL, xR = mJ_vals.min(), mJ_vals.max()
    y_ground = min(E_vals[0], E_vals[-1]) + y_offset

    # shorten endpoints
    x0 = xL + shorten*(xR - xL)
    x1 = xR - shorten*(xR - xL)

    arrowstyle = "<->" if style == "double" else "-|>"
    patch = FancyArrowPatch((x0, y_ground), (x1, y_ground),
                            arrowstyle=arrowstyle, mutation_scale=head,
                            linewidth=lw, color=color, zorder=8,
                            shrinkA=0, shrinkB=0,
                            connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(patch)

    if label:
        ax.text(0.0, y_ground + 0.06*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                "QTM", color=color, ha="center", va="bottom", fontsize=15)


# -------- grid of mJ states --------
mJ = np.arange(-6, 7, 1)

# --- fake inversion barrier ---
B = inverted_parabola(mJ, apex_height=22, curvature=0.55, apex_mj=0.0)
fig3, ax3 = plt.subplots(figsize=(7, 5))
draw_levels(ax3, mJ, B)

# two ground-state dots (you already have this)
minE3 = B.min()
ground_idxs = np.where(np.isclose(B, minE3))[0]
ax3.scatter(mJ[ground_idxs], B[ground_idxs], s=140, alpha=0.45,
            edgecolors="k", linewidths=0.8, zorder=4)

# --- NEW: arrows like the figure ---
# make a callable parabola for convenience
parabola = lambda x: inverted_parabola(np.asarray(x), apex_height=22, curvature=0.55, apex_mj=0.0)

# After draw_levels(ax3, mJ, E3) and the ground-state dots:
draw_orbach_discrete(ax3, mJ, B, head=12, lw=2.4, color="tab:orange", label=True)

# Raman arrows that start/end exactly on the ground-state levels:
draw_raman_ends(ax3, mJ, B,
                mid_y=B.min() + 0.35*(B.max()-B.min()),
                head=12, lw=2.4, color="tab:green", label=True)
# QTM between ground states (slightly shortened, tiny upward offset to avoid overlap)
draw_qtm(ax3, mJ, B,
         head=12, lw=2.4, color="tab:blue",
         style="double", curve=0.0, shorten=0.03, y_offset=0.0)


style_axes(ax3, "")
plt.show()
