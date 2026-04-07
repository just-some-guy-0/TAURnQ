#!/usr/bin/env python3
"""
slope_change_breakpoints.py

Detect up to 2 breakpoints as the largest slope changes computed from rolling
linear fits over a window of N points (default N=5).

Uses k = 1/tau and two physics-relevant transforms:
  1) log-log Raman view:   y = log10(k), x = log10(T)
  2) Orbach view:          y = log10(k), x = 1/T

A breakpoint score at index i is:
  score(i) = max( |m_left - m_right| ) over the chosen views
where m_left is slope fit on points [i-N+1 .. i] and m_right on [i+1 .. i+N].

Returns 1 or 2 breakpoints depending on --n-breaks (0/1/2) or --auto
(auto uses a simple relative-threshold test).

Example:
  python slope_change_breakpoints.py data.txt --show
"""

from __future__ import annotations
import argparse
from typing import Optional, Tuple, List
import numpy as np


def load_xy(path: str, delimiter: Optional[str], skiprows: int, xcol: int, ycol: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim == 1:
        raise ValueError("File appears to contain only one row; need 2 columns minimum.")
    if max(xcol, ycol) >= arr.shape[1]:
        raise ValueError(f"File has {arr.shape[1]} columns; requested xcol={xcol}, ycol={ycol}.")
    return arr[:, xcol], arr[:, ycol]


def slope_ols(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of y ~ a + b x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def compute_kink_scores(T: np.ndarray, tau: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kink scores at each index i where both left and right windows exist.
    Returns (valid_indices, scores).
    """
    eps = 1e-300
    k = 1.0 / np.clip(tau, eps, None)

    # Views
    x1 = np.log10(T)
    y1 = np.log10(k)      # Raman view
    x2 = 1.0 / T
    y2 = np.log10(k)      # Orbach view

    n = len(T)
    idxs = []
    scores = []

    for i in range(window - 1, n - window):
        L = slice(i - window + 1, i + 1)
        R = slice(i + 1, i + 1 + window)

        m1L = slope_ols(x1[L], y1[L])
        m1R = slope_ols(x1[R], y1[R])
        d1 = abs(m1L - m1R)

        m2L = slope_ols(x2[L], y2[L])
        m2R = slope_ols(x2[R], y2[R])
        d2 = abs(m2L - m2R)

        idxs.append(i)
        scores.append(max(d1, d2))

    return np.array(idxs, int), np.array(scores, float)


def pick_top_breaks(
    idxs: np.ndarray,
    scores: np.ndarray,
    n_breaks: int,
    min_separation: int,
    min_pts_each_side: int,
    n_total: int,
) -> List[int]:
    """
    Pick top n_breaks indices by score with constraints:
      - at least min_separation apart (in index units)
      - each break leaves at least min_pts_each_side on both sides globally
    """
    order = np.argsort(scores)[::-1]  # descending
    chosen: List[int] = []

    for p in order:
        i = int(idxs[p])

        # global edge constraint
        if i < min_pts_each_side - 1:
            continue
        if i > n_total - min_pts_each_side - 1:
            continue

        # separation constraint
        if any(abs(i - c) < min_separation for c in chosen):
            continue

        chosen.append(i)
        if len(chosen) >= n_breaks:
            break

    return sorted(chosen)


def auto_choose_n_breaks(scores: np.ndarray, ratio: float = 0.6) -> int:
    """
    Simple auto rule:
      - 0 breaks if no scores
      - 1 break if top score is prominent
      - 2 breaks only if second score is at least `ratio` of the top score
    """
    if scores.size == 0:
        return 0
    s = np.sort(scores)[::-1]
    if s[0] <= 0:
        return 0
    if s.size == 1:
        return 1
    return 2 if (s[1] / s[0] >= ratio) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Breakpoint detection via largest slope change in rolling windows.")
    p.add_argument("file", help="Path to data file with at least 2 columns (T, tau).")
    p.add_argument("--delimiter", default=None, help="Delimiter for loadtxt (default: whitespace).")
    p.add_argument("--skiprows", type=int, default=0, help="Rows to skip at top (e.g., header).")
    p.add_argument("--xcol", type=int, default=0, help="Column index for T (0-based).")
    p.add_argument("--ycol", type=int, default=1, help="Column index for tau (0-based).")

    p.add_argument("--window", type=int, default=5, help="Window size for slope fits (default 5).")
    p.add_argument("--n-breaks", type=int, choices=[0, 1, 2], default=1, help="Number of breakpoints to return.")
    p.add_argument("--auto", action="store_true", help="Automatically choose 0/1/2 breakpoints from scores.")
    p.add_argument("--auto-ratio", type=float, default=0.6, help="Second/top score ratio to allow 2 breaks in --auto.")

    p.add_argument("--min-sep", type=int, default=5, help="Minimum separation between breakpoints (in points).")
    p.add_argument("--min-pts-side", type=int, default=6, help="Minimum points on each side of a breakpoint.")
    p.add_argument("--show", action="store_true", help="Show plot.")
    p.add_argument("--save-plot", default=None, help="Save plot to path.")

    args = p.parse_args(argv)

    T, tau = load_xy(args.file, args.delimiter, args.skiprows, args.xcol, args.ycol)

    mask = np.isfinite(T) & np.isfinite(tau) & (tau > 0)
    T = T[mask]
    tau = tau[mask]

    if T.size < 2 * args.window + 1:
        raise ValueError("Not enough points for the requested window size.")

    # sort by T
    order = np.argsort(T)
    T = T[order]
    tau = tau[order]

    idxs, scores = compute_kink_scores(T, tau, window=args.window)

    if args.auto:
        n_breaks = auto_choose_n_breaks(scores, ratio=args.auto_ratio)
    else:
        n_breaks = args.n_breaks

    breaks = pick_top_breaks(
        idxs=idxs,
        scores=scores,
        n_breaks=n_breaks,
        min_separation=args.min_sep,
        min_pts_each_side=args.min_pts_side,
        n_total=len(T),
    )

    print(f"Selected breakpoints (count={len(breaks)}):")
    for i in breaks:
        print(f"  index={i}  T={T[i]:.6g} K  score={scores[np.where(idxs==i)[0][0]]:.6g}")

    if args.show or args.save_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(T, tau, "o-", label="data")
        for i in breaks:
            ax.axvline(T[i], linestyle="--", linewidth=1)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("tau (s)")
        ax.set_yscale("log")
        ax.legend()

        if args.save_plot:
            fig.savefig(args.save_plot, dpi=200, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
