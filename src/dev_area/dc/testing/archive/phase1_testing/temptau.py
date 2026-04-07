#!/usr/bin/env python3
"""
piecewise_breakpoints_physical.py

Drop-in script to detect 0–2 breakpoints in relaxation data using PHYSICALLY
motivated linearity tests on the RATE k = 1/tau.

Model:
    k(T) = 10^{-A} exp(-Ueff/T) + 10^{R} T^{n} + 10^{-Q}

Dominant-term linearizations (used per segment):
  - QTM:    k vs T (lin-lin)                     k = b0 + b1*T  (expects b1 ~ 0)
  - Raman:  log10(k) vs log10(T) (log-log)       logk = b0 + b1*logT
  - Orbach: log10(k) vs (1/T) (lin-log in 1/T)   logk = b0 + b1*(1/T)  (expects b1 < 0)

We fit piecewise segments (K=1..3 segments -> 0..2 breakpoints) via dynamic
programming, where each segment cost is the BEST SSE among {QTM,Raman,Orbach}.
Then select K by BIC (default) or AIC.

Input: file with at least 2 columns (T, tau).
Output: chosen breakpoints, segment mechanisms, coefficients; optional plot.

Examples
--------
python piecewise_breakpoints_physical.py data.txt --show
python piecewise_breakpoints_physical.py data.csv --delimiter , --skiprows 1 --show
python piecewise_breakpoints_physical.py data.txt --min-pts 6 --criterion bic --save-plot fit.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


# -----------------------------
# OLS helpers
# -----------------------------
def _ols_sse(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Ordinary least squares SSE for y ~ X @ beta."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    sse = float(np.sum((y - yhat) ** 2))
    return sse, beta


def _segment_best_mechanism_sse(
    Tseg: np.ndarray,
    kseg: np.ndarray,
    penalize_unphysical: bool = True,
) -> Tuple[float, str, Tuple[float, float]]:
    """
    Return (best_sse, mech, (b0,b1)) where mech in {"QTM","Raman","Orbach"} with:

      QTM:    k        = b0 + b1*T
      Raman:  log10(k) = b0 + b1*log10(T)
      Orbach: log10(k) = b0 + b1*(1/T)
    """
    eps = 1e-300
    kseg = np.clip(kseg, eps, None)

    # QTM (lin-lin)
    Xq = np.vstack([np.ones_like(Tseg), Tseg]).T
    sse_q, bq = _ols_sse(Xq, kseg)

    # Raman (log-log)
    logk = np.log10(kseg)
    logT = np.log10(Tseg)
    Xr = np.vstack([np.ones_like(logT), logT]).T
    sse_r, br = _ols_sse(Xr, logk)

    # Orbach (logk vs 1/T)
    invT = 1.0 / Tseg
    Xo = np.vstack([np.ones_like(invT), invT]).T
    sse_o, bo = _ols_sse(Xo, logk)

    if penalize_unphysical:
        # Orbach slope should be negative in log10(k) vs 1/T
        if bo[1] > 0:
            sse_o *= 5.0

        # QTM should be ~constant k: penalize large relative variation explained by slope
        level = float(np.median(kseg))
        if level > 0:
            rel_slope = abs(float(bq[1])) * (float(Tseg.max() - Tseg.min())) / level
            if rel_slope > 0.3:
                sse_q *= 2.0

    sses = [sse_q, sse_r, sse_o]
    mechs = ["QTM", "Raman", "Orbach"]
    betas = [bq, br, bo]

    m = int(np.argmin(sses))
    b0 = float(betas[m][0])
    b1 = float(betas[m][1])
    return float(sses[m]), mechs[m], (b0, b1)


def precompute_segment_costs_physical(
    T: np.ndarray,
    tau: np.ndarray,
    min_pts: int,
    penalize_unphysical: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    cost[i,j]   = min SSE among {QTM,Raman,Orbach} linearizations on k=1/tau for segment i..j.
    params[i,j] = (mech, b0, b1) for the chosen mechanism on that segment.
    """
    n = len(T)
    cost = np.full((n, n), np.inf, dtype=float)
    params = np.empty((n, n), dtype=object)

    k = 1.0 / tau

    for i in range(n):
        j0 = i + min_pts - 1
        if j0 >= n:
            continue
        for j in range(j0, n):
            sse, mech, (b0, b1) = _segment_best_mechanism_sse(
                T[i : j + 1],
                k[i : j + 1],
                penalize_unphysical=penalize_unphysical,
            )
            cost[i, j] = sse
            params[i, j] = (mech, b0, b1)

    return cost, params


# -----------------------------
# DP segmentation (optimal K segments)
# -----------------------------
def fit_k_segments(cost: np.ndarray, K: int, min_pts: int) -> Tuple[float, List[int], List[Tuple[int, int]]]:
    """
    Optimal partition into K segments minimizing total cost via dynamic programming.

    Returns:
      total_cost, breaks, segments
        breaks: end index of each segment except last (length K-1)
        segments: list of (i,j) inclusive bounds (length K)
    """
    n = cost.shape[0]
    dp = np.full((K, n), np.inf, dtype=float)
    prev = np.full((K, n), -1, dtype=int)

    # Base: 1 segment 0..j
    for j in range(min_pts - 1, n):
        dp[0, j] = cost[0, j]

    # Fill DP
    for k in range(1, K):
        j_start = (k + 1) * min_pts - 1
        for j in range(j_start, n):
            t_min = k * min_pts - 1
            t_max = j - min_pts
            best_val = np.inf
            best_t = -1
            for t in range(t_min, t_max + 1):
                val = dp[k - 1, t] + cost[t + 1, j]
                if val < best_val:
                    best_val = val
                    best_t = t
            dp[k, j] = best_val
            prev[k, j] = best_t

    total = dp[K - 1, n - 1]
    if not np.isfinite(total):
        raise ValueError("No feasible segmentation for this K/min_pts/data length.")

    # Backtrack breaks
    breaks: List[int] = []
    j = n - 1
    for k in range(K - 1, 0, -1):
        t = prev[k, j]
        if t < 0:
            raise RuntimeError("Backtrack failed (unexpected).")
        breaks.append(int(t))
        j = t
    breaks.reverse()

    # Segments
    segments: List[Tuple[int, int]] = []
    start = 0
    for b in breaks:
        segments.append((start, b))
        start = b + 1
    segments.append((start, n - 1))

    return float(total), breaks, segments


def ic_from_sse(sse: float, n: int, K: int, criterion: str) -> float:
    """
    Information criterion for Gaussian residuals with unknown variance.
    K segments -> p = 2*K parameters (b0,b1 per segment, mech chosen by min SSE).
    """
    p = 2 * K
    sse = max(sse, 1e-300)
    crit = criterion.lower()
    if crit == "bic":
        return n * np.log(sse / n) + p * np.log(n)
    if crit == "aic":
        return n * np.log(sse / n) + 2 * p
    raise ValueError("criterion must be 'bic' or 'aic'")


# -----------------------------
# Fit result structs
# -----------------------------
@dataclass
class SegmentFit:
    i: int
    j: int
    mech: str
    b0: float
    b1: float


@dataclass
class FitResult:
    x: np.ndarray  # sorted T
    tau: np.ndarray  # sorted tau
    K_segments: int
    n_breakpoints: int
    break_indices: List[int]
    break_temperatures: List[float]
    segments: List[Tuple[int, int]]
    segment_fits: List[SegmentFit]
    models: List[Dict[str, Any]]


def fit_piecewise_physical_0_to_2_breakpoints(
    T: np.ndarray,
    tau: np.ndarray,
    min_pts: int = 2,
    criterion: str = "bic",
    penalize_unphysical: bool = True,
) -> FitResult:
    """
    Fit among K in {1,2,3} segments -> breakpoints in {0,1,2}.
    Segment cost = best SSE among QTM/Raman/Orbach linearizations on k=1/tau.
    """
    T = np.asarray(T, float)
    tau = np.asarray(tau, float)
    if T.shape != tau.shape:
        raise ValueError("T and tau must have the same shape.")

    order = np.argsort(T)
    x = T[order]
    tau_s = tau[order]

    n = len(x)
    if n < min_pts:
        raise ValueError("Not enough points for min_pts.")

    cost, params = precompute_segment_costs_physical(
        x, tau_s, min_pts=min_pts, penalize_unphysical=penalize_unphysical
    )

    # Only allow 0..2 breakpoints => K=1..3 segments, if feasible
    K_candidates = [K for K in (1, 2, 3) if K * min_pts <= n]
    if not K_candidates:
        raise ValueError("min_pts too large for dataset length to fit even 1 segment.")

    models: List[Dict[str, Any]] = []
    for K in K_candidates:
        sse, breaks, segments = fit_k_segments(cost, K, min_pts=min_pts)
        ic = ic_from_sse(sse, n, K, criterion=criterion)
        models.append({"K": K, "sse": sse, "ic": ic, "breaks": breaks, "segments": segments})

    best = min(models, key=lambda d: d["ic"])

    segfits: List[SegmentFit] = []
    for (i, j) in best["segments"]:
        mech, b0, b1 = params[i, j]
        segfits.append(SegmentFit(i=int(i), j=int(j), mech=str(mech), b0=float(b0), b1=float(b1)))

    break_T = [float(x[b]) for b in best["breaks"]]

    return FitResult(
        x=x,
        tau=tau_s,
        K_segments=int(best["K"]),
        n_breakpoints=int(best["K"] - 1),
        break_indices=[int(b) for b in best["breaks"]],
        break_temperatures=break_T,
        segments=[(int(i), int(j)) for (i, j) in best["segments"]],
        segment_fits=segfits,
        models=models,
    )


# -----------------------------
# Plotting
# -----------------------------
def build_tau_fit(res: FitResult) -> np.ndarray:
    """Construct tau_fit(T) from per-segment mechanism fits."""
    tau_fit = np.full_like(res.x, np.nan, dtype=float)
    eps = 1e-300

    for sf in res.segment_fits:
        i, j = sf.i, sf.j
        Tseg = res.x[i : j + 1]

        if sf.mech == "QTM":
            # k = b0 + b1*T
            kseg = sf.b0 + sf.b1 * Tseg
        elif sf.mech == "Raman":
            # log10(k) = b0 + b1*log10(T) => k = 10^b0 * T^b1
            kseg = (10.0 ** sf.b0) * (Tseg ** sf.b1)
        elif sf.mech == "Orbach":
            # log10(k) = b0 + b1*(1/T) => k = 10^(b0 + b1/T)
            kseg = 10.0 ** (sf.b0 + sf.b1 * (1.0 / Tseg))
        else:
            raise ValueError(f"Unknown mechanism: {sf.mech}")

        kseg = np.clip(kseg, eps, None)
        tau_fit[i : j + 1] = 1.0 / kseg

    return tau_fit


def maybe_plot(
    res: FitResult,
    save_path: Optional[str],
    show: bool,
):
    if (save_path is None) and (not show):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib required for plotting: {e}") from e

    tau_fit = build_tau_fit(res)

    fig, ax = plt.subplots()
    ax.plot(res.x, res.tau, "o", label="data")
    ax.plot(res.x, tau_fit, "-", label="physical piecewise fit")

    for tb in res.break_temperatures:
        ax.axvline(tb, linestyle="--", linewidth=1)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("tau (s)")
    ax.set_yscale("log")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------
# I/O and CLI
# -----------------------------
def load_xy(
    path: str,
    delimiter: Optional[str],
    skiprows: int,
    xcol: int,
    ycol: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim == 1:
        raise ValueError("File appears to contain only one row; need 2 columns minimum.")
    if max(xcol, ycol) >= arr.shape[1]:
        raise ValueError(f"File has {arr.shape[1]} columns; requested xcol={xcol}, ycol={ycol}.")
    return arr[:, xcol], arr[:, ycol]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Detect 0–2 breakpoints using mechanism-consistent linearity on k=1/tau."
    )
    p.add_argument("file", help="Path to data file with at least 2 columns (T, tau).")
    p.add_argument("--delimiter", default=None, help="Delimiter for loadtxt (default: whitespace).")
    p.add_argument("--skiprows", type=int, default=0, help="Rows to skip at top (e.g., header).")
    p.add_argument("--xcol", type=int, default=0, help="Column index for T (0-based).")
    p.add_argument("--ycol", type=int, default=1, help="Column index for tau (0-based).")

    p.add_argument("--min-pts", type=int, default=6, help="Minimum points per segment.")
    p.add_argument("--criterion", choices=["bic", "aic"], default="bic", help="Model selection criterion.")
    p.add_argument("--no-physics-penalty", action="store_true",
                   help="Disable mild penalties for unphysical Orbach/QTM behavior.")

    p.add_argument("--save-plot", default=None, help="Path to save plot (optional).")
    p.add_argument("--show", action="store_true", help="Show plot interactively.")

    args = p.parse_args(argv)

    T, tau = load_xy(args.file, args.delimiter, args.skiprows, args.xcol, args.ycol)

    # Basic sanity
    mask = np.isfinite(T) & np.isfinite(tau) & (tau > 0)
    T = T[mask]
    tau = tau[mask]
    if len(T) < args.min_pts:
        raise ValueError("Not enough valid points after filtering (finite and tau>0).")

    res = fit_piecewise_physical_0_to_2_breakpoints(
        T=T,
        tau=tau,
        min_pts=args.min_pts,
        criterion=args.criterion,
        penalize_unphysical=(not args.no_physics_penalty),
    )

    # Print summary
    print(f"Chosen segments: {res.K_segments}  (breakpoints: {res.n_breakpoints})")
    if res.n_breakpoints > 0:
        print("Breakpoint temperatures (K):")
        for tb in res.break_temperatures:
            print(f"  {tb:.6g}")
    else:
        print("No breakpoints selected.")

    print("\nSegments and mechanism fits (on k = 1/tau):")
    for idx, sf in enumerate(res.segment_fits, start=1):
        if sf.mech == "QTM":
            form = "k = b0 + b1*T"
        elif sf.mech == "Raman":
            form = "log10(k) = b0 + b1*log10(T)"
        else:
            form = "log10(k) = b0 + b1*(1/T)"
        print(f"  seg {idx}: i={sf.i:>3} j={sf.j:>3}  mech={sf.mech:<6}  {form}  b0={sf.b0:.6g}  b1={sf.b1:.6g}")

    print("\nModel comparison (K segments => K-1 breakpoints; only K=1..3 considered):")
    for m in sorted(res.models, key=lambda d: d["K"]):
        nb = m["K"] - 1
        print(f"  K={m['K']} (breakpoints={nb})  SSE={m['sse']:.6g}  {args.criterion.upper()}={m['ic']:.6g}")

    maybe_plot(res, save_path=args.save_plot, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
