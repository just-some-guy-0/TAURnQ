#!/usr/bin/env python3
"""
Breakpoint detection in log–log space with a forced first segment.

Constraint:
- The first segment is forced to include the first `force_first_n` points.
- Breakpoints are only searched after that.

Supports 0/1/2 breakpoints overall (i.e. 1/2/3 segments),
but the first segment is always anchored to the first `force_first_n` points.
"""

import numpy as np
import matplotlib.pyplot as plt


def fit_line(x: np.ndarray, y: np.ndarray):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)  # [a, m]
    yhat = A @ coef
    sse = float(np.sum((y - yhat) ** 2))
    return (float(coef[0]), float(coef[1])), yhat, sse


def bic(n: int, sse: float, k_params: int):
    eps = 1e-300
    return n * np.log(max(sse / n, eps)) + k_params * np.log(n)


def eval_0_breaks_forced(x, y, force_first_n=2):
    """
    No breakpoints: single line fit across all points.
    force_first_n is irrelevant here but kept for interface symmetry.
    """
    (a, m), _, sse = fit_line(x, y)
    return {
        "nbreaks": 0,
        "break_idx": (),
        "break_T": (),
        "slopes": (m,),
        "intercepts": (a,),
        "sse": sse,
        "k_params": 2,
    }


def eval_1_break_forced(x, y, T, force_first_n=2, min_seg_pts=3, min_slope_jump=0.0):
    """
    One breakpoint at k:
      seg1 = [0:k], must satisfy k >= force_first_n
      seg2 = [k:n], must satisfy (n-k) >= min_seg_pts
    """
    n = len(x)
    best = None

    k_start = max(force_first_n, min_seg_pts)  # seg1 must have at least these points
    for k in range(k_start, n - min_seg_pts + 1):
        (a1, m1), _, sse1 = fit_line(x[:k], y[:k])
        (a2, m2), _, sse2 = fit_line(x[k:], y[k:])

        if abs(m1 - m2) < min_slope_jump:
            continue

        total = sse1 + sse2
        if (best is None) or (total < best["sse"]):
            best = {
                "nbreaks": 1,
                "break_idx": (k,),
                "break_T": (float(T[k]),),
                "slopes": (m1, m2),
                "intercepts": (a1, a2),
                "sse": total,
                "k_params": 4,
            }
    return best


def eval_2_breaks_forced(x, y, T, force_first_n=2, min_seg_pts=3, min_slope_jump=0.0):
    """
    Two breakpoints at (i, j):
      seg1 = [0:i], must satisfy i >= force_first_n and i >= min_seg_pts
      seg2 = [i:j], must satisfy (j-i) >= min_seg_pts
      seg3 = [j:n], must satisfy (n-j) >= min_seg_pts
    """
    n = len(x)
    best = None

    i_start = max(force_first_n, min_seg_pts)
    for i in range(i_start, n - 2 * min_seg_pts + 1):
        for j in range(i + min_seg_pts, n - min_seg_pts + 1):
            (a1, m1), _, sse1 = fit_line(x[:i], y[:i])
            (a2, m2), _, sse2 = fit_line(x[i:j], y[i:j])
            (a3, m3), _, sse3 = fit_line(x[j:], y[j:])

            if abs(m1 - m2) < min_slope_jump:
                continue
            if abs(m2 - m3) < min_slope_jump:
                continue

            total = sse1 + sse2 + sse3
            if (best is None) or (total < best["sse"]):
                best = {
                    "nbreaks": 2,
                    "break_idx": (i, j),
                    "break_T": (float(T[i]), float(T[j])),
                    "slopes": (m1, m2, m3),
                    "intercepts": (a1, a2, a3),
                    "sse": total,
                    "k_params": 6,
                }
    return best


def detect_breakpoints_forced_first(
    T, tau,
    allow=(0, 1, 2),
    force_first_n=2,
    min_seg_pts=3,
    min_slope_jump=0.0,
):
    """
    Detect 0/1/2 breakpoints in log10(tau) vs log10(T), with a forced first segment
    containing the first `force_first_n` points. Select by BIC.
    """
    T = np.asarray(T, float)
    tau = np.asarray(tau, float)
    if np.any(T <= 0) or np.any(tau <= 0):
        raise ValueError("All T and tau must be > 0.")

    x = np.log10(T)
    y = np.log10(tau)
    n = len(T)

    if force_first_n < 2:
        raise ValueError("force_first_n should be >= 2 to meaningfully 'force' the first line.")

    candidates = []

    if 0 in allow:
        candidates.append(eval_0_breaks_forced(x, y, force_first_n=force_first_n))

    if 1 in allow:
        c1 = eval_1_break_forced(
            x, y, T,
            force_first_n=force_first_n,
            min_seg_pts=min_seg_pts,
            min_slope_jump=min_slope_jump,
        )
        if c1 is not None:
            candidates.append(c1)

    if 2 in allow:
        c2 = eval_2_breaks_forced(
            x, y, T,
            force_first_n=force_first_n,
            min_seg_pts=min_seg_pts,
            min_slope_jump=min_slope_jump,
        )
        if c2 is not None:
            candidates.append(c2)

    if not candidates:
        raise RuntimeError("No valid models found. Relax constraints.")

    for c in candidates:
        c["bic"] = bic(n, c["sse"], c["k_params"])

    best = min(candidates, key=lambda d: d["bic"])
    best["x"] = x
    best["y"] = y
    return best, candidates


def plot_with_fit(T, tau, best):
    T = np.asarray(T, float)
    tau = np.asarray(tau, float)
    x = np.log10(T)

    plt.figure()
    plt.plot(T, tau, marker="o")

    # if best["nbreaks"] == 0:
    #     a, m = best["intercepts"][0], best["slopes"][0]
    #     plt.plot(T, 10 ** (a + m * x))
    # elif best["nbreaks"] == 1:
    #     k = best["break_idx"][0]
    #     (a1, a2) = best["intercepts"]
    #     (m1, m2) = best["slopes"]
    #     plt.plot(T[:k], 10 ** (a1 + m1 * x[:k]))
    #     plt.plot(T[k:], 10 ** (a2 + m2 * x[k:]))
    #     plt.axvline(best["break_T"][0])
    # else:
    #     i, j = best["break_idx"]
    #     (a1, a2, a3) = best["intercepts"]
    #     (m1, m2, m3) = best["slopes"]
    #     plt.plot(T[:i], 10 ** (a1 + m1 * x[:i]))
    #     plt.plot(T[i:j], 10 ** (a2 + m2 * x[i:j]))
    #     plt.plot(T[j:], 10 ** (a3 + m3 * x[j:]))
    #     plt.axvline(best["break_T"][0])
    #     plt.axvline(best["break_T"][1])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Temperature (K)")
    plt.ylabel("τ (s)")
    plt.title("τ vs T with forced-first-segment breakpoint selection (BIC)")
    plt.tight_layout()
    plt.show()


def main():
   
    T = np.array([2, 4, 6, 9, 13, 16, 20, 23], float)
    tau = np.array([814.3, 534.2, 410.9, 302.8, 222.1, 177.3, 132.1, 106.4], float)

    data = np.array([
        # T (K),    value
        [ 40.0,   8.66980000E-01],
        [ 42.0,   7.18500000E-01],
        [ 44.0,   5.99880000E-01],
        [ 46.0,   5.09510000E-01],
        [ 48.0,   4.30310000E-01],
        [ 50.0,   3.72380000E-01],
        [ 52.0,   3.19370000E-01],
        [ 54.0,   2.77660000E-01],
        [ 56.0,   2.42380000E-01],
        [ 58.0,   2.13850000E-01],
        [ 60.0,   1.88770000E-01],
        [ 62.0,   1.64800000E-01],
        [ 64.0,   1.43050000E-01],
        [ 66.0,   1.19030000E-01],
        [ 68.0,   9.57400000E-02],
        [ 70.0,   7.23400000E-02],
        [ 71.9,   5.08100000E-02],
        [ 73.81,  3.43000000E-02],
        [ 75.69,  2.13700000E-02],
        [ 77.6,   1.35800000E-02],
        [ 79.49,  8.37000000E-03],
        [ 81.38,  5.21000000E-03],
        [ 83.29,  3.25000000E-03],
        [ 85.19,  2.03000000E-03],
        [ 87.11,  1.27000000E-03],
        [ 89.0,   8.28237000E-04],
        [ 90.0,   6.58876000E-04],
        [ 91.0,   5.28963000E-04],
        [ 92.0,   4.27569000E-04],
        [ 93.0,   3.43770000E-04],
        [ 94.0,   2.83283000E-04],
        [ 95.0,   2.33609000E-04],
        [ 96.0,   1.94080000E-04],
        [ 97.0,   1.61801000E-04],
        [ 98.0,   1.33383000E-04],
        [ 99.0,   1.04436000E-04],
        [100.0,   8.53422000E-05],
    ])

    T   = data[:, 0]
    tau = data[:, 1]


    # With only 8 points, prefer allow=(0,1) or allow=(0,1,2) but keep min_seg_pts small.
    best, candidates = detect_breakpoints_forced_first(
        T, tau,
        allow=(0, 1,2),
        force_first_n=2,     # force first line to include first 2 points
        min_seg_pts=2,       # small because dataset is small
        min_slope_jump=0.3,  # increase to force sharper kinks
    )

    print("Model comparison (lower BIC is better):")
    for c in sorted(candidates, key=lambda d: d["bic"]):
        print(f"  {c['nbreaks']} break(s): BIC={c['bic']:.3f}, SSE={c['sse']:.4g}, breaks={c['break_T']}")

    print("\nSelected model:")
    print(f"  nbreaks = {best['nbreaks']}")
    print(f"  break_T = {best['break_T']}")
    print(f"  slopes (dlog10(tau)/dlog10(T)) = {tuple(round(s, 3) for s in best['slopes'])}")

    plot_with_fit(T, tau, best)


if __name__ == "__main__":
    main()
