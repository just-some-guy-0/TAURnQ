#!/usr/bin/env python3
"""
orbach_spread.py — independent, closed-form estimate of the Orbach-channel
parameter spread implied by the per-temperature alpha values (Reta-Chilton
log-widths), with NO Monte Carlo, NO rho parameter, NO domain regulariser.

Use it to decide whether global_fit_combined's sU is the honest spread its
own inputs imply, or an artefact of the pinned rho_AU / the rho>=0 clamp.

PRINCIPLE
---------
In the Orbach-dominated regime the rate law linearises:

    tau = tau0 * exp(Ueff / T)   =>   ln tau = A*ln10 + Ueff*(1/T)

so ln(tau) is a straight line in x = 1/T, slope = Ueff, intercept = A*ln10.

R&C assign each temperature a 1-sigma log-normal width on ln(tau)

    gamma_i = 1.82 * sqrt(alpha_i) / (1 - alpha_i)          (Table 1, 1 sigma)

Feed those as per-point errors into a weighted straight-line fit. The slope's
standard error is the barrier spread implied purely by the alpha values and
the temperatures you include. In the linear-Gaussian regime the estimation
covariance and the "spread of U consistent with the data at 1 sigma" are the
SAME Gaussian, so sigma_U here is an apples-to-apples target for the
generative sU that global_fit reports — not merely R&C's fit error bar.

READ-OUT
--------
Compare sigma_U(this)  with  global_fit's sU:

  sigma_U(this) ~ sU     -> the fit is honest. sU is small because the window
                            is narrow / high-T (small alpha), NOT because rho
                            ate it. To grow it, widen the T-range that informs
                            the barrier (include lower-T, higher-alpha points).

  sigma_U(this) >> sU    -> the fit suppresses a spread its own inputs imply
                            -> the rho_AU pin (or the rho>=0 clamp) is eating
                               it. That is a bug, not a category difference.

BONUS: the fit also returns corr(A, U). For a straight line in 1/T (x>0) the
estimation correlation is NEGATIVE. Compare its sign to your pinned
rho_AU=+0.999 — if they disagree, the clamp isn't merely collapsing the
magnitude, it's stuck at the wrong sign, which is itself the mechanism that
forces sU to collapse (you can't use the intercept/slope anti-correlation to
keep the Orbach tau-width tight, so the only way left to keep it tight is to
shrink sU).
"""

import numpy as np
import pandas as pd
import argparse, sys

LN10 = np.log(10.0)


def gamma_lognormal(alpha):
    """R&C 1-sigma log-normal half-width of ln(tau) for generalised-Debye alpha."""
    a = np.asarray(alpha, float)
    return 1.82 * np.sqrt(a) / (1.0 - a)


def load_tsv(path):
    df = pd.read_csv(path, sep=None, engine="python", header="infer")
    df.columns = [c.lower().strip() for c in df.columns]
    if "tau_mu" in df.columns:
        df = df.rename(columns={"tau_mu": "tau_mean"})
    if "t" in df.columns:
        df = df.rename(columns={"t": "T"})
    if not {"T", "tau_mean", "alpha"}.issubset(df.columns):
        df = pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, :3]
        df.columns = ["T", "tau_mean", "alpha"]
    return df[["T", "tau_mean", "alpha"]].astype(float)


def weighted_line(x, y, w):
    """Weighted least squares y = c0 + c1*x, errors taken as ABSOLUTE 1 sigma
    (unscaled covariance — trusts the given weights, as R&C do)."""
    x, y, w = map(lambda v: np.asarray(v, float), (x, y, w))
    S   = np.sum(w)
    Sx  = np.sum(w * x)
    Sy  = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)
    D   = S * Sxx - Sx * Sx
    c1  = (S * Sxy - Sx * Sy) / D          # slope  = Ueff
    c0  = (Sxx * Sy - Sx * Sxy) / D        # intercept = A*ln10
    var_c1 = S   / D
    var_c0 = Sxx / D
    cov    = -Sx / D
    corr   = cov / np.sqrt(var_c0 * var_c1)
    # goodness of fit against the supplied errors
    resid = y - (c0 + c1 * x)
    chi2  = np.sum(w * resid ** 2)
    dof   = max(len(x) - 2, 1)
    return dict(slope=c1, intercept=c0, sd_slope=np.sqrt(var_c1),
                sd_intercept=np.sqrt(var_c0), corr=corr,
                chi2_red=chi2 / dof, n=len(x))


def orbach_spread(df, window, subtract=None):
    """Closed-form (A, sA, Ueff, sU, corr) from the alpha-weighted Orbach line.

    subtract : optional (R, N, Q) to strip the Raman+QTM rate before taking
               ln(tau), isolating the Orbach relaxation time. Inflates the
               per-point width at contaminated points accordingly.
    """
    lo, hi = window
    m = (df["T"] >= lo) & (df["T"] <= hi)
    sub = df[m].sort_values("T")
    if len(sub) < 2:
        raise ValueError(f"window {window} has < 2 points")

    T   = sub["T"].values
    tau = sub["tau_mean"].values
    a   = sub["alpha"].values
    x   = 1.0 / T
    g   = gamma_lognormal(a)                 # 1 sigma on ln(tau_observed)

    if subtract is not None:
        R, N, Q = subtract
        rate_obs   = 1.0 / tau
        rate_other = 10.0 ** R * T ** N + 10.0 ** (-Q)
        rate_orb   = rate_obs - rate_other
        if np.any(rate_orb <= 0):
            bad = T[rate_orb <= 0]
            print(f"  WARNING: Raman+QTM >= observed rate at T={bad} — "
                  f"drop these from the Orbach window.", file=sys.stderr)
        tau_orb = 1.0 / rate_orb
        # d ln tau_orb = [1/(1 - rate_other/rate_obs)] d ln tau_obs
        g = g / (1.0 - rate_other / rate_obs)
        y = np.log(tau_orb)
    else:
        y = np.log(tau)

    w  = 1.0 / g ** 2
    fit = weighted_line(x, y, w)
    return dict(A=fit["intercept"] / LN10,
                sA=fit["sd_intercept"] / LN10,
                Ueff=fit["slope"],
                sU=fit["sd_slope"],
                corr_AU=fit["corr"],
                chi2_red=fit["chi2_red"],
                n=fit["n"],
                alpha_range=(float(a.min()), float(a.max())),
                T_range=(float(T.min()), float(T.max())))


def lower_edge_sweep(df, hi, lo_grid, subtract=None):
    """Show how sU grows as you push the lower window edge down into the
    higher-alpha data — i.e. where R&C's +-100 K actually comes from."""
    rows = []
    for lo in lo_grid:
        try:
            r = orbach_spread(df, (lo, hi), subtract=subtract)
            rows.append((lo, r["T_range"][0], r["n"], r["alpha_range"][1],
                         r["Ueff"], r["sU"], r["corr_AU"], r["chi2_red"]))
        except ValueError:
            continue
    return rows


def _print(r, label=""):
    print(f"  {label}")
    print(f"    T range   : {r['T_range'][0]:.1f}-{r['T_range'][1]:.1f} K "
          f"({r['n']} pts)   alpha {r['alpha_range'][0]:.3f}-{r['alpha_range'][1]:.3f}")
    print(f"    Ueff      : {r['Ueff']:8.1f}  +/- {r['sU']:6.1f} K")
    print(f"    A         : {r['A']:8.3f}  +/- {r['sA']:6.3f}")
    print(f"    corr(A,U) : {r['corr_AU']:+.3f}   (chi2_red = {r['chi2_red']:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, help="AC data (T, tau_mu/tau_mean, alpha)")
    ap.add_argument("--orbach_window", required=True, metavar="Tlo,Thi")
    ap.add_argument("--subtract_RNQ", default=None, metavar="R,N,Q",
                    help="strip Raman+QTM using these fitted params before the "
                         "Orbach line, e.g. --subtract_RNQ=-4.76,3.81,-0.32 "
                         "(use the = form so the leading minus isn't read as a flag)")
    ap.add_argument("--sweep_to", type=float, default=None,
                    help="also sweep the lower edge down to this T, to show "
                         "where the barrier spread comes from")
    # argparse treats any token starting with '-' as a possible option, which
    # breaks --subtract_RNQ -4.76,... . Pre-join the value so the minus is safe.
    argv = sys.argv[1:]
    for i, tok in enumerate(argv[:-1]):
        if tok == "--subtract_RNQ":
            argv[i] = f"--subtract_RNQ={argv[i + 1]}"
            del argv[i + 1]
            break
    args = ap.parse_args(argv)

    df = load_tsv(args.tsv)
    lo, hi = [float(v) for v in args.orbach_window.split(",")]
    sub = None
    if args.subtract_RNQ:
        sub = tuple(float(v) for v in args.subtract_RNQ.split(","))

    print(f"\nLoaded {len(df)} rows, T {df['T'].min():.1f}-{df['T'].max():.1f} K\n")
    print("Alpha-implied Orbach spread (closed form, no MC / no rho):\n")
    _print(orbach_spread(df, (lo, hi), subtract=sub), "as-configured window:")
    if sub:
        print("    (Raman+QTM stripped before the Orbach line)")

    if args.sweep_to is not None:
        grid = sorted(set(df["T"][(df["T"] >= args.sweep_to) & (df["T"] < hi)].values))
        rows = lower_edge_sweep(df, hi, grid, subtract=sub)
        print(f"\n  Lower-edge sweep (Thi fixed at {hi:.0f} K):")
        print(f"    {'Tlo':>6} {'npts':>5} {'amax':>6} {'Ueff':>8} {'sU':>7} "
              f"{'corr':>6} {'chi2r':>6}")
        for lo_, tmin, n, amax, U, sU, corr, chi2 in rows:
            print(f"    {tmin:6.1f} {n:5d} {amax:6.3f} {U:8.1f} {sU:7.1f} "
                  f"{corr:+6.2f} {chi2:6.2f}")


if __name__ == "__main__":
    main()