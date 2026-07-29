#!/usr/bin/env python3
"""
ac_pipeline.py  –  TAURnQ AC phases 2 + 3 merged into one command.

Runs the per-temperature moment fits (phase 2) and the Monte Carlo
distribution fits (phase 3) back-to-back, passing phase-2 priors —
means, spreads AND correlations, as complete correlated blocks — directly
into phase 3 in memory. No intermediate CSV, no hand-copied --pr_* flags.

    raw AC data (T, tau_mu, alpha)
          │
          ▼  phase 2: fit Orbach / Raman / QTM moments per window
    priors {A,sA,Ueff,sU,rho_AU, R,n,sR,sN,rho_RN, Q,sQ} + windows
          │
          ▼  phase 3: per-row Monte Carlo, cross-terms sampled from priors
    ac_mc_params.csv   (schema unchanged: T, Au,Uu,Ru,Nu,Qu, As,Us,Rs,Ns,Qs, …)

USAGE
    # full pipeline (phase 2 → phase 3)
    python ac_pipeline.py --infile tBuOCl.tsv \\
        --orbach_window 45,58 --raman_window 30,34 --qtm_window 9.7,12

    # stop after phase 2 (write the phase-2 summary, skip the Monte Carlo)
    python ac_pipeline.py --infile tBuOCl.tsv --orbach_window 45,58 \\
        --phase2_only

    # override any single phase-2 prior going into phase 3
    python ac_pipeline.py --infile tBuOCl.tsv --orbach_window 45,58 \\
        --raman_window 30,34 --pr_sR 0.3 --pr_sN 0.2 --pr_rho_RN -0.9

The phase-2 and phase-3 mathematics live in ac_phase2.py and
ac_montecarlo.py; this script imports and orchestrates them, so there is a
single source of truth for every formula. Both remain runnable standalone.
"""

import numpy as np
import pandas as pd
import argparse, os, sys

import ac_phase2 as p2
import ac_montecarlo as p3


# ---------------------------------------------------------------------------
# Phase 2 → priors/windows, entirely in memory
# ---------------------------------------------------------------------------

def run_phase2(df, windows, w_mu=1.0, w_sd=1.0, verbose=True):
    """Fit Orbach / Raman / QTM moment models within their windows.

    Returns (priors, phase2_row):
      priors     : dict over {A,sA,Ueff,sU,rho_AU, R,n,sR,sN,rho_RN, Q,sQ},
                   only for processes whose window was set and fitted.
                   Correlated blocks are kept intact (rho beside its pair).
      phase2_row : dict for the phase-2 summary CSV (same schema as the
                   standalone ac_phase2.py output, incl. sd_eff columns).
    """
    priors, row = {}, {}

    if windows['orbach_w']:
        dfw = p2.slice_window(df, windows['orbach_w'])
        if len(dfw) >= 2:
            r = p2.fit_orbach(dfw, w_mu, w_sd, verbose=verbose)
            priors.update({'A': r['mu_A'], 'sA': r['sd_A'],
                           'Ueff': r['mu_U'], 'sU': r['sd_U'],
                           'rho_AU': r['rho_AU']})
            row.update({'mu_A': r['mu_A'], 'mu_U': r['mu_U'],
                        'sd_A': r['sd_A'], 'sd_U': r['sd_U'],
                        'rho_AU': r['rho_AU'],
                        'sd_A_lo': r.get('sd_A_lo', np.nan),
                        'sd_A_hi': r.get('sd_A_hi', np.nan),
                        'sd_U_lo': r.get('sd_U_lo', np.nan),
                        'sd_U_hi': r.get('sd_U_hi', np.nan),
                        'sd_eff_orbach': r.get('sd_eff', np.nan)})
        elif verbose:
            print(f"  Orbach window {windows['orbach_w']} has <2 rows — skipped.",
                  file=sys.stderr)

    if windows['raman_w']:
        dfw = p2.slice_window(df, windows['raman_w'])
        if len(dfw) >= 2:
            r = p2.fit_raman(dfw, w_mu, w_sd, verbose=verbose)
            priors.update({'R': r['mu_R'], 'n': r['mu_N'],
                           'sR': r['sd_R'], 'sN': r['sd_N'],
                           'rho_RN': r['rho_RN']})
            row.update({'mu_R': r['mu_R'], 'mu_N': r['mu_N'],
                        'sd_R': r['sd_R'], 'sd_N': r['sd_N'],
                        'rho_RN': r['rho_RN'],
                        'sd_R_lo': r.get('sd_R_lo', np.nan),
                        'sd_R_hi': r.get('sd_R_hi', np.nan),
                        'sd_N_lo': r.get('sd_N_lo', np.nan),
                        'sd_N_hi': r.get('sd_N_hi', np.nan),
                        'sd_eff_raman': r.get('sd_eff', np.nan)})
        elif verbose:
            print(f"  Raman window {windows['raman_w']} has <2 rows — skipped.",
                  file=sys.stderr)

    if windows['qtm_w']:
        dfw = p2.slice_window(df, windows['qtm_w'])
        if len(dfw) >= 2:
            r = p2.fit_qtm(dfw, w_mu, w_sd, verbose=verbose)
            priors.update({'Q': r['mu_Q'], 'sQ': r['sd_Q']})
            row.update({'mu_Q': r['mu_Q'], 'sd_Q': r['sd_Q']})
        elif verbose:
            print(f"  QTM window {windows['qtm_w']} has <2 rows — skipped.",
                  file=sys.stderr)

    # window metadata (dash-separated to keep the CSV comma-clean)
    for name, key in [('orbach', 'orbach_w'), ('raman', 'raman_w'),
                      ('qtm', 'qtm_w')]:
        w = windows[key]
        row[f'{name}_window'] = f"{w[0]}-{w[1]}" if w else ""
    return priors, row


def apply_cli_overrides(priors, args):
    """Overlay explicit --pr_* flags on top of phase-2 priors, warning when a
    spread is overridden without its correlation (the ridge-degeneracy trap)."""
    cli = {'A': args.pr_A, 'Ueff': args.pr_Ueff, 'sA': args.pr_sA,
           'sU': args.pr_sU, 'R': args.pr_R, 'n': args.pr_n,
           'sR': args.pr_sR, 'sN': args.pr_sN, 'Q': args.pr_Q,
           'sQ': args.pr_sQ, 'rho_AU': args.pr_rho_AU,
           'rho_RN': args.pr_rho_RN}
    src = {k: ('phase2' if k in priors else None) for k in cli}
    for k, v in cli.items():
        if v is not None:
            priors[k], src[k] = float(v), 'cli'

    for rk, sks, label in [('rho_RN', ('sR', 'sN'), 'Raman'),
                           ('rho_AU', ('sA', 'sU'), 'Orbach')]:
        if any(src.get(s) == 'cli' for s in sks) and src.get(rk) != 'cli':
            rho_p2 = priors.get(rk, 0.0)
            if abs(rho_p2) > 0.5:
                print(f"  WARNING: {label} spreads overridden on the CLI but "
                      f"--pr_{rk} was not — phase 2 fitted these as a "
                      f"correlated triple ({rk} = {rho_p2:+.3f}). Overriding "
                      f"spreads without the correlation changes the induced "
                      f"width drastically.")

    if 'rho_AU' in priors:
        priors['rho_AU'] = float(np.clip(priors['rho_AU'], 0.0, 0.999))
    if 'rho_RN' in priors:
        priors['rho_RN'] = float(np.clip(priors['rho_RN'], -0.999, 0.0))
    return priors, src


def print_priors(priors, src, windows):
    def f(k):
        return f"{priors[k]:.4f} [{src.get(k) or 'phase2'}]" if k in priors else "absent"
    print("\nPriors carried into phase 3 (value [source]):")
    if 'A' in priors and 'Ueff' in priors:
        print(f"  A    = {f('A')}    sA = {f('sA')}")
        print(f"  Ueff = {f('Ueff')}    sU = {f('sU')}    rho_AU = {f('rho_AU')}")
    else:
        print("  Orbach (A, Ueff): absent")
    if 'R' in priors and 'n' in priors:
        print(f"  R    = {f('R')}    sR = {f('sR')}")
        print(f"  n    = {f('n')}    sN = {f('sN')}    rho_RN = {f('rho_RN')}")
    else:
        print("  Raman (R, n): absent")
    print(f"  Q    = {f('Q')}" + (f"    sQ = {f('sQ')}" if 'Q' in priors else ""))
    for name, key in [('orbach', 'orbach_w'), ('raman', 'raman_w'),
                      ('qtm', 'qtm_w')]:
        w = windows[key]
        print(f"  {name:>6} window: {f'{w[0]:g}-{w[1]:g} K' if w else '—'}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ AC pipeline — phases 2 + 3 in one command")

    ap.add_argument("--infile", required=True,
                    help="Raw AC data TSV/CSV (T, tau_mu/tau_mean, alpha)")
    ap.add_argument("--phase2_only", action="store_true",
                    help="Run phase 2 only; write the moment-fit summary and "
                         "stop before the Monte Carlo.")
    ap.add_argument("--p2_out", default="ac_phase2_out.csv",
                    help="Phase-2 summary CSV (also written in the full run)")
    ap.add_argument("--out", default="ac_mc_params.csv",
                    help="Phase-3 per-row Monte Carlo output CSV")
    ap.add_argument("--clear", action="store_true",
                    help="Truncate --out before writing (phase 3)")
    ap.add_argument("--seed_base", type=int, default=14322)
    ap.add_argument("--plot", action="store_true",
                    help="Phase-3 diagnostic plot (single-row runs only)")

    win = ap.add_argument_group("Temperature windows")
    win.add_argument("--orbach_window", default="", metavar="Tlo,Thi")
    win.add_argument("--raman_window",  default="", metavar="Tlo,Thi")
    win.add_argument("--qtm_window",    default="", metavar="Tlo,Thi")

    p2g = ap.add_argument_group("Phase-2 fit weights")
    p2g.add_argument("--w_mu", type=float, default=1.0)
    p2g.add_argument("--w_sd", type=float, default=1.0)

    pr = ap.add_argument_group(
        "Phase-3 prior overrides (optional)",
        "By default every phase-3 prior comes from the phase-2 fit above. "
        "Set any of these to override a single value; spreads and their "
        "correlations are safest overridden together.")
    pr.add_argument("--pr_A",    type=float, default=None)
    pr.add_argument("--pr_Ueff", type=float, default=None)
    pr.add_argument("--pr_sA",   type=float, default=None)
    pr.add_argument("--pr_sU",   type=float, default=None)
    pr.add_argument("--pr_R",    type=float, default=None)
    pr.add_argument("--pr_n",    type=float, default=None)
    pr.add_argument("--pr_sR",   type=float, default=None)
    pr.add_argument("--pr_sN",   type=float, default=None)
    pr.add_argument("--pr_Q",    type=float, default=None)
    pr.add_argument("--pr_sQ",   type=float, default=None)
    pr.add_argument("--pr_rho_AU", type=float, default=None)
    pr.add_argument("--pr_rho_RN", type=float, default=None)
    pr.add_argument("--fix_cross", action="store_true",
                    help="Phase 3: fix cross terms at their means instead of "
                         "sampling with prior spreads (legacy behaviour).")

    args = ap.parse_args()

    windows = {
        'orbach_w': p2.parse_window(args.orbach_window),
        'raman_w':  p2.parse_window(args.raman_window),
        'qtm_w':    p2.parse_window(args.qtm_window),
    }
    if not any(windows.values()):
        print("ERROR: set at least one of --orbach_window / --raman_window / "
              "--qtm_window.", file=sys.stderr)
        sys.exit(1)

    # ── load raw data ────────────────────────────────────────────────────────
    df = p2.load_data(args.infile)
    print(f"Loaded {len(df)} rows from {args.infile}  "
          f"(T {df['T'].min():.1f}–{df['T'].max():.1f} K)\n")

    # ═══ PHASE 2 ═════════════════════════════════════════════════════════════
    print("=" * 62)
    print("  PHASE 2 — per-temperature moment fits")
    print("=" * 62)
    priors, p2_row = run_phase2(df, windows, args.w_mu, args.w_sd, verbose=True)
    p2_row['infile'] = args.infile
    pd.DataFrame([p2_row]).to_csv(args.p2_out, index=False)
    print(f"\n  Phase-2 summary → {os.path.abspath(args.p2_out)}")

    if args.phase2_only:
        print("\n--phase2_only set — stopping before the Monte Carlo.")
        return

    # ── overlay CLI overrides, keeping correlated blocks intact ──────────────
    priors, src = apply_cli_overrides(priors, args)
    print_priors(priors, src, windows)

    # ═══ PHASE 3 ═════════════════════════════════════════════════════════════
    print("=" * 62)
    print("  PHASE 3 — per-row Monte Carlo")
    print("=" * 62)

    # inject window-flag columns so phase-3 dispatch matches the phase-2 windows
    for col, key in [("in_orbach_window", "orbach_w"),
                     ("in_raman_window",  "raman_w"),
                     ("in_qtm_window",    "qtm_w")]:
        w = windows[key]
        df[col] = (df["T"].between(w[0], w[1]) if w else
                   pd.Series(False, index=df.index))

    n_o = int(df["in_orbach_window"].sum())
    n_r = int(df["in_raman_window"].sum())
    n_q = int(df["in_qtm_window"].sum())
    any_w = df["in_orbach_window"] | df["in_raman_window"] | df["in_qtm_window"]
    print(f"Rows: {len(df)} total  |  Orbach: {n_o}  Raman: {n_r}  QTM: {n_q}  "
          f"Skip: {len(df) - int(any_w.sum())}\n")
    if n_o or n_r:
        print("NOTE: per-row (mean, sd) pairs at a SINGLE temperature are "
              "degenerate — only the location and total width of ln(tau) at "
              "that T are identified. Per-row values are anchors; the global "
              "stage does the decomposition across the full T range.\n")

    if args.clear and os.path.exists(args.out):
        open(args.out, "w").close()

    make_plot = args.plot and (len(df) == 1)
    for idx in range(len(df)):
        try:
            p3.fit_one_row(df, idx, args.out, priors, windows,
                           seed_base=args.seed_base, make_plot=make_plot,
                           fix_cross=args.fix_cross)
        except Exception as e:
            print(f"[idx={idx}] ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Phase-3 results → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
