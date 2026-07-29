#!/usr/bin/env python3
"""
dc_pipeline.py  –  TAURnQ DC phases 2 + 3 merged into one command.

Runs the two-piece-normal / exact-SEF moment fits (phase 2) and the Monte
Carlo distribution fits (phase 3) back-to-back. Phase-2 compiled priors
(Q, and R/n when estimable) pass straight into phase 3 in memory; optional
AC priors from ac_phase2 / ac_pipeline fill in the Orbach (and Raman) blocks.
No intermediate CSV round-trip, no hand-copied --pr_* flags.

    raw DC data (T, tau_star, beta)
          │
          ▼  phase 2: exact SEF density → two-piece-normal + exact quantiles,
          │            compiled Q (+ R/n) priors
          ▼  phase 3: per-row Monte Carlo against exact SEF quantile targets
    dc_mc_params.csv   (schema unchanged; Au/Uu/As/Us always NaN)

USAGE
    # full pipeline (phase 2 → phase 3), DC-only
    python dc_pipeline.py --infile dc_data.tsv --qtm_window 2,9 --raman_window 13,23

    # with AC-side priors (Orbach term in the rate model)
    python dc_pipeline.py --infile dc_data.tsv --qtm_window 2,9 \\
        --ac_phase2 ac_phase2_out.csv

    # stop after phase 2 (write the SEF/TPN summary, skip the Monte Carlo)
    python dc_pipeline.py --infile dc_data.tsv --qtm_window 2,9 --phase2_only

The phase-2 and phase-3 mathematics live in dc_phase2.py and
dc_montecarlo.py; this script imports and orchestrates them (single source
of truth). Both remain runnable standalone.
"""

import numpy as np
import pandas as pd
import argparse, os, sys

import dc_phase2 as p2
import dc_montecarlo as p3


def _parse_window(s):
    if not s or str(s).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must be 'Tlo,Thi'")
    return (min(parts), max(parts))


def collect_priors(p2_df, acp2, cli):
    """Resolve phase-3 priors group-wise (own modality first), keeping each
    correlated block from a single source. Mirrors dc_montecarlo's logic but
    reads the phase-2 DataFrame in memory instead of a re-loaded CSV.

    p2_df : the DataFrame returned by dc_phase2.process_file (carries the
            compiled pr_* columns).
    acp2  : dict of AC priors (or {}).
    cli   : dict of explicit --pr_* overrides (values or None).
    """
    infile_pri = {}
    for key, col in [('Q', 'pr_Q'), ('sQ', 'pr_sQ'), ('R', 'pr_R'),
                     ('n', 'pr_n'), ('sR', 'pr_sR'), ('sN', 'pr_sN')]:
        if col in p2_df.columns:
            v = pd.to_numeric(p2_df[col], errors="coerce").dropna()
            if len(v):
                infile_pri[key] = float(v.iloc[0])

    DEFAULTS = {'sA': 0.15, 'sU': 20.0, 'sR': 0.35, 'sN': 0.25, 'sQ': 0.21,
                'rho_AU': 0.0, 'rho_RN': 0.0}
    priors, src = {}, {}

    def take(key, *cands):
        for val, s in cands:
            if val is not None:
                priors[key], src[key] = float(val), s
                return
        if key in DEFAULTS:
            priors[key], src[key] = DEFAULTS[key], 'default'

    # Q group — dc_phase2 compiled beats ac_phase2
    q_d, q_s = ((infile_pri, 'dc_phase2') if 'Q' in infile_pri
                else (acp2, 'ac_phase2'))
    take('Q',  (cli['Q'],  'cli'), (q_d.get('Q'),  q_s))
    take('sQ', (cli['sQ'], 'cli'), (q_d.get('sQ'), q_s))

    # Raman group — one base source; rho only travels with ac_phase2
    if 'R' in infile_pri and 'n' in infile_pri:
        rn_d, rn_s = infile_pri, 'dc_phase2'
    elif 'R' in acp2 and 'n' in acp2:
        rn_d, rn_s = acp2, 'ac_phase2'
    else:
        rn_d, rn_s = {}, None
    for k in ('R', 'n', 'sR', 'sN'):
        take(k, (cli[k], 'cli'), (rn_d.get(k), rn_s))
    take('rho_RN', (cli['rho_RN'], 'cli'),
         (rn_d.get('rho_RN') if rn_s == 'ac_phase2' else None, 'ac_phase2'))

    # Orbach group — AC only
    for k in ('A', 'Ueff', 'sA', 'sU'):
        take(k, (cli[k], 'cli'), (acp2.get(k), 'ac_phase2'))
    take('rho_AU', (cli['rho_AU'], 'cli'), (acp2.get('rho_AU'), 'ac_phase2'))

    priors['rho_AU'] = float(np.clip(priors.get('rho_AU', 0.0),  0.0, 0.999))
    priors['rho_RN'] = float(np.clip(priors.get('rho_RN', 0.0), -0.999, 0.0))

    for rk, sks, label in [('rho_RN', ('sR', 'sN'), 'Raman'),
                           ('rho_AU', ('sA', 'sU'), 'Orbach')]:
        if any(src.get(s) == 'cli' for s in sks) and src.get(rk) != 'cli' \
                and abs(acp2.get(rk, 0.0)) > 0.5:
            print(f"  WARNING: {label} spreads overridden on the CLI but "
                  f"--pr_{rk} was not — ac_phase2 fitted these as a correlated "
                  f"triple ({rk} = {acp2[rk]:+.3f}). Overriding spreads without "
                  f"the correlation changes the induced width drastically.")
    return priors, src


def print_priors(priors, src):
    def f(k):
        return f"{priors[k]:.4f} [{src.get(k, '?')}]" if k in priors else "absent"
    print("\nPriors carried into phase 3 (value [source]):")
    if 'A' in priors and 'Ueff' in priors:
        print(f"  A    = {f('A')}    sA = {f('sA')}")
        print(f"  Ueff = {f('Ueff')}    sU = {f('sU')}    rho_AU = {f('rho_AU')}")
    else:
        print("  Orbach (A, Ueff): absent — term dropped from model")
    if 'R' in priors and 'n' in priors:
        print(f"  R    = {f('R')}    sR = {f('sR')}")
        print(f"  n    = {f('n')}    sN = {f('sN')}    rho_RN = {f('rho_RN')}")
    else:
        print("  Raman (R, n): absent — term dropped from QTM rows")
    print(f"  Q    = {f('Q')}" + (f"    sQ = {f('sQ')}" if 'Q' in priors else ""))
    print()


def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ DC pipeline — phases 2 + 3 in one command")

    ap.add_argument("--infile", required=True,
                    help="Raw DC data TSV/CSV (T, tau_star, beta)")
    ap.add_argument("--phase2_only", action="store_true",
                    help="Run phase 2 only; write the SEF/TPN summary and stop.")
    ap.add_argument("--p2_out", default="dc_phase2_out.csv",
                    help="Phase-2 summary CSV (also written in the full run)")
    ap.add_argument("--out", default="dc_mc_params.csv",
                    help="Phase-3 per-row Monte Carlo output CSV")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--seed_base", type=int, default=14322)
    ap.add_argument("--plot", action="store_true")

    win = ap.add_argument_group("Temperature windows")
    win.add_argument("--qtm_window",   default="", metavar="Tlo,Thi")
    win.add_argument("--raman_window", default="", metavar="Tlo,Thi")

    pr = ap.add_argument_group(
        "Phase-3 priors (optional)",
        "Q (and R/n when estimable) come from phase 2 automatically. "
        "--ac_phase2 supplies the AC Orbach/Raman blocks with their "
        "correlations. Individual --pr_* flags override single values.")
    pr.add_argument("--ac_phase2", default=None,
                    help="ac_phase2 / ac_pipeline output CSV — AC priors")
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

    qtm_w   = _parse_window(args.qtm_window)
    raman_w = _parse_window(args.raman_window)
    if not (qtm_w or raman_w):
        print("ERROR: set at least one of --qtm_window / --raman_window.",
              file=sys.stderr)
        sys.exit(1)

    # ═══ PHASE 2 ═════════════════════════════════════════════════════════════
    print("=" * 62)
    print("  PHASE 2 — SEF density → two-piece-normal + exact quantiles")
    print("=" * 62)
    p2_df = p2.process_file(args.infile, args.p2_out,
                            make_plot=False,
                            qtm_window=qtm_w, raman_window=raman_w)
    print(f"\n  Phase-2 summary → {os.path.abspath(args.p2_out)}")

    if args.phase2_only:
        print("\n--phase2_only set — stopping before the Monte Carlo.")
        return

    # ── AC priors (optional) ─────────────────────────────────────────────────
    acp2 = {}
    if args.ac_phase2:
        acp2 = p3.load_ac_phase2(args.ac_phase2)
        print(f"\nLoaded AC priors from {args.ac_phase2}")

    cli = {'A': args.pr_A, 'Ueff': args.pr_Ueff, 'sA': args.pr_sA,
           'sU': args.pr_sU, 'R': args.pr_R, 'n': args.pr_n,
           'sR': args.pr_sR, 'sN': args.pr_sN, 'Q': args.pr_Q,
           'sQ': args.pr_sQ, 'rho_AU': args.pr_rho_AU,
           'rho_RN': args.pr_rho_RN}
    priors, src = collect_priors(p2_df, acp2, cli)
    print_priors(priors, src)

    # ═══ PHASE 3 ═════════════════════════════════════════════════════════════
    print("=" * 62)
    print("  PHASE 3 — per-row Monte Carlo (exact SEF quantile targets)")
    print("=" * 62)

    df = p2_df.reset_index(drop=True)
    # process_file already set in_qtm_window / in_raman_window
    n_q = int(df["in_qtm_window"].sum())
    n_r = int(df["in_raman_window"].sum())
    n_skip = len(df) - int((df["in_qtm_window"] | df["in_raman_window"]).sum())
    print(f"Rows: {len(df)} total  |  QTM: {n_q}  Raman: {n_r}  Skip: {n_skip}\n")
    if n_r:
        print("NOTE: per-row (R, n, sR, sN) at a SINGLE temperature are "
              "degenerate — only R + n*log10(T) and the total width are "
              "identified. Use dc_phase2's slope-based compiled R/n and the "
              "global stage for the decomposition.\n")

    if args.clear and os.path.exists(args.out):
        open(args.out, "w").close()

    make_plot = args.plot and (len(df) == 1)
    for idx in range(len(df)):
        try:
            p3.fit_one_row(df, idx, args.out, priors,
                           seed_base=args.seed_base, make_plot=make_plot,
                           fix_cross=args.fix_cross)
        except Exception as e:
            print(f"[idx={idx}] ERROR: {e}", file=sys.stderr)

    print(f"\nDone. Phase-3 results → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
