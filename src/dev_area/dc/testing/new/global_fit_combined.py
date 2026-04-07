#!/usr/bin/env python3
"""
global_fit_combined.py  -  TAURnQ phase 4 for combined AC + DC data.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import argparse, os, sys

SQRT2PI = np.sqrt(2.0 / np.pi)

# quantile targets
def fk_ln_quantiles(tau_mean, alpha, qs):
    g = 1.82 * np.sqrt(alpha) / (1.0 - alpha)
    return np.log(tau_mean) + norm.ppf(qs) * g

def tpn_ln_quantiles(mu_ln, s1, s2, qs):
    p_left = s1 / (s1 + s2)
    qs = np.asarray(qs, dtype=float)
    result = np.empty_like(qs)
    for i, q in enumerate(qs):
        if q <= p_left:
            result[i] = mu_ln + norm.ppf(q*(s1+s2)/(2.*s1)) * s1
        else:
            result[i] = mu_ln + norm.ppf(1.-(1.-q)*(s1+s2)/(2.*s2)) * s2
    return result

# rate model + MC
def rate_model(T, A, Ueff, R, N, Q):
    Ts = np.maximum(T, 1e-300)
    return (10.**(-A)*np.exp(-Ueff/np.maximum(Ts,1e-12))
            + 10.**R * Ts**N + 10.**(-Q))

def draw_params(mu, sigmas, rho_AU, rho_RN, Z):
    A,Ueff,R,N,Q = mu
    sA,sU,sR,sN,sQ = sigmas
    L_AU = np.array([[sA,0.],[rho_AU*sU, sU*np.sqrt(max(1-rho_AU**2,1e-12))]])
    L_RN = np.array([[sR,0.],[rho_RN*sN, sN*np.sqrt(max(1-rho_RN**2,1e-12))]])
    eps_AU = Z[:,:2] @ L_AU.T
    eps_RN = Z[:,2:4] @ L_RN.T
    return np.column_stack([A+eps_AU[:,0], Ueff+eps_AU[:,1],
                             R+eps_RN[:,0], N+eps_RN[:,1], Q+Z[:,4]*sQ])

def mc_quantiles(T, mu, sigmas, rho_AU, rho_RN, qs, Z):
    th = draw_params(mu, sigmas, rho_AU, rho_RN, Z)
    r = rate_model(T, *th.T)
    return np.quantile(np.log(1./np.maximum(r,1e-300)), qs)

def unpack(x):
    mu     = x[0:5]
    sigmas = 1e-3 + np.exp(x[5:10])
    rho_AU = np.tanh(x[10])**2
    rho_RN = -(np.tanh(x[11])**2)
    return mu, sigmas, rho_AU, rho_RN

# window helpers
def parse_window(s):
    if not s or s.strip()=="": return None
    lo,hi = [float(v.strip()) for v in s.split(",")]
    return (min(lo,hi), max(lo,hi))

def in_window(T_arr, window):
    if window is None: return np.ones(len(T_arr), dtype=bool)
    lo,hi = window
    return (np.asarray(T_arr)>=lo) & (np.asarray(T_arr)<=hi)

def ivw_mean(vals, sds):
    v,s = np.asarray(vals,float), np.asarray(sds,float)
    w = 1./np.maximum(s**2,1e-12)
    return float(np.nansum(w*v)/np.maximum(np.nansum(w),1e-12))

def robust_med(vals):
    v = np.asarray(vals,float); v = v[np.isfinite(v)]
    return float(np.nanmedian(v)) if v.size>0 else np.nan

def compile_window(df, mask, which):
    sub = df[mask]
    if sub.empty: return None
    if which=="AU":
        return ((ivw_mean(sub["Au"],sub["As"]), ivw_mean(sub["Uu"],sub["Us"])),
                (robust_med(sub["As"]),         robust_med(sub["Us"])))
    elif which=="RN":
        return ((ivw_mean(sub["Ru"],sub["Rs"]), ivw_mean(sub["Nu"],sub["Ns"])),
                (robust_med(sub["Rs"]),         robust_med(sub["Ns"])))
    elif which=="Q":
        return ((ivw_mean(sub["Qu"],sub["Qs"]),),
                (robust_med(sub["Qs"]),))

def build_initials(df_ac, df_dc, AU_w, RN_w, Q_w):
    def _c(df, w, which):
        if df is None or df.empty: return None
        mask = in_window(df["T"].values, w)
        if not mask.any(): mask = np.ones(len(df),dtype=bool)
        return compile_window(df, mask, which)

    au = _c(df_ac,AU_w,"AU") or _c(df_dc,AU_w,"AU")
    rn = _c(df_ac,RN_w,"RN") or _c(df_dc,RN_w,"RN")
    q  = _c(df_dc,Q_w, "Q")  or _c(df_ac,Q_w, "Q")

    def safe(res,n):
        if res is None: return tuple([0.]*n), tuple([1.]*n)
        return res

    (A0,U0),(sA0,sU0) = safe(au,2)
    (R0,N0),(sR0,sN0) = safe(rn,2)
    (Q0,),  (sQ0,)    = safe(q, 1)
    clamp = lambda v: max(v if np.isfinite(v) else 1e-3, 1e-3)
    sA0,sU0,sR0,sN0,sQ0 = map(clamp,[sA0,sU0,sR0,sN0,sQ0])

    frames = [d for d in [df_ac,df_dc] if d is not None]
    df_all = pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    def safe_corr(c1,c2,fb=0.):
        if df_all.empty or c1 not in df_all or c2 not in df_all: return fb
        x,y = df_all[c1].values, df_all[c2].values
        m = np.isfinite(x)&np.isfinite(y)
        return float(np.corrcoef(x[m],y[m])[0,1]) if m.sum()>=3 else fb

    rAU = np.clip( safe_corr("Au","Uu"), 0., 0.95)
    rRN = np.clip(-abs(safe_corr("Ru","Nu")), -0.95, 0.)

    x0 = np.array([A0,U0,R0,N0,Q0,
                   np.log(max(sA0,1e-6)), np.log(max(sU0,1e-6)),
                   np.log(max(sR0,1e-6)), np.log(max(sN0,1e-6)),
                   np.log(max(sQ0,1e-6)),
                   np.arctanh(np.sqrt(np.clip(rAU,     0,0.99))),
                   np.arctanh(np.sqrt(np.clip(abs(rRN),0,0.99)))],
                  dtype=float)
    compiled = {
        "AU": {"mean":np.array([A0,U0]),  "sd":np.array([sA0,sU0])},
        "RN": {"mean":np.array([R0,N0]),  "sd":np.array([sR0,sN0])},
        "Q":  {"mean":np.array([Q0]),      "sd":np.array([sQ0])},
    }
    return x0, compiled

# penalties
def penalty(mu, sigmas, rho_AU, rho_RN, ridge_Q=0.):
    A,Ueff,R,N,Q = mu
    pen = 0.
    def qout(v,lo,hi,sc):
        if v<lo: return sc*(lo-v)**2
        if v>hi: return sc*(v-hi)**2
        return 0.
    pen += qout(-A,  0, 30,  1e-3)
    pen += qout(Ueff,0,3000, 1e-6)
    pen += qout(R, -20, 10,  1e-3)
    pen += qout(N,   0, 12,  1e-3)
    pen += qout(Q, -20, 10,  1e-3)
    pen += 1e-5*float(np.sum(sigmas**2))
    for r in (rho_AU,rho_RN):
        if abs(r)>0.995: pen += 1e-4*(abs(r)-0.995)**2
    if ridge_Q>0: pen += ridge_Q*Q**2
    return pen

def domain_reg(mu, compiled, lam_AU, lam_RN, lam_Q, eps=1e-6):
    A,Ueff,R,N,Q = mu
    pen = 0.
    if lam_AU>0:
        tgt,sd = compiled["AU"]["mean"], np.maximum(compiled["AU"]["sd"],eps)
        pen += lam_AU*((A-tgt[0])**2/sd[0]**2 + (Ueff-tgt[1])**2/sd[1]**2)
    if lam_RN>0:
        tgt,sd = compiled["RN"]["mean"], np.maximum(compiled["RN"]["sd"],eps)
        pen += lam_RN*((R-tgt[0])**2/sd[0]**2 + (N-tgt[1])**2/sd[1]**2)
    if lam_Q>0:
        tgt,sd = compiled["Q"]["mean"], np.maximum(compiled["Q"]["sd"],eps)
        pen += lam_Q*(Q-tgt[0])**2/sd[0]**2
    return float(pen)

# combined objective
def objective(x, ac_rows, dc_rows, qs, Z, compiled,
              lam_AU, lam_RN, lam_Q, ridge_Q=0.):
    mu,sigmas,rho_AU,rho_RN = unpack(x)
    total, n = 0., 0
    for (T,tau_mean,alpha) in ac_rows:
        lnq = mc_quantiles(T,mu,sigmas,rho_AU,rho_RN,qs,Z)
        res = lnq - fk_ln_quantiles(tau_mean,alpha,qs)
        total += float(np.dot(res,res)); n += len(qs)
    for (T,mu_ln,s1,s2) in dc_rows:
        lnq = mc_quantiles(T,mu,sigmas,rho_AU,rho_RN,qs,Z)
        res = lnq - tpn_ln_quantiles(mu_ln,s1,s2,qs)
        total += float(np.dot(res,res)); n += len(qs)
    loss  = total/max(n,1)
    loss += penalty(mu,sigmas,rho_AU,rho_RN,ridge_Q=ridge_Q)
    loss += domain_reg(mu,compiled,lam_AU,lam_RN,lam_Q)
    return loss

# skewness
def tpn_skewness(s1, s2):
    Var = (1.-2./np.pi)*(s2-s1)**2 + s1*s2
    std = np.sqrt(max(Var,1e-30))
    m3  = SQRT2PI*(s2-s1)*((4./np.pi-1.)*Var + s1*s2*(1.-2./np.pi))
    return float(m3/std**3)

def compute_skewness(dc_phase2_path, Q_w, RN_w):
    null = {p:0. for p in ["A","Ueff","R","N","Q"]}
    if dc_phase2_path is None or not os.path.exists(dc_phase2_path):
        return null
    df = pd.read_csv(dc_phase2_path)
    if not {"T","sigma1_ln","sigma2_ln"}.issubset(df.columns):
        return null
    def _wskew(w):
        sub = df[in_window(df["T"].values,w)] if w else df
        if sub.empty: return 0.
        return float(np.mean([tpn_skewness(r["sigma1_ln"],r["sigma2_ln"])
                               for _,r in sub.iterrows()]))
    return {"A":0.,"Ueff":0.,"R":_wskew(RN_w),"N":_wskew(RN_w),"Q":_wskew(Q_w)}

# data loaders
def load_ac(params_csv, tsv_path, fit_window=None):
    dfp = pd.read_csv(params_csv)
    dft = pd.read_csv(tsv_path,sep=None,engine="python",header=None).iloc[:,:3]
    dft.columns = ["T","tau_mu","alpha"]; dft = dft.astype(float)
    shared = set(dfp["T"].values)&set(dft["T"].values)
    dfp = dfp[dfp["T"].isin(shared)].sort_values("T").reset_index(drop=True)
    dft = dft[dft["T"].isin(shared)].sort_values("T").reset_index(drop=True)
    rows = [(r["T"],r["tau_mu"],r["alpha"]) for _,r in dft.iterrows()]
    if fit_window:
        rows = [(T,tm,a) for T,tm,a in rows if fit_window[0]<=T<=fit_window[1]]
    return dfp, rows

def load_dc(params_csv, phase2_csv, fit_window=None):
    dfp  = pd.read_csv(params_csv)
    dfp2 = pd.read_csv(phase2_csv)
    shared = set(dfp["T"].values)&set(dfp2["T"].values)
    dfp  = dfp[dfp["T"].isin(shared)].sort_values("T").reset_index(drop=True)
    dfp2 = dfp2[dfp2["T"].isin(shared)].sort_values("T").reset_index(drop=True)
    rows = [(r["T"],r["mu_ln"],r["sigma1_ln"],r["sigma2_ln"])
            for _,r in dfp2.iterrows()]
    if fit_window:
        rows = [(T,m,s1,s2) for T,m,s1,s2 in rows
                if fit_window[0]<=T<=fit_window[1]]
    return dfp, rows

# main
def main():
    ap = argparse.ArgumentParser(
        description="TAURnQ phase 4 - combined AC + DC global fit with skewness")
    ap.add_argument("--ac_params",     default=None)
    ap.add_argument("--dc_params",     default=None)
    ap.add_argument("--ac_tsv",        default=None)
    ap.add_argument("--dc_phase2",     default=None)
    ap.add_argument("--AU_window",     default="")
    ap.add_argument("--RN_window",     default="")
    ap.add_argument("--Q_window",      default="")
    ap.add_argument("--ac_fit_window", default="")
    ap.add_argument("--dc_fit_window", default="")
    ap.add_argument("--lambda_AU",     type=float, default=1.0)
    ap.add_argument("--lambda_RN",     type=float, default=1.0)
    ap.add_argument("--lambda_Q",      type=float, default=1.0)
    ap.add_argument("--dc_Q_boost",    type=float, default=3.0,
                    help="Multiply lambda_Q when DC data present (default 3)")
    ap.add_argument("--ac_AU_boost",   type=float, default=3.0,
                    help="Multiply lambda_AU when AC data present (default 3)")
    ap.add_argument("--K",             type=int,   default=30000)
    ap.add_argument("--seed",          type=int,   default=202)
    ap.add_argument("--maxiter",       type=int,   default=1200)
    ap.add_argument("--ridgeQ",        type=float, default=0.0)
    ap.add_argument("--qs", default="0.02,0.10,0.25,0.50,0.75,0.90,0.98")
    ap.add_argument("--out",           default="global_combined.csv")
    args = ap.parse_args()

    has_ac = args.ac_params is not None and args.ac_tsv     is not None
    has_dc = args.dc_params is not None and args.dc_phase2  is not None
    if not has_ac and not has_dc:
        print("ERROR: supply (--ac_params + --ac_tsv) "
              "and/or (--dc_params + --dc_phase2)", file=sys.stderr)
        sys.exit(1)

    AU_w = parse_window(args.AU_window)
    RN_w = parse_window(args.RN_window)
    Q_w  = parse_window(args.Q_window)
    ac_fw = parse_window(args.ac_fit_window)
    dc_fw = parse_window(args.dc_fit_window)

    df_ac, ac_rows, df_dc, dc_rows = None, [], None, []
    if has_ac:
        df_ac, ac_rows = load_ac(args.ac_params, args.ac_tsv, ac_fw)
        print(f"  AC rows: {len(ac_rows)} in quantile fit")
    if has_dc:
        df_dc, dc_rows = load_dc(args.dc_params, args.dc_phase2, dc_fw)
        print(f"  DC rows: {len(dc_rows)} in quantile fit")

    lam_AU = args.lambda_AU * (args.ac_AU_boost if has_ac else 1.)
    lam_RN = args.lambda_RN
    lam_Q  = args.lambda_Q  * (args.dc_Q_boost  if has_dc else 1.)
    print(f"  lambda_AU={lam_AU:.2f}  lambda_RN={lam_RN:.2f}  lambda_Q={lam_Q:.2f}")

    x0, compiled = build_initials(df_ac, df_dc, AU_w, RN_w, Q_w)
    qs = np.array([float(q.strip()) for q in args.qs.split(",")], dtype=float)
    Z  = np.random.default_rng(args.seed).standard_normal((args.K, 5))

    def obj(x):
        return objective(x, ac_rows, dc_rows, qs, Z,
                         compiled, lam_AU, lam_RN, lam_Q,
                         ridge_Q=args.ridgeQ)

    print(f"\n  Nelder-Mead  maxiter={args.maxiter}  K={args.K}...")
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter":args.maxiter,
                            "xatol":2e-3,"fatol":2e-4,"disp":True})

    mu,sigmas,rho_AU,rho_RN = unpack(res.x)
    A,Ueff,R,N,Q = mu
    sA,sU,sR,sN,sQ = sigmas
    skews = compute_skewness(args.dc_phase2, Q_w, RN_w)

    out_row = pd.DataFrame([{
        "A":A, "Ueff":Ueff, "R":R, "N":N, "Q":Q,
        "sA":sA, "sU":sU, "sR":sR, "sN":sN, "sQ":sQ,
        "skew_A":skews["A"], "skew_Ueff":skews["Ueff"],
        "skew_R":skews["R"], "skew_N":skews["N"], "skew_Q":skews["Q"],
        "rho_AU":rho_AU, "rho_RN":rho_RN,
        "loss":float(res.fun), "success":bool(res.success),
        "nit":int(getattr(res,"nit",-1)),
        "n_ac_rows":len(ac_rows), "n_dc_rows":len(dc_rows),
        "AU_window":args.AU_window, "RN_window":args.RN_window,
        "Q_window":args.Q_window,
        "lambda_AU":lam_AU, "lambda_RN":lam_RN, "lambda_Q":lam_Q,
    }])
    header_needed = not os.path.exists(args.out) or os.path.getsize(args.out)==0
    out_row.to_csv(args.out, index=False, mode="a", header=header_needed)

    print("\n" + "="*58)
    print("  TAURnQ Phase 4 - Combined AC + DC global fit")
    print("="*58)
    print(f"  AC rows: {len(ac_rows)}   DC rows: {len(dc_rows)}")
    print(f"  Windows  AU={args.AU_window or 'ALL'}  "
          f"RN={args.RN_window or 'ALL'}  Q={args.Q_window or 'ALL'}")
    print()
    print(f"  {'Param':>6}  {'mean':>10}  {'sigma':>10}  {'skew':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name,val,sig,sk in [
        ("A",    A,    sA, skews["A"]),
        ("Ueff", Ueff, sU, skews["Ueff"]),
        ("R",    R,    sR, skews["R"]),
        ("N",    N,    sN, skews["N"]),
        ("Q",    Q,    sQ, skews["Q"]),
    ]:
        print(f"  {name:>6}  {val:>10.5f}  {sig:>10.5f}  {sk:>+8.4f}")
    print()
    print(f"  rho_AU={rho_AU:.4f}  rho_RN={rho_RN:.4f}")
    print(f"  loss={float(res.fun):.5g}  success={res.success}  "
          f"nit={getattr(res,'nit','?')}")
    print(f"\n  Saved -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
