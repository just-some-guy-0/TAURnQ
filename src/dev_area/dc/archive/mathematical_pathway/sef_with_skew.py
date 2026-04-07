import numpy as np

EULER_GAMMA = 0.5772156649015328606
ZETA3 = 1.2020569031595942854  # Apery’s constant ζ(3)

def sef_moments_q(tau_star: float, beta: float):
    """
    For KWW/SEF intrinsic distribution of relaxation times, with q = log10(tau):

    Zorn (JCP 2002) gives for Kohlrausch/KWW:
      E[ln tau]      = ln(tau*) + (1/beta - 1) * gamma_E
      Var(ln tau)    = (pi^2/6) * (1/beta^2 - 1)
      mu3_central(ln tau) = 2*zeta(3) * (1 - 1/beta^3)

    Skewness of ln(tau):
      g1_ln = mu3 / Var^(3/2)

    Since q = ln(tau)/ln(10) is a linear rescaling, skewness is invariant:
      g1_q = g1_ln
    """
    if not (0 < beta <= 1):
        raise ValueError("beta must be in (0, 1].")
    if tau_star <= 0:
        raise ValueError("tau_star must be positive.")

    # Mean/variance in ln-space
    mu_ln  = np.log(tau_star) + (1.0 / beta - 1.0) * EULER_GAMMA
    var_ln = (np.pi**2 / 6.0) * (1.0 / beta**2 - 1.0)

    # Third central moment in ln-space (Zorn, Table I, Kohlrausch row)
    mu3_ln = 2.0 * ZETA3 * (1.0 - 1.0 / beta**3)

    # Skewness in ln-space, and hence in q-space
    if var_ln <= 0:
        # beta=1 gives var=0; distribution collapses to a delta, skewness undefined.
        skew_q = np.nan
    else:
        skew_q = mu3_ln / (var_ln ** 1.5)

    # Convert mean/std to q=log10(tau)
    ln10 = np.log(10.0)
    mu_q = mu_ln / ln10
    sigma_q = np.sqrt(var_ln) / ln10

    return mu_q, sigma_q, skew_q


# -----------------------------
# Testing
# -----------------------------
tau_star = 106.4
beta = 0.857

mu_q, sigma_q, skew_q = sef_moments_q(tau_star, beta)
print("Intrinsic q mean (E[q])      :", mu_q)
print("Intrinsic q sigma (Std)      :", sigma_q)
print("Intrinsic q skewness (gamma1):", skew_q)
