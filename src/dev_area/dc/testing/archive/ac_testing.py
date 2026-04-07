#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Reta–Chilton width parameter
# ==========================================================
def sigma_from_alpha(alpha):
    """σ_lnτ from Reta & Chilton (2019)"""
    return 1.82 * np.sqrt(alpha) / (1 - alpha)


# ==========================================================
# 2. Log-normal PDF in τ-space
# ==========================================================
def lognormal_tau_pdf(tau, mu_ln, sigma_ln):
    """
    Log-normal distribution:
        tau ~ LogNormal(mu_ln, sigma_ln)
    PDF(tau) = 1/(tau*sigma*sqrt(2π)) * exp(-(ln tau - mu)^2/(2 sigma^2))
    """
    return (1.0 / (tau * sigma_ln * np.sqrt(2*np.pi))
            * np.exp(-(np.log(tau) - mu_ln)**2 / (2 * sigma_ln**2)))


# ==========================================================
# 3. Plot both distributions
# ==========================================================
def plot_two_lognormals(
        tau_mu_ref = 7.042600000E-01,
        alpha_ref  = 2.871300000E-01,
        tau_mu_user = 0.70,
        sigma_user  = 0.50
    ):
    """
    tau_mu_ref  : mean relaxation time from experiment (Reta–Chilton)
    alpha_ref   : width parameter used to compute σ_lnτ
    tau_mu_user : mean predicted from Q
    sigma_user  : sigma predicted from Q
    """

    # Convert Reta–Chilton τμ to μ_ln
    mu_ln_ref = np.log(tau_mu_ref)
    sigma_ln_ref = sigma_from_alpha(alpha_ref)

    # Convert editable τμ (from Q) to μ_ln
    mu_ln_user = np.log(tau_mu_user)
    sigma_ln_user = sigma_user

    # τ-domain for plotting
    tau = np.logspace(-4, 3, 4000)

    # Compute PDFs
    pdf_ref  = lognormal_tau_pdf(tau, mu_ln_ref, sigma_ln_ref)
    pdf_user = lognormal_tau_pdf(tau, mu_ln_user, sigma_ln_user)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(tau, pdf_ref,  label=f"Reta–Chilton LN: τμ={tau_mu_ref:.4f}, σ={sigma_ln_ref:.4f}", lw=2)
    plt.plot(tau, pdf_user, label=f"User LN: τμ={tau_mu_user:.4f}, σ={sigma_ln_user:.4f}", lw=2)

    plt.xscale("log")
    plt.xlabel("τ (s)")
    plt.ylabel("PDF(τ)")
    plt.title("Comparison of Two Log-normal Relaxation-Time Distributions")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 4. Run test
# ==========================================================
if __name__ == "__main__":

    # Reference experimental distribution (fixed)
    tau_mu_ref = 7.042600000E-01
    alpha_ref  = 2.871300000E-01

    # USER-EDITABLE distribution (derived from Q)
    tau_mu_user = -0.152267     # edit freely
    sigma_user  = 0.594134     # edit freely

    plot_two_lognormals(
        tau_mu_ref, alpha_ref,
        tau_mu_user, sigma_user
    )
