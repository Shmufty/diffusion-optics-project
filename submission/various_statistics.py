"""Temp script: three related correlation statistics, for BOTH the
corrected and uncorrected paths.

1. |g1(tau)| -- the field-correlation MAGNITUDE, obtained via Siegert
   inversion from intensity data, NOT by directly spatially-averaging the
   complex field (we established that gives a physically meaningless,
   mostly-cancelled result -- see the conversation this script comes
   from). Siegert: G2(tau) = <I>^2 + beta*|G1(tau)|^2, so
   |g1(tau)| = sqrt((g2_classical(tau) - 1) / beta), with
   beta = g2_classical(0) - 1.

   IMPORTANT: this inversion needs the CLASSICAL normalization
   g2_classical(tau) = <I(0)I(tau)> / <I(0)>^2 (mean of I0, THEN
   squared) -- NOT main.py's compute_g2_spatial, which deliberately uses
   <I(0)^2> (mean of I0 squared) instead, specifically so that its own
   g2(0)=1 exactly by construction (mirroring g1(0)=1). That's a
   different, valid statistic (see item 3), but it makes beta=0 under
   this script's Siegert inversion (division by zero) -- the classical
   <I(0)>^2 normalization is what makes g2_classical(0)=1+beta with a
   generally-nonzero beta, which is what Siegert inversion needs.

2. G2(tau) = <I(0)I(tau)> -- the RAW, unnormalized intensity
   cross-correlation, exactly as defined on course_material/"8. Laser
   speckle contrast imaging and multiple scattering theory2.pdf" slide 3
   (no division by anything). Note: corrected and uncorrected G2 differ
   in absolute scale by ~2 orders of magnitude (the corrected path's
   ideal t=0 spot is far more concentrated) -- both are still plotted on
   one shared linear axis below, so the uncorrected curve will look
   nearly flat next to the corrected one; that's a real scale
   difference, not a plotting bug.

3. The CURRENTLY-USED g2(tau) from main.py's compute_g2_spatial:
   g2(tau) = <I(0)I(tau)> / <I(0)^2>, i.e. G2(tau) (item 2) divided by
   the numerator's OWN value at tau=0 -- guarantees g2(0)=1 by
   construction, the same identity as g1(0)=1. This is the version used
   everywhere else in this project (main.py's third figure).

All three use the same SPATIAL averaging convention (mean over all
pixels of the element-wise product, referenced to the t=0 frame) as the
rest of the project -- not a temporal average over multiple (t,t+tau)
pairs.
"""
import numpy as np
import matplotlib.pyplot as plt

from main import setup, compute_output_series, compute_g2_spatial


def compute_three_stats(fields_by_t):
    """|g1(tau)| (Siegert inversion), G2(tau) (raw), g2(tau) (current) --
    see module docstring for the definitions and why |g1| needs a
    different normalization than g2."""
    I0 = np.abs(fields_by_t[0]) ** 2
    mean_I0 = I0.mean()

    G2_raw = np.array([np.mean(I0 * np.abs(Et) ** 2) for Et in fields_by_t])

    g2_classical = G2_raw / (mean_I0 ** 2)
    beta = g2_classical[0] - 1.0
    g1_mag = np.sqrt(np.clip((g2_classical - 1.0) / beta, 0.0, None))

    g2_current = compute_g2_spatial(fields_by_t)

    return g1_mag, G2_raw, g2_current, beta


def print_table(label, timesteps, g1_mag, G2_raw, g2_current):
    print(f"--- {label} ---")
    print(f"{'tau':>6} {'|g1(tau)|':>12} {'G2(tau)':>12} {'g2(tau) current':>16}")
    for t, g1v, g2v, g2c in zip(timesteps, g1_mag, G2_raw, g2_current):
        print(f"{t:6.2f} {g1v:12.4f} {g2v:12.3e} {g2c:16.4f}")
    print()


def main():
    s = setup()
    timesteps = s["timesteps"]

    u_out_by_t, u_out_uncorrected_by_t = compute_output_series(s)

    g1_corr, G2_corr, g2_corr, beta_corr = compute_three_stats(u_out_by_t)
    g1_unc, G2_unc, g2_unc, beta_unc = compute_three_stats(u_out_uncorrected_by_t)

    print(f"beta: corrected={beta_corr:.4f}, uncorrected={beta_unc:.4f}\n")
    print_table("Corrected path", timesteps, g1_corr, G2_corr, g2_corr)
    print_table("Uncorrected path", timesteps, g1_unc, G2_unc, g2_unc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(timesteps, g1_corr, "o-", color="C0", label="corrected")
    axes[0].plot(timesteps, g1_unc, "s-", color="C1", label="uncorrected")
    axes[0].set_xlabel(r"$\tau$ [s]")
    axes[0].set_ylabel(r"$|g_1(\tau)|$")
    axes[0].set_title("Field autocorrelation magnitude\n(Siegert inversion)", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(timesteps, G2_corr, "o-", color="C0", label="corrected")
    axes[1].plot(timesteps, G2_unc, "s-", color="C1", label="uncorrected")
    axes[1].set_xlabel(r"$\tau$ [s]")
    axes[1].set_ylabel(r"$G_2(\tau)=\langle I(0)I(\tau)\rangle$")
    axes[1].set_title("Raw intensity correlation\n(unnormalized)", fontsize=10)
    axes[1].legend(fontsize=8)

    axes[2].plot(timesteps, g2_corr, "o-", color="C0", label="corrected")
    axes[2].plot(timesteps, g2_unc, "s-", color="C1", label="uncorrected")
    axes[2].set_xlabel(r"$\tau$ [s]")
    axes[2].set_ylabel(r"$g_2(\tau)$")
    axes[2].set_title("Currently-used $g_2$\n($\\langle I(0)I(\\tau)\\rangle/\\langle I(0)^2\\rangle$)", fontsize=10)
    axes[2].legend(fontsize=8)

    fig.suptitle("Corrected vs. uncorrected -- three correlation statistics")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
