"""Debugging script (interactive only, nothing saved to disk).

Computes g2(tau) for the CORRECTED propagation only (see main.py for the
full g2 derivation: spatial average of I(0)*I(tau), referenced to the
t=0 frame). For every tau where g2 INCREASES relative to the previous
sample (g2[k] > g2[k-1]), shows a side-by-side comparison of the
intensity |u_out|^2 at t=0 (reference) vs. t=tau, to visually inspect
what's driving that particular uptick. Each pair of subplots shares one
color scale (referenced to the larger of the two frames' own peaks), so
the two intensities are directly comparable within that figure.
"""
import numpy as np
import matplotlib.pyplot as plt

from main import compute_g2_spatial, compute_output_series, setup


def main():
    s = setup()
    timesteps = s["timesteps"]

    u_out_by_t, _ = compute_output_series(s)
    g2 = compute_g2_spatial(u_out_by_t)

    I0 = np.abs(u_out_by_t[0]) ** 2

    n_events = 0
    for k in range(1, len(timesteps)):
        if g2[k] <= g2[k - 1]:
            continue
        n_events += 1
        Ik = np.abs(u_out_by_t[k]) ** 2
        vmax = max(I0.max(), Ik.max())

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

        im0 = axes[0].imshow(I0, cmap="inferno", vmin=0, vmax=vmax)
        axes[0].set_title("t=0")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(Ik, cmap="inferno", vmin=0, vmax=vmax)
        axes[1].set_title(f"t={timesteps[k]:g}s")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"g2 increase at tau={timesteps[k]:g}s: "
            f"{g2[k-1]:.4f} (tau={timesteps[k-1]:g}s) -> {g2[k]:.4f}"
        )
        fig.tight_layout()

    print(f"Found {n_events} g2-increase event(s) out of {len(timesteps) - 1} consecutive pairs")
    plt.show()


if __name__ == "__main__":
    main()
