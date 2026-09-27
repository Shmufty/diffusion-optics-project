"""Figure: BOTH aberration planes (A -- nearest the tissue entrance, and
B -- at the exit face, per the M=2 forward-propagation convention: plane
A is index 0, plane B is index 1) at three specific timesteps: t=0, 1,
2.5s, for the two-RBC scenario. Shape (2,3): top row plane A, bottom row
plane B, same timestamps in both rows.

Uses compute_planes_at via the same setup() dict main.py's own pipeline
uses, so this is exactly the same A(t)/B(t) main.py computes -- not a
re-derivation. Phase-only (A, B are unit-magnitude everywhere), same
plotting convention as step_two_two_rbcs.py's aberration-plane figures
(twilight colormap, fixed -pi..pi scale).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from main import setup, compute_planes_at

TIMES = [0.0, 1.0, 2.5]
OUTPUT_PATH = "step 2 results/first_aberration_layer_snapshots.png"


def main():
    s = setup()

    planes_by_t = [compute_planes_at(t, s["bg"], s["x_grid1"], s["t_entry_2"], s["h_z_stp"]) for t in TIMES]

    fig, axes = plt.subplots(2, len(TIMES), figsize=(4 * len(TIMES), 8))
    for col, t in enumerate(TIMES):
        A, B = planes_by_t[col]
        for row, (label, plane) in enumerate([("A", A), ("B", B)]):
            ax = axes[row, col]
            im = ax.imshow(np.angle(plane), cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"plane {label}, t={t:g}s", fontsize=11)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")

    fig.suptitle("Aberration planes A (top) and B (bottom) at t=0, 1, 2.5s")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
