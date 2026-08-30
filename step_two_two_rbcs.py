"""Sim step 2 variant: TWO red blood cells through the same capillary.

Extends step_two.py's single-RBC scenario (same tissue-generation and
carve-then-reconstruct-to-M=2-planes approach; diameter/width/length/
speed formulas are imported, not duplicated, to avoid drift). A fresh
35x35x15um background tissue is used (capillary_system/
tissue_background_35x35x15_two_rbcs.mat), same generation parameters as
before, new random sphere realization.

Scenario (user-specified, confirmed before implementing):
  - t=0: capillary present but genuinely empty (no cell at all).
  - RBC 1 starts entering at t=0 (its leading edge begins at the tissue's
    Y-entrance boundary and advances from there -- GRADUAL entry: only
    part of the cell is visible while crossing the boundary, growing to
    full length once fully inside, and symmetrically shrinking on exit).
  - RBC 2 starts entering once RBC 1's leading edge reaches the tissue's
    Y-midpoint ("halfway through" the tissue, spatially -- not a fixed
    elapsed time, since speed varies with the capillary's dilation).
  - Both cells share the SAME capillary (single shared diameter(t)); at
    any given absolute time t, both use that same instant's width(t) for
    their cross-section, just at their own, independently-tracked
    Y-positions.
  - Recording continues at 0.25s steps from t=0 until BOTH cells have
    fully exited (trailing edge of each past the far Y boundary) -- the
    frame count is therefore computed, not fixed at 4 like step_two.py.

Output: two figures (one per plane, per the user's choice), each a grid
of all recorded timesteps (4 columns, as many rows as needed).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from step_two import (
    build_x_grid,
    diameter_um,
    length_um,
    load_background,
    reconstruct_coarse,
    speed_um_s,
    width_um,
    RBC_DELTA_N,
)

MAT_PATH = "capillary_system/tissue_background_35x35x15_two_rbcs.mat"
OUTPUT_DIR = "step 2 results"
TIME_STEP = 0.25
GRID_COLS = 4


def cumulative_displacement_um(t_start, t_end):
    """Integral of speed_um_s over absolute time [t_start, t_end] (fine
    trapezoidal quadrature). Uses ABSOLUTE time throughout -- the
    capillary's dilation (and thus speed) is a single shared clock, not
    reset per-cell, so a cell that enters later experiences whatever
    phase of the sinusoid is current at its own entry time."""
    if t_end <= t_start:
        return 0.0
    n = max(2001, int((t_end - t_start) * 4000) + 1)
    tt = np.linspace(t_start, t_end, n)
    return float(np.trapezoid(speed_um_s(tt), tt))


def find_time_for_displacement(target_displacement, t_start):
    """Smallest t_end > t_start such that cumulative_displacement_um(t_start, t_end)
    == target_displacement. speed_um_s > 0 always, so displacement is strictly
    increasing in t_end -- plain bisection is exact and robust."""
    lo, hi = t_start, t_start + 1.0
    while cumulative_displacement_um(t_start, hi) < target_displacement:
        hi = t_start + (hi - t_start) * 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if cumulative_displacement_um(t_start, mid) < target_displacement:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cell_edges(t, t_entry, y_max):
    """(trailing, leading) edge Y-position of a cell that started entering
    at t_entry, evaluated at absolute time t. Leading edge = -y_max at
    t=t_entry and advances from there (gradual entry); trailing edge
    trails behind by the cell's CURRENT length (evaluated at t, since
    size depends on the capillary's current dilation, not time-since-
    entry). Before t_entry, the cell hasn't started at all -- returns an
    interval entirely outside the tissue, which the carving step's
    boolean clipping naturally renders as nothing."""
    if t < t_entry:
        return (-y_max - 1.0, -y_max - 1.0)
    y1 = -y_max + cumulative_displacement_um(t_entry, t)
    y0 = y1 - length_um(t)
    return (y0, y1)


def cell_fully_exited(y0, y_max):
    return y0 > y_max


def build_timeline(y_max):
    """Entry time for cell 2 (cell 1's leading edge reaches y=0), then the
    full list of recording timesteps from t=0 until both cells have fully
    exited."""
    t_entry_2 = find_time_for_displacement(y_max, t_start=0.0)

    timesteps = []
    t = 0.0
    safety_cap_s = 120.0
    while True:
        timesteps.append(round(t, 10))
        y0_1, _ = cell_edges(t, 0.0, y_max)
        y0_2, _ = cell_edges(t, t_entry_2, y_max)
        if cell_fully_exited(y0_1, y_max) and cell_fully_exited(y0_2, y_max):
            break
        t += TIME_STEP
        if t > safety_cap_s:
            raise RuntimeError(f"Neither cell exited within {safety_cap_s}s -- check formulas/parameters")
    return t_entry_2, timesteps


def carve_fine_volume_multi(t, mask_f_hr, x_grid1, z_grid1, cell_intervals, rbc_delta_n):
    """Like step_two.carve_fine_volume, generalized to N simultaneous cells
    (each an independent (y0,y1) interval) sharing one capillary."""
    combined = mask_f_hr.copy()
    Nx = x_grid1.shape[0]
    Nz = z_grid1.shape[0]

    xx = x_grid1[:, None]
    zz = z_grid1[None, :]

    r_tube = diameter_um(t) / 2.0
    tube_xz = (xx ** 2 + zz ** 2) <= r_tube ** 2

    w = width_um(t)
    box_xz = (np.abs(xx) <= w / 2.0) & (np.abs(zz) <= w / 2.0) & tube_xz

    tube_3d = np.broadcast_to(tube_xz[:, None, :], (Nx, Nx, Nz))
    combined[tube_3d] = 1.0 + 0.0j
    touched = tube_3d.copy()

    rbc_value = np.exp(1j * 2 * np.pi * rbc_delta_n)
    for y0, y1 in cell_intervals:
        cell_y_mask = (x_grid1 >= y0) & (x_grid1 <= y1)
        box_3d = np.broadcast_to(box_xz[:, None, :], (Nx, Nx, Nz)) & cell_y_mask[None, :, None]
        combined[box_3d] = rbc_value
        touched = touched | box_3d

    return combined, touched


def compute_planes_at(t, bg, x_grid1, t_entry_2, h_z_stp):
    """A, B phase-only planes (M=2) at time t for the two-RBC scenario --
    carves both cells' current geometry into the fine background volume
    and reconstructs the coarse planes. Factored out so other scripts
    (e.g. rbc_flow_correction.py) can reuse this exact computation rather
    than re-deriving it."""
    y_max = bg["x_max"]
    edges_1 = cell_edges(t, 0.0, y_max)
    edges_2 = cell_edges(t, t_entry_2, y_max)
    combined, touched = carve_fine_volume_multi(
        t, bg["mask_f_hr"], x_grid1, bg["z_grid1"], [edges_1, edges_2], RBC_DELTA_N)
    coarse = reconstruct_coarse(
        combined, touched, bg["mask_f"], bg["z_grid1"], bg["z_grid1_sps"], h_z_stp, bg["x_stp"])
    A = np.exp(1j * np.angle(coarse[:, :, 0]))
    B = np.exp(1j * np.angle(coarse[:, :, 1]))
    return A, B


def main():
    bg = load_background(MAT_PATH)
    Nx = bg["mask_f_hr"].shape[0]
    x_grid1 = build_x_grid(bg["x_max"], bg["x_stp"], Nx)
    y_max = bg["x_max"]
    h_z_stp = bg["Delta"] / 2.0

    t_entry_2, timesteps = build_timeline(y_max)
    print(f"Cell 2 entry time: {t_entry_2:.3f}s")
    print(f"Recording {len(timesteps)} timesteps: t=0 to {timesteps[-1]:.2f}s")

    planes_by_t = [compute_planes_at(t, bg, x_grid1, t_entry_2, h_z_stp) for t in timesteps]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dims = bg["tissue_dims_um"]
    n_rows = -(-len(timesteps) // GRID_COLS)  # ceil

    for plane_idx, plane_label in [(0, "A"), (1, "B")]:
        fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(4 * GRID_COLS, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for i, t in enumerate(timesteps):
            ax = axes[i // GRID_COLS, i % GRID_COLS]
            plane = planes_by_t[i][plane_idx]
            im = ax.imshow(np.angle(plane), cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"t={t:g}s", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")
        for i in range(len(timesteps), n_rows * GRID_COLS):
            axes[i // GRID_COLS, i % GRID_COLS].axis("off")
        fig.suptitle(f"Sim step 2 (two RBCs): plane {plane_label}, {dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue")
        fig.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"step_two_two_rbcs_plane_{plane_label}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
