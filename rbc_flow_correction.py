"""Sim step 3: apply a t=0-calibrated correction to the flowing two-RBC
system (35x35x15um tissue, capillary_system/tissue_background_35x35x15_
two_rbcs.mat), across all 17 timesteps from step_two_two_rbcs.py's
dynamic stopping rule (t=0 to t=4.0s, 0.25s steps).

Mirrors step_one.py's pipeline:
  u_in -> propagate(Delta) -> xA(t) -> propagate(Delta) -> xB(t)
        -> xB~ -> propagate(-Delta) -> xA~ -> propagate(-Delta) -> u_out(t)

But instead of an "ideal" per-timestep correction (A~=conj(A(t)),
B~=conj(B(t)), as in step_one.py), the correction planes are FIXED,
calibrated ONCE from the t=0 snapshot (empty capillary, no RBCs yet) and
applied UNCHANGED to every subsequent timestep's actual (A(t), B(t)).
This tests how a wavefront correction calibrated on a "clean" sample
degrades as blood actually flows through it.

All 17 |u_out(t)| panels share ONE fixed color scale (vmin=0,
vmax=|u_out(t=0)|.max()) so they're directly comparable to each other.
t=0 is the "ground truth" reference: since the fixed correction is
exactly the conjugate of that same t=0 aberration, u_out(t=0) is the
best-case, ideal-correction result (same diffraction-limited refocus as
step_one.py) -- every later frame's peak is expected to be <= this,
visually encoding how much the correction has gone stale.

A second figure plots an UNCORRECTED baseline for comparison: light
through the tissue (same forward pass, xA(t) then xB(t)), then
backpropagated 2*epsilon in one step with NO correction planes applied
at all (no B~, no A~). Uses the SAME shared color scale (vmax from the
corrected t=0 ground truth) as the first figure, so the two are directly,
visually comparable on one absolute scale.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from step_one import LAMBDA_UM, angular_spectrum_propagate
from step_two import build_x_grid, load_background
from step_two_two_rbcs import MAT_PATH, build_timeline, compute_planes_at

OUTPUT_DIR = "step 3 results"
GRID_COLS = 4


def run_pipeline(u_in, A_t, B_t, A_tilde_fixed, B_tilde_fixed, eps_um, dx_um):
    """One timestep's full propagate-aberrate-correct-backpropagate pass,
    identical in structure to step_one.py's main(), except the
    correction planes are passed in fixed rather than derived from A_t/B_t."""
    u1 = angular_spectrum_propagate(u_in, eps_um, LAMBDA_UM, dx_um)
    u_after_A = A_t * u1

    u2 = angular_spectrum_propagate(u_after_A, eps_um, LAMBDA_UM, dx_um)
    u_after_B = B_t * u2  # this timestep's actual tissue output

    u_after_Btilde = B_tilde_fixed * u_after_B
    u3 = angular_spectrum_propagate(u_after_Btilde, -eps_um, LAMBDA_UM, dx_um)
    u_after_Atilde = A_tilde_fixed * u3
    u_out = angular_spectrum_propagate(u_after_Atilde, -eps_um, LAMBDA_UM, dx_um)
    return u_out


def run_pipeline_uncorrected(u_in, A_t, B_t, eps_um, dx_um):
    """Same forward pass through the tissue as run_pipeline (propagate,
    xA(t), propagate, xB(t)), but then backpropagated 2*eps_um in ONE
    step with NO correction planes applied -- an uncorrected baseline.
    Two separate -eps_um backpropagations with nothing in between would
    give the identical result (free-space propagation composes:
    H(eps)*H(eps) = H(2*eps)), so a single -2*eps_um step is used
    directly, matching how the request is phrased."""
    u1 = angular_spectrum_propagate(u_in, eps_um, LAMBDA_UM, dx_um)
    u_after_A = A_t * u1

    u2 = angular_spectrum_propagate(u_after_A, eps_um, LAMBDA_UM, dx_um)
    u_after_B = B_t * u2  # tissue output

    u_out_uncorrected = angular_spectrum_propagate(u_after_B, -2 * eps_um, LAMBDA_UM, dx_um)
    return u_out_uncorrected


def plot_grid(values_by_t, timesteps, vmax, title, out_path):
    n_rows = -(-len(timesteps) // GRID_COLS)  # ceil
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(4 * GRID_COLS, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(timesteps):
        ax = axes[i // GRID_COLS, i % GRID_COLS]
        im = ax.imshow(np.abs(values_by_t[i]), cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"t={t:g}s", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(timesteps), n_rows * GRID_COLS):
        axes[i // GRID_COLS, i % GRID_COLS].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    bg = load_background(MAT_PATH)
    Nx = bg["mask_f_hr"].shape[0]
    x_grid1 = build_x_grid(bg["x_max"], bg["x_stp"], Nx)
    h_z_stp = bg["Delta"] / 2.0
    dx_um = bg["x_stp"]
    eps_um = bg["Delta"]

    t_entry_2, timesteps = build_timeline(bg["x_max"])
    print(f"{len(timesteps)} timesteps: t=0 to {timesteps[-1]:.2f}s")

    # Fixed correction planes, calibrated ONCE from t=0 (empty capillary).
    A_t0, B_t0 = compute_planes_at(0.0, bg, x_grid1, t_entry_2, h_z_stp)
    A_tilde_fixed = np.conj(A_t0)
    B_tilde_fixed = np.conj(B_t0)

    u_in = np.zeros((Nx, Nx), dtype=complex)
    u_in[Nx // 2, Nx // 2] = 1.0 + 0j

    u_out_by_t = []
    u_out_uncorrected_by_t = []
    for t in timesteps:
        A_t, B_t = compute_planes_at(t, bg, x_grid1, t_entry_2, h_z_stp)
        u_out_by_t.append(run_pipeline(u_in, A_t, B_t, A_tilde_fixed, B_tilde_fixed, eps_um, dx_um))
        u_out_uncorrected_by_t.append(run_pipeline_uncorrected(u_in, A_t, B_t, eps_um, dx_um))

    # t=0 is the ground truth: the fixed correction is exact there, so
    # this is the best-case (ideal-correction) peak -- shared across all
    # panels of BOTH figures, so corrected vs. uncorrected is directly
    # comparable on one absolute scale.
    vmax = np.abs(u_out_by_t[0]).max()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dims = bg["tissue_dims_um"]

    plot_grid(
        u_out_by_t, timesteps, vmax,
        f"Sim step 3: t=0-calibrated (fixed) correction applied through the flow, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out|",
        os.path.join(OUTPUT_DIR, "rbc_flow_correction_u_out.png"),
    )
    plot_grid(
        u_out_uncorrected_by_t, timesteps, vmax,
        f"Sim step 3: NO correction (light through tissue, backpropagated 2*epsilon), "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out|",
        os.path.join(OUTPUT_DIR, "rbc_flow_no_correction_u_out.png"),
    )

    plt.show()


if __name__ == "__main__":
    main()
