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

A third figure adds a speckle correlation statistic from course_material/
"8. Laser speckle contrast imaging and multiple scattering theory2.pdf":
  g2(tau) = <I(0)I(tau)> / <I(0)^2>
where I(t) = |u_out(t)|^2 (real, non-negative intensity image) and <...>
is a SPATIAL average (mean over all pixels of the element-wise product,
e.g. I(0)[x,y]*I(tau)[x,y]) referenced to the t=0 frame -- NOT a temporal
average over multiple (t,t+tau) pairs. The denominator is the NUMERATOR'S
OWN value at tau=0 (<I(0)*I(0)>=<I(0)^2>, mean of I0 squared -- NOT
<I(0)>^2, mean of I0 then squared; the two differ whenever I(0) isn't
spatially uniform), so g2(0)=1 is guaranteed by construction, same as
g1(0)=1 in the analogous field-correlation formula this mirrors. This
intentionally does not attempt a field correlation g1: spatially
averaging the complex field E(0)[x,y]*E*(tau)[x,y] over many independent
speckle grains (each with essentially uncorrelated absolute phase)
mostly cancels out and isn't physically meaningful without a common
phase reference across the image -- which is exactly why real
camera-based speckle imaging (the PDF's slides 7-9) works with intensity
statistics, not field statistics: a camera only ever records |E|^2,
never E itself. Using intensity avoids that problem (no cancellation:
I>=0 everywhere) and gives a constant, large sample count (~Nx^2 pixels)
at every tau lag. Computed for the corrected path only for now (u_out_by_t);
u_out_uncorrected_by_t is still computed (used for the second figure
above) but its g2 is left out of the plot.

Note: g2(tau) for tau>0 is NOT bounded by 1 -- the asymmetric,
t=0-referenced normalization only guarantees g2(0)=1. By Cauchy-Schwarz,
g2(tau) <= sqrt(<I(tau)^2>/<I(0)^2>), which exceeds 1 whenever frame tau
has more spatial contrast than the t=0 reference frame; this is a
legitimate property of the formula, not a bug.

--- Submission note ---
This is a course-submission copy: all supporting functions this file
needs (imported below from functions.py) are consolidated in that one
sibling file instead of the full project's multi-module structure. The
generated tissue data (mask_f_hr etc.) is NOT included in this folder --
see the accompanying docx.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from functions import LAMBDA_UM, angular_spectrum_propagate, build_x_grid, load_background, MAT_PATH, build_timeline, compute_planes_at

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


def compute_g2_spatial(fields_by_t):
    """g2(tau) = <I(0)I(tau)>/<I(0)I(0)>, with <...> a SPATIAL average
    (mean over all pixels of the element-wise product) referenced to the
    t=0 frame -- not a temporal average over multiple (t,t+tau) pairs.
    I(t)=|fields_by_t[t]|**2. The denominator is the NUMERATOR'S OWN
    value at tau=0 (<I(0)*I(0)>=<I(0)^2>, mean of I0 squared), not
    <I(0)>^2 (mean of I0, then squared) -- same construction as
    g1(tau)=<E(0)E*(tau)>/<E(0)E*(0)>, so g2(0)=1 is guaranteed by
    construction, exactly like g1(0)=1. Real, non-negative throughout (no
    phase-cancellation issue, unlike a field correlation), and every tau
    lag gets the same, large (~Nx^2 pixel) sample count."""
    I0 = np.abs(fields_by_t[0]) ** 2
    denom = np.mean(I0 * I0)  # = <I(0)^2>, i.e. the numerator at tau=0
    G2 = np.array([np.mean(I0 * np.abs(Et) ** 2) for Et in fields_by_t])
    g2 = G2 / denom
    return g2


def plot_speckle_statistics(tau, g2_corr, out_path):
    # Uncorrected g2 left out for now (still computable via compute_g2_spatial
    # on u_out_uncorrected_by_t if needed later -- just not plotted here).
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    ax.plot(tau, g2_corr, "o-", color="C0", label="corrected")
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"$g_2(\tau)$")
    ax.set_title("Intensity autocorrelation $g_2$\n(spatial average, referenced to t=0)", fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def setup():
    """Shared setup: load the background tissue, build coordinate grids,
    compute the t=0-calibrated fixed correction planes, and the point-
    source input field. Returns everything needed to propagate through
    any timestep -- factored out so other scripts (e.g. a debugging
    script that only needs a subset of what main() computes) can reuse
    it without duplicating this setup."""
    bg = load_background(MAT_PATH)
    Nx = bg["mask_f_hr"].shape[0]
    x_grid1 = build_x_grid(bg["x_max"], bg["x_stp"], Nx)
    h_z_stp = bg["Delta"] / 2.0
    dx_um = bg["x_stp"]
    eps_um = bg["Delta"]

    t_entry_2, timesteps = build_timeline(bg["x_max"])

    # Fixed correction planes, calibrated ONCE from t=0 (empty capillary).
    A_t0, B_t0 = compute_planes_at(0.0, bg, x_grid1, t_entry_2, h_z_stp)
    A_tilde_fixed = np.conj(A_t0)
    B_tilde_fixed = np.conj(B_t0)

    u_in = np.zeros((Nx, Nx), dtype=complex)
    u_in[Nx // 2, Nx // 2] = 1.0 + 0j

    return dict(
        bg=bg, x_grid1=x_grid1, h_z_stp=h_z_stp, dx_um=dx_um, eps_um=eps_um,
        t_entry_2=t_entry_2, timesteps=timesteps,
        A_tilde_fixed=A_tilde_fixed, B_tilde_fixed=B_tilde_fixed, u_in=u_in,
    )


def compute_output_series(s):
    """u_out_by_t (corrected) and u_out_uncorrected_by_t for every
    timestep, given the shared setup dict from setup()."""
    u_out_by_t = []
    u_out_uncorrected_by_t = []
    for t in s["timesteps"]:
        A_t, B_t = compute_planes_at(t, s["bg"], s["x_grid1"], s["t_entry_2"], s["h_z_stp"])
        u_out_by_t.append(run_pipeline(
            s["u_in"], A_t, B_t, s["A_tilde_fixed"], s["B_tilde_fixed"], s["eps_um"], s["dx_um"]))
        u_out_uncorrected_by_t.append(run_pipeline_uncorrected(
            s["u_in"], A_t, B_t, s["eps_um"], s["dx_um"]))
    return u_out_by_t, u_out_uncorrected_by_t


def main():
    s = setup()
    timesteps = s["timesteps"]
    bg = s["bg"]
    print(f"{len(timesteps)} timesteps: t=0 to {timesteps[-1]:.2f}s")

    u_out_by_t, u_out_uncorrected_by_t = compute_output_series(s)

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

    # Speckle correlation statistic g2(tau), spatially averaged (referenced
    # to t=0) intensity autocorrelation, per course_material/"8. Laser
    # speckle contrast imaging and multiple scattering theory2.pdf".
    # Corrected path only for now (see plot_speckle_statistics).
    tau = np.array(timesteps)
    g2_corr = compute_g2_spatial(u_out_by_t)

    plot_speckle_statistics(
        tau, g2_corr,
        os.path.join(OUTPUT_DIR, "speckle_correlation_statistics.png"),
    )

    plt.show()


if __name__ == "__main__":
    main()
