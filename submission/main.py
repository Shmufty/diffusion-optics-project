"""Apply a t=0-calibrated correction to the flowing two-RBC
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

Three figures show the aberration planes themselves -- the input to the
correction problem: plane A (tissue entrance) and plane B (exit face)
across all timesteps, one figure each, plus a 4-panel summary of plane A
spanning t=0 to the final timestep. The planes are phase-only, so these
plot np.angle on a fixed -pi..pi twilight scale, making every panel
comparable both within and across the figures. Read together with the
fixed correction described above, they are what the correction is losing
track of: A~ and B~ stay frozen at their t=0 conjugates while these drift.

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

A final figure is the actual robustness estimate. Correction efficiency
is measured by a Strehl-like peak intensity at the refocus plane (both
paths normalised to the corrected t=0 frame, the ideal-correction ground
truth), NOT by g2 -- which the figures below show is nearly blind to
whether the correction is on at all. That efficiency is then plotted
against how far the SAMPLE has drifted from its calibration state,
D = 1 - g2_uncorrected, rather than against elapsed time. Time is only a
proxy for sample change and a poor one here (the flow speeds up, the tube
dilates, the cells enter and leave at particular moments), so a
"correction lifetime in seconds" read off this run would describe the
clip rather than the method. Plotting against drift removes all of that
from the x-axis and makes the estimate comparable against other flow
conditions and geometries. This is where g2 earns its place: not as a
measure of correction quality, but as the yardstick for the perturbation.
See functions.py's robustness section for the full argument.

Caveat for interpreting it: the degradation measured here mixes TWO
causes, which this run does not separate -- the RBCs flowing through, AND
the capillary itself dilating (functions.diameter_um sweeps 4um at t=0 to
10um at t=4.0s, an 8s sinusoid, with RBC speed tracking it), so the
t=0-calibrated correction is being judged against a sample whose tube
geometry has also changed out from under it.

Before computing any correlation statistic, BOTH u_out_by_t and
u_out_uncorrected_by_t are carried one step further: both already sit at
the same plane as u_in (the corrected path's two -eps_um
backpropagations, and the uncorrected path's single -2*eps_um
backpropagation, both cancel the +2*eps_um forward pass through the
tissue). From there, each is propagated FORWARD by another 2*eps_um of
free space (no aberration/correction planes -- plain angular-spectrum
propagation), landing 2*epsilon past the refocus plane. Two amplitude-
grid figures show this for each path. g2(tau) is deliberately computed on
THIS propagated field, not the raw refocus-plane field: at zero
propagation distance the corrected field is a near-singular case for a
spatial-average intensity correlation (almost all signal energy sits in
a handful of central pixels -- the "spot"), which isn't representative of
an actual extended speckle pattern; propagating it forward turns it into
one, which is what g2 is meant to characterize. The speckle animation
(see save_speckle_animation) is likewise built from this propagated
field, for the same reason.

A figure adds speckle correlation statistics from course_material/
"8. Laser speckle contrast imaging and multiple scattering theory2.pdf":
  g2(tau) = <I(0)I(tau)> / <I(0)^2>
where I(t) = |u_out_prop(t)|^2 (real, non-negative intensity image, on
the +2*epsilon-propagated field described above) and <...> is a SPATIAL
average (mean over all pixels of the element-wise product, e.g.
I(0)[x,y]*I(tau)[x,y]) referenced to the t=0 frame -- NOT a temporal
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
at every tau lag. The figure overlays both corrected and uncorrected for
a direct decorrelation-speed comparison.

Note: g2(tau) for tau>0 is NOT bounded by 1 -- the asymmetric,
t=0-referenced normalization only guarantees g2(0)=1. By Cauchy-Schwarz,
g2(tau) <= sqrt(<I(tau)^2>/<I(0)^2>), which exceeds 1 whenever frame tau
has more spatial contrast than the t=0 reference frame; this is a
legitimate property of the formula, not a bug.

--- Submission note ---
This is a course-submission copy: all supporting functions this file
needs (imported below from functions.py) are consolidated in that one
sibling file instead of the full project's multi-module structure. The
generated tissue data (mask_f_hr etc.) IS included in this folder, at
capillary_system/tissue_background_35x35x15_two_rbcs.mat, so this folder
is fully self-contained and can be run as-is.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from functions import (LAMBDA_UM, angular_spectrum_propagate, build_x_grid, load_background,
                       MAT_PATH, build_timeline, compute_planes_at, crop_center,
                       compute_focus_quality, compute_robustness, plot_robustness,
                       report_robustness)

GRID_COLS = 4

# Anchored to this file's own directory (not the process cwd) so results
# always land inside the submission folder itself, regardless of where
# main.py is invoked from.
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SUBMISSION_DIR, "results")
ANIMATION_PATH = os.path.join(SUBMISSION_DIR, "rbc_flow_speckle_animation.gif")


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


def plot_grid(values_by_t, timesteps, vmax, title, out_path, crop=None):
    n_rows = -(-len(timesteps) // GRID_COLS)  # ceil
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(4 * GRID_COLS, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(timesteps):
        ax = axes[i // GRID_COLS, i % GRID_COLS]
        amp = np.abs(values_by_t[i])
        if crop:
            amp = crop_center(amp, crop)
        im = ax.imshow(amp, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"t={t:g}s", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(timesteps), n_rows * GRID_COLS):
        axes[i // GRID_COLS, i % GRID_COLS].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_phase_grid(planes_by_t, timesteps, title, out_path):
    """Aberration-plane phase across a set of timesteps. The planes are
    phase-only (unit magnitude everywhere), so |plane| carries no
    information and np.angle is what gets plotted -- twilight on a fixed
    -pi..pi scale, matching the project's other aberration-plane figures,
    so panels are directly comparable to each other and across figures."""
    n_rows = -(-len(timesteps) // GRID_COLS)  # ceil
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(4 * GRID_COLS, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(timesteps):
        ax = axes[i // GRID_COLS, i % GRID_COLS]
        im = ax.imshow(np.angle(planes_by_t[i]), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"t={t:g}s", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")
    for i in range(len(timesteps), n_rows * GRID_COLS):
        axes[i // GRID_COLS, i % GRID_COLS].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_input_field(u_in, out_path, crop=20):
    """u_in is an ideal point source: a single bright pixel at the array
    center, zero everywhere else. Two panels: the full frame (to show its
    location/scale relative to the Nx x Nx grid) and a tight crop around
    the center (to actually see the nonzero pixel)."""
    amp = np.abs(u_in)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    im0 = axes[0].imshow(amp, cmap="inferno", vmin=0, vmax=amp.max())
    axes[0].set_title("full frame", fontsize=10)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(crop_center(amp, crop), cmap="inferno", vmin=0, vmax=amp.max())
    axes[1].set_title(f"cropped {crop}x{crop}px around center", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Input field u_in: ideal point source, |u_in|")
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


def save_speckle_animation(u_out_prop_by_t, u_out_uncorrected_prop_by_t, timesteps, out_path, crop=None):
    """1x2 animated GIF: |u_out_prop(t)| corrected (left) vs. uncorrected
    (right) -- the +2*epsilon-propagated speckle field (same field the g2
    statistics are computed on), NOT the raw refocus-plane "spot": at zero
    propagation distance the corrected field is a tight, near-single-pixel
    peak, which isn't representative of an extended speckle pattern:
    propagating it forward is what actually turns it into one. No crop by
    default (unlike the spot, this pattern is already spread across most
    of the frame -- cropping would cut off real structure); pass crop=N
    for an NxN window around the center if needed.
    Each panel is scaled to its OWN global max across all 17 frames
    (independent vmax per panel) -- unlike the static comparison figures,
    which intentionally share one absolute scale to show the ~2-orders-of-
    magnitude intensity gap; here the goal is just to make each path's own
    speckle dynamics visible side by side, which a shared scale would
    mostly hide for the uncorrected (much dimmer) panel."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    if crop:
        corr_frames = [crop_center(np.abs(u), crop) for u in u_out_prop_by_t]
        uncorr_frames = [crop_center(np.abs(u), crop) for u in u_out_uncorrected_prop_by_t]
    else:
        corr_frames = [np.abs(u) for u in u_out_prop_by_t]
        uncorr_frames = [np.abs(u) for u in u_out_uncorrected_prop_by_t]
    vmax_corr = max(a.max() for a in corr_frames)
    vmax_uncorr = max(a.max() for a in uncorr_frames)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    im0 = axes[0].imshow(corr_frames[0], cmap="inferno", vmin=0, vmax=vmax_corr)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    title0 = axes[0].set_title(f"corrected, t={timesteps[0]:g}s", fontsize=11)

    im1 = axes[1].imshow(uncorr_frames[0], cmap="inferno", vmin=0, vmax=vmax_uncorr)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    title1 = axes[1].set_title(f"uncorrected, t={timesteps[0]:g}s", fontsize=11)

    crop_note = f" (cropped {crop}x{crop}px around center)" if crop else ""
    fig.suptitle(f"|u_out(t)|, +2*epsilon past refocus: corrected vs. uncorrected{crop_note}")
    fig.tight_layout()

    def update(i):
        im0.set_data(corr_frames[i])
        title0.set_text(f"corrected, t={timesteps[i]:g}s")
        im1.set_data(uncorr_frames[i])
        title1.set_text(f"uncorrected, t={timesteps[i]:g}s")
        return im0, im1, title0, title1

    ani = FuncAnimation(fig, update, frames=len(timesteps), interval=400)
    ani.save(out_path, writer=PillowWriter(fps=2))
    plt.close(fig)
    print(f"Saved animation to {out_path}")


def plot_g2_comparison(tau, g2_corr, g2_uncorr, title, out_path):
    """g2(tau) as defined in compute_g2_spatial, with both series
    (corrected, uncorrected) overlaid on one axes for a direct
    decorrelation-speed comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    ax.plot(tau, g2_corr, "o-", color="C0", label="corrected")
    ax.plot(tau, g2_uncorr, "s-", color="C1", label="uncorrected")
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"$g_2(\tau)$")
    ax.set_title(title, fontsize=11)
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
    timestep, given the shared setup dict from setup(), plus the
    aberration planes A(t) and B(t) each was propagated through --
    returned rather than discarded so they can be plotted without paying
    for a second pass of compute_planes_at, which dominates the runtime."""
    u_out_by_t = []
    u_out_uncorrected_by_t = []
    A_by_t = []
    B_by_t = []
    for t in s["timesteps"]:
        A_t, B_t = compute_planes_at(t, s["bg"], s["x_grid1"], s["t_entry_2"], s["h_z_stp"])
        A_by_t.append(A_t)
        B_by_t.append(B_t)
        u_out_by_t.append(run_pipeline(
            s["u_in"], A_t, B_t, s["A_tilde_fixed"], s["B_tilde_fixed"], s["eps_um"], s["dx_um"]))
        u_out_uncorrected_by_t.append(run_pipeline_uncorrected(
            s["u_in"], A_t, B_t, s["eps_um"], s["dx_um"]))
    return u_out_by_t, u_out_uncorrected_by_t, A_by_t, B_by_t


def main():
    s = setup()
    timesteps = s["timesteps"]
    bg = s["bg"]
    print(f"{len(timesteps)} timesteps: t=0 to {timesteps[-1]:.2f}s")

    u_out_by_t, u_out_uncorrected_by_t, A_by_t, B_by_t = compute_output_series(s)

    # t=0 is the ground truth: the fixed correction is exact there, so
    # this is the best-case (ideal-correction) peak -- shared across all
    # panels of BOTH figures, so corrected vs. uncorrected is directly
    # comparable on one absolute scale.
    vmax = np.abs(u_out_by_t[0]).max()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dims = bg["tissue_dims_um"]

    plot_input_field(s["u_in"], os.path.join(OUTPUT_DIR, "input_field.png"))

    # The aberration planes themselves -- the input to the correction
    # problem, i.e. what the flowing cells actually do to the wavefront and
    # how far the fixed t=0-calibrated correction's target drifts from what
    # it was calibrated on.
    plot_phase_grid(
        A_by_t, timesteps,
        f"Aberration plane $A_1$ (tissue entrance) over time, phase, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue",
        os.path.join(OUTPUT_DIR, "aberration_plane_A.png"),
    )
    plot_phase_grid(
        B_by_t, timesteps,
        f"Aberration plane B (tissue exit) over time, phase, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue",
        os.path.join(OUTPUT_DIR, "aberration_plane_B.png"),
    )
    # Same plane A, thinned to 4 panels spanning the full run (first t=0,
    # last the final timestep, the middle two as evenly spaced as 17
    # timesteps allow) -- a readable summary of the 17-panel figure above.
    sel = np.linspace(0, len(timesteps) - 1, 4).round().astype(int)
    plot_phase_grid(
        [A_by_t[i] for i in sel], [timesteps[i] for i in sel],
        f"Aberration plane $A_1$ (tissue entrance), phase, 4 timesteps spanning "
        f"t=0 to t={timesteps[-1]:g}s",
        os.path.join(OUTPUT_DIR, "aberration_plane_A_4_timesteps.png"),
    )

    plot_grid(
        u_out_by_t, timesteps, vmax,
        f"t=0-calibrated (fixed) correction applied through the flow, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out|",
        os.path.join(OUTPUT_DIR, "rbc_flow_correction_u_out.png"),
    )
    # Same data, cropped to a 50x50 PIXEL window (5x5um, since x_stp=0.1um)
    # around the frame center -- the full-frame view above makes the spot
    # too small to see clearly against the 351x351 pixel field.
    plot_grid(
        u_out_by_t, timesteps, vmax,
        f"t=0-calibrated (fixed) correction applied through the flow, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out| (cropped 50x50px around center)",
        os.path.join(OUTPUT_DIR, "rbc_flow_correction_u_out_cropped50.png"),
        crop=50,
    )
    plot_grid(
        u_out_uncorrected_by_t, timesteps, vmax,
        f"NO correction (light through tissue, backpropagated 2*epsilon), "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out|",
        os.path.join(OUTPUT_DIR, "rbc_flow_no_correction_u_out.png"),
    )

    strehl_corr, _, _, _ = compute_focus_quality(u_out_by_t, u_out_uncorrected_by_t)

    # --- Further forward propagation past the refocus/backpropagation
    # plane, both paths --- (see module docstring for why: this is the
    # actual, extended speckle pattern the g2 statistics and the animation
    # below are computed/built on, NOT the raw refocus-plane "spot").
    u_out_prop_by_t = [
        angular_spectrum_propagate(u, 2 * s["eps_um"], LAMBDA_UM, s["dx_um"])
        for u in u_out_by_t
    ]
    u_out_uncorrected_prop_by_t = [
        angular_spectrum_propagate(u, 2 * s["eps_um"], LAMBDA_UM, s["dx_um"])
        for u in u_out_uncorrected_by_t
    ]

    save_speckle_animation(u_out_prop_by_t, u_out_uncorrected_prop_by_t, timesteps, ANIMATION_PATH)

    # t=0 corrected-propagated frame is the new "ground truth" peak, shared
    # by both grids below, same convention as vmax above.
    vmax_prop = np.abs(u_out_prop_by_t[0]).max()

    plot_grid(
        u_out_prop_by_t, timesteps, vmax_prop,
        f"corrected path, further propagated +2*epsilon past refocus, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out|",
        os.path.join(OUTPUT_DIR, "rbc_flow_correction_u_out_propagated2eps.png"),
    )
    plot_grid(
        u_out_uncorrected_prop_by_t, timesteps, vmax_prop,
        f"uncorrected path, further propagated +2*epsilon past refocus, "
        f"{dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, |u_out|",
        os.path.join(OUTPUT_DIR, "rbc_flow_no_correction_u_out_propagated2eps.png"),
    )

    # Speckle correlation statistic g2(tau), spatially averaged (referenced
    # to t=0) intensity autocorrelation, per course_material/"8. Laser
    # speckle contrast imaging and multiple scattering theory2.pdf".
    # Computed on the +2*epsilon-propagated field above, for both paths.
    tau = np.array(timesteps)
    g2_corr = compute_g2_spatial(u_out_prop_by_t)
    g2_uncorr = compute_g2_spatial(u_out_uncorrected_prop_by_t)

    plot_g2_comparison(
        tau, g2_corr, g2_uncorr,
        "Intensity autocorrelation $g_2$, +2*epsilon past refocus\n"
        "(spatial average, referenced to t=0)",
        os.path.join(OUTPUT_DIR, "speckle_correlation_statistics_propagated2eps.png"),
    )

    # Robustness: the correction's efficiency (Strehl) re-parametrised by how
    # far the sample has drifted from calibration (D = 1 - g2_uncorrected)
    # instead of by elapsed time, so the estimate is not tied to this run's
    # flow speed and timing. See functions.py's robustness section.
    r = compute_robustness(timesteps, g2_uncorr, strehl_corr, bg["x_max"])
    report_robustness(r)
    plot_robustness(r, os.path.join(OUTPUT_DIR, "correction_robustness_vs_drift.png"))

    plt.show()


if __name__ == "__main__":
    main()
