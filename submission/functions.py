"""Consolidated support functions for main.py (course submission).

This file gathers every function/constant that main.py imports, directly
or transitively, from the full project's step_one.py, step_two.py and
step_two_two_rbcs.py modules -- so this submission folder is self-
contained (main.py + this one file) without needing the project's full,
scattered module structure. Everything up to and including the
compute_planes_at section is copied unchanged from its original module
(noted per section below); the final section, the robustness analysis, is
new to this submission and has no counterpart in the project's step_*.py
modules.

The generated tissue data (mask_f_hr etc., the aberration/correction
plane data) IS included in this submission, at
submission/capillary_system/tissue_background_35x35x15_two_rbcs.mat, so
this folder is fully self-contained and runnable as-is. See the
accompanying docx for how to regenerate it from scratch via the project's
tissue_3d_generator.m / capillary_system/run_tissue_3d_generator_
capillary_35x35x15_two_rbcs.m, if ever needed.
"""
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt

# =============================================================================
# From step_one.py: wavelength constant and angular-spectrum propagation.
# =============================================================================

LAMBDA_NM = 532.0
LAMBDA_UM = LAMBDA_NM * 1e-3


def angular_spectrum_propagate(u, z_um, lam_um, dx_um):
    """Propagate a 2D field using the angular-spectrum method.

    Parameters
    ----------
    u : numpy.ndarray
        Complex-valued square input field sampled on a uniform 2D grid.
    z_um : float
        Propagation distance in micrometres. Positive values propagate
        forward according to the chosen phase convention.
    lam_um : float
        Wavelength in micrometres.
    dx_um : float
        Spatial sampling interval in both transverse directions, in
        micrometres.

    Returns
    -------
    numpy.ndarray
        Complex-valued propagated field with the same shape as ``u``.
        Evanescent spatial-frequency components are discarded.
    """
    n = u.shape[0]
    fx = np.fft.fftfreq(n, d=dx_um)
    fx_grid, fy_grid = np.meshgrid(fx, fx, indexing="ij")
    arg = 1.0 - (lam_um * fx_grid) ** 2 - (lam_um * fy_grid) ** 2
    propagating = arg >= 0
    kz = np.sqrt(np.clip(arg, 0, None))
    H = np.where(propagating, np.exp(1j * 2 * np.pi * z_um / lam_um * kz), 0.0)
    return np.fft.ifft2(np.fft.fft2(u) * H)


# =============================================================================
# From step_two.py: background-tissue loading, RBC/capillary dynamics
# formulas, and coarse-plane reconstruction.
# =============================================================================

# RBC phase convention: Delta_n = 0.38 (docx: RBC index 1.38 at 532nm;
# plasma/background index = 1), applied directly as phase-in-cycles
# (exp(i*2*pi*RBC_DELTA_N)) -- matches how tissue_3d_generator.m's own
# background-sphere phase accumulation has no separate path-length term
# either. See the accompanying docx for the full justification.
RBC_DELTA_N = 0.38


def _load_complex(f, name):
    """h5py reads MATLAB's column-major arrays with all axes reversed."""
    raw = f[name][:]
    arr = raw["real"] + 1j * raw["imag"]
    return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))


def load_background(mat_path):
    with h5py.File(mat_path, "r") as f:
        mask_f_hr = _load_complex(f, "mask_f_hr")    # (Nx, Nx, Nz) fine
        mask_f0_hr = _load_complex(f, "mask_f0_hr")  # (Nx, Nx, Nz) fine
        mask_f = _load_complex(f, "mask_f")          # (Nx, Nx, M) coarse background
        z_grid1 = np.asarray(f["z_grid1"]).squeeze()          # (Nz,) fine, generator-internal (0=tissue z-center)
        z_grid1_sps = np.asarray(f["z_grid1_sps"]).squeeze()  # (M,) coarse bin centers, generator-internal
        x_max = float(np.asarray(f["x_max"]).squeeze())
        x_stp = float(np.asarray(f["x_stp"]).squeeze())
        Delta = float(np.asarray(f["Delta"]).squeeze())
        tissue_dims_um = np.asarray(f["tissue_dims_um"]).squeeze()
    return dict(
        mask_f_hr=mask_f_hr, mask_f0_hr=mask_f0_hr, mask_f=mask_f,
        z_grid1=z_grid1, z_grid1_sps=z_grid1_sps,
        x_max=x_max, x_stp=x_stp, Delta=Delta, tissue_dims_um=tissue_dims_um,
    )


def build_x_grid(x_max, x_stp, expected_n):
    n = round(2 * x_max / x_stp) + 1
    assert n == expected_n, f"x_grid length mismatch: computed {n}, expected {expected_n}"
    return np.linspace(-x_max, x_max, n)


def diameter_um(t):
    """Capillary diameter, um. Sinusoidal, 8s period, min (4um) at t=0."""
    return 7.0 - 3.0 * np.cos(2 * np.pi * t / 8.0)


def width_um(t):
    """RBC width, um -- linear in instantaneous diameter: 4um@d=4 -> 8um@d=10."""
    return 4.0 + (diameter_um(t) - 4.0) * (2.0 / 3.0)


def length_um(t):
    """RBC length, um -- linear in instantaneous diameter: 10um@d=4 -> 12um@d=10."""
    return 10.0 + (diameter_um(t) - 4.0) * (1.0 / 3.0)


def speed_um_s(t):
    """RBC speed, um/s -- linear in instantaneous diameter: 8@d=4 -> 25@d=10."""
    return 8.0 + (diameter_um(t) - 4.0) * (17.0 / 6.0)


def reconstruct_coarse(combined, touched, mask_f_background, z_grid1, z_grid1_sps, h_z_stp, x_stp):
    """Per plane: untouched columns keep MATLAB's own coarse value;
    touched columns get the plain mean of the carved fine volume over
    that bin's fine layers. Bin membership uses the exact same tolerance
    as the fixed tissue_3d_generator.m, so bins line up 1:1."""
    M = z_grid1_sps.shape[0]
    coarse = mask_f_background.copy()
    for j in range(M):
        ii0 = np.where(np.abs(z_grid1 - z_grid1_sps[j]) <= h_z_stp + x_stp * 0.1)[0]
        bin_touched = np.any(touched[:, :, ii0], axis=2)
        bin_mean = np.mean(combined[:, :, ii0], axis=2)
        coarse[:, :, j] = np.where(bin_touched, bin_mean, coarse[:, :, j])
    return coarse


# =============================================================================
# From step_two_two_rbcs.py: two-RBC timing/geometry and M=2 plane
# reconstruction for the flowing capillary scenario.
# =============================================================================

# Anchored to this file's own directory (not the process cwd), so the
# bundled copy of the tissue data resolves correctly regardless of where
# main.py is invoked from -- matches main.py's ANIMATION_PATH convention.
MAT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "capillary_system", "tissue_background_35x35x15_two_rbcs.mat",
)
TIME_STEP = 0.25


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
    """Generalized to N simultaneous cells (each an independent (y0,y1)
    interval) sharing one capillary. The capillary's axis runs along Y
    (lateral, NOT the z/propagation axis), so its diameter is a circular
    cross-section in the X-Z plane, uniform across Y. Each cell is a box,
    square in cross-section, intersected with the tube's circle so it
    never pokes past the vessel wall."""
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
    and reconstructs the coarse planes."""
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


# =============================================================================
# Robustness analysis (new in this submission -- no counterpart in the
# project's step_*.py modules).
#
# The question is how long a correction calibrated ONCE, at t=0, keeps
# working. Two choices make that estimate transferable rather than a fact
# about this particular clip:
#
#   1. Correction efficiency is measured by a focus metric (Strehl-like
#      peak intensity, plus core energy fraction), NOT by g2. g2 of the
#      propagated field turns out to be nearly blind to whether the
#      correction is on at all -- it tracks the sample's own evolution,
#      differing by <0.03 between the corrected and uncorrected paths
#      while the underlying focus quality differs by ~25x.
#
#   2. That efficiency is plotted against how far the SAMPLE has drifted
#      from its calibration state, NOT against elapsed time. Drift is
#      measured as D = 1 - g2_uncorrected: g2 on the uncorrected path is
#      the cleanest available measure of how much of the calibration
#      state survives, and is uncontaminated by the correction's own
#      ordered t=0 field. This is where g2 earns its place -- as the
#      yardstick for the perturbation, not as the measurement.
#
# Time is only a proxy for sample change, and a poor one here: RBC speed
# varies (8 um/s at t=0 to 25 um/s at t=4s), the capillary dilates (4um
# to 10um over the same window), and the cells enter and leave at
# particular moments, so one second near t=0 corresponds to far less
# sample change than one second near t=3. Any "correction lifetime in
# seconds" read off this run describes the clip's timing. Re-parametrising
# by D removes flow speed, dilation schedule and cell timing from the
# x-axis -- they act only by driving D -- so the resulting curve estimates
# a property of the correction method, and can be compared against runs
# with different flow conditions or geometry.
# =============================================================================

CORE_PX = 5


def crop_center(arr, size):
    """size x size window centered on arr's own center pixel (arr is
    square, same convention as the point source placed at [Nx//2, Nx//2])."""
    n = arr.shape[0]
    half = size // 2
    start = n // 2 - half
    return arr[start:start + size, start:start + size]


def compute_focus_quality(u_corr_by_t, u_uncorr_by_t, core_px=CORE_PX):
    """Focus quality at the refocus plane (i.e. on the raw u_out fields,
    BEFORE the +2*epsilon forward propagation the g2 statistic uses), for
    both paths: peak intensity, and the fraction of total energy inside a
    core_px x core_px window at the frame centre.

    Both peak curves are normalised to the CORRECTED t=0 frame -- the
    ideal-correction ground truth, where the fixed correction is exact by
    construction -- so the two paths sit on one absolute scale, the same
    convention as the shared vmax in the amplitude grids. The corrected
    curve is then a Strehl ratio: 1.0 at t=0, decaying as the fixed
    correction goes stale."""
    I_corr = [np.abs(u) ** 2 for u in u_corr_by_t]
    I_uncorr = [np.abs(u) ** 2 for u in u_uncorr_by_t]
    peak_ref = I_corr[0].max()
    strehl_corr = np.array([I.max() for I in I_corr]) / peak_ref
    strehl_uncorr = np.array([I.max() for I in I_uncorr]) / peak_ref
    core_corr = np.array([crop_center(I, core_px).sum() / I.sum() for I in I_corr])
    core_uncorr = np.array([crop_center(I, core_px).sum() / I.sum() for I in I_uncorr])
    return strehl_corr, strehl_uncorr, core_corr, core_uncorr


def compute_robustness(timesteps, g2_uncorr, strehl, x_max):
    """Pair the correction's efficiency with the sample drift that caused
    it, and work out how much of the run is usable for the drift plot.

    Only the leg up to the moment cell 1's leading edge reaches the far
    edge of the tissue is a clean "sample drifting away from calibration"
    record. Past it, the cell starts clearing the beam: drift keeps rising
    for a few more timesteps because cell 2 is still advancing, but it is
    now a mix of two opposing effects, and shortly after, the sample turns
    around and drifts back TOWARDS its calibration state -- which would
    make the curve double back on itself."""
    t = np.asarray(timesteps, dtype=float)
    g2_uncorr = np.asarray(g2_uncorr, dtype=float)
    strehl = np.asarray(strehl, dtype=float)

    t_exit_start = find_time_for_displacement(2 * x_max, t_start=0.0)
    return dict(
        t=t,
        g2_uncorr=g2_uncorr,
        strehl=strehl,
        D=1.0 - g2_uncorr,
        t_exit_start=t_exit_start,
        n_plot=int(np.searchsorted(t, t_exit_start, side="right")),
    )


def plot_robustness(r, out_path):
    """(a) the two clocks against time, over the whole run, to motivate the
    reparametrisation -- correction efficiency and the sample's own
    similarity to t=0 run down at visibly different rates.

    (b) the same efficiency against drift, truncated to the clean leg."""
    t, D, S, n = r["t"], r["D"], r["strehl"], r["n_plot"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    ax = axes[0]
    ax.plot(t, S, "o-", color="C0", label="correction efficiency (Strehl)")
    ax.plot(t, r["g2_uncorr"], "s-", color="C1",
            label=r"sample similarity to $t{=}0$  ($g_2$, uncorrected)")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("normalised to $t=0$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) Normalised Strehl and $g_2$ over time", fontsize=11)

    ax = axes[1]
    ax.plot(D[:n], S[:n], "o-", color="C0")
    ax.set_xlabel(r"sample drift   $D = 1 - g_2^{\,\mathrm{uncorr}}$")
    ax.set_ylabel("Strehl")
    ax.set_xlim(0, D[:n].max() * 1.08)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"(b) efficiency vs. drift, while the sample decorrelates "
                 f"($t \\leq {t[n - 1]:g}$s)", fontsize=11)

    fig.suptitle("Robustness of the $t=0$-calibrated plane correction: efficiency "
                 "per unit of sample decorrelation, not per second")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def report_robustness(r):
    """Console summary: the per-timestep table, the drift at which half the
    correction is gone, and the fragility gain G = (1-Strehl)/D -- the
    efficiency lost per unit of sample drift. G=1 would mean the correction
    is exactly as robust as the medium allows; G>1 means it is more fragile
    than the sample's own change accounts for."""
    t, D, S = r["t"], r["D"], r["strehl"]
    G = np.full_like(S, np.nan)
    G[1:] = (1.0 - S[1:]) / D[1:]

    print(f"{'t':>6} {'g2_unc':>8} {'drift D':>9} {'Strehl':>8} {'1-Strehl':>9} {'G':>7}")
    for i in range(len(t)):
        g = "     --" if i == 0 else f"{G[i]:7.2f}"
        print(f"{t[i]:6.2f} {r['g2_uncorr'][i]:8.4f} {D[i]:9.4f} "
              f"{S[i]:8.4f} {1 - S[i]:9.4f} {g}")

    i = int(np.argmax(S < 0.5))
    D_half = np.interp(0.5, [S[i], S[i - 1]], [D[i], D[i - 1]])
    print(f"\nHalf the correction is gone by D = {D_half:.3f} "
          f"(sample only {100 * D_half:.1f}% decorrelated)")
    print(f"Fragility gain G: {np.nanmin(G):.2f} to {np.nanmax(G):.2f}, "
          f"median {np.nanmedian(G):.2f}")

    turn = int(np.argmax(D))
    print(f"\nDrift plot truncated at t={r['t_exit_start']:.2f}s, when cell 1's leading "
          f"edge reaches the far edge and it starts clearing the beam. Drift keeps rising "
          f"to D={D[turn]:.3f} at t={t[turn]:g}s (cell 2 still advancing) before turning.")
    print("Beyond the turn the sample drifts back toward calibration -- retrace check:")
    for j in range(turn + 1, len(t)):
        S_out = np.interp(D[j], D[:turn + 1], S[:turn + 1])
        print(f"  t={t[j]:4.2f}s  D={D[j]:.3f}  Strehl={S[j]:.3f}  "
              f"vs {S_out:.3f} outbound  (diff {S[j] - S_out:+.3f})")
