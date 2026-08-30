"""Consolidated support functions for main.py (course submission).

This file gathers every function/constant that main.py imports, directly
or transitively, from the full project's step_one.py, step_two.py and
step_two_two_rbcs.py modules -- so this submission folder is self-
contained (main.py + this one file) without needing the project's full,
scattered module structure. Nothing here is new code: each piece is
copied unchanged from its original module (noted per section below).

NOT included on purpose: the generated tissue data (mask_f_hr etc., the
aberration/correction plane data) is NOT attached to this submission --
see the accompanying docx for why, and for how to regenerate it via the
project's tissue_3d_generator.m / capillary_system/run_tissue_3d_
generator_capillary_35x35x15_two_rbcs.m.
"""
import numpy as np
import h5py

# =============================================================================
# From step_one.py: wavelength constant and angular-spectrum propagation.
# =============================================================================

LAMBDA_NM = 532.0
LAMBDA_UM = LAMBDA_NM * 1e-3


def angular_spectrum_propagate(u, z_um, lam_um, dx_um):
    """Exact angular spectrum propagation by z_um; evanescent components dropped."""
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

MAT_PATH = "capillary_system/tissue_background_35x35x15_two_rbcs.mat"
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
