"""Sim step 2 (see 'Diffusion optics project guidelines.docx').

M=2 aberration planes A, B for a 35x35x15um tissue (capillary_system/
tissue_background_35x35x15.mat), with a capillary + single red blood
cell (RBC) carved in at 4 timesteps (t=0,0.25,0.5,0.75s -- 4Hz sampling,
one second total). The background random-sphere tissue is generated
ONCE (in MATLAB, see capillary_system/run_tissue_3d_generator_capillary_
35x35x15.m) and is identical across all 4 timesteps; only the capillary
width and the cell's position/size change per the docx's sinusoidal
dilation model.

Capillary/cell geometry: the capillary's axis runs along Y (lateral,
NOT the z/propagation axis), so its stated diameter is a circular
cross-section in the X-Z plane, centered at the tissue's lateral AND
depth center (x=0, z=0 in tissue_3d_generator's own generator-internal,
z-centered coordinates -- NOT the forward z=0-at-entrance convention
used for propagation in step_one.py). The cell is a box (docx: "model
the cell as a rectangular box"), square in cross-section (width x width,
intersected with the tube's circular cross-section so it never poke
past the vessel wall), moving along Y.

Coarse-plane reconstruction: once M=2 planes are computed with the FIXED
tissue_3d_generator.m averaging (see that file's changelog), plane A's
averaging window is forward z in [0, Z_tissue/2] and plane B's is
[Z_tissue/2, Z_tissue] -- contiguous halves of the tissue depth, not
thin slices at arbitrary z. A capillary centered at the tissue's z-
midpoint therefore straddles exactly the boundary between the two
planes' windows, giving both a meaningful share.

Rather than reimplementing the full fine-to-coarse averaging formula in
Python (risking drift from the MATLAB logic), untouched (x,y) columns
just reuse MATLAB's own already-correct coarse `mask_f` directly; only
columns actually touched by the carved capillary/cell get recomputed,
as a plain mean of the carved fine volume over that bin's fine layers.

RBC phase: Delta_n = 0.38 (docx: RBC index 1.38 at 532nm; user-specified
plasma/background index = 1). Applied DIRECTLY as phase-in-cycles
(exp(i*2*pi*RBC_DELTA_N)), not via an explicit 2*pi*Delta_n*L/lambda
formula -- this matches the only existing precedent in this codebase:
tissue_3d_generator.m's own background-sphere phase accumulation has no
separate path-length term either (see "choices made.docx" / the plan
docx for the full justification and the flagged alternative reading).
RBC_DELTA_N is a prominent, easily-changed constant below for exactly
this reason.
"""
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

MAT_PATH = "capillary_system/tissue_background_35x35x15.mat"
OUTPUT_DIR = "step 2 results"
TIMESTEPS = [0.0, 0.25, 0.5, 0.75]

# RBC phase convention -- see module docstring. Change to a
# 2*pi*RBC_DELTA_N*L/LAMBDA_UM formula here if the direct-cycles reading
# turns out to be wrong; everything downstream only depends on this value.
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


def cumulative_displacement_um(t):
    """Integral of speed_um_s from 0 to t (fine trapezoidal quadrature,
    not a hand-derived closed form -- avoids arithmetic-slip risk)."""
    if t <= 0:
        return 0.0
    tt = np.linspace(0.0, t, 2001)
    return float(np.trapezoid(speed_um_s(tt), tt))


def cell_y_interval(t, y_max):
    """RBC's [y_start, y_end] at time t. Starts with its bottom at the
    tissue's Y-entrance edge (y=-y_max) at t=0, moves toward +y. A cell
    that has moved past y_max simply produces an empty (fully-outside)
    interval -- carving naturally clips it away, no special-case needed."""
    y0 = -y_max + cumulative_displacement_um(t)
    y1 = y0 + length_um(t)
    return y0, y1


def carve_fine_volume(t, mask_f_hr, x_grid1, z_grid1, y_max, rbc_delta_n):
    """Overwrite the capillary (plasma, 1+0j) and RBC (exp(i*2*pi*Delta_n))
    voxels in a copy of the fine background volume. Returns (combined,
    touched) -- touched marks every fine voxel that was overwritten."""
    combined = mask_f_hr.copy()
    Nx = x_grid1.shape[0]
    Nz = z_grid1.shape[0]

    xx = x_grid1[:, None]  # (Nx,1)
    zz = z_grid1[None, :]  # (1,Nz)

    r_tube = diameter_um(t) / 2.0
    tube_xz = (xx ** 2 + zz ** 2) <= r_tube ** 2  # (Nx, Nz), uniform across Y

    w = width_um(t)
    box_xz = (np.abs(xx) <= w / 2.0) & (np.abs(zz) <= w / 2.0) & tube_xz

    y0, y1 = cell_y_interval(t, y_max)
    cell_y_mask = (x_grid1 >= y0) & (x_grid1 <= y1)  # Y uses the same grid as X (square tissue)

    tube_3d = np.broadcast_to(tube_xz[:, None, :], (Nx, Nx, Nz))
    box_3d = np.broadcast_to(box_xz[:, None, :], (Nx, Nx, Nz)) & cell_y_mask[None, :, None]

    combined[tube_3d] = 1.0 + 0.0j
    combined[box_3d] = np.exp(1j * 2 * np.pi * rbc_delta_n)

    touched = tube_3d | box_3d
    return combined, touched


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


def build_x_grid(x_max, x_stp, expected_n):
    n = round(2 * x_max / x_stp) + 1
    assert n == expected_n, f"x_grid length mismatch: computed {n}, expected {expected_n}"
    return np.linspace(-x_max, x_max, n)


def main():
    bg = load_background(MAT_PATH)
    Nx = bg["mask_f_hr"].shape[0]
    x_grid1 = build_x_grid(bg["x_max"], bg["x_stp"], Nx)
    y_max = bg["x_max"]  # square tissue: Y uses the same extent/grid as X
    h_z_stp = bg["Delta"] / 2.0

    planes_by_t = []
    for t in TIMESTEPS:
        combined, touched = carve_fine_volume(
            t, bg["mask_f_hr"], x_grid1, bg["z_grid1"], y_max, RBC_DELTA_N)
        coarse = reconstruct_coarse(
            combined, touched, bg["mask_f"], bg["z_grid1"], bg["z_grid1_sps"], h_z_stp, bg["x_stp"])
        A = np.exp(1j * np.angle(coarse[:, :, 0]))
        B = np.exp(1j * np.angle(coarse[:, :, 1]))
        planes_by_t.append((A, B))

    fig, axes = plt.subplots(4, 2, figsize=(9, 17))
    for i, t in enumerate(TIMESTEPS):
        A, B = planes_by_t[i]
        for j, (label, plane) in enumerate([("A", A), ("B", B)]):
            ax = axes[i, j]
            im = ax.imshow(np.angle(plane), cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"t={t}s, plane {label}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")
    dims = bg["tissue_dims_um"]
    fig.suptitle(f"Sim step 2: capillary + RBC, {dims[0]:g}x{dims[1]:g}x{dims[2]:g}um tissue, M=2 planes")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "step_two_phases_35x35x15.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
