"""Sim step 1 (see 'Diffusion optics project guidelines.docx').

M=2 aberration planes A, B (from 2_planes_system/tissue_output_2planes_*.mat).
Ideal correction: A~ = conj(A), B~ = conj(B).

u_out = P(-eps)@A~ @ P(-eps)@B~ @ B @ P(eps)@A @ P(eps) @ u_in

Propagation P(z) is the exact (non-paraxial) angular spectrum kernel,
evanescent components dropped (see the added "Propagation kernel used in
step 1" section of the guidelines docx). lambda is given in nm; all
spatial quantities (x_stp, eps, ...) are in um.
"""
import glob
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

MAT_PATH = "2_planes_system/tissue_output_2planes_20x20x10.mat"
OUTPUT_DIR = "step 1 results"
LAMBDA_NM = 532.0
LAMBDA_UM = LAMBDA_NM * 1e-3

correct_plane_A = 0
correct_plane_B = 0

# Multiplies the plotted (not physical) amplitude for a brighter image.
# Each panel's color scale is still capped at that panel's own true peak
# (so the colorbar stays physically meaningful), but the gain is applied
# *before* the clip -- boosting sub-peak detail (e.g. faint speckle) into
# visible range faster, instead of just rescaling (which would be a no-op
# under plain autoscale, since scaling data and its own max cancels out).
amplitude_gain = 5


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


def load_planes(mat_path):
    with h5py.File(mat_path, "r") as f:
        raw = f["mask_f"][:]  # MATLAB complex -> compound dtype [('real',),('imag',)], shape (M, Ny, Nx)
        mask_f = raw["real"] + 1j * raw["imag"]
        mask_f = np.transpose(mask_f, (2, 1, 0))  # -> (Nx, Ny, M), matches MATLAB mask_f(:,:,k)
        dx_um = float(np.asarray(f["x_stp"]).squeeze())
        eps_um = float(np.asarray(f["Delta"]).squeeze())
        tissue_dims_um = np.asarray(f["tissue_dims_um"]).squeeze()  # [width_x, width_y, depth_z]
    return mask_f, dx_um, eps_um, tissue_dims_um


def dims_tag(tissue_dims_um):
    """e.g. [20.,20.,10.] -> '20x20x10', for filenames/titles."""
    return "x".join(f"{d:g}" for d in tissue_dims_um)


def compute_planes(mask_f):
    """Aberration planes A, B and their ideal (conjugate) corrections A~, B~.

    A, B, A~, B~ are diagonal N^2xN^2 matrices (guidelines' "Dimensions"
    section); we store just their diagonal as an NxN phase image, since
    applying a diagonal matrix to u is an elementwise product with u, and
    composing two diagonal matrices (e.g. A @ A~) is an elementwise product
    of their diagonals.
    """
    A = np.exp(1j * np.angle(mask_f[:, :, 0]))
    B = np.exp(1j * np.angle(mask_f[:, :, 1]))
    A_tilde = np.conj(A)
    B_tilde = np.conj(B)
    return A, B, A_tilde, B_tilde


def main(mat_path=MAT_PATH, show=True):
    mask_f, dx_um, eps_um, tissue_dims_um = load_planes(mat_path)
    tag = dims_tag(tissue_dims_um)
    n = mask_f.shape[0]

    A, B, A_tilde, B_tilde = compute_planes(mask_f)

    u_in = np.zeros((n, n), dtype=complex)
    u_in[n // 2, n // 2] = 1.0 + 0j

    u1 = angular_spectrum_propagate(u_in, eps_um, LAMBDA_UM, dx_um)
    u_after_A = A * u1

    u2 = angular_spectrum_propagate(u_after_A, eps_um, LAMBDA_UM, dx_um)
    u_after_B = B * u2  # = field leaving the tissue

    if correct_plane_B:
        u_after_Btilde = B_tilde * u_after_B
    else:
        u_after_Btilde = u_after_B

    u3 = angular_spectrum_propagate(u_after_Btilde, -eps_um, LAMBDA_UM, dx_um)
    
    if correct_plane_A:
        u_after_Atilde = A_tilde * u3
    else:
        u_after_Atilde = u3

    u_out = angular_spectrum_propagate(u_after_Atilde, -eps_um, LAMBDA_UM, dx_um)

    fields = [
        ("input field |u_in|", u_in, False),
        ("after aberration A", u_after_A, False),
        ("after aberration B (tissue output)", u_after_B, False),
        ("after correction B~" if correct_plane_B else "B~ correction skipped", u_after_Btilde, False),
        ("after correction A~" if correct_plane_A else "A~ correction skipped", u_after_Atilde, False),
        ("after further eps propagation (u_out)", u_out, True),
    ]

    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4))
    for ax, (title, field, apply_gain) in zip(axes, fields):
        amp = np.abs(field)
        vmax = amp.max()
        if apply_gain:
            amp = np.clip(amp * amplitude_gain, 0, vmax)
        im = ax.imshow(amp, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Sim step 1: M=2 aberration planes, tissue {tag} um, ideal conjugate correction")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    skipped = [name for name, on in [("noA", correct_plane_A), ("noB", correct_plane_B)] if not on]
    suffix = "" if not skipped else "_" + "_".join(skipped)
    filename = f"step_one_fields_{tag}{suffix}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    residual = np.linalg.norm(np.abs(u_out) - np.abs(u_in)) / np.linalg.norm(np.abs(u_in))
    print(f"||u_out| - |u_in|| / ||u_in|| = {residual:.4f}  "
          f"(nonzero: exact correction only holds within the propagating/non-evanescent band)")

    if show:
        plt.show()


if __name__ == "__main__":
    all_tissues = sorted(glob.glob("2_planes_system/tissue_output_2planes_*.mat"))
    skip = ["50x50x10"]  # skip plotting these tissues for now
    for path in all_tissues:
        if any(tag in path for tag in skip):
            continue
        main(mat_path=path)
