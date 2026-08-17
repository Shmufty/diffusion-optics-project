"""Sim step 1 (see 'Diffusion optics project guidelines.docx').

M=2 aberration planes A, B (from 2_planes_system/tissue_output_2planes.mat).
Ideal correction: A~ = conj(A), B~ = conj(B).

u_out = P(-eps)@A~ @ P(-eps)@B~ @ B @ P(eps)@A @ P(eps) @ u_in

Propagation P(z) is the exact (non-paraxial) angular spectrum kernel,
evanescent components dropped (see the added "Propagation kernel used in
step 1" section of the guidelines docx). lambda is given in nm; all
spatial quantities (x_stp, eps, ...) are in um.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

MAT_PATH = "2_planes_system/tissue_output_2planes.mat"
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


def load_planes(mat_path):
    with h5py.File(mat_path, "r") as f:
        raw = f["mask_f"][:]  # MATLAB complex -> compound dtype [('real',),('imag',)], shape (M, Ny, Nx)
        mask_f = raw["real"] + 1j * raw["imag"]
        mask_f = np.transpose(mask_f, (2, 1, 0))  # -> (Nx, Ny, M), matches MATLAB mask_f(:,:,k)
        dx_um = float(np.asarray(f["x_stp"]).squeeze())
        eps_um = float(np.asarray(f["Delta"]).squeeze())
    return mask_f, dx_um, eps_um


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


def main():
    mask_f, dx_um, eps_um = load_planes(MAT_PATH)
    n = mask_f.shape[0]

    A, B, A_tilde, B_tilde = compute_planes(mask_f)

    u_in = np.zeros((n, n), dtype=complex)
    u_in[n // 2, n // 2] = 1.0 + 0j

    u1 = angular_spectrum_propagate(u_in, eps_um, LAMBDA_UM, dx_um)
    u_after_A = A * u1

    u2 = angular_spectrum_propagate(u_after_A, eps_um, LAMBDA_UM, dx_um)
    u_after_B = B * u2  # = field leaving the tissue
    u_after_Btilde = B_tilde * u_after_B

    u3 = angular_spectrum_propagate(u_after_Btilde, -eps_um, LAMBDA_UM, dx_um)
    u_after_Atilde = A_tilde * u3

    u_out = angular_spectrum_propagate(u_after_Atilde, -eps_um, LAMBDA_UM, dx_um)

    planes = [
        ("A (aberration, plane 1)", A),
        ("B (aberration, plane 2)", B),
        ("A~ = conj(A) (correction)", A_tilde),
        ("B~ = conj(B) (correction)", B_tilde),
    ]

    fig, axes = plt.subplots(1, len(planes), figsize=(4 * len(planes), 4))
    for ax, (title, plane) in zip(axes, planes):
        im = ax.imshow(np.angle(plane), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")
    fig.suptitle("Sim step 1: aberration / correction planes (phase, since |A|=|B|=1)")
    fig.tight_layout()
    fig.savefig("step_one_planes.png", dpi=150)
    print("Saved plot to step_one_planes.png")

    fields = [
        ("input field |u_in|", u_in),
        ("after aberration A", u_after_A),
        ("after aberration B (tissue output)", u_after_B),
        ("after correction B~", u_after_Btilde),
        ("after correction A~", u_after_Atilde),
        ("after further eps propagation (u_out)", u_out),
    ]

    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4))
    for ax, (title, field) in zip(axes, fields):
        im = ax.imshow(np.abs(field), cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Sim step 1: M=2 aberration planes, ideal conjugate correction")
    fig.tight_layout()
    fig.savefig("step_one_fields.png", dpi=150)
    print("Saved plot to step_one_fields.png")

    residual = np.linalg.norm(np.abs(u_out) - np.abs(u_in)) / np.linalg.norm(np.abs(u_in))
    print(f"||u_out| - |u_in|| / ||u_in|| = {residual:.4f}  "
          f"(nonzero: exact correction only holds within the propagating/non-evanescent band)")

    plt.show()


if __name__ == "__main__":
    main()
