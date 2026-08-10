import numpy as np

from step_one import MAT_PATH, compute_planes, load_planes


def test_correction_is_inverse_of_aberration():
    """A~ = A* should exactly cancel A: A @ A~ = I (elementwise, since both
    are diagonal matrices -- see compute_planes)."""
    mask_f, _, _ = load_planes(MAT_PATH)
    A, B, A_tilde, B_tilde = compute_planes(mask_f)

    np.testing.assert_allclose(A * A_tilde, np.ones_like(A), atol=1e-12)
    np.testing.assert_allclose(B * B_tilde, np.ones_like(B), atol=1e-12)
