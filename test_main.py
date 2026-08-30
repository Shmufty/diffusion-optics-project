import numpy as np
import pytest

from step_one import LAMBDA_UM, angular_spectrum_propagate
from main import run_pipeline


def _random_phase_plane(n, seed):
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(n, n)))


def test_matching_correction_reproduces_step_one_style_pipeline():
    """When the 'fixed' correction is the exact conjugate of the SAME
    A/B used for the forward pass (i.e. t=0 evaluated at t=0), running
    through run_pipeline must equal manually composing the identical
    8-operator sequence step_one.py uses -- two independent ways of
    computing the same physics should agree exactly."""
    n = 64
    eps_um = 5.0
    dx_um = 0.1
    A = _random_phase_plane(n, seed=1)
    B = _random_phase_plane(n, seed=2)
    A_tilde = np.conj(A)
    B_tilde = np.conj(B)

    u_in = np.zeros((n, n), dtype=complex)
    u_in[n // 2, n // 2] = 1.0 + 0j

    got = run_pipeline(u_in, A, B, A_tilde, B_tilde, eps_um, dx_um)

    u1 = angular_spectrum_propagate(u_in, eps_um, LAMBDA_UM, dx_um)
    u_after_A = A * u1
    u2 = angular_spectrum_propagate(u_after_A, eps_um, LAMBDA_UM, dx_um)
    u_after_B = B * u2
    u_after_Btilde = B_tilde * u_after_B
    u3 = angular_spectrum_propagate(u_after_Btilde, -eps_um, LAMBDA_UM, dx_um)
    u_after_Atilde = A_tilde * u3
    expected = angular_spectrum_propagate(u_after_Atilde, -eps_um, LAMBDA_UM, dx_um)

    np.testing.assert_allclose(got, expected)


def test_matching_correction_at_own_timestep_beats_mismatched_correction():
    """A correction computed FOR the same A/B it's applied to should
    refocus at least as well (peak amplitude) as a correction computed
    for a DIFFERENT, unrelated A/B (the 'stale correction' case this
    script is built to visualize)."""
    n = 64
    eps_um = 5.0
    dx_um = 0.1
    A = _random_phase_plane(n, seed=10)
    B = _random_phase_plane(n, seed=11)
    A_other = _random_phase_plane(n, seed=20)
    B_other = _random_phase_plane(n, seed=21)

    u_in = np.zeros((n, n), dtype=complex)
    u_in[n // 2, n // 2] = 1.0 + 0j

    matched = run_pipeline(u_in, A, B, np.conj(A), np.conj(B), eps_um, dx_um)
    mismatched = run_pipeline(u_in, A, B, np.conj(A_other), np.conj(B_other), eps_um, dx_um)

    assert np.abs(matched).max() > np.abs(mismatched).max()
