import numpy as np
import pytest

from main import compute_g2_spatial


def test_g2_zero_lag_is_always_exactly_one():
    # g2(0) = <I(0)*I(0)>/<I(0)*I(0)> = 1 BY CONSTRUCTION, for any field,
    # varying or not -- the denominator is the numerator's own value at
    # tau=0, same as g1(0)=1 in the field-correlation formula this
    # mirrors. Not "g2(0)>=1 with equality only if constant" (that was
    # the old, incorrect <I(0)>^2 normalization).
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)) for _ in range(5)]
    g2 = compute_g2_spatial(fields)
    assert g2[0] == pytest.approx(1.0)


def test_g2_constant_intensity_gives_flat_unity():
    # |field|=1 everywhere, every timestep (only phase varies) -> I(t) is
    # the same constant image at every t -> g2(tau)=1 for every lag
    rng = np.random.default_rng(1)
    fields = [np.exp(1j * rng.uniform(-np.pi, np.pi, size=(6, 6))) for _ in range(4)]
    g2 = compute_g2_spatial(fields)
    np.testing.assert_allclose(g2, 1.0, atol=1e-12)


def test_g2_matches_hand_computed_value():
    I0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    I1 = np.array([[4.0, 3.0], [2.0, 1.0]])
    fields = [np.sqrt(I0).astype(complex), np.sqrt(I1).astype(complex)]
    g2 = compute_g2_spatial(fields)
    expected_G2_1 = np.mean(I0 * I1)
    expected_denom = np.mean(I0 * I0)  # <I(0)^2>, NOT <I(0)>^2
    assert g2[1] == pytest.approx(expected_G2_1 / expected_denom)


def test_g2_uses_only_t0_as_reference_not_sliding_pairs():
    # a field that changes a lot AFTER t=0 but has I(0) constant should
    # still normalize by <I(0)^2> computed once (here I(0)=1 everywhere,
    # so <I(0)^2>=1, same numeric value as <I(0)>^2=1 in this special
    # case -- but see test_g2_matches_hand_computed_value for a case
    # where the two genuinely differ)
    const = np.ones((5, 5), dtype=complex)
    varying = np.array([[10.0, 0, 0, 0, 0]] * 5, dtype=complex)
    fields = [const, varying, const]
    g2 = compute_g2_spatial(fields)
    assert g2[0] == pytest.approx(1.0)
    expected_g2_1 = np.mean(np.abs(varying) ** 2) / 1.0
    assert g2[1] == pytest.approx(expected_g2_1)
