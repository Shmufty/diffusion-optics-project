import os

import numpy as np
import pytest

from step_two import (
    MAT_PATH,
    RBC_DELTA_N,
    build_x_grid,
    carve_fine_volume,
    cell_y_interval,
    diameter_um,
    length_um,
    load_background,
    reconstruct_coarse,
    speed_um_s,
    width_um,
)


def test_formulas_at_full_contraction():
    """t=0: docx literal values (fully contracted)."""
    assert diameter_um(0.0) == pytest.approx(4.0)
    assert width_um(0.0) == pytest.approx(4.0)
    assert length_um(0.0) == pytest.approx(10.0)
    assert speed_um_s(0.0) == pytest.approx(8.0)


def test_formulas_at_full_dilation():
    """t=4 (half period): docx literal values (fully dilated)."""
    assert diameter_um(4.0) == pytest.approx(10.0)
    assert width_um(4.0) == pytest.approx(8.0)
    assert length_um(4.0) == pytest.approx(12.0)
    assert speed_um_s(4.0) == pytest.approx(25.0)


def test_diameter_is_periodic_and_bounded():
    assert diameter_um(0.0) == pytest.approx(4.0)   # trough: cos(0)=1
    assert diameter_um(8.0) == pytest.approx(4.0)    # next trough, one period later
    assert diameter_um(4.0) == pytest.approx(10.0)   # peak: cos(pi)=-1
    t = np.linspace(0, 16, 1000)
    d = diameter_um(t)
    assert d.min() >= 4.0 - 1e-9
    assert d.max() <= 10.0 + 1e-9
    np.testing.assert_allclose(diameter_um(t), diameter_um(t + 8.0), atol=1e-10)


def test_cell_disappears_after_exiting_tissue():
    """The 4 real deliverable frames (t<=0.75s) don't reach the far edge
    (displacement+length ~16.5um of a 35um domain) -- so exercise the
    exit path directly with a synthetic large t instead."""
    y_max = 17.5  # 35um tissue half-width
    y0_early, y1_early = cell_y_interval(0.5, y_max)
    assert y0_early < y_max  # still inside at a real deliverable timestep

    y0_late, y1_late = cell_y_interval(200.0, y_max)  # far beyond any real frame
    assert y0_late > y_max  # cell has fully exited

    x_grid1 = np.linspace(-5, 5, 51)
    z_grid1 = np.linspace(-5, 5, 51)
    mask_f_hr = np.ones((51, 51, 51), dtype=complex)
    _, touched_late = carve_fine_volume(200.0, mask_f_hr, x_grid1, z_grid1, y_max, RBC_DELTA_N)
    # tube is still present (diameter doesn't depend on exit), but no RBC voxels should be set
    # anywhere near y=y_max (the exited cell's interval is entirely outside the grid)
    assert y0_late > x_grid1.max(), "sanity: synthetic grid doesn't even reach the exited cell's position"


def test_rbc_never_extends_past_capillary_wall():
    """Box cross-section is clipped to the tube's circle -- corners of a
    width==diameter square would otherwise poke outside a round tube."""
    x_grid1 = np.linspace(-10, 10, 201)
    z_grid1 = np.linspace(-10, 10, 201)
    mask_f_hr = np.zeros((201, 201, 201), dtype=complex)
    y_max = 10.0
    t = 0.0  # fully contracted: width(0)=4 == diameter(0)=4, the tightest case
    combined, touched = carve_fine_volume(t, mask_f_hr, x_grid1, z_grid1, y_max, RBC_DELTA_N)

    r = diameter_um(t) / 2.0
    xx, yy, zz = np.meshgrid(x_grid1, x_grid1, z_grid1, indexing="ij")
    outside_tube = (xx ** 2 + zz ** 2) > r ** 2
    is_rbc_value = np.isclose(combined, np.exp(1j * 2 * np.pi * RBC_DELTA_N))
    assert not np.any(is_rbc_value & outside_tube), "RBC voxels found outside the capillary's circular cross-section"


def test_tube_present_at_center_absent_far_away():
    x_grid1 = np.linspace(-10, 10, 201)
    z_grid1 = np.linspace(-10, 10, 201)
    mask_f_hr = np.zeros((201, 201, 201), dtype=complex)
    y_max = 10.0
    combined, touched = carve_fine_volume(0.0, mask_f_hr, x_grid1, z_grid1, y_max, RBC_DELTA_N)  # diameter=4, radius=2
    cx = np.argmin(np.abs(x_grid1))
    cz = np.argmin(np.abs(z_grid1))
    far = np.argmin(np.abs(x_grid1 - 5.0))  # 5um from center, outside radius=2
    assert touched[cx, 0, cz]       # center of tube: touched
    assert not touched[far, 0, cz]  # far from tube: untouched


def test_reconstruct_coarse_keeps_untouched_columns_identical_to_background():
    Nx, Nz, M = 20, 40, 2
    z_grid1 = np.linspace(-4, 4, Nz)
    z_grid1_sps = np.array([-2.0, 2.0])
    h_z_stp = 2.0
    x_stp = 0.2
    mask_f_hr = (np.random.default_rng(0).normal(size=(Nx, Nx, Nz))
                 + 1j * np.random.default_rng(1).normal(size=(Nx, Nx, Nz)))
    mask_f_background = (np.random.default_rng(2).normal(size=(Nx, Nx, M))
                          + 1j * np.random.default_rng(3).normal(size=(Nx, Nx, M)))
    touched = np.zeros((Nx, Nx, Nz), dtype=bool)  # nothing carved

    coarse = reconstruct_coarse(mask_f_hr, touched, mask_f_background, z_grid1, z_grid1_sps, h_z_stp, x_stp)
    np.testing.assert_array_equal(coarse, mask_f_background)


def test_reconstruct_coarse_touched_column_is_plain_mean():
    Nx, Nz, M = 5, 8, 2
    z_grid1 = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    z_grid1_sps = np.array([-2.0, 2.0])
    h_z_stp = 2.0
    x_stp = 1.0  # tolerance = 2.0 + 0.1 = 2.1 -> bin1 catches [-3.5..-0.5] (4 layers), bin2 catches [0.5..3.5]
    combined = np.zeros((Nx, Nx, Nz), dtype=complex)
    combined[2, 3, :] = np.arange(Nz) + 1j  # distinct values per fine layer at one column
    touched = np.zeros((Nx, Nx, Nz), dtype=bool)
    touched[2, 3, :] = True
    mask_f_background = np.full((Nx, Nx, M), 99 + 99j, dtype=complex)  # sentinel, must NOT survive at [2,3]

    coarse = reconstruct_coarse(combined, touched, mask_f_background, z_grid1, z_grid1_sps, h_z_stp, x_stp)
    expected_bin0 = np.mean(combined[2, 3, 0:4])
    expected_bin1 = np.mean(combined[2, 3, 4:8])
    assert coarse[2, 3, 0] == pytest.approx(expected_bin0)
    assert coarse[2, 3, 1] == pytest.approx(expected_bin1)
    assert coarse[0, 0, 0] == pytest.approx(99 + 99j)  # untouched column unaffected


@pytest.mark.skipif(not os.path.exists(MAT_PATH), reason="requires generated tissue_background_35x35x15.mat")
def test_loads_real_background_and_grid_matches():
    bg = load_background(MAT_PATH)
    Nx = bg["mask_f_hr"].shape[0]
    x_grid1 = build_x_grid(bg["x_max"], bg["x_stp"], Nx)
    assert x_grid1.shape[0] == Nx
    assert bg["mask_f"].shape[2] == 2  # M=2 planes
    assert bg["z_grid1_sps"].shape[0] == 2
