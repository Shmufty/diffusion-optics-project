import numpy as np
import pytest

from step_two import RBC_DELTA_N, diameter_um, length_um, speed_um_s
from step_two_two_rbcs import (
    build_timeline,
    carve_fine_volume_multi,
    cell_edges,
    cell_fully_exited,
    cumulative_displacement_um,
    find_time_for_displacement,
)

Y_MAX = 17.5


def test_cumulative_displacement_zero_length_interval():
    assert cumulative_displacement_um(1.0, 1.0) == 0.0
    assert cumulative_displacement_um(2.0, 1.0) == 0.0  # end before start -> 0


def test_cumulative_displacement_matches_manual_integration_over_subinterval():
    # displacement(0,2) should equal displacement(0,1) + displacement(1,2)
    d_full = cumulative_displacement_um(0.0, 2.0)
    d_split = cumulative_displacement_um(0.0, 1.0) + cumulative_displacement_um(1.0, 2.0)
    assert d_full == pytest.approx(d_split, rel=1e-6)


def test_find_time_for_displacement_round_trips():
    target = 10.0
    t_found = find_time_for_displacement(target, t_start=0.5)
    d = cumulative_displacement_um(0.5, t_found)
    assert d == pytest.approx(target, rel=1e-6)


def test_cell2_entry_time_matches_cell1_leading_edge_at_midpoint():
    t_entry_2 = find_time_for_displacement(Y_MAX, t_start=0.0)
    _, y1_cell1_at_trigger = cell_edges(t_entry_2, 0.0, Y_MAX)
    assert y1_cell1_at_trigger == pytest.approx(0.0, abs=1e-6)  # tissue Y-midpoint


def test_cell_not_present_before_entry():
    y0, y1 = cell_edges(0.5, t_entry=1.0, y_max=Y_MAX)
    assert y0 < -Y_MAX and y1 < -Y_MAX  # entirely outside the tissue


def test_cell_gradual_entry_starts_at_boundary_and_grows():
    # at t=t_entry, leading edge is exactly at the boundary, trailing edge is behind it
    # (i.e. mostly/fully outside) -- confirms "gradual" entry, not instant full-length appearance
    y0_at_entry, y1_at_entry = cell_edges(1.0, t_entry=1.0, y_max=Y_MAX)
    assert y1_at_entry == pytest.approx(-Y_MAX, abs=1e-9)
    assert y0_at_entry < y1_at_entry  # trailing edge behind (outside), not coincident

    # shortly after, the cell has advanced -- more of it is inside than at t=t_entry
    y0_later, y1_later = cell_edges(1.25, t_entry=1.0, y_max=Y_MAX)
    assert y1_later > y1_at_entry
    visible_len_at_entry = min(y1_at_entry, Y_MAX) - max(y0_at_entry, -Y_MAX)
    visible_len_later = min(y1_later, Y_MAX) - max(y0_later, -Y_MAX)
    assert visible_len_later > visible_len_at_entry


def test_build_timeline_starts_at_zero_and_stops_when_both_exited():
    t_entry_2, timesteps = build_timeline(Y_MAX)
    assert timesteps[0] == 0.0
    # strictly increasing, 0.25s steps
    diffs = np.diff(timesteps)
    np.testing.assert_allclose(diffs, 0.25, atol=1e-9)
    # both cells confirmed exited at the final recorded step
    y0_1, _ = cell_edges(timesteps[-1], 0.0, Y_MAX)
    y0_2, _ = cell_edges(timesteps[-1], t_entry_2, Y_MAX)
    assert cell_fully_exited(y0_1, Y_MAX)
    assert cell_fully_exited(y0_2, Y_MAX)
    # ...but NOT yet at the second-to-last step (otherwise the loop should have stopped earlier)
    y0_1_prev, _ = cell_edges(timesteps[-2], 0.0, Y_MAX)
    y0_2_prev, _ = cell_edges(timesteps[-2], t_entry_2, Y_MAX)
    assert not (cell_fully_exited(y0_1_prev, Y_MAX) and cell_fully_exited(y0_2_prev, Y_MAX))


def test_carve_multi_two_separated_cells_both_present():
    x_grid1 = np.linspace(-10, 10, 201)
    z_grid1 = np.linspace(-10, 10, 201)
    mask_f_hr = np.zeros((201, 201, 201), dtype=complex)
    t = 0.0
    cell_a = (-8.0, -6.0)
    cell_b = (2.0, 4.0)
    combined, touched = carve_fine_volume_multi(t, mask_f_hr, x_grid1, z_grid1, [cell_a, cell_b], RBC_DELTA_N)

    cx = np.argmin(np.abs(x_grid1))  # x=0, within tube (radius=2 at t=0)
    cz = np.argmin(np.abs(z_grid1))
    y_in_a = np.argmin(np.abs(x_grid1 - (-7.0)))
    y_in_b = np.argmin(np.abs(x_grid1 - 3.0))
    y_between = np.argmin(np.abs(x_grid1 - 0.0))  # plasma, no cell here

    rbc_value = np.exp(1j * 2 * np.pi * RBC_DELTA_N)
    assert combined[cx, y_in_a, cz] == pytest.approx(rbc_value)
    assert combined[cx, y_in_b, cz] == pytest.approx(rbc_value)
    assert combined[cx, y_between, cz] == pytest.approx(1.0 + 0.0j)  # tube present, but plasma (no cell)


def test_diameter_still_periodic_after_import():
    # sanity: reused formulas from step_two behave as expected here too
    assert diameter_um(0.0) == pytest.approx(4.0)
    assert length_um(0.0) == pytest.approx(10.0)
    assert speed_um_s(0.0) == pytest.approx(8.0)
