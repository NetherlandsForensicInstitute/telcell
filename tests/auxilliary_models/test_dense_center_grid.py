from typing import Dict

import numpy as np
import pytest

from telcell.auxilliary_models.extended_geography import DenseCenterGrid, EmptyDenseCenterGrid
from telcell.auxilliary_models.geography import GridPoint


def create_dummy_grid():
    return DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000),
                           np.ones((50, 50)), np.ones((50, 50)))


def create_dummy_grid_normalized():
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000),
                            np.ones((50, 50)), np.ones((50, 50)))
    return _grid.normalize(1)


def _test_corners(grid: DenseCenterGrid) -> Dict[str, float]:
    """
    Retrieves values for each corner section within the outer values of a DenseCenterGrid
    """
    low_res = grid.outer.resolution
    center_distance = low_res / 2

    # get values
    sw_value = grid.get_value_for_center(grid.southwest.move(center_distance, center_distance))
    nw_value = grid.get_value_for_center(
        GridPoint(grid.southwest.move(center_distance, center_distance).x,
                  grid.northeast.move(-center_distance, -center_distance).y))
    ne_value = grid.get_value_for_center(grid.northeast.move(-center_distance, -center_distance))
    se_value = grid.get_value_for_center(
        GridPoint(grid.northeast.move(-center_distance, -center_distance).x,
                  grid.southwest.move(center_distance, center_distance).y))

    return {'sw_corner': sw_value, 'nw_corner': nw_value,
            'ne_corner': ne_value, 'se_corner': se_value}


def test_normalization():
    """
    Test that normalization to 1, sums to 1
    """
    assert pytest.approx(create_dummy_grid_normalized().sum()) == 1


def test_move_values():
    """
    Test that movement of values results in correct values (asserted by summing values)
    """
    dummy_grid_normalized = create_dummy_grid_normalized()
    dummy_grid_normalized = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(0, 0))
    assert pytest.approx(dummy_grid_normalized.sum()) == 1

    # moves values half the diameter to the right, removing half of valid values
    # padding with zeros, should make the sum half of what it was
    dummy_grid_normalized = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(250, 0))
    assert pytest.approx(dummy_grid_normalized.sum()) == .5
    # moving it back doesn't bring back it values
    dummy_grid_normalized = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(-250, 0))
    assert pytest.approx(dummy_grid_normalized.sum()) == .5

    dummy_grid_normalized = create_dummy_grid_normalized()
    dummy_grid_normalized = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(250, 250))
    assert pytest.approx(dummy_grid_normalized.sum()) == .25

    dummy_grid_normalized = create_dummy_grid_normalized()
    dummy_grid_normalized = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(100, 300))
    assert pytest.approx(dummy_grid_normalized.sum()) == .32

    dummy_grid_normalized = create_dummy_grid_normalized()
    dummy_grid_normalized = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(500, 500))
    assert pytest.approx(dummy_grid_normalized.sum()) == 0

    dummy_grid_normalized = create_dummy_grid_normalized()

    with pytest.raises(ValueError):
        _ = dummy_grid_normalized.move(dummy_grid_normalized.southwest.move(505, 505))


def test_move_properties():
    """
    Test that movement of DenseCenterGrid results in correct (same) parameters
    """
    _grid = create_dummy_grid()
    _grid = _grid.move(_grid.southwest.move(250, 0))
    assert _grid.southwest == GridPoint(1250, 1000)
    assert _grid.diameter == 500

    # movement should have valid steps
    with pytest.raises(ValueError, match="Southwest point of Grid is not aligned with resolution"):
        _grid.move(_grid.southwest.move(5, 5))

    with pytest.raises(ValueError, match="Southwest point of Grid is not aligned with resolution"):
        _grid.move(_grid.southwest.move(50, 5))

    # movement of empty DenseCenterGrid result in an EmptyDenseCenterGrid
    _grid = EmptyDenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000))
    _grid = _grid.move(_grid.southwest.move(10, 10))
    assert isinstance(_grid, EmptyDenseCenterGrid)

    _grid = EmptyDenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000))
    _grid_out_of_bounds = _grid.move(_grid.southwest.move(1000, 1000))
    assert not _grid.intersect(_grid_out_of_bounds)


def test_passe_partout():
    """
    Test that correct Grid (inner or outer) is used for selecting values
    """
    # check that inner is selected (which is 0 not 1)
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000), np.zeros((50, 50)), np.ones((50, 50)))
    assert pytest.approx(_grid.get_value_for_coord(GridPoint(1250, 1250))) == 0.
    assert pytest.approx(_grid.get_value_for_coord(GridPoint(1005, 1005))) == 1.


def test_get_value_for_coord():
    """
    Test that values outside DenseCenterGrid are invalid
    """
    _grid = create_dummy_grid()
    with pytest.raises(ValueError):
        _grid.get_value_for_coord(GridPoint(2000, 2000))


def test_get_value_for_center():
    """
    Test that correct Grid (inner or outer) is used for selecting values (using center)
    """
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000), np.zeros((50, 50)), np.ones((50, 50)))
    assert pytest.approx(_grid.get_value_for_center(GridPoint(1251, 1251))) == 0.
    with pytest.raises(ValueError):
        _grid.get_value_for_center(GridPoint(2000, 2000))


def test_invalid_parameters():
    """
    Test that no invalid DenseCenterGrid can be initialized
    """
    with pytest.raises(ValueError, match="Resolution should be an even number"):
            EmptyDenseCenterGrid(500, 100, 5, 1, GridPoint(1000, 1000))
    with pytest.raises(ValueError, match="Size of Grid is not aligned with resolution"):
            EmptyDenseCenterGrid(505, 100, 10, 2, GridPoint(1000, 1000))
    with pytest.raises(ValueError, match="Southwest point of Grid is not aligned with resolution"):
            EmptyDenseCenterGrid(500, 100, 10, 2, GridPoint(1005, 1005))
    with pytest.raises(ValueError, match="Low res 10 should be a multiple of the high res 9"):
            EmptyDenseCenterGrid(500, 100, 10, 9, GridPoint(1000, 1000))


def test_upsampling():
    """
    Test correct upsampling of sections
    """
    _grid = create_dummy_grid()
    _grid = _grid.move(_grid.southwest.move(250, 250))
    assert pytest.approx(_grid.get_value_for_coord(GridPoint(1500, 1500))) == 1 / 25

    _grid = create_dummy_grid()
    _grid = _grid.move(_grid.southwest.move(250, 250), normalized=False)
    assert pytest.approx(_grid.get_value_for_coord(GridPoint(1500, 1500))) == 1


def test_downsampling():
    """
    Test correct downsampling of sections
    """
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000), np.ones((50, 50)), np.ones((50, 50)))
    _grid = _grid.move(_grid.southwest.move(250, 250))
    assert pytest.approx(_grid.get_value_for_coord(GridPoint(1251, 1251))) == 25

    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000), np.ones((50, 50)), np.ones((50, 50)))
    _grid = _grid.move(_grid.southwest.move(250, 250), normalized=False)
    assert pytest.approx(_grid.get_value_for_coord(GridPoint(1251, 1251))) == 1


def test_operators():
    """
    Test operators for DenseCenterGrids
    """
    _norm_grid = create_dummy_grid_normalized()
    _grid_summed = _norm_grid + _norm_grid
    assert pytest.approx(_grid_summed.sum()) == 2.

    _grid = create_dummy_grid()
    _grid_multiplied = _grid * _grid
    assert pytest.approx(_grid_multiplied.sum()) == 4900.

    _grid_minus = _grid - _grid
    assert pytest.approx(_grid_minus.sum()) == 0.

    _grid_div = _grid / _grid
    assert pytest.approx(_grid_div.sum()) == 4900.

    _grid_2 = DenseCenterGrid(500, 100, 10, 2, GridPoint(1500, 1500), np.ones((50, 50)), np.ones((50, 50)))
    with pytest.raises(ValueError, match='Grids are not aligned'):
        _grid_summed = _grid + _grid_2

    # Test (in)equality
    assert _grid == _grid
    assert not _grid == _grid_minus


def test_iter():
    """
    Test that looping over DenseCenterGrid will return the inner and outer values
    """
    _grid = DenseCenterGrid(1000, 100, 10, 2, GridPoint(1500, 1500), np.ones((50, 50)), np.zeros((100, 100)))
    values = [x for x in _grid]
    assert 100 * 100 - 10 * 10 + 50 * 50 == len(values)


def test_orientation():
    """
    Test that coordinates correspond to expected section within values
    """
    _grid = create_dummy_grid()

    # Fetch all corner values (of outer values)
    corner_values = _test_corners(_grid)
    # should all be one
    for corner_value in corner_values.values():
        assert corner_value == 1

    # move values northeast, making all corners 0 except southwest corner
    _grid = _grid.move(_grid.southwest.move(250, 250))
    corner_values = _test_corners(_grid)
    for corner, value in corner_values.items():
        if corner == 'sw_corner':
            assert pytest.approx(value) == 25
        else:
            assert pytest.approx(value) == 0

    # move values northwest, making all corners 0 except southeast corner
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1250, 1250), np.ones((50, 50)), np.ones((50, 50)))
    _grid = _grid.move(_grid.southwest.move(-250, 250))
    corner_values = _test_corners(_grid)
    for corner, value in corner_values.items():
        if corner == 'se_corner':
            assert pytest.approx(value) == 25
        else:
            assert pytest.approx(value) == 0


def test_meshgrid_calc():
    """
    Test that coordinates correspond to expected section within values
    """
    _grid = EmptyDenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000))
    x_vals, y_vals = _grid.outer.coords_mesh_grid()
    # computes distance to point right of southeast corner
    dist = np.linalg.norm(np.stack([x_vals, y_vals], axis=-1) - np.array([[2000, 1000]]), axis=-1)
    # assign distance to this point as values of the values
    # _grid = _grid.set_values(dist, dist)
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000), dist, dist)
    c_values = _test_corners(_grid)
    # check ordening of (corner) values
    assert c_values['se_corner'] < c_values['ne_corner'] < c_values['sw_corner'] < c_values['nw_corner']


def test_multiple_grids():
    """
    Test that using multiple DenseCenterGrids provide valid results
    """
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(2000, 2000), np.ones((50, 50)), np.ones((50, 50)))
    _norm_grid = _grid.normalize(1.)

    # move values in each direction, so that each quarter is covered
    _grid_ne = _norm_grid.move(_norm_grid.southwest.move(250, 250)).move(_norm_grid.southwest)
    _grid_se = _norm_grid.move(_norm_grid.southwest.move(250, -250)).move(_norm_grid.southwest)
    _grid_sw = _norm_grid.move(_norm_grid.southwest.move(-250, -250)).move(_norm_grid.southwest)
    _grid_nw = _norm_grid.move(_norm_grid.southwest.move(-250, 250)).move(_norm_grid.southwest)
    _grid_empty = _norm_grid.move(_norm_grid.southwest.move(500, 500)).move(_norm_grid.southwest)

    # summing four quarters result in fully filled DenseCenterGrid
    _sum_grid = _grid_ne + _grid_se + _grid_sw + _grid_nw + _grid_empty
    assert pytest.approx(_sum_grid.sum()) == 1.

    # Average values
    _avg_grid = _sum_grid / 5
    assert pytest.approx(_avg_grid.sum()) == .2

    # Multiplying with other values
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(2000, 2000), np.ones((50, 50)), np.ones((50, 50)))
    _mult_grid = _avg_grid * _grid
    assert pytest.approx(_mult_grid.sum()) == 0.2

    # invalid operations
    with pytest.raises(AssertionError):
        _grid + np.zeros((50, 2, 50))

    with pytest.raises(AssertionError):
        _grid + [np.zeros((1, 2)), np.zeros((2, 1))]


def test_intersection():
    """
    Test that Grids do (not) intersect
    """
    _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(2000, 2000), np.ones((50, 50)), np.ones((50, 50)))
    assert _grid.intersect(_grid)
    assert not _grid.intersect(_grid.move(_grid.northeast))
    assert _grid.intersect(_grid.move(_grid.northeast.move(-10, -10)))
