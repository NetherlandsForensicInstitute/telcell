import unittest

import numpy as np
import pytest
import tensorflow as tf

from telcell.auxilliary_models.geography import GridPoint, manhattan_distance, Grid, EmptyGrid


def test_distance():
    """
    Test distance between points
    """
    p = GridPoint(10000, 11000)
    assert manhattan_distance(p, p.move(10, 0)) == 10
    assert manhattan_distance(p, p.move(0, 10)) == 10
    assert manhattan_distance(p, p.move(5, 5)) == 10


def test_move():
    """
    Test if no movement results in equal points
    """
    p = GridPoint(10000, 11000)
    assert p == p
    assert p == p.move(0, 0)
    assert p == p.move(.0, .0)


def test_stick_to_resolution():
    """
    Test if stick to resolution results in correct GridPoint coordinates
    """
    p = GridPoint(10000, 11000)
    assert p == GridPoint(9510, 10510).stick_to_resolution(1000)
    assert p == GridPoint(10000.4, 11000.4).stick_to_resolution(1000)
    assert GridPoint(100000, 100000) == GridPoint(90000, 110000).stick_to_resolution(100000)


def test_grid():
    """
    Test that values (with cut-out) has correct number of np.nan and 1.
    Also tests specific exceptions
    """
    _grid = Grid(100, 10, GridPoint(1000, 1000), np.ones((10, 10)),
                 (GridPoint(1040, 1040), GridPoint(1060, 1060)))
    assert np.sum(np.isnan(_grid.values)) == 4
    assert np.sum(_grid.values == 1.) == 96

    # Cut out (sw) does not align with resolutions
    with pytest.raises(ValueError):
        _grid = Grid(100, 10, GridPoint(1000, 1000), np.ones((10, 10)),
                     (GridPoint(1045, 1045), GridPoint(1060, 1060)))

    # Cut out (ne) does not align with resolutions
    with pytest.raises(ValueError):
        _grid = Grid(100, 10, GridPoint(1000, 1000), np.ones((10, 10)),
                     (GridPoint(1040, 1040), GridPoint(1065, 1065)))

    # Array does not fit shape of values
    with pytest.raises(ValueError):
        _grid = Grid(100, 10, GridPoint(1000, 1000), np.ones((11, 11)),
                     (GridPoint(1040, 1040), GridPoint(1060, 1060)))


def test_edges():
    """
    Test that southwest and northeast of Grid are correct
    """
    _grid = EmptyGrid(100, 10, GridPoint(1000, 1000),
                      (GridPoint(1040, 1040), GridPoint(1060, 1060)))

    assert _grid.southwest == GridPoint(1000, 1000)
    assert _grid.northeast == GridPoint(1100, 1100)


def test_coords():
    """
    Test correct y and x coordinates
    """
    _grid = EmptyGrid(100, 10, GridPoint(1000, 10000),
                      (GridPoint(1040, 10040), GridPoint(1060, 10060)))
    assert _grid.x_coords == [1005, 1015, 1025, 1035, 1045, 1055, 1065, 1075, 1085, 1095]
    assert _grid.y_coords == [10005, 10015, 10025, 10035, 10045, 10055, 10065, 10075, 10085, 10095]


def test_value_for_coord():
    """
    Test that values assigned to values match the expected coordinates (GridPoint)
    """
    _grid = EmptyGrid(100, 10, GridPoint(1000, 10000),
                      (GridPoint(1040, 10040), GridPoint(1060, 10060)))

    # Set random values to values
    random_grid = np.random.rand(*_grid.grid_shape)
    _grid = Grid(100, 10, GridPoint(1000, 10000),
                 random_grid,
                 (GridPoint(1040, 10040), GridPoint(1060, 10060)))

    assert _grid.get_value_for_coord(GridPoint(1001, 10001)) == random_grid[0, 0]
    assert _grid.get_value_for_coord(GridPoint(1001, 10099)) == random_grid[-1, 0]
    assert _grid.get_value_for_coord(GridPoint(1099, 10099)) == random_grid[-1, -1]
    assert _grid.get_value_for_coord(GridPoint(1099, 10001)) == random_grid[0, -1]
    # Value within cut-out are np.nan
    assert np.isnan(_grid.get_value_for_coord(GridPoint(1049, 10049)))

    # Coordinates outside values are invalid
    with pytest.raises(Exception, match='is not within a section of Grid'):
        _grid.get_value_for_coord(GridPoint(2001, 10001))


def test_value_for_center():
    """
    Test that values assigned to the values match the expected sections based on the coordinates
    of the centers of these sections
    """
    _grid = EmptyGrid(100, 10, GridPoint(1000, 10000),
                      (GridPoint(1040, 10040), GridPoint(1060, 10060)))

    random_grid = np.random.rand(*_grid.grid_shape)
    _grid = Grid(100, 10, GridPoint(1000, 10000),
                 random_grid,
                 (GridPoint(1040, 10040), GridPoint(1060, 10060)))

    assert _grid.get_value_for_center(GridPoint(1005, 10005)) == random_grid[0, 0]
    assert np.isnan(_grid.get_value_for_center(GridPoint(1045, 10045)))

    # is not a center of a section
    with pytest.raises(ValueError, match='is not a center within Grid'):
        _grid.get_value_for_center(GridPoint(1001, 10005))

    # is not (a center) within the values
    with pytest.raises(ValueError, match='is not a center within Grid'):
        _grid.get_value_for_center(GridPoint(2005, 10005))


def test_sum():
    """
    Test that sum is valid (and ignores np.nan)
    """
    _grid = Grid(100, 10, GridPoint(1000, 1000), np.ones((10, 10)),
                 (GridPoint(1040, 1040), GridPoint(1060, 1060)))
    assert _grid.sum() == 96


def test_scale_grid_values():
    """
    Test hat values are correctly scaled
    """
    _grid = Grid(100, 10, GridPoint(1000, 1000), np.ones((10, 10)),
                 (GridPoint(1040, 1040), GridPoint(1060, 1060)))
    _scaled_grid = _grid.scale_grid_values(4)
    assert _scaled_grid.sum() == 96 * 4


def test_meshgrid_coords():
    """
    Test that meshgrid has correct coordinates and correspond with expected section in values
    """
    _grid = EmptyGrid(500, 10, GridPoint(1000, 1000),
                      (GridPoint(1200, 1200), GridPoint(1300, 1300)))

    x_vals, y_vals = _grid.coords_mesh_grid()

    # first row and first column of values is the southwest
    x_sw, y_sw = x_vals[0, 0], y_vals[0, 0]
    assert pytest.approx(x_sw) == 1005  # west
    assert pytest.approx(y_sw) == 1005  # south

    # last row and first column of values is the northwest
    x_nw, y_nw = x_vals[-1, 0], y_vals[-1, 0]
    assert pytest.approx(x_nw) == 1005  # west
    assert pytest.approx(y_nw) == 1495  # north

    # last row and last column of the values is the northeast
    x_ne, y_ne = x_vals[-1, -1], y_vals[-1, -1]
    assert pytest.approx(x_ne) == 1495  # east
    assert pytest.approx(y_ne) == 1495  # north

    # first row and last column of the values is the southeast
    x_se, y_se = x_vals[0, -1], y_vals[0, -1]
    assert pytest.approx(x_se) == 1495  # east
    assert pytest.approx(y_se) == 1005  # south

    # Check that tf and numpy implementation are equal
    x_tf_vals, y_tf_vals = _grid.coords_mesh_grid('tf')
    assert np.array_equal(x_tf_vals.numpy(), x_vals)
    assert np.array_equal(y_tf_vals.numpy(), y_vals)

    # check that reshaping result in same (original) meshgrid
    coords = tf.stack([x_tf_vals, y_tf_vals], axis=-1)
    coords_reshaped = tf.reshape(coords, [-1, 2, 1]).numpy()
    assert np.array_equal(x_vals, coords_reshaped[:, 0].reshape(x_vals.shape))
    assert np.array_equal(y_vals, coords_reshaped[:, 1].reshape(y_vals.shape))


def test_area():
    """
    Test Area of corresponding Grid
    """
    _grid = EmptyGrid(500, 10, GridPoint(1000, 1000),
                      (GridPoint(1200, 1200), GridPoint(1300, 1300)))

    assert GridPoint(1000, 1000) == _grid.southwest


def test_move_values():
    """
    Test that movement of values results in correct values (asserted by summing values)
    """
    _norm_grid = Grid(500, 10, GridPoint(1000, 1000), np.ones((50, 50))).normalize(1)
    _norm_grid = _norm_grid.move(_norm_grid.southwest.move(0, 0))
    assert pytest.approx(_norm_grid.sum()) == 1

    # moves values half the diameter to the right, removing half of valid values
    # padding with zeros, should make the sum half of what it was
    _norm_grid = _norm_grid.move(_norm_grid.southwest.move(250, 0))
    assert pytest.approx(_norm_grid.sum()) == .5
    # moving it back doesn't bring back it values
    _norm_grid = _norm_grid.move(_norm_grid.southwest.move(-250, 0))
    assert pytest.approx(_norm_grid.sum()) == .5

    _norm_grid = Grid(500, 10, GridPoint(1000, 1000), np.ones((50, 50))).normalize(1)
    _norm_grid = _norm_grid.move(_norm_grid.southwest.move(250, 250))
    assert pytest.approx(_norm_grid.sum()) == .25

    _norm_grid = Grid(500, 10, GridPoint(1000, 1000), np.ones((50, 50))).normalize(1)
    _norm_grid = _norm_grid.move(_norm_grid.southwest.move(100, 300))
    assert pytest.approx(_norm_grid.sum()) == .32

    _norm_grid = Grid(500, 10, GridPoint(1000, 1000), np.ones((50, 50))).normalize(1)
    _norm_grid = _norm_grid.move(_norm_grid.southwest.move(500, 500))
    assert pytest.approx(_norm_grid.sum()) == 0

    _norm_grid = Grid(500, 10, GridPoint(1000, 1000), np.ones((50, 50))).normalize(1)

    with pytest.raises(ValueError):
        _norm_grid = _norm_grid.move(_norm_grid.southwest.move(505, 505))
