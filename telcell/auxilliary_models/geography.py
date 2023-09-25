import math
import operator
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from telcell.data.models import RDPoint


class GridPoint(RDPoint):
    """
    histo.Point with some extra attributes, that facilitate the point living on a grid.
    """

    def __init__(self, rdx: float, rdy: float):
        super().__init__(x=rdx, y=rdy)

    def move(self, delta_rdx: float, delta_rdy: float) -> "GridPoint":
        return GridPoint(self.x + delta_rdx, self.y + delta_rdy)

    def stick_to_resolution(self, resolution: int) -> "GridPoint":
        rdx = round(self.x / resolution) * resolution
        rdy = round(self.y / resolution) * resolution
        return GridPoint(rdx, rdy)

    def __eq__(self, obj) -> bool:
        return self.x == obj.x and self.y == obj.y

    def __ne__(self, obj) -> bool:
        return self.x != obj.x or self.y != obj.y

    def __hash__(self):
        return hash(self.xy)


class Area:
    def __init__(self, diameter: int, southwest: GridPoint):
        """
        :param diameter: Height and width of the Area
        :param southwest: southwest point of the Area
        """
        self.diameter = diameter
        self._southwest = southwest

    def intersect(self, other: 'Area') -> bool:
        """
        Checks if area has an intersection with current area

        :param other: Area
        :return: boolean that indicates if there is an intersection
        """
        intersection, _ = grids_intersection(self, other)
        if intersection:
            return True
        else:
            return False

    @property
    def southwest(self) -> GridPoint:
        """
        Southwest coordinate of values
        :return: GridPoint
        """
        return self._southwest

    @property
    def northeast(self) -> GridPoint:
        """
        Northeast coordinate of values
        :return: GridPoint
        """
        return self.southwest.move(self.diameter, self.diameter)


class EmptyGrid(Area):
    def __init__(self, diameter: int, resolution: int, southwest: GridPoint,
                 cut_out: Optional[Tuple[GridPoint, GridPoint]] = None):
        super().__init__(diameter, southwest)
        if southwest.x % resolution != 0 or southwest.y % resolution != 0:
            raise ValueError("Southwest point of Grid is not aligned with resolution")
        if diameter % resolution != 0:
            raise ValueError("Size of Grid is not aligned with resolution")
        if resolution % 2 != 0:
            raise ValueError("Resolution should be an even number")

        self.resolution = resolution
        self.cut_out = cut_out
        self.grid_shape = (diameter // resolution, diameter // resolution)
        if self.cut_out:
            if any(coord % self.resolution != 0 for coord in
                   self.cut_out[0].xy):
                raise ValueError(
                    'Cut out (southwest) is not aligned with resolution')
            if any(coord % self.resolution != 0 for coord in
                   self.cut_out[1].xy):
                raise ValueError(
                    'Cut out (northwest) is not aligned with resolution')

    @property
    def x_coords(self) -> List[int]:
        """
        X coordinates of values. These apply to the centers of the
        sections of the values
        :return: list of unique x coordinates for the sections
        """
        # move to center of the section (hence //2)
        distance_to_center = self.resolution // 2
        return list(range(int(self.southwest.move(distance_to_center, distance_to_center).x),
                          int(self.northeast.move(distance_to_center, distance_to_center).x),
                          self.resolution))

    @property
    def y_coords(self) -> List[int]:
        """
        Y coordinates of values. These apply to the centers of the
        sections of the values
        :return: list of unique y coordinates for the sections
        """
        # move to center of the section (hence //2)
        distance_to_center = self.resolution // 2
        return list(range(int(self.southwest.move(distance_to_center, distance_to_center).y),
                          int(self.northeast.move(distance_to_center, distance_to_center).y),
                          self.resolution))

    def coords_mesh_grid(self, mode: str = 'np'):
        """
        Mesh values of x and y coordinates.
        Tuple of arrays with all x and y coordinates.
        (see `np.meshgrid` or `tf.meshgrid`)

        :param mode: specifies the format, `np` for arrays, `tf` for tensors
        :return: Array with x coordinates (either array or tensor),
            Array with y coordinates (either array or tensor)
        """
        if mode == 'np':
            return np.meshgrid(self.x_coords, self.y_coords)
        elif mode == 'tf':
            return tf.meshgrid(self.x_coords, self.y_coords)

    def move(self, southwest: GridPoint) -> 'EmptyGrid':
        """
        Moves anchor (southwest) point of the values to a new (southwest) location, copying
        values for overlapping sections.

        :param southwest: new southwest (anchor) of values
        :return: EmptyGrid with new southwest (anchor) with values of original values
        """
        return EmptyGrid(self.diameter, self.resolution, southwest, self.cut_out)

    def zeros(self) -> 'Grid':
        """
        Sets all values to zero for the sections of the EmptyGrid, returning a Grid

        :return: A Grid with the same properties but all values set to 0
        """
        return Grid(self.diameter, self.resolution, self.southwest, np.zeros(self.grid_shape), self.cut_out)

    def __repr__(self) -> str:
        return f"EmptyGrid({self.southwest},{self.northeast})"

    def __hash__(self):
        if self.cut_out:
            return hash((self.diameter, self.southwest, self.resolution,
                         self.cut_out[0], self.cut_out[1]))
        else:
            return hash((self.diameter, self.southwest, self.resolution))


class Grid(EmptyGrid):
    def __init__(self, diameter: int, resolution: int, southwest: GridPoint,
                 values: Optional[np.ndarray] = None,
                 cut_out: Optional[Tuple[GridPoint, GridPoint]] = None):
        super().__init__(diameter, resolution, southwest, cut_out)
        if self.grid_shape != values.shape:
            raise ValueError('Shape of array is incompatible with shape of Grid')

        if cut_out:
            # Failsafe
            values = set_values_on_array(values, self, cut_out, np.nan)
        self._values = values
        # Failsafe
        self._values.flags.writeable = False

    @property
    def values(self) -> np.ndarray:
        """
        Values of the grid

        :return: numpy array with values of the grid
        """
        return self._values

    def get_empty_grid(self) -> 'EmptyGrid':
        """
        Return a new EmptyGrid with the values removed from this Grid instance.

        :return: an EmptyGrid with the same coordinates as the Grid instance.
        """
        return EmptyGrid(self.diameter, self.resolution, self.southwest, self.cut_out)

    def get_value_for_coord(self, point: GridPoint) -> float:
        """
        Get value for a specific coordinate.

        :param point: point within Grid
        :return: value
        """
        if not (self.northeast.x > point.x > self.southwest.x) \
                or not (self.northeast.y > point.y > self.southwest.y):
            raise ValueError(f"Point {point} is not within a section of Grid {self}")

        x_center = min(self.x_coords, key=lambda x: abs(x - point.x))
        y_center = min(self.y_coords, key=lambda y: abs(y - point.y))
        value = self.get_value_for_center(GridPoint(x_center, y_center))
        return value

    def get_value_for_center(self, point: GridPoint) -> float:
        """
        Get value for a specific section by its center coordinate.

        :param point: point within Grid
        :return: value
        """
        if point.x not in self.x_coords or point.y not in self.y_coords:
            raise ValueError(f"Point {point} is not a center within Grid {self}")

        row_offset, column_offset = point_offset(self, point)
        return self.values[row_offset, column_offset]

    def _check_alignment(self, other: 'Grid') -> bool:
        """
        Checks if Grids are aligned

        :param other: Grid
        :return: True if aligned
        """
        if self.southwest != other.southwest or self.diameter != other.diameter or \
                self.resolution != other.resolution or \
                self.diameter != other.diameter:
            return False
        else:
            return True

    def move(self, southwest: GridPoint) -> 'Grid':
        """
        Moves anchor (southwest) point of the values to a new (southwest) location, copying
        values for overlapping sections.

        :param southwest: new southwest (anchor) of values
        :return: Grid with new southwest (anchor) with values of original values
        """
        """
        Returns a new grid with the same values as the current grid, but
        covering a different area. Any new values will be 0.
        """
        new_grid = self.get_empty_grid().move(southwest)
        values = self._copy_intersection(np.zeros(self.grid_shape), new_grid, self)
        return Grid(self.diameter, self.resolution, southwest, values, self.cut_out)

    def sum(self) -> float:
        """
        Sum of probabilities of values
        :return: sum value
        """
        return float(np.nansum(self.values))

    @staticmethod
    def sum_all(*grids: 'Grid') -> 'Grid':
        """
        Summing the values of a sequence of grids (element-wise).
        Speed up over:
        >>> slow_sum = grid0 + grid1 + grid2 + grid3
        >>> fast_sum = grid_sum(grid0, grid1, grid2, grid3)

        :param grids: aligned values for which the values are summed element wise
        :return: Grid with the values from the summed grids
        """
        if len(grids) < 1:
            raise ValueError("One or more grids should be provided")
        ref_grid = grids[0]
        if not all(ref_grid._check_alignment(grid) for grid in grids):
            raise ValueError("Grids are not aligned")

        values = np.sum(np.stack([g.values for g in grids], axis=-1), axis=2)

        return Grid(ref_grid.diameter, ref_grid.resolution, ref_grid.southwest, values, ref_grid.cut_out)

    def normalize(self, sum_value: float = 1.) -> 'Grid':
        """
        Normalize values within sections, in order that they sum to a given value

        :param sum_value: value that values should sum to
        """
        return self.scale_grid_values(sum_value / self.sum())

    def scale_grid_values(self, constant: float) -> 'Grid':
        """
        Multiplies values of the grid with
        a specific (constant) value
        :param constant: factor that is used to multiply values with
        """
        return Grid(self.diameter, self.resolution, self.southwest,
                    self.values * constant, self.cut_out)

    def _copy_intersection(self, base_array: np.ndarray, base_grid: EmptyGrid, source_grid: "Grid") -> np.ndarray:
        """
        Copy values from the source values to the base values for sections that intersect

        :param base_array: values for which values are replaced
        :param base_grid: empty grid for which values are replaced
        :param source_grid: empty grid from which values are used
        :return: numpy array with replaced values
        """
        if base_grid.resolution != source_grid.resolution:
            raise ValueError("Unable to copy intersection for different resolutions")

        sw_intersection, ne_intersection = grids_intersection(base_grid, source_grid)
        if not sw_intersection and not ne_intersection:
            return base_array

        # select overlapping sections
        cropped_grid = crop_grid(source_grid, (sw_intersection, ne_intersection))

        # place new values on overlapping sections of the base values
        base_array = set_values_on_array(
            base_array, base_grid, (sw_intersection, ne_intersection), cropped_grid)

        return base_array

    def __add__(self, other: Union['Grid', float]) -> 'Grid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.add)

    def __sub__(self, other: Union['Grid', float]) -> 'Grid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.sub)

    def __mul__(self, other: Union['Grid', float]) -> 'Grid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.mul)

    def __truediv__(self, other: Union['Grid', float]) -> 'Grid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.truediv)

    def _operate_grids(self, other: Union['Grid', float], op: operator) -> 'Grid':
        """
        Applies operator between Grid and `other`
        If other is an aligned Grid, the operation is applied on the inner and outer grids
        respectively of both Grid, resulting in a new Grid.
        If the other is a different instance it is broadcast to both the inner and outer values
        using the selected operator

        :param other: instance use for the operation with the current Grid
            (if Grid it should be aligned with the current Grid)
        :param op: operation to apply between current Grid and other
        :return: Grid with updated values
        """
        # check alignment and select outer and inner values
        if isinstance(other, Grid):
            if not self._check_alignment(other):
                raise ValueError("Grids are not aligned")
            other = other.values
        elif not isinstance(other, (float, int)):
            raise AssertionError(f'Invalid type ({type(other)}) for operation {op} with Grid')
        return Grid(self.diameter, self.resolution, self.southwest,
                    op(self.values, other), self.cut_out)

    def __iter__(self) -> Tuple[float, GridPoint]:
        """
        Iterates over the grid and returns the probability at the location and the grid point.
        """
        distance_to_center = self.resolution // 2
        for (r, c), prob in np.ndenumerate(self._values):
            if not np.isnan(prob):
                yield prob, self.southwest.move(r * self.resolution + distance_to_center,
                                                c * self.resolution + distance_to_center)

    def __repr__(self) -> str:
        return f"Grid({self.southwest},{self.northeast})"


def grids_intersection(grid_a: Area, grid_b: Area) -> \
        Tuple[Optional[GridPoint], Optional[GridPoint]]:
    """
    Determine sw and ne points of intersection of two grids or areas. If no intersection is present,
    None is returned for both points

    :param grid_a: Grid or Area used to determine intersection
    :param grid_b: Grid or Area used to determine intersection
    :return: sw intersection point, ne intersection point
    """
    sw_intersection = GridPoint(max([grid_a.southwest.x, grid_b.southwest.x]),
                                max([grid_a.southwest.y, grid_b.southwest.y]))
    ne_intersection = GridPoint(min([grid_a.northeast.x, grid_b.northeast.x]),
                                min([grid_a.northeast.y, grid_b.northeast.y]))
    if sw_intersection.x >= ne_intersection.x or sw_intersection.y >= ne_intersection.y:
        return None, None
    return sw_intersection, ne_intersection


def point_offset(grid: Grid, point: GridPoint) -> Tuple[int, int]:
    """
    Computes offset of a point given a values

    :param grid: values for which offset should be determined
    :param point: point for which offset should be determined
    :return: row offset within values, column offset within values
    """
    return \
        int((point.y - grid.southwest.y) // grid.resolution), \
        int((point.x - grid.southwest.x) // grid.resolution)


def crop_grid(grid: Grid, points: Sequence[GridPoint]) -> np.ndarray:
    """
    Crops values according to points and returns crop

    :param grid: values for which array should be cropped
    :param points: points that indicate
    :return: numpy array
    """
    row_start, row_end, column_start, column_end = _extract_crop_offsets(grid, points)
    return grid.values[row_start:row_end, column_start:column_end]


def set_values_on_array(array: np.ndarray, grid: EmptyGrid, points: Sequence[GridPoint], vals) -> np.ndarray:
    """
    Updates crop of values according to points and returns full array
    with vals insert on crop

    :param array: (section of) array to set values of
    :param grid: underling grid of array
    :param points: points that define the crop
    :param vals: values that should be inserted in (section of) array (compliant with numpy operation)
    :return: numpy array
    """
    row_start, row_end, column_start, column_end = _extract_crop_offsets(grid, points)
    new_array = np.copy(array)
    new_array[row_start:row_end, column_start:column_end] = vals
    return new_array


def _extract_crop_offsets(grid: EmptyGrid, points: Sequence[GridPoint]) -> Tuple[int, int, int, int]:
    """
    Computes offsets for set of points for a given values, to index its corresponding crop

    :param grid: Grid for which offsets should be computed
    :param points: set of points for which crop should be selected
    :return: start and end offset for row, start and end offset for column
    """
    row_start, column_start = point_offset(grid, points[0])
    row_end, column_end = point_offset(grid, points[1])
    return row_start, row_end, column_start, column_end


def relative_angle(angle: float) -> float:
    """
    converts an angle between the values of -180 and 180
    :param angle: angle to be converted
    :return: angle in degrees
    """
    angle = angle % 360
    angle = abs(angle)
    angle = angle if angle < 180 else angle - 360
    return angle


def manhattan_distance(coord1: GridPoint, coord2: GridPoint) -> float:
    """
    manhattan distance between two GridPoints

    :param coord1: GridPoint
    :param coord2: GridPoint
    :return: distance between the two points
    """
    return abs(coord1.x - coord2.x) + abs(coord1.y - coord2.y)


def euclidean_distance(coord1: GridPoint, coord2: GridPoint) -> float:
    """
    euclidean distance between two GridPoints

    :param coord1: GridPoint
    :param coord2: GridPoint
    :return: distance between the two points
    """
    return math.hypot(abs(coord1.x - coord2.x), abs(coord1.y - coord2.y))


class AreaSize:
    """
    We sometimes need the radius of the area around an antenna and sometimes the diameter. When calling the database,
    we usually need the radius; for the grid, we usually need the diameter.
    """

    def __init__(self, radius: Optional[int] = None, diameter: Optional[int] = None):
        if not radius and not diameter:
            raise ValueError("You cannot create an AreaSize without either a radius or a diameter")
        if radius and diameter and radius != 2 * diameter:
            raise ValueError(f"Both radius and diameter specified for AreaSize; radius must be 2x diameter"
                             f"but this is not the case (radius: {radius} and diameter: {diameter})")
        self.radius = radius or int(diameter // 2)
        self.diameter = diameter or radius * 2
