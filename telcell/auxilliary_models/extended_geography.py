import operator
from functools import lru_cache
from itertools import chain
from typing import Dict, Tuple, Union

import numpy as np
from keras.layers import AveragePooling2D, UpSampling2D

from .geography import EmptyGrid, Grid, GridPoint, crop_grid, grids_intersection, \
    set_values_on_array


class EmptyDenseCenterGrid(EmptyGrid):
    """
    A grid of values in which the center (inner) has a higher (dense) resolution compared to the edges (outer) of the
    values. The sections within the inner part of this values represent, therefore, smaller areas compared to the
    sections in the outer part of this values.

    Values for the sections can be set using `set_values`, in which the inner and outer values can be provided
    using an array.

    There are multiple constraints to guarantee successful alignment between (Empty)DenseCenterGrids:
    * The low resolution should be a multiple of the high resolution
    * The inner diameter should be a multiple of the high resolution
    * The outer diameter ($(diameter - inner diameter) / 2$) should be a multiple of the low resolution
    * All constraints that are placed on a default Grid (see `Grid`)
    """
    def __init__(self, diameter: int, inner_diameter: int,
                 outer_resolution: int, inner_resolution: int,
                 southwest: GridPoint):
        """
        :param diameter: Height/width of (outer) values
        :param inner_diameter: height/width of inner values
        :param outer_resolution: resolution of outer values
        :param inner_resolution: resolution of inner values
        :param southwest: southwest point of (outer) values
        """
        super().__init__(diameter, outer_resolution, southwest)
        if outer_resolution % inner_resolution != 0:
            raise ValueError(f"Low res {outer_resolution} should be a multiple of the high res {inner_resolution}")

        # determine location of inner values
        outer_size = (diameter - inner_diameter) // 2
        cut_out_sw = southwest.move(outer_size, outer_size)
        cut_out_ne = cut_out_sw.move(inner_diameter, inner_diameter)

        # init outer and inner
        self.inner = EmptyGrid(inner_diameter, inner_resolution, cut_out_sw)
        self.outer = EmptyGrid(diameter, outer_resolution, southwest, (cut_out_sw, cut_out_ne))

        # scaling between low- and high resolution (used for down- and upsampling)
        self._scaling = outer_resolution // inner_resolution

        # checks if (part of) the inner fits in (part of) the outer
        if any(coord % outer_resolution != 0 for coord in self.inner.southwest.xy):
            raise ValueError("Low res is not a multiple of the inner diameter")
        if any(coord % outer_resolution != 0 for coord in self.inner.northeast.xy):
            raise ValueError("Low res is not a multiple of the inner diameter")
        if outer_size % outer_resolution != 0 or inner_diameter % outer_resolution != 0:
            raise ValueError("Low res is not a multiple of the outer or inner diameter")

    @property
    def grids(self) -> Tuple[EmptyGrid, EmptyGrid]:
        """
        The inner and outer values

        :return: inner values, outer values
        """
        return self.inner, self.outer

    @property
    def grid(self) -> EmptyGrid:
        """
        Outer values defining the edges of the DenseCenterGrid

        :return: outer values
        """
        return self.outer

    def move(self, southwest: GridPoint) -> 'EmptyDenseCenterGrid':
        """
        Moves anchor (southwest) point of the values to a new (southwest) location, copying
        values for overlapping sections.
        Up- and downsampling is applied to copy values between section with different resolutions.

        :param southwest: new southwest (anchor) of values
        :return: EmptyDenseCenterGrid with new southwest (anchor) with values of original values
        """
        # init empty values, with same resolutions and diameters, with new southwest
        # initialized with zeros to cover sections that are not present in original values
        return self.copy(southwest=southwest)

    def zeros(self) -> 'DenseCenterGrid':
        """
        Sets all values to zero for the sections of the EmptyDenseCenterGrid, returning a DenseCenterGrid

        :return: A DenseCenterGrid with the same properties but all values set to 0
        """
        return DenseCenterGrid(**self.meta_attributes,
                               inner_values=np.zeros(self.inner.grid_shape),
                               outer_values=np.zeros(self.outer.grid_shape))

    def copy(self, **kwargs):
        """
        Creates a copy of the current grid, values are altered if they are provided as key word arguments
        :return: a copy of the grid with altered values (if provided)
        """
        new_arguments = {**self.meta_attributes, **kwargs}
        return EmptyDenseCenterGrid(**new_arguments)

    def _check_alignment(self, other: 'EmptyDenseCenterGrid') -> bool:
        """
        Checks if EmptyDenseCenterGrids are aligned

        :param other: EmptyDenseCenterGrid
        :return: True if aligned
        """
        if self.southwest != other.southwest or self.diameter != other.diameter or \
                self.outer.resolution != other.outer.resolution or \
                self.inner.resolution != other.inner.resolution or \
                self.inner.diameter != other.inner.diameter:
            return False
        else:
            return True

    @property
    def meta_attributes(self) -> Dict:
        return {'diameter': self.outer.diameter,
                'inner_diameter': self.inner.diameter,
                'outer_resolution': self.outer.resolution,
                'inner_resolution': self.inner.resolution,
                'southwest': self.southwest}

    def __repr__(self) -> str:
        return f"EmptyDenseCenterGrid({self.grid}); " \
               f"inner({self.inner}, {self.inner.resolution}), " \
               f"outer({self.outer}, {self.outer.resolution})"

    def __iter__(self):
        """
        Iterates over the inner and outer values of the DenseCenterGrid (in that order)
        """
        yield from chain(iter(self.outer), iter(self.inner))

    def __hash__(self):
        return hash((hash(self.inner), hash(self.outer)))


class DenseCenterGrid(EmptyDenseCenterGrid):
    """
    An EmptyDenseCenterGrid with values for each section.

    Default operators can be applied on aligned DenseCenterGrids (same properties except it values,
    see `self._check_alignment`). Other operations will be broadcast to the inner and outer values.
    Depending on the operation and shape of the grids, this will result in (valid) results.

    Minimal example
    >>> _grid = DenseCenterGrid(500, 100, 10, 2, GridPoint(1000, 1000),
    >>>                   np.ones((50, 50)), np.ones((50, 50)))
    >>> _grid_summed = _grid + _grid

    """
    def __init__(self, diameter: int, inner_diameter: int,
                 outer_resolution: int, inner_resolution: int,
                 southwest: GridPoint,
                 inner_values: np.ndarray, outer_values: np.ndarray):
        """
        :param diameter: Height/width of (outer) values
        :param inner_diameter: height/width of inner values
        :param outer_resolution: resolution of outer values
        :param inner_resolution: resolution of inner values
        :param southwest: southwest point of (outer) values
        :param inner_values: values for inner grid
        :param outer_values: values for outer grid
        """
        super().__init__(diameter, inner_diameter, outer_resolution, inner_resolution, southwest)
        # init outer and inner
        self.inner = Grid(inner_diameter, inner_resolution, self.inner.southwest, inner_values)
        self.outer = Grid(diameter, outer_resolution, self.outer.southwest, outer_values, self.outer.cut_out)

    def drop_values(self) -> 'EmptyDenseCenterGrid':
        """
        Return a new EmptyDenseCenterGrid with the values removed from this DenseCenterGrid instance.

        :return: an EmptyDenseCenterGrid with the same coordinates as the DenseCenterGrid instance.
        """
        return EmptyDenseCenterGrid(**self.meta_attributes)

    def copy(self, **kwargs):
        inner = kwargs.pop('inner_values', self.inner.values)
        outer = kwargs.pop('outer_values', self.outer.values)
        new_arguments = {**self.meta_attributes, **kwargs}
        return DenseCenterGrid(**new_arguments,
                               inner_values=inner, outer_values=outer)

    def get_value_for_coord(self, point: GridPoint) -> float:
        """
        Get value for a specific coordinate.
        Based on the coordinate the value will be from a section of the inner or the outer values.

        :param point: point within DenseCenterGrid
        :return: value
        """
        if self.inner.southwest.x < point.x < self.inner.northeast.x:
            return self.inner.get_value_for_coord(point)
        elif self.outer.southwest.x < point.x < self.outer.northeast.x:
            return self.outer.get_value_for_coord(point)
        else:
            raise ValueError(f"Point {point} is not within DenseCenterGrid {self}")

    def get_value_for_center(self, point: GridPoint) -> float:
        """
        Get value for a specific section of the inner or outer values by its center coordinate.

        :param point: point within DenseCenterGrid
        :return: value
        """
        if point.x in self.inner.x_coords and point.y in self.inner.y_coords:
            return self.inner.get_value_for_center(point)
        elif point.x in self.outer.x_coords and point.y in self.outer.y_coords:
            return self.outer.get_value_for_center(point)
        else:
            raise ValueError(f"Point {point} is not a center within DenseCenterGrid {self}")

    def move(self, southwest: GridPoint, normalized: bool = True) -> 'DenseCenterGrid':
        """
        Moves anchor (southwest) point of the values to a new (southwest) location, copying
        values for overlapping sections.
        Sections for which no values are present from the original values will be filled with 0.
        Up- and downsampling is applied to copy values between sections with different resolutions.

        :param southwest: new southwest (anchor) of values
        :param normalized: indicates if normalization should be preserved
        :return: DenseCenterGrid with new southwest (anchor) with values of original values
        """
        # init empty values, with same resolutions and sizes, with new southwest
        # initialized with zeros to cover sections that are not present in original values
        _new_multi_grid = self.drop_values().move(southwest)

        inner, outer = np.zeros(self.inner.grid_shape), np.zeros(self.outer.grid_shape)

        # fill outer values with original (outer & inner) grids
        for source_grid in (self.outer, self.inner):
            inner = self._copy_intersection(inner, _new_multi_grid.inner, source_grid, normalized)
            outer = self._copy_intersection(outer, _new_multi_grid.outer, source_grid, normalized)

        return DenseCenterGrid(**_new_multi_grid.meta_attributes, inner_values=inner, outer_values=outer)

    def normalize(self, sum_value: float = 1.) -> 'DenseCenterGrid':
        """
        Normalize values within sections, in order that they sum to a given value

        :param sum_value: value that values should sum to
        """
        aligned_grid = self._align_low_high_res_values()
        # Scale scores in order that they sum 1
        total_prop = aligned_grid.sum()

        return self.copy(inner_values=aligned_grid.inner.scale_grid_values(sum_value / total_prop).values,
                         outer_values=aligned_grid.outer.scale_grid_values(sum_value / total_prop).values)

    @staticmethod
    def _up_sample(grid: np.ndarray, scaling: int, normalized: bool) -> np.ndarray:
        """
        Upsample an array with a specific scaling

        :param grid: array that should be upsampled
        :param scaling: factor of upsampling
        :param normalized: indicates if normalization should be preserved
        :return: upsampled array
        """
        grid = grid.reshape((1, *grid.shape, 1))
        up_samp_2d = UpSampling2D(scaling)
        _grid = up_samp_2d(grid)
        if normalized:
            _grid = _grid / (scaling ** 2)
        return _grid.numpy().squeeze(axis=(0, 3))

    @staticmethod
    def _down_sample(grid: np.ndarray, scaling: int, normalized: bool) -> np.ndarray:
        """
        Downsample an array with a specific scaling

        :param grid: array that should be downsampled
        :param scaling: factor of downsampling
        :param normalized: indicates if normalization should be preserved
        :return: downsampled array
        """
        grid = grid.reshape((1, *grid.shape, 1)).astype(float)
        avg_pool_2d = AveragePooling2D(scaling)
        _grid = avg_pool_2d(grid)
        if normalized:
            _grid = _grid * (scaling ** 2)
        return _grid.numpy().squeeze(axis=(0, 3))

    def sum(self) -> float:
        """
        Sum of all sections within values

        :return: sum value
        """
        return self.outer.sum() + self.inner.sum()

    def intersect(self, other: 'DenseCenterGrid') -> bool:
        """
        Checks if DenseCenterGrid has an intersection with current DenseCenterGrid (based on outer grids)

        :param other: DenseCenterGrid
        :return: boolean that indicates if there is an intersection
        """
        intersection, _ = grids_intersection(self.outer, other.outer)
        if intersection:
            return True
        else:
            return False

    @property
    def grid(self) -> EmptyGrid:
        """
        Outer values defining the edges of the DenseCenterGrid

        :return: outer grid
        """
        return self.outer.get_empty_grid()

    def get_empty_grid(self) -> EmptyGrid:
        """
        Return a new EmptyGrid with the values removed from the outer grid instance.

        :return: an EmptyGrid with the same coordinates as the Grid instance.
        """
        return EmptyDenseCenterGrid(**self.meta_attributes)

    def _align_low_high_res_values(self) -> 'DenseCenterGrid':
        """
        Align values in outer and inner grids based on difference in resolutions.
        Values in (outer values) sections with a lower resolution cover a larger area.
        This operation corrects for this difference by applying the scaling factor
        to the values in the outer values.
        """
        return self.copy(inner_values=self.inner.values,
                         outer_values=self.outer.scale_grid_values(self._scaling ** 2).values)

    def _copy_intersection(self,
                           base_array: np.ndarray,
                           base_grid: EmptyGrid,
                           source_grid: Grid,
                           normalized: bool) -> np.ndarray:
        """
        Copy values from the source values to the base values for sections that intersect

        :param base_array: values for which values are replaced
        :param base_grid: empty grid for which values are replaced
        :param source_grid: empty grid from which values are used
        :param normalized: indicates if normalization should be preserved
        :return: numpy array with replaced values
        """
        sw_intersection, ne_intersection = grids_intersection(base_grid, source_grid)
        if not sw_intersection and not ne_intersection:
            return base_array

        # select overlapping sections
        cropped_source_grid = crop_grid(source_grid, (sw_intersection, ne_intersection))

        # correct for difference in resolutions between sections
        if source_grid.resolution > base_grid.resolution:
            source_grid = self._up_sample(cropped_source_grid, self._scaling, normalized)
        elif source_grid.resolution < base_grid.resolution:
            source_grid = self._down_sample(cropped_source_grid, self._scaling, normalized)
        else:
            source_grid = cropped_source_grid

        # place new values on overlapping sections of the base values
        base_array = set_values_on_array(base_array, base_grid,
                                         (sw_intersection, ne_intersection),
                                         source_grid)

        return base_array

    def __add__(self, other: Union['DenseCenterGrid', float]) -> 'DenseCenterGrid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.add)

    def __sub__(self, other: Union['DenseCenterGrid', float]) -> 'DenseCenterGrid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.sub)

    def __mul__(self, other: Union['DenseCenterGrid', float]) -> 'DenseCenterGrid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.mul)

    def __truediv__(self, other: Union['DenseCenterGrid', float]) -> 'DenseCenterGrid':
        """
        see `._operate_grids`
        """
        return self._operate_grids(other, operator.truediv)

    def _operate_grids(self, other: Union['DenseCenterGrid', float], op: operator) -> 'DenseCenterGrid':
        """
        Applies operator between DenseCenterGrid and `other`
        If other is an aligned DenseCenterGrid, the operation is applied on the inner and outer grids
        respectively of both DenseCenterGrids, resulting in a new DenseCenterGrid.
        If the other is a different instance it is broadcast to both the inner and outer values
        using the selected operator

        :param other: instance use for the operation with the current DenseCenterGrid
            (if DenseCenterGrid it should be aligned with the current DenseCenterGrid)
        :param op: operation to apply between current DenseCenterGrid and other
        :return: DenseCenterGrid with updated values
        """
        # check alignment and select outer and inner values
        if isinstance(other, DenseCenterGrid):
            if not self._check_alignment(other):
                raise ValueError("Grids are not aligned")
            other_inner = other.inner.values
            other_outer = other.outer.values
        elif isinstance(other, (float, int)):
            other_inner, other_outer = other, other
        else:
            raise AssertionError(f'Invalid type ({type(other)}) for operation {op} with DenseCenterGrid')

        return self.copy(inner_values=op(self.inner.values, other_inner),
                         outer_values=op(self.outer.values, other_outer))

    def __eq__(self, other: 'DenseCenterGrid') -> bool:
        return np.array_equal(self.inner.values, other.inner.values) and \
            np.array_equal(self.outer.values, other.outer.values, equal_nan=True) and \
            self._check_alignment(other)

    def __ne__(self, other: 'DenseCenterGrid') -> bool:
        return not self == other

    def __repr__(self) -> str:
        return f"DenseCenterGrid({self.grid}); " \
               f"inner({self.inner}, {self.inner.resolution}), " \
               f"outer({self.outer}, {self.outer.resolution})"


def sum_dense_center_grids(*grids: 'DenseCenterGrid') -> 'DenseCenterGrid':
    """
    Summing the values of a sequence of multi grids (element-wise).
    Speed up over:
    >>> slow_sum = multi_grid0 + multi_grid1 + multi_grid2 + multi_grid3
    >>> fast_sum = sum_dense_center_grids(multi_grid0, multi_grid1, multi_grid2, multi_grid3)

    :param grids: aligned values for which the values are summed element wise
    :return: DenseCenterGrid in containing the summed grids
    """
    if len(grids) < 1:
        raise ValueError("One or more grids should be provided")
    ref_grid = grids[0]
    if not all(ref_grid._check_alignment(grid) for grid in grids):
        raise ValueError("Grids are not aligned")

    inner = np.sum(np.stack([g.inner.values for g in grids], axis=-1), axis=2)
    outer = np.sum(np.stack([g.outer.values for g in grids], axis=-1), axis=2)

    return DenseCenterGrid(**ref_grid.meta_attributes, inner_values=inner, outer_values=outer)


@lru_cache(maxsize=None)
def get_measurement_grid(coords: GridPoint,
                         outer_size: int,
                         outer_resolution: int,
                         inner_size: int,
                         inner_resolution: int) -> EmptyDenseCenterGrid:
    """Gets the measurement values of an antenna based on its coordinates"""
    move = -outer_size // 2
    southwest = coords.move(move, move).stick_to_resolution(outer_resolution)
    return EmptyDenseCenterGrid(outer_size, inner_size,
                                outer_resolution, inner_resolution,
                                southwest)
