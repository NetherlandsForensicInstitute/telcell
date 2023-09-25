from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List

import numpy as np
import tensorflow as tf
from more_itertools import first
from sklearn.base import BaseEstimator
from tensorflow.python.framework.ops import EagerTensor

from telcell.auxilliary_models.extended_geography import DenseCenterGrid, EmptyDenseCenterGrid
from telcell.auxilliary_models.geography import Grid, GridPoint, \
    Area, EmptyGrid
from telcell.auxilliary_models.rare_pair.corrected_priors import \
    CorrectedPriorsModel
from telcell.auxilliary_models.rare_pair.features import extract_azimuth, \
    extract_features, extract_antenna_coordinates, calculate_angle_distance_cache
from telcell.auxilliary_models.rare_pair.utils import DISTANCE_STEP
from telcell.data.models import Measurement


class CoverageModel(ABC):
    """
    An abstract baseclass that specifies the interface that all coverage models
     should adhere to.
    """
    @abstractmethod
    def normalized_probabilities(self, measurement: Measurement) -> Grid:
        """
        Returns the normalized probability on a grid for a given antenna.

        :param measurement: antenna of interest
        :return: Grid with normalized probabilities for the antenna of interest for each location
        """
        raise NotImplementedError

    @abstractmethod
    def probabilities(self, measurement: Measurement) -> Grid:
        """
        Returns the probability on a grid for a given antenna.

        :param measurement: antenna of interest
        :return: Grid with probabilities for the antenna of interest for each location
        """
        raise NotImplementedError

    @abstractmethod
    def measurement_area(self, measurement: Measurement) -> Area:
        """
        Get area for which the model computes the coverage for a specific measurement

        :param measurement: Antenna for which the coverage is retrieved
        :return: Area which is covered by the model
        """
        raise NotImplementedError

    def cache_clear(self) -> None:
        """
        Clears all lru caches of this model.
        """
        return None


class AngleDistanceClassificationCoverageModel(CoverageModel):
    """
    Coverage model that models the service area of an Antenna
    with a (scikit-learn based) classification model with the
    following features:
        * angle
        * distance

    :param resolution: resolution of grid used for coverage model
    :param diameter: diameter of circle for which coverage is determined
    :param model: model used to classify antennas
    :param calibrator: calibrator used to calibrate probabilities
    """
    def __init__(
            self,
            resolution: int,
            diameter: int,
            model: BaseEstimator,
            calibrator: Optional[BaseEstimator] = None
    ):
        self.resolution = resolution
        self.diameter = diameter
        self.model = model
        self.calibrator = calibrator
        self._probabilities: Optional[tf.Tensor] = None

    @property
    def radius(self):
        """
        radius of coverage model
        """
        return self.diameter / 2

    def normalized_probabilities(self, measurement: Measurement) -> Grid:
        return self.probabilities(measurement).normalize(1.)

    @lru_cache(maxsize=None)
    def probabilities(self, measurement: Measurement) -> Grid:
        locations = self.measurement_locations(measurement)
        return Grid(locations.diameter, locations.resolution, locations.southwest,
                    self.get_probabilities(measurement, locations), locations.cut_out)

    def get_probabilities(self, measurement: Measurement, locations: EmptyGrid) -> np.ndarray:
        """
        Get probabilities for each location from a grid for a specific measurement.

        :param measurement: antenna for which the probabilities are computed
        :param locations: locations for which the probabilities are computed
        :return: array with same size (and shape) as the locations grid with the probabilities
        """
        angles, distances = self._extract_features(measurement, locations)
        probabilities = self.predict(angles=angles, distances=distances)
        probabilities = probabilities.numpy().reshape(*locations.grid_shape)

        if self.calibrator:
            probabilities = self.calibrator.predict_proba(probabilities.flatten())[:, 1].reshape(locations.grid_shape)

        distances = distances.numpy().reshape(*locations.grid_shape)
        # Set probabilities outside radius to zero
        # to prevent differences in operating coverage grids between antennas
        # based on their relative positioning towards each other
        probabilities[distances > self.radius] = 0

        return probabilities

    @lru_cache(maxsize=None)
    def measurement_area(self, measurement: Measurement) -> Area:
        move = -self.diameter // 2
        measurement_grid_point = GridPoint(*measurement.xy)
        southwest = measurement_grid_point.move(move, move).stick_to_resolution(self.resolution)
        return Area(self.diameter, southwest)

    def measurement_locations(self, measurement: Measurement) -> EmptyGrid:
        """
        Get locations for which the model computes the coverage for a specific measurement

        :param measurement: Antenna for which the coverage is retrieved
        :return: Locations which are covered by the model
        """
        area = self.measurement_area(measurement)
        return EmptyGrid(area.diameter, self.resolution, area.southwest)

    @lru_cache(maxsize=1)
    def _extract_features(self, measurement: Measurement, locations: EmptyGrid) -> EagerTensor:
        """
        Get features for each location from a grid for a specific measurement.

        :param measurement: antenna used to compute features
        :param locations: locations for which features are to be computed
        :return: angle array and distance array with features representing
            the same antenna on the last index
        """
        # get a meshgrid containing all coordinates for the values and reshape (n-coords, x-y, 1)
        grid_coords = tf.cast(
            tf.reshape(tf.stack(locations.coords_mesh_grid('tf'), axis=-1),
                       [-1, 2, 1]), tf.float32)

        # get antenna(s) coordinates
        antenna_coords = extract_antenna_coordinates(measurement)
        azimuth = extract_azimuth(measurement)
        return extract_features(grid_coords, antenna_coords,  azimuth)

    def predict(self, angles: EagerTensor, distances: EagerTensor) -> EagerTensor:
        """
        Retrieve (cached) probabilities for the angle and distance (pairs)
        Features from the same location should have the same index

        :param angles: angle features
        :param distances: distance features
        :return: probability for each location (based on its index)
        """
        return tf.gather_nd(self._probabilities, tf.stack([angles, distances // DISTANCE_STEP], -1))

    def fit(self, x: np.ndarray, y: np.ndarray, x_cal: np.ndarray = None, y_cal: np.ndarray = None, **kwargs) -> None:
        """
        Fit model used to classify antennas and (optionally) calibrate these scores

        :param x: features of antennas to fit model on
        :param y: label (connection yes/no) for antennas to fit model on
        :param x_cal: features of antennas to calibrate model on
        :param y_cal: label (connection yes/no) for antennas to calibrate model on
        """
        if isinstance(self.model, CorrectedPriorsModel):
            if 'fake_prior_odds' not in kwargs or 'true_prior_odds' not in kwargs:
                raise ValueError(
                    "Cannot fit CorrectedPriorsModel without fake_prior_odds or true_prior_odds")
            self.model.fit(x, y, fake_prior_odds=kwargs['fake_prior_odds'],
                           true_prior_odds=kwargs['true_prior_odds'])
        else:
            self.model.fit(x, y)
        if x_cal is not None and y_cal is not None:
            if self.calibrator is None:
                raise ValueError('Calibration model not specified')
            self._fit_calibrator(x_cal, y_cal, **kwargs)
        self._probabilities = calculate_angle_distance_cache(self.model, self.diameter)

    def _fit_calibrator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Calibrate calibration model

        :param x: features of antennas to calibrate model on
        :param y: label (connection yes/no) for antennas to calibrate model on
        """
        # retrieve normalized scores
        predictions = self.model.predict_proba(x, )[:, 1]
        # add misleading evidence to make sure changes are never 0 or 1
        predictions = np.append(predictions, [0, 1])
        y = np.append(y, [1, 0])

        # init and fit calibrator
        if 'fake_prior_odds' not in kwargs or 'true_prior_odds' not in kwargs:
            raise ValueError(
                "Cannot fit CorrectedPriorsModel without fake_prior_odds or true_prior_odds")
        self.calibrator.fit(predictions, y,
                            fake_prior_odds=kwargs['fake_prior_odds'],
                            true_prior_odds=kwargs['true_prior_odds'])

    def cache_clear(self):
        self._extract_features.cache_clear()
        self.measurement_area.cache_clear()
        self.probabilities.cache_clear()


class ExtendedAngleDistanceClassificationCoverageModel(AngleDistanceClassificationCoverageModel):
    """
    Extended coverage model that models the service area of an Antenna
    with a (scikit-learn based) classification model with the
    following features:
        * angle
        * distance

    :param inner_diameter: diameter of inner grid used for coverage model
    :param inner_resolution: resolution of inner grid used for coverage model
    :param outer_diameter: diameter of outer grid used for coverage model
    :param outer_resolution: resolution of outer grid used for coverage model
    :param model: model used to classify antennas
    :param calibrator: calibrator used to calibrate probabilities
    """
    def __init__(self, inner_diameter: int, inner_resolution: int, outer_diameter: int, outer_resolution: int,
                 model: BaseEstimator, calibrator: Optional[BaseEstimator] = None):
        super().__init__(outer_resolution, outer_diameter, model, calibrator)
        self.inner_diameter = inner_diameter
        self.inner_resolution = inner_resolution

    @property
    def outer_diameter(self):
        """
        Size of the outer grid
        """
        return self.diameter

    @property
    def outer_resolution(self):
        """
        Resolution of the outer grid
        """
        return self.resolution

    @lru_cache(maxsize=None)
    def probabilities(self, measurement: Measurement) -> DenseCenterGrid:
        locations = self.measurement_locations(measurement)

        return DenseCenterGrid(**locations.meta_attributes,
                               inner_values=self.get_probabilities(measurement, locations.inner),
                               outer_values=self.get_probabilities(measurement, locations.outer))

    def measurement_locations(self, measurement: Measurement) -> EmptyDenseCenterGrid:
        area = self.measurement_area(measurement)
        return EmptyDenseCenterGrid(area.diameter, inner_diameter=self.inner_diameter,
                                    outer_resolution=self.outer_resolution, inner_resolution=self.inner_resolution,
                                    southwest=area.southwest)


@dataclass
class CoverageData:
    """Data to train the coverage model with. Should consist of a location, with a positive antenna and negative
    antennas"""
    location: Measurement
    positive_antenna: Measurement
    negative_antennas: List[Measurement]
    # optionally give the time diff explicitly, this is useful when raw data are intervals rather than time points
    time_diff_s: Optional[int] = None

    def get_bin(self, possible_bins):
        if self.time_diff_s is not None:
            # we may only have time diffs, but not actual time stamps. use that one
            delta_t = self.time_diff_s
        else:
            if not self.positive_antenna.timestamp or not self.location.timestamp:
                raise ValueError(f'could not get a timediff for coveragedata {self.location}, {self.positive_antenna}')
            delta_t = abs(self.positive_antenna.timestamp - self.location.timestamp).total_seconds()
        return first((bin for bin in possible_bins if bin[0] <= int(delta_t) <= bin[1]), None)

    @property
    def antennas(self):
        return [self.positive_antenna] + self.negative_antennas
