from abc import abstractmethod
from typing import List, Tuple, Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from telcell.auxilliary_models.rare_pair.coverage_model import CoverageData
from telcell.auxilliary_models.rare_pair.features import extract_angles, \
    extract_distances
from telcell.data.models import Measurement


class BaseTransformer:
    @abstractmethod
    def get_features(self, measurements: List[Measurement]) \
            -> Tuple[List[Tuple[Any]], List[int]]:
        """
        Takes a list of measurements and returns per measurement a tuple with the features and the y values.
        :param measurements: list of measurements containing the gps locations and the positive and negative antennas
        :return: a list of tuples containing the features and the y values
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        Get the feature variable names.
        :return: Returns a list of the feature names.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class AngleDistanceTransformer(BaseTransformer):
    """
    Transformer that computes angles and distances between a gps location and antennas
    """

    @staticmethod
    def prepare_measurement(gps_location: Measurement,
                            measurements: List[Measurement]) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Takes a gps location and a list of antennas and returns them in tensor format for speedy computations
        """
        gps_location_tensor = tf.convert_to_tensor(
            np.expand_dims(np.array(gps_location.xy), (-1, 0)), tf.float32)
        antennas_coords = np.array([antenna.xy for antenna in measurements])
        antennas_coords_tensor = tf.convert_to_tensor(
            np.expand_dims(antennas_coords.transpose(), 0), tf.float32)
        antenna_azimuths = np.array(
            [antenna.extra['azimuth'] for antenna in measurements])
        antenna_azimuths_tensor = tf.convert_to_tensor(
            np.expand_dims(antenna_azimuths, 0), tf.float32)
        return gps_location_tensor, antennas_coords_tensor, antenna_azimuths_tensor

    @staticmethod
    def get_angles(gps_location_tensor: tf.Tensor,
                   antennas_coords_tensor: tf.Tensor,
                   antenna_azimuths_tensor: tf.Tensor) -> List[int]:
        """
        Computes the angle between a gps location and the corresponding antennas
        """
        angles = extract_angles(gps_location_tensor, antennas_coords_tensor,
                                antenna_azimuths_tensor)
        return angles.numpy().flatten().tolist()

    @staticmethod
    def get_distances(gps_location_tensor: tf.Tensor,
                      antennas_coords_tensor: tf.Tensor) -> List[int]:
        """
        Computes the distance between a gps location and the corresponding antennas
        """
        distances = extract_distances(gps_location_tensor,
                                      antennas_coords_tensor)
        return distances.numpy().flatten().tolist()

    def get_features(self, measurements: List[CoverageData]) \
            -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        For each measurement a list containing the angle and the distance per antenna is returned.
        :param measurements: List of measurements to calculate features for.
        :return: A list of feature tuples and a list of outputs.
        The length of the lists is:
        the number of measurements * number of antennas (positive & negative) per measurement.
            - Features: tuple of:
                - Distance (m) between point and antenna
                - Angular offset (degrees) of the point relative to the antenna's azimuth (viewing direction).
            - Output: binary, i.e. was this antenna connected to or not.
        """
        features = []
        y = []
        for measurement in tqdm(measurements,
                                desc='Generating features for each measurement'):
            gps_location_tensor, antennas_coords_tensor, antenna_azimuths_tensor = self.prepare_measurement(
                measurement.location, measurement.antennas)
            distances = self.get_distances(gps_location_tensor,
                                           antennas_coords_tensor)
            angles = self.get_angles(gps_location_tensor,
                                     antennas_coords_tensor,
                                     antenna_azimuths_tensor)

            features.extend(list(zip(distances, angles)))
            y.extend([1] + [0] * len(measurement.negative_antennas))
        return features, y

    @property
    def feature_names(self) -> List[str]:
        """
        Get the feature variable names.
        :return: Returns a list of the feature names.
        """
        return ['distance', 'angle']
