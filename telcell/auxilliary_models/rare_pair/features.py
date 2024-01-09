from math import pi
from typing import Tuple, Any, Optional, Sequence

import numpy as np
import tensorflow as tf

from telcell.auxilliary_models.antenna import Antenna
from telcell.auxilliary_models.rare_pair.utils import round_to_nearest, \
    DISTANCE_STEP
from telcell.data.models import Measurement

PI = tf.constant(pi)


def extract_features(grid_coords: tf.Tensor, antenna_coords: tf.Tensor,
                     azimuths: tf.Tensor) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert coordinates of values points to:
     * distance to antenna
     * angle to antenna (corrected for azimuth of antenna)

    :param grid_coords: coordinates [n-coordinates, 2 (x-y), 1](float)
    :param antenna_coords: coordinates of antennas [1, 2 (x-y), n-antennas](float)
    :param azimuths: azimuths of antennas [1, n-antennas](float)
    :return: angle array and distance array with features representing
        the same antenna on the last index
    """
    distance = grid_coords - antenna_coords
    angle = tf.math.atan2(distance[:, 0, :], distance[:, 1, :])
    distance = tf.norm(distance, axis=1)
    distance = round_to_nearest(distance)

    # correct angle for azimuth of antenna and take absolute value
    # TODO: floormod should not be needed if database provide only 'valid' azimuths. could be doable
    # TODO: by modelling cell measurements and putting requirements on the azimuth.
    angle = tf.math.floormod(abs(angle * 180 / PI - azimuths), 360)
    angle = tf.where(angle > 180, 360 - angle, angle)

    return tf.cast(angle, dtype=tf.int64), tf.cast(distance, dtype=tf.int64)


def calculate_angle_distance_cache(model: Any,
                                   max_distance: int) -> tf.Tensor:
    """
    Precalculate probabilities for all possible distance/angle combinations of a sklearn model

    :param model: sklearn model used to predict scores
    :param max_distance: maximum distance to calculate cache for
    :return: Tensor of probabilities
    """
    distance_steps = list(range(0, max_distance, DISTANCE_STEP))

    # cache results for all possible distance/angle combinations
    dist, angle = np.meshgrid(distance_steps, list(range(181)))
    feature_array = np.array((dist.flatten(), angle.flatten())).transpose()
    predictions = model.predict_proba(feature_array)[:, 1]

    # switch distance -- angle order for implementation
    probabilities = np.zeros((181, len(distance_steps) + 1))
    probabilities[:, :len(distance_steps)] = np.reshape(predictions, (
        181, len(distance_steps)))

    return tf.convert_to_tensor(probabilities)


def extract_azimuth(measurement: Measurement,
                    antennas: Optional[Sequence[Antenna]] = None) -> tf.Tensor:
    """
    Extract  azimuth for antenna(s)

    :param measurement: antenna
    :param antennas: antennas within the proximity of the antenna
    :return: azimuth array [1, n-antennas](float)
    """
    if antennas:
        azimuth = np.array([measurement.extra['azimuth']] + [
            antenna.azimuth if antenna.azimuth <= 180 else antenna.azimuth - 360
            for antenna in antennas])
    else:
        azimuth = np.array([measurement.extra['azimuth']])

    # transpose to keep correct (row wise ordening)
    return tf.convert_to_tensor(np.expand_dims(azimuth, 0), tf.float32)


def extract_antenna_coordinates(measurement: Measurement,
                                antennas: Optional[Sequence[Antenna]] = None) \
        -> tf.Tensor:
    """
    Extract coordinates antenna(s)

    :param measurement: antenna
    :param antennas: antennas within the proximity of the antenna
    :return: coordinates array of antennas [1, 2 (x-y), n-antennas](float)
    """
    if antennas:
        antennas_coords = np.array(
            [measurement.xy] + [antenna.coords.xy() for antenna in
                                antennas])
    else:
        antennas_coords = np.array([measurement.xy])

    # transpose to keep correct (row wise ordening)
    return tf.expand_dims(
        tf.transpose(tf.convert_to_tensor(antennas_coords, tf.float32)), 0)


def extract_angles(coords: tf.Tensor, antennas_coordinates: tf.Tensor,
                   azimuths: tf.Tensor) -> tf.Tensor:
    """
    convert coordinates of points to angle to antenna
    (corrected for azimuth of the antenna)

    :param coords: coordinates for grid [n-coordinates, 2 (x-y), 1](float)
    :param antennas_coordinates: coordinates array of antennas [1, 2 (x-y), n-antennas](float)
    :param azimuths: azimuth array [1, n-antennas](float)
    :return: angle array [resolution, resolution, n-antennas](int)
        with features representing the same antenna on the last index
    """
    t_dif = coords - antennas_coordinates
    t_angle = tf.math.atan2(t_dif[:, 0, :], t_dif[:, 1, :])
    angle = tf.reshape(t_angle, [*(1, 1), -1])

    # radians to degrees
    angle = angle * 180 / PI
    # take difference between angle and azimuth
    angle = abs(angle - azimuths)
    # only use half a circle because of mirror symmetry.
    angle = abs(tf.where(angle > 180, 360 - angle, angle))
    return tf.cast(angle, dtype=tf.int64)


def extract_distances(coords: tf.Tensor, antennas_coordinates: tf.Tensor,
                      clip_distance: int = 629000) -> tf.Tensor:
    """
    convert coordinates of points to distance to antenna

    :param coords: coordinates for grid [n-coordinates, 2 (x-y), 1](float)
    :param antennas_coordinates: coordinates array of antennas [1, 2 (x-y), n-antennas](float)
    :param clip_distance: distances equal or larger than this value are clipped to his value
    :return: distance array [resolution, resolution, n-antennas](int)
        with features representing the same antenna on the last index
    """
    t_dif = coords - antennas_coordinates
    t_dist = tf.norm(t_dif, axis=1)
    dist = tf.cast(tf.reshape(t_dist, [*(1, 1), -1]), dtype=tf.int64)
    # clip distance (clipped_distance is mapped to 0 probability upon cache retrieval)
    dist = tf.clip_by_value(dist, 0, clip_value_max=int(clip_distance))
    return dist
