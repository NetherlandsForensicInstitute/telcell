import datetime
import numpy as np
from typing import List

from telcell.data.models import Measurement, Track, MeasurementPair


def get_measurement_with_minimum_time_difference(track: Track,
                                                 timestamp: datetime) \
        -> Measurement:
    """
    Finds the measurement in the track with the smallest time
    difference compared to the timestamp.

    @param track: A history of measurements for a single device.
    @param timestamp: The timestamp used to find the closest measurement.
    @return: The measurement that is the closest to the timestamp
    """
    track_a_times = [x.timestamp for x in track.measurements]
    time_difference_with_timestamp = [abs(timestamp - x)
                                      for x in track_a_times]
    index = np.argmin(time_difference_with_timestamp)
    return track.measurements[index]


def make_pair_based_on_time_difference(track: Track,
                                       measurement: Measurement) \
        -> MeasurementPair:
    """
    Creates a pair based on time difference. The closest measurement in
    absolute time will be paired to the measurement.

    @param track: A history of measurements for a single device.
    @param measurement: A single measurement of a device at a certain
                        place and time.
    @return: A pair of measurements.
    """
    closest_measurement = get_measurement_with_minimum_time_difference(
        track,
        measurement.timestamp)
    return MeasurementPair(closest_measurement, measurement)


def pair_measurements_based_on_time(track_a: Track,
                                    track_b: Track) \
        -> List[MeasurementPair]:
    """
    Pairs two tracks based on the time difference between the measurements.

    @param track_a: A history of measurements for a single device.
    @param track_b: A history of measurements for a single device.
    @return: A list with all paired measurements.
    """
    paired_measurements = []
    for measurement in track_b.measurements:
        new_pair = make_pair_based_on_time_difference(track_a,
                                                      measurement)
        paired_measurements.append(new_pair)
    return paired_measurements


def filter_delay(paired_measurements: List[MeasurementPair],
                 min_delay: datetime.timedelta,
                 max_delay: datetime.timedelta) \
        -> List[MeasurementPair]:
    """
    Filter the paired measurements based on a specified delay range. Can
    return an empty list.

    @param paired_measurements: A list with all paired measurements.
    @param min_delay: the minimum amount of delay that is allowed.
    @param max_delay: the maximum amount of delay that is allowed.
    @return: A filtered list with all paired measurements.
    """
    return [x for x in paired_measurements
            if min_delay <= x.time_difference <= max_delay]
