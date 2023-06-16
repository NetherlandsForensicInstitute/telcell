import datetime
from typing import List, Tuple
from collections import Counter

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
    return min(track.measurements, key=lambda m: abs(m.timestamp - timestamp))


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
    It pairs all measurements from track_b to the closest pair of track_a,
    meaning that not all measurements from track_a have to be present in the
    final list!

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


def measurement_pairs_with_rarest_location_per_interval_based_on_track_history(
        paired_measurements: List[MeasurementPair],
        interval: Tuple[datetime.datetime, datetime.datetime],
        history_track: Track,
        round_lon_lats: bool) -> MeasurementPair:
    """
    @param paired_measurements: A list with all paired measurements to
           consider.
    @param interval: the interval of time for which one measurement pair must
           be chosen.
    @param history_track: the whole history of the right track to find rarity
           of locations in the interval considered.
    @param round_lon_lats: Can be toggled to round the lon/lats to two decimals
    @return: The measurement pair that has the rarest location based on the
             history.

    TODO There is a problem with testdata, because those are almost continuous
         lat/lon data,
    making rarity of locations not as straightforward.
    Pseudo-solution for now: round lon/lats to two decimals and determine
    rarity of those.
    This should not be used if locations are actual cell-ids
    """

    def in_interval(timestamp, interval):
        return interval[0] <= timestamp <= interval[1]

    def pair_in_interval(pair, interval):
        return any((in_interval(pair.measurement_a.timestamp, interval),
                    in_interval(pair.measurement_b.timestamp, interval)))

    def location_key(measurement):
        if round_lon_lats:
            return f'{measurement.lon:.2f}_{measurement.lat:.2f}'
        else:
            return f'{measurement.lon}_{measurement.lat}'

    def sort_key(element):
        rarity, pair = element
        return rarity, pair.time_difference

    pairs_in_interval = [x for x in paired_measurements
                         if pair_in_interval(x, interval)]
    history_outside_interval = [x for x in history_track.measurements
                                if not in_interval(x.timestamp, interval)]

    location_counts = Counter(location_key(m) for m in
                              history_outside_interval)
    min_rarity, rarest_pair = min(((location_counts.get(
        location_key(pair.measurement_b), 0), pair)
        for pair in pairs_in_interval), key=sort_key)
    return rarest_pair
