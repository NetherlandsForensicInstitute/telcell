import datetime
from typing import List, Tuple
from collections import defaultdict

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
        interval: List[datetime.datetime],
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

    def is_timestamp_in_interval(track_timestamp, interval):
        return interval[0] <= track_timestamp <= interval[1]

    def is_pair_in_interval(measurement_pair, interval):
        return any((is_timestamp_in_interval(measurement_pair.measurement_a.timestamp, interval),
                    is_timestamp_in_interval(measurement_pair.measurement_b.timestamp, interval)))

    measurement_pairs_to_consider = [x for x in paired_measurements
                                     if is_pair_in_interval(x, interval)]
    history_to_consider = [x for x in history_track.measurements
                           if not is_timestamp_in_interval(x.timestamp,
                                                           interval)]

    dict_location_bins = defaultdict(int)
    for h in history_to_consider:
        if round_lon_lats:
            lon_lat_from_h = str(round(h.lon, 2)) + "_" + str(round(h.lat, 2))
        else:
            lon_lat_from_h = str(h.lon) + "_" + str(h.lat)

        dict_location_bins[lon_lat_from_h] += 1

    rarest_m = None
    rarity_m = len(history_to_consider)

    for m in measurement_pairs_to_consider:

        if round_lon_lats:
            lon_lat_from_m = str(round(m.measurement_a.lon, 2)) + "_" + \
                             str(round(m.measurement_a.lat, 2))
        else:
            lon_lat_from_m = str(m.measurement_a.lon) + "_" + str(m.measurement_a.lat)

        if dict_location_bins[lon_lat_from_m] < rarity_m:
            rarity_m = dict_location_bins[lon_lat_from_m]
            rarest_m = m

    return rarest_m
