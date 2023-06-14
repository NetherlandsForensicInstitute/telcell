import datetime
from collections import Counter
from itertools import combinations
from typing import List, Tuple

import lir
import numpy as np
import pyproj
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from telcell.data.models import Measurement, Track, MeasurementPair
from telcell.models import Model


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

    location_counts = Counter(location_key(m) for m in history_outside_interval)
    min_rarity, rarest_pair = min(
        ((location_counts.get(location_key(pair.measurement_b), 0), pair)
         for pair in pairs_in_interval), key=sort_key)
    return rarest_pair


def calculate_distance_for_pair(pair: MeasurementPair) -> float:
    latlon_a = pair.measurement_a.latlon
    latlon_b = pair.measurement_b.latlon
    return calculate_distance_lat_lon(latlon_a, latlon_b)


def calculate_distance_lat_lon(latlon_a: Tuple[float, float],
                               latlon_b: Tuple[float, float]) -> float:
    geod = pyproj.Geod(ellps='WGS84')
    lat_a, lon_a = latlon_a
    lat_b, lon_b = latlon_b
    _, _, distance = geod.inv(lon_a, lat_a, lon_b, lat_b)
    return distance


def select_colocated_pairs(tracks: List[Track],
                           min_delay: datetime.timedelta = datetime.timedelta(
                               seconds=0),
                           max_delay: datetime.timedelta = datetime.timedelta(
                               seconds=120)) -> List[MeasurementPair]:
    final_pairs = []
    for track_a, track_b in combinations(tracks, 2):
        if track_a.owner == track_b.owner and track_a.name != track_b.name:
            pairs = pair_measurements_based_on_time(track_a, track_b)
            pairs = filter_delay(pairs, min_delay, max_delay)
            final_pairs.extend(pairs)
    return final_pairs


def generate_dislocated_pairs(measurement: Measurement, track: Track) -> List[
    MeasurementPair]:
    pairs = []
    for measurement_a in track:
        pairs.append(MeasurementPair(measurement, measurement_a))
    return pairs


class CalibratedEstimator:
    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.calibrator.fit(self.estimator.predict_proba(X)[:, 1], y)
        return self

    def predict_proba(self, X):
        p1 = self.estimator.predict_proba(X)[:, 1]
        p1 = lir.util.to_probability(self.calibrator.transform(p1))
        return np.stack([1 - p1, p1], axis=1)


class CellDistance(Model):
    def __init__(self, colocated_training_data: List[Track]):
        self.training_data = colocated_training_data

    def predict_lr(self, track_a: Track, track_b: Track, interval: Tuple[datetime, datetime], background: Track,
                   **kwargs) -> float:
        pairs = pair_measurements_based_on_time(track_a, track_b)
        pair = measurement_pairs_with_rarest_location_per_interval_based_on_track_history(
            # TODO: fix name :D
            pairs,
            interval=interval,
            history_track=background,
            round_lon_lats=True,
        )

        colocated_training_pairs = select_colocated_pairs(self.training_data)
        # resulting pairs need not be really dislocated, but simulated dislocation by temporally
        # shifting track a's history towards the timestamp of the singular measurement of track b
        dislocated_training_pairs = generate_dislocated_pairs(
            pair.measurement_b, background)
        training_pairs = colocated_training_pairs + dislocated_training_pairs
        training_labels = [1] * len(colocated_training_pairs) + [0] * len(
            dislocated_training_pairs)

        training_features = np.array(list(map(calculate_distance_for_pair,
                                              training_pairs))).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(training_features)
        training_features = scaler.transform(training_features)
        comparison_features = np.array(
            [calculate_distance_for_pair(pair)]).reshape(-1, 1)
        comparison_features = scaler.transform(comparison_features)

        estimator = LogisticRegression()
        calibrator = lir.ELUBbounder(lir.KDECalibrator(bandwidth=1.0))
        calibrated_estimator = CalibratedEstimator(estimator, calibrator)
        calibrated_estimator.fit(training_features, np.array(training_labels))

        return float(lir.to_odds(calibrated_estimator.predict_proba(
            comparison_features)[:, 1]))
