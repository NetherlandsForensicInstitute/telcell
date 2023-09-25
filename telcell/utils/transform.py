from collections import Counter, defaultdict
from datetime import datetime, timedelta, time
from typing import Iterator, Tuple, Mapping, Any, List

from more_itertools import pairwise

from telcell.data.models import Measurement, Track, MeasurementPair
from telcell.data.utils import extract_intervals, split_track_by_interval


def create_track_pairs(tracks: List[Track],
                       all_different_source: bool = False) \
        -> Iterator[Tuple[Track, Track]]:
    """
    Takes a set of tracks and returns track pairs:
     - all same source (i.e. same 'owner' attribute) track pairs
     - if all_different_source = True: all different source track pairs ()
     - if all_different_source = False: all different source where the name is different. This makes sense
     if name can be 'personal' and 'burner' for example.
    """
    for i, track_a in enumerate(tracks):
        for track_b in tracks[i + 1:]:
            if is_colocated(track_a, track_b):
                yield track_a, track_b
            else:
                if all_different_source:
                    yield track_a, track_b
                elif track_a.device != track_b.device:
                    yield track_a, track_b


def slice_track_pairs_to_intervals(track_pairs: Iterator[Tuple[Track, Track]],
                                   interval_length_h: int = 1) \
        -> Iterator[Tuple[Track, Track, Mapping[str, Any]]]:
    """
    Takes a set of pairs of tracks `(track_a,
    track_b)` and splits them into slices of length interval_length_h hours. The first
    slice starts on 5 am before the first data point. The function yields a 3-tuple
    for each such slice that contains the following:
        - A `Track` consisting of the `track_a` measurements for that interval;
        - A `Track` consisting of the `track_b` measurements for that interval;
        - A mapping with two `Track`s containing all other measurements ("background_a"
        and "background_b") and start and end datetime of the interval ("interval").
    """

    for track_a, track_b in track_pairs:
        # For our `start` we use 5:00 AM on the day before the start of our
        # measurements.
        earliest = next(iter(track_a)).timestamp
        start = datetime.combine(
            earliest.date() - timedelta(days=1),
            time(5, tzinfo=earliest.tzinfo),
        )

        # Find all intervals of an hour represented in the data.
        intervals = extract_intervals(
            timestamps=(m.timestamp for m in track_a),
            start=start,
            duration=timedelta(hours=interval_length_h)
        )

        for start, end in intervals:
            track_a_interval, other_a = split_track_by_interval(track_a, start,
                                                                end)
            track_b_interval, other_b = split_track_by_interval(track_b, start,
                                                                end)
            yield track_a_interval, track_b_interval, {"background_a": other_a,
                                                       "background_b": other_b,
                                                       "interval": (start, end)}


def is_colocated(track_a: Track, track_b: Track) -> bool:
    """Checks if two tracks are colocated to each other."""
    if track_a is track_b:
        return True

    return track_a.owner is not None and track_a.owner == track_b.owner


def get_switches(track_a: Track, track_b: Track) -> List[MeasurementPair]:
    """
    Retrieves subsequent registrations of different devices (e.g. 'names'). For
    example, if we have to devices A, B with registrations A1-A2-B1-B2-B3-A3-B4
    then we retrieve the pairs A2-B1, B3-A3 and A3-B4. Finally, for each pair,
    we check whether the first measurement originates from track_a. If this is
    not the case, we change the order of the two measurements, so that the first
    measurement is always from track_a and the second from track_b.

    :param track_a: A history of measurements for a single device.
    :param track_b: A history of measurements for a single device.
    :return: A list with all paired measurements.
    """
    if track_a.device == track_b.device and track_a.owner == track_b.owner:
        raise ValueError('No switches exist if the tracks are from the same device')
    combined_tracks = [(m, 'a') for m in track_a.measurements] + [(m, 'b') for m in track_b.measurements]
    combined_tracks = sorted(combined_tracks, key=lambda x: x[0].timestamp)
    paired_measurements = []
    for (measurement_first, origin_first), (measurement_second, origin_second) in pairwise(combined_tracks):
        # check this pair is from the two different tracks
        if origin_first != origin_second:
            # put the 'a' track first
            if origin_first == 'a':
                paired_measurements.append(
                    MeasurementPair(measurement_first, measurement_second)
                )
            elif origin_first == 'b':
                paired_measurements.append(
                    MeasurementPair(measurement_second, measurement_first)
                )
            else:
                raise ValueError(f'unclear origin for {origin_first}')
    return paired_measurements


def filter_delay(paired_measurements: List[MeasurementPair],
                 max_delay: timedelta) \
        -> List[MeasurementPair]:
    """
    Filter the paired measurements based on a specified maximum delay. Can
    return an empty list.

    :param paired_measurements: A list with all paired measurements.
    :param max_delay: the maximum amount of delay that is allowed.
    :return: A filtered list with all paired measurements.
    """
    return [x for x in paired_measurements
            if x.time_difference <= max_delay]


def sort_pairs_based_on_rarest_location(
        switches: List[MeasurementPair],
        history_track_b: Track,
        round_lon_lats: bool,
        max_delay: int = None
) -> List[Tuple[int, MeasurementPair]]:
    """
    Pairs are first filtered on allowed time interval of the two registrations
    of a single pair. Then, sort pairs based on the rarest location of the
    track history first and secondly by time difference of the pair.

    :param switches: A list with all paired measurements to consider.
    :param history_track_b: the history of track_b to find the rarity of locations.
    :param round_lon_lats: boolean indicating whether to round the lon/lats
            to two decimals.
    :param max_delay: maximum allowed time difference (seconds) in a pair.
                      Default: no max_delay, show all possible pairs.
    :return: The location counts and measurement pairs that are sorted on the
            rarest location based on the history and time difference. The
            location count is the number of occurrences of the coordinates from
            measurement_b in the track history that is provided.

    TODO There is a problem with testdata, because those are almost continuous
    lat/lon data, making rarity of locations not as straightforward.
    Pseudo-solution for now: round lon/lats to two decimals and determine
    rarity of those.
    This should not be used if locations are actual cell-ids
    """

    def location_key(measurement):
        if round_lon_lats:
            return f'{measurement.lon:.2f}_{measurement.lat:.2f}'
        else:
            return f'{measurement.lon}_{measurement.lat}'

    def sort_key(element):
        rarity, pair = element
        return rarity, pair.time_difference

    location_counts = Counter(
        location_key(m) for m in history_track_b.measurements)

    if max_delay:
        switches = filter_delay(switches, timedelta(seconds=max_delay))

    sorted_pairs = sorted(
        ((location_counts.get(location_key(pair.measurement_b), 0), pair) for
         pair in switches), key=sort_key)

    return sorted_pairs


def select_colocated_pairs(tracks: List[Track],
                           max_delay: timedelta = timedelta(seconds=120)) \
        -> List[MeasurementPair]:
    """
    For a list of tracks, find pairs of measurements that are colocated, i.e.
    that do not share the same track name, but do share the owner. Also filter
    the pairs based on a maximum time delay.

    :param tracks: the tracks to find pairs of.
    :param max_delay: the maximum amount of delay that is allowed.
    :return: A filtered list with all colocated paired measurements.
    """
    tracks_per_owner = defaultdict(list)
    for track in tracks:
        tracks_per_owner[track.owner].append(track)

    final_pairs = []
    for tracks in tracks_per_owner.values():
        if len(tracks) == 2:
            pairs = get_switches(*tracks)
            pairs = filter_delay(pairs, max_delay)
            final_pairs.extend(pairs)
    return final_pairs


def generate_all_pairs(measurement: Measurement, track: Track) \
        -> List[MeasurementPair]:
    """
    Created all measurement pairs of the specific measurement with every
    measurement of the given track.

    :param measurement: the measurement that will be linked to other
     measurements
    :param track: the measurements of this track will be linked to the given
     measurement
    :return: A list with paired measurements.
    """
    pairs = []
    for measurement_a in track:
        pairs.append(MeasurementPair(measurement, measurement_a))
    return pairs
