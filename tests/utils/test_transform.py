import random
from datetime import timedelta, datetime
from functools import partial

from telcell.data.models import Measurement, Track, Point
from telcell.utils.transform import get_switches, create_track_pairs, \
    is_colocated, sort_pairs_based_on_rarest_location, MeasurementPair


def test_get_switches(test_data_3days):
    track_a, track_b, _ = test_data_3days
    switches = get_switches(track_a, track_b)

    # check list of manually inserted first few switches
    correct_switches = [
        (0, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 4),
        (5, 5),
        (6, 5),
        (6, 6),
        (7, 6),
        (7, 7),
        (8, 8),
        (8, 9),
        (9, 9),
        (10, 10)

    ]
    for correct_switch, switch in zip(correct_switches, switches):
        assert switch.measurement_a == track_a.measurements[correct_switch[0]]
        assert switch.measurement_b == track_b.measurements[correct_switch[1]]

    # check last switch
    assert switches[-1].measurement_a == track_a.measurements[-1]
    assert switches[-1].measurement_b == track_b.measurements[-1]


def test_create_track_pairs(test_data):
    tracks = [
        Track(owner='A', device='1', measurements=[]),
        Track(owner='A', device='2', measurements=[]),
        Track(owner='B', device='1', measurements=[]),
        Track(owner='B', device='2', measurements=[]),
        Track(owner='B', device='3', measurements=[]),
        Track(owner='C', device='1', measurements=[]),
    ]
    pairs = list(create_track_pairs(tracks))
    assert len([pair for pair in pairs if is_colocated(*pair)]) == 4 # same source: 1 A and 3 B
    assert len(pairs) == 4 + 7 # diff source: A1-B2/B3 A2-B1/B3/C1 B2-C1 B3-C1

    pairs = list(create_track_pairs(tracks, all_different_source=True))
    assert len([pair for pair in pairs if is_colocated(*pair)]) == 4 # same source: 1 A and 3 B
    assert len(pairs) == 4 + 11


def test_sort_by_time_diff_for_same_location_rarity(max_delay):
    measurement_tmp = partial(Measurement, Point(lat=1, lon=1), extra={})
    t_0 = datetime(2023, 8, 3, 12, 0)
    seed = random.Random(42)

    # create switches where the locations are identical and only timestamp
    # differs
    switches_one_location = [
        MeasurementPair(measurement_tmp(timestamp=t_0),
                        measurement_tmp(timestamp=t_0 + timedelta(
                            seconds=seed.randint(1, 30))))
        for _ in range(10)
    ]
    background_b = Track('', '',
                         [s.measurement_b for s in switches_one_location])
    sorted_pairs = sort_pairs_based_on_rarest_location(switches_one_location,
                                                       background_b,
                                                       False,
                                                       max_delay)

    # manually check correct timestamps of measurement_b
    correct_timestamps_b = [1, 4, 4, 5, 8, 8, 9, 21, 24, 24]
    for correct_time_diff, (count, pair) in zip(correct_timestamps_b, sorted_pairs):
        assert pair.measurement_b.timestamp.second == correct_time_diff

    # all time differences are within 2 minutes so we should retrieve all pairs
    assert len(sorted_pairs) == 10
    # the time differences should be increasing
    assert all(x[1].measurement_b.timestamp <= y[1].measurement_b.timestamp
               for x, y in zip(sorted_pairs, sorted_pairs[1:]))
    # we have ten times the same location, so location count should be equal for all
    assert all(count == 10 for count, _ in sorted_pairs)


def test_sort_by_location_rarity_for_same_time_diff(max_delay):
    t_0 = datetime(2023, 8, 3, 12, 0)
    measurement_tmp = partial(Measurement, timestamp=t_0, extra={})
    seed = random.Random(42)

    # create switches where the timestamps are identical and only locations differs
    switches_one_time_diff = [
        MeasurementPair(measurement_tmp(coords=Point(lat=1, lon=1)),
                        measurement_tmp(
                            coords=Point(lat=2, lon=seed.randint(1, 4))))
        for _ in range(10)
    ]
    background_b = Track('', '',
                         [s.measurement_b for s in switches_one_time_diff])
    sorted_pairs = sort_pairs_based_on_rarest_location(switches_one_time_diff,
                                                       background_b,
                                                       False,
                                                       max_delay)

    # manually check correct location counts and longitude of measurement_b
    correct_counts_lon_b = [(1, 3.0), (1, 4.0), (3, 2.0), (3, 2.0), (3, 2.0),
                            (5, 1.0), (5, 1.0), (5, 1.0), (5, 1.0), (5, 1.0)]
    for (correct_count, correct_lon_b), (count, pair) in zip(correct_counts_lon_b,
                                                sorted_pairs):
        assert count == correct_count
        assert pair.measurement_b.lon == correct_lon_b

    # all timestamps of measurement a and b are equal so we should retrieve all pairs
    assert len(sorted_pairs) == 10
    # the counts should be increasing since the pairs should be sorted by count
    assert all(x[0] <= y[0] for x, y in zip(sorted_pairs, sorted_pairs[1:]))

    background_a = Track('', '',
                         [s.measurement_a for s in switches_one_time_diff])
    sorted_pairs = sort_pairs_based_on_rarest_location(switches_one_time_diff,
                                                       background_a,
                                                       False,
                                                       max_delay)
    # with a wrong background that has no intersection with 'track_b',
    # the counts should be zero
    assert all(count == 0 for count, _ in sorted_pairs)


def test_sort_outside_bin(max_delay):
    t_0 = datetime(2023, 8, 3, 12, 0)
    measurement_tmp = partial(Measurement, Point(lat=1, lon=1), extra={})
    seed = random.Random(42)
    # create switches with negative and positive time differences that can be
    # larger than the maximum bin
    switches = [
        MeasurementPair(measurement_tmp(timestamp=t_0),
                        measurement_tmp(timestamp=t_0 + timedelta(
                            minutes=seed.randint(-4, 4))))
        for _ in range(10)
    ]
    background_b = Track('', '', [s.measurement_b for s in switches])
    sorted_pairs = sort_pairs_based_on_rarest_location(switches,
                                                       background_b,
                                                       False,
                                                       max_delay)

    assert len(sorted_pairs) == 5  # some pairs with too large time difference are filtered
    assert max(s[1].time_difference for s in sorted_pairs).seconds == 120


def test_round_lat_lons(max_delay):
    t_0 = datetime(2023, 8, 3, 12, 0)
    measurement_tmp = partial(Measurement, timestamp=t_0, extra={})

    # create switches where the longitude has more than 2 digits after the decimal point
    switches = [
        MeasurementPair(measurement_tmp(coords=Point(lat=1, lon=1)),
                        measurement_tmp(
                            coords=Point(lat=1, lon=1 + (i / 10000))))
        for i in range(10)
    ]
    background_b = Track('', '', [s.measurement_b for s in switches])
    sorted_pairs = sort_pairs_based_on_rarest_location(switches,
                                                       background_b,
                                                       False,
                                                       max_delay)
    # without rounding, we have unique locations
    assert all(count == 1 for count, _ in sorted_pairs)

    sorted_pairs = sort_pairs_based_on_rarest_location(switches,
                                                       background_b,
                                                       True,
                                                       max_delay)
    # with rounding, the locations are identical
    assert all(count == 10 for count, _ in sorted_pairs)

