from datetime import timedelta, datetime, timezone

from telcell.data.parsers import parse_measurements_csv
from telcell.models.simplemodel import get_switches, filter_delay,\
    make_pair_based_on_rarest_location_within_interval


def test_simplemodel(testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = get_switches(track_a, track_b)

    min_delay_td = timedelta(seconds=0)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)

    assert len(paired_measurements) == 4320
    assert len(filtered_measurement_pairs) == 4320

    min_delay_td = timedelta(seconds=10)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)

    assert len(filtered_measurement_pairs) == 1926

    start_of_interval = datetime(2023, 5, 18, 00, 00, 00, tzinfo=timezone.utc)
    end_of_interval = datetime(2023, 5, 19, 00, 00, 00, tzinfo=timezone.utc)

    rarest_measurement_pair = \
        make_pair_based_on_rarest_location_within_interval(
            filtered_measurement_pairs,
            (start_of_interval, end_of_interval),
            track_b,
            round_lon_lats=True)
    assert rarest_measurement_pair
