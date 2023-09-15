from datetime import timedelta, datetime, timezone

from telcell.data.parsers import parse_measurements_csv
from telcell.utils.transform import get_switches, filter_delay, sort_pairs_based_on_rarest_location


def test_simplemodel(testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = get_switches(track_a, track_b)

    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              max_delay_td)

    assert len(paired_measurements) == len(filtered_measurement_pairs)

    rarest_measurement_pair = \
        sort_pairs_based_on_rarest_location(
            filtered_measurement_pairs,
            track_b,
            round_lon_lats=True)[0][1]
    assert rarest_measurement_pair
