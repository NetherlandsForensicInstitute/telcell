from datetime import timedelta

from telcell.data.parsers import parse_measurements_csv
from telcell.utils.transform import get_switches, filter_delay, \
    get_pair_with_rarest_measurement_b, categorize_measurement_by_rounded_coordinates


def test_simplemodel(testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = get_switches(track_a, track_b)

    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              max_delay_td)

    assert len(paired_measurements) == len(filtered_measurement_pairs)

    _, rarest_measurement_pair = \
        get_pair_with_rarest_measurement_b(
            filtered_measurement_pairs,
            track_b,
            categorize_measurement_for_rarity=categorize_measurement_by_rounded_coordinates)
    assert rarest_measurement_pair
