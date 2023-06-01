from datetime import timedelta

from telcell.data.parsers import parse_measurements_csv
from telcell.models.simplemodel import pair_measurements_based_on_time,\
    filter_delay


def test_simplemodel(testdata_path):
    tracks = parse_measurements_csv(testdata_path)
    track_a = tracks[0]
    track_b = tracks[1]

    paired_measurements = pair_measurements_based_on_time(track_a, track_b)

    min_delay_td = timedelta(seconds=0)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)

    assert len(paired_measurements) == 50
    assert len(filtered_measurement_pairs) == 50

    min_delay_td = timedelta(seconds=10)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)
    assert len(filtered_measurement_pairs) == 0
