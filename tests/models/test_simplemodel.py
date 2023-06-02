from datetime import timedelta, datetime, timezone


from telcell.data.parsers import parse_measurements_csv
from telcell.models.simplemodel import pair_measurements_based_on_time,\
    filter_delay,\
    measurement_pairs_with_rarest_location_per_interval_based_on_track_history


def test_simplemodel(testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = pair_measurements_based_on_time(track_a, track_b)

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

    rarest_measurement_pair = measurement_pairs_with_rarest_location_per_interval_based_on_track_history(
        filtered_measurement_pairs,
        [start_of_interval,
         end_of_interval],
        track_b,
        round_lon_lats=True)
    assert rarest_measurement_pair
