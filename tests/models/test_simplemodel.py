from datetime import timedelta, datetime, timezone
import pytest

from telcell.crunchers.dummycruncher import dummy_cruncher
from telcell.data.parsers import parse_measurements_csv
from telcell.models.simplemodel import get_switches, filter_delay,\
    make_pair_based_on_rarest_location_within_interval, \
    MeasurementPairClassifier
from telcell.pipeline import run_pipeline


def test_get_switchets(testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = get_switches(track_a, track_b)
    assert len(paired_measurements) == 4320


def test_filter_delay(testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = get_switches(track_a, track_b)

    min_delay_td = timedelta(seconds=0)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)

    assert len(filtered_measurement_pairs) == 4320

    min_delay_td = timedelta(seconds=10)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)

    assert len(filtered_measurement_pairs) == 1926


def test_make_pair_based_on_rarest_location_within_interval(
        testdata_3days_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_3days_path)

    paired_measurements = get_switches(track_a, track_b)
    min_delay_td = timedelta(seconds=10)
    max_delay_td = timedelta(seconds=120)

    filtered_measurement_pairs = filter_delay(paired_measurements,
                                              min_delay_td,
                                              max_delay_td)

    start_of_interval = datetime(2023, 5, 18, 00, 00, 00, tzinfo=timezone.utc)
    end_of_interval = datetime(2023, 5, 19, 00, 00, 00, tzinfo=timezone.utc)

    rarest_measurement_pair = \
        make_pair_based_on_rarest_location_within_interval(
            filtered_measurement_pairs,
            (start_of_interval, end_of_interval),
            track_b,
            round_lon_lats=True)
    assert rarest_measurement_pair


def test_select_collocated_pairs(test_data, test_measurements_path):
    model = MeasurementPairClassifier(
        colocated_training_data=parse_measurements_csv(
            test_measurements_path))

    assert len(model.colocated_training_pairs) == 419
    assert model.colocated_training_pairs[0].is_colocated is True
    assert model.colocated_training_pairs[212].distance == \
           pytest.approx(20458.72)


def test_MeasurementPairClassifier(test_data, test_measurements_path):
    model = MeasurementPairClassifier(
        colocated_training_data=parse_measurements_csv(
            test_measurements_path))
    data = list(dummy_cruncher(test_data))
    lrs, y_true = run_pipeline(data, model)

    assert lrs == [1.0, 1.0, 1.0, 1.0]
    assert y_true == [True, True, False, False]
