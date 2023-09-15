from telcell.utils.transform import is_colocated
from telcell.data.parsers import parse_measurements_csv


def test_is_colocated(testdata_path):
    track_a, track_b, track_c = parse_measurements_csv(testdata_path)

    assert is_colocated(track_a, track_a) is True
    assert is_colocated(track_a, track_b) is True
    assert is_colocated(track_b, track_a) is True
    assert is_colocated(track_a, track_c) is False
    assert is_colocated(track_c, track_a) is False
