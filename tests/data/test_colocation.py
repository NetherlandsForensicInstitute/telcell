from telcell.data.models import is_colocated
from telcell.data.parsers import parse_measurements_csv


def test_is_colocated(testdata_path):
    tracks = parse_measurements_csv(testdata_path)
    track_a = tracks[0]
    track_b = tracks[1]
    track_c = tracks[2]

    assert is_colocated(track_a, track_b) is True
    assert is_colocated(track_a, track_c) is False
