from pathlib import Path

from telcell.data.parsers import parse_measurements_csv


def testdata_path():
    return Path(__file__).parents[1] / 'testdata.csv'


def test_parse_measurements_csv(path=testdata_path()):
    tracks = parse_measurements_csv(path)

    # 3 different tracks in testdata
    assert len(tracks) == 3

    # all tracks have 50 records
    assert [len(x) for x in tracks] == [50, 50, 50]
