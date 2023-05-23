from telcell.data.parsers import parse_measurements_csv


def test_parse_measurements_csv(testdata_path):
    tracks = parse_measurements_csv(testdata_path)

    # 3 different tracks in testdata
    assert len(tracks) == 3

    # all tracks have 50 records
    assert [len(x) for x in tracks] == [50, 50, 50]
