from pathlib import Path

from telcell.models.basicmodel import BasicModel
from telcell.data.parsers import parse_measurements_csv


def testdata_path():
    return Path(__file__).parents[1] / 'testdata.csv'


def test_basicmodel(path=testdata_path()):
    tracks = parse_measurements_csv(path)
    track_a = tracks[0]
    track_b = tracks[1]

    dummy_model = BasicModel()
    prediction = dummy_model.predict_lr(track_a, track_b)

    assert prediction == 1.0
