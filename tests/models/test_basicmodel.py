from telcell.models.basicmodel import BasicModel
from telcell.data.parsers import parse_measurements_csv

def test_basicmodel(path='./tests/testdata.csv'):
    tracks = parse_measurements_csv(path)
    trackA = tracks[0]
    trackB = tracks[1]

    dummyModel = BasicModel()
    prediction = dummyModel.predict(trackA, trackB)

    assert prediction == 1.0

