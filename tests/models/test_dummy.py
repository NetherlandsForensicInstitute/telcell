from telcell.data.parsers import parse_measurements_csv
from telcell.models import DummyModel


def test_dummy_model(testdata_path):
    tracks = parse_measurements_csv(testdata_path)
    track_a = tracks[0]
    track_b = tracks[1]

    dummy_model = DummyModel()
    prediction, _ = dummy_model.predict_lr(track_a, track_b)

    assert prediction == 1.0
