from telcell.crunchers.dummycruncher import dummy_cruncher
from telcell.models.dummy import DummyModel
from telcell.pipeline import run_pipeline


def test_pipeline(test_data):
    data = list(dummy_cruncher(test_data))

    model = DummyModel()
    lrs, y_true = run_pipeline(data, model)

    assert lrs == [1.0, 1.0, 1.0, 1.0]
    assert y_true == [True, True, False, False]
