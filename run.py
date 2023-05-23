from pathlib import Path

from telcell.data.parsers import parse_measurements_csv
from telcell.models.basicmodel import BasicModel


def mock_selection(data):
    return data[0], data[1]


def mock_evaluation(models, lrs):
    for model, lr in zip(models, lrs):
        print(f"the lr of model {model} is: {lr}")


# Load data
datapath = Path(__file__).parent / 'tests/testdata.csv'
data = parse_measurements_csv(datapath)

# Select tracks and background information
track_a, track_b = mock_selection(data)

# Choose which models to evaluate
models = [BasicModel()]

# Predict LR for all the models
LR = [model.predict_lr(track_a, track_b) for model in models]

# Evaluate the LR
mock_evaluation(models, LR)
