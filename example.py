import datetime
from pathlib import Path

from telcell.data.parsers import parse_measurements_csv
from telcell.data.utils import extract_intervals, split_track_by_interval
from telcell.models import DummyModel
from telcell.pipeline import run_pipeline

# Load data.
path = Path(__file__).parent / 'tests' / 'testdata.csv'
tracks = parse_measurements_csv(path)

# Only consider the first two tracks in `tracks` for this example.
track_a, track_b, *_ = tracks

# Dummy data cruncher that splits `track_a` into slices for each day that is
# represented in the data.
start = datetime.datetime.combine(
    next(iter(track_a)).timestamp.date(),
    datetime.time(5)
)
intervals = extract_intervals(
    timestamps=(m.timestamp for m in track_a),
    start=start,
    duration=datetime.timedelta(days=1)
)

data = []
for start, end in intervals:
    a, background = split_track_by_interval(track_a, start, end)
    data.append((a, track_b, {"background": background}))

models = [DummyModel()]
lrs = run_pipeline(data, models)
print(lrs)
