import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Tuple

from telcell.data.models import Track
from telcell.data.parsers import parse_measurements_csv
from telcell.data.utils import extract_intervals, split_track_by_interval
from telcell.models import DummyModel
from telcell.pipeline import run_pipeline


def dummy_cruncher(tracks: Iterable[Track]) \
        -> Iterator[Tuple[Track, Track, Mapping[str, Any]]]:
    """
    Dummy data cruncher that takes each consecutive pair of tracks `(track_a,
    track_b)` and splits `track_a` into slices for each day. Yields a 3-tuple
    for each such slice that contains the following:
        - A `Track` consisting of the `track_a` measurements for that day;
        - A reference to `track_b`;
        - A mapping with a `Track` containing all other `track_a` measurements.
    """

    for track_a, track_b in pairwise(tracks):
        # For our `start` we use 5:00 AM on the first day of the first
        # measurement.
        start = datetime.datetime.combine(
            next(iter(track_a)).timestamp.date(),
            datetime.time(5)
        )

        # Find all intervals of a day represented in the data.
        intervals = extract_intervals(
            timestamps=(m.timestamp for m in track_a),
            start=start,
            duration=datetime.timedelta(days=1)
        )

        for start, end in intervals:
            single_day, other = split_track_by_interval(track_a, start, end)
            yield single_day, track_b, {"background": other}


def main():
    # Load data.
    path = Path(__file__).parent / 'tests' / 'testdata.csv'
    tracks = parse_measurements_csv(path)

    # Crunch the data so that it fits our desired format.
    data = list(dummy_cruncher(tracks))

    # Specify the models that we want to evaluate.
    models = [DummyModel()]

    # Run the pipeline and print results.
    lrs = run_pipeline(data, models, output_dir="scratch")
    print(lrs)


if __name__ == '__main__':
    main()
