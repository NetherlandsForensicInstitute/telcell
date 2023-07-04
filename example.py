"""Script containing an example how to use telcell."""

import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Tuple

from lrbenchmark.evaluation import Setup

from telcell.data.models import Track
from telcell.data.parsers import parse_measurements_csv
from telcell.data.utils import extract_intervals, split_track_by_interval
from telcell.models import DummyModel
from telcell.models.simplemodel import MeasurementPairClassifier
from telcell.pipeline import run_pipeline
from telcell.utils.savefile import make_output_plots


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
        # For our `start` we use 5:00 AM on the day before the start of our
        # measurements.
        earliest = next(iter(track_a)).timestamp
        start = datetime.datetime.combine(
            earliest.date() - datetime.timedelta(days=1),
            datetime.time(5, tzinfo=earliest.tzinfo),
        )

        # Find all intervals of an hour represented in the data.
        intervals = extract_intervals(
            timestamps=(m.timestamp for m in track_a),
            start=start,
            duration=datetime.timedelta(hours=1)
        )

        for start, end in intervals:
            single_hour, other = split_track_by_interval(track_a, start, end)
            yield single_hour, track_b, {"background": other,
                                         "interval": (start, end)}


def main():
    """Main function that deals with the whole process. Three steps: loading,
    crunching and evaluation."""
    # Loading data.
    path = Path(__file__).parent / 'tests' / 'testdata.csv'
    tracks = parse_measurements_csv(path)

    # Crunch the data so that it fits our desired format.
    data = list(dummy_cruncher(tracks))

    # Specify the models that we want to evaluate.
    models = [DummyModel(), MeasurementPairClassifier(
        colocated_training_data=parse_measurements_csv(
            'tests/test_measurements.csv'))]

    # Create an experiment setup using run_pipeline as the evaluation function
    setup = Setup(run_pipeline)
    # Specify the constant parameters for evaluation
    setup.parameter('data', data)
    # Specify the main output_dir. Each model/parameter combination gets a
    # directory in the main output directory.
    main_output_dir = Path('scratch')

    # Specify the variable parameters for evaluation in the variable 'grid'.
    # This grid is a dict of iterables and all combinations will be used
    # during the evaluation. An example is a list of all different models
    # that need to be evaluated, or a list of different parameter settings
    # for the models.
    grid = {'model': models}
    for variable, parameters, (predicted_lrs, y_true) in \
            setup.run_full_grid(grid):
        model_name = parameters['model'].__class__.__name__
        print(f"{model_name}: {predicted_lrs}")

        unique_dir = '_'.join(f'{key}-{value}'
                              for key, value in variable.items())
        output_dir = main_output_dir / unique_dir
        make_output_plots(predicted_lrs,
                          y_true,
                          output_dir)


if __name__ == '__main__':
    main()
