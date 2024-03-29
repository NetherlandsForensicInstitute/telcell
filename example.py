"""Script containing an example how to use telcell."""

from pathlib import Path

from lrbenchmark.evaluation import Setup

from telcell.data.parsers import parse_measurements_csv
from telcell.models import DummyModel
from telcell.models.simplemodel import MeasurementPairClassifier
from telcell.pipeline import run_pipeline
from telcell.utils.savefile import make_output_plots
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs


def main():
    """Main function that deals with the whole process. Three steps: loading,
    transforming and evaluation."""
    # Loading data
    path = Path(__file__).parent / 'tests' / 'testdata.csv'
    tracks = parse_measurements_csv(path)

    # Create pairs of tracks, where the both are sliced per `interval_length` (default 24 hours), starting at
    # `interval_start` (default 5 am).
    data = list(slice_track_pairs_to_intervals(create_track_pairs(tracks)))

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
    for variable, parameters, (predicted_lrs, y_true, extras) in \
            setup.run_full_grid(grid):
        model_name = parameters['model'].__class__.__name__
        print(f"{model_name}: {predicted_lrs}")

        unique_dir = '_'.join(f'{key}-{value}'
                              for key, value in variable.items())
        output_dir = main_output_dir / unique_dir
        make_output_plots(predicted_lrs,
                          y_true,
                          output_dir,
                          ignore_missing_lrs=True)


if __name__ == '__main__':
    main()
