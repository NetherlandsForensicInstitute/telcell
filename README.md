# Telcell

Telcell is a collection of scripts than can be used to determine the evidence that a pair of any phones is used by the
same person.

## Requirements

1. Python 3.10

## Pre-run

1. Make sure the requirements are installed

```bash
pip install -r requirements.txt
```

## Tests

To run the tests do:

```bash
pip install -r test-requirements.txt
coverage run --branch --source telcell --module pytest --strict-markers tests/
```

## Run

```bash
python example.py
```

The script example.py contains information and an example pipeline to run the library. It uses testdata that is included
in the repository and should return output like:

```
DummyModel: [1.0, 1.0, 1.0, 1.0]
MeasurementPairClassifier: [1.0, 1.0, 1.0, 1.0]
```

### Input data

At this moment only a csv file can be used as input, but an extra input source can be easily realised if necessary.
The following columns are expected to be present in the csv file:

        - owner
        - device
        - cellinfo.wgs84.lat
        - cellinfo.wgs84.lon
        - timestamp

Any additional columns are stored under the `extra` attribute of each resulting `Measurement` object.

After parsing, the data is stored as `Tracks` and `Measurement` objects. Each `Track` has an `owner`, `device` and a
sequence of `Measurement`. A `Measurement` consists of `coords` (a `Point` object), a timestamp (`datetime`) and
an `extra` (`Mapping`) for additional information.

### Processing

The next step is data processing or crunching. The data will be transformed in a format that can be used by the models.

### Models
Even though custom models can be used with telcell, a number of models have been implemented.

- `Dummy model`: always returns an LR of 1.
- `MeasurementPairClassifier`: generates colocated and dislocated training pairs from the tracks based on the time 
  interval and the rarity of the location. Fits a Logistic Regression model and an ELUB bounded KDE calibrator. 
  Returns LRs for the training data.
- `RarePairModel`: uses coverage data (gps locations with corresponding antennas) to fit coverage_models for each 
  time interval bin. For the validation/case data, a registration pair is chosen from a pair of tracks, for which 
  the registrations are within a specific time interval and the location for the second measurement is the rarest 
  with respect to the background of `track_b`. For this pair a calibrated score is given by the model/calibrator 
  which is subsequently used to calculate the LR via a number of statistical calculations.

### Evaluation

We use the evaluation of the library lrbenchmark. A `Setup` object is created with the `run_pipeline` function, the
models that have to be evaluated, the necessary parameters and the data itself. All different combinations will be
evaluated, resulting in multiple lrs that can be used to determine if the two phones were carried by the same person or
not.  


#### Generating LRs
Instead of using the pipeline for evaluation, functionality is also present to write the generated LRs to file. 
Replace `make_output_plots` with `write_lrs` with the required parameters and the LRs will be written to file.


## Dashboards
The command to run the dashboard is `PYTHONPATH=. streamlit run data_analysis/dashboard.py -- --file-name <PATH_TO_FILE>` for the repository root.
The `PYTHONPATH` is necessary to set, so streamlit can find the imported files correctly.
In this dashboard, there exist two applications:
1) `Tracks and pairs`, to visualize the registrations and pairs from a `measurements.csv` file (so this can both be casework as validation measurements). 
   The `<PATH_TO_FILE>` refers to the path of this `measurements.csv`. You can overwrite the default path 
   to this file with a different one. The extra `--` in the above command is necessary for streamlit to 
   parse the arguments belonging to the script, instead of to streamlit itself.
2) `Travel speed`, to visualize the travel speeds found in a `measurements.csv` file, that is provided via `<PATH_TO_FILE>`. 
