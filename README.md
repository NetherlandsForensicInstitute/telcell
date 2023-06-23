# Telcell
Telcell is a collection of scripts than can be used to determine the evidence that a pair of any phones is used by the same person. 

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

The script example.py contains information and an example pipeline to run the library. It uses testdata that is included in the repository and should return output like:
```
DummyModel: [1.0, 1.0, 1.0, 1.0]
MeasurementPairClassifier: [1.0, 1.0, 1.0, 1.0]
```

### Inputdata
At this moment only a csv file can be used as input, but an extra inputsource can be easily realised if necessary. After parsing, the data is stored as `Tracks` and `Measurement` objects. The following columns are expected to be present:

        - track
        - sensor
        - celldb.wgs84.lat
        - celldb.wgs84.lon
        - timestamp

Any additional columns are stored under the `extra` attribute of each resulting `Measurement` object. 

### Processing
The next step is data processing or crunching. The data will be transformed in a format that can be used by the models.

### Evaluation
We use the evaluation of the library lrbenchmark. A `Setup` object is created with the `run_pipeline` function, the models that have to be evaluated, the necessary parameters and the data itself. All different combination will be evaluated, resulting in multiple lrs that can be used to determine if the two phones were carried by the same person or not.  

