from datetime import datetime

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from telcell.auxilliary_models.rare_pair.coverage_model import AngleDistanceClassificationCoverageModel
from telcell.auxilliary_models.rare_pair.predictor import Predictor
from telcell.auxilliary_models.rare_pair.utils import DISTANCE_STEP
from telcell.data.models import Measurement, Point


def test_angle_distance_coverage_model():
    test_measurement = Measurement(Point(lat=52.0449566305567, lon=4.3585472613577965),
                                   datetime.strptime('2023-01-01', '%Y-%m-%d'), {'mnc': 4, 'azimuth': 0})

    clf = DecisionTreeClassifier()
    diameter = 2000
    resolution = 50
    classification_model = AngleDistanceClassificationCoverageModel(resolution, diameter, clf)
    classification_model._probabilities = np.ones([180, diameter // DISTANCE_STEP])

    predictor = Predictor({(4, (0, 0)): classification_model})
    assert pytest.approx(predictor.get_probability_e_h(test_measurement, test_measurement)) == 1.
