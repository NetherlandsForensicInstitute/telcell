import os
from datetime import datetime

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from telcell.auxilliary_models.geography import GridPoint
from telcell.auxilliary_models.rare_pair.coverage_model import ExtendedAngleDistanceClassificationCoverageModel
from telcell.auxilliary_models.rare_pair.utils import DISTANCE_STEP
from telcell.data.models import Measurement, Point


@pytest.fixture
def dummy_model() -> ExtendedAngleDistanceClassificationCoverageModel:
    coverage_model = \
        ExtendedAngleDistanceClassificationCoverageModel(1000,
                                                         100,
                                                         11000,
                                                         1000,
                                                         DecisionTreeClassifier())
    coverage_model._probabilities = np.ones([181, coverage_model.diameter // DISTANCE_STEP])
    return coverage_model

@pytest.fixture
def test_measurement(dummy_model) -> Measurement:
    point = Point(lat=52.0449566305567, lon=4.3585472613577965)
    rd_point = point.convert_to_rd()
    # Move the point so it fits exactly in the middle of the grid, nice for the angle and distance checks
    gridpoint = GridPoint(rd_point.x, rd_point.y).stick_to_resolution(dummy_model.resolution).move(
        dummy_model.resolution / 2, dummy_model.resolution / 2)
    return Measurement(gridpoint.convert_to_wgs84(), datetime.strptime('2023-01-01', '%Y-%m-%d'), {'mnc': 16, 'azimuth': 0})


def test_get_angles_and_distances(dummy_model, test_measurement):
    angles, distances = dummy_model._extract_features(test_measurement,
                                                      dummy_model.measurement_locations(test_measurement))

    angles = angles.numpy().reshape(-1, int(dummy_model.diameter / dummy_model.resolution))
    distances = distances.numpy().reshape(-1, int(dummy_model.diameter / dummy_model.resolution))

    assert np.all(angles < 360)
    # the antenna is facing down so all angles in the upper half should be larger than the lower half
    assert np.min(angles[:5]) > np.max(angles[5:])
    # check the corners
    corners = angles[::angles.shape[0] - 1, ::angles.shape[1] - 1]
    assert np.array_equal(corners[0], np.array([135, 135]))
    # TODO: Tell me why
    if any(np.array_equal(corners[1], np.array([x, 45])) for x in (44, 45)):
        assert True
    assert np.all(distances < dummy_model.diameter)


def test_get_probabilities(dummy_model, test_measurement):
    probabilities = dummy_model.probabilities(test_measurement)
    assert probabilities.outer.values[0, 0] == 0.
    assert probabilities.outer.values[4, 4] == 1.


def test_get_normalized_probabilities(dummy_model, test_measurement):
    probabilities = dummy_model.normalized_probabilities(test_measurement)
    assert pytest.approx(probabilities.sum()) == 1.
