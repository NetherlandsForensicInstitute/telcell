from pathlib import Path
from typing import List

import pytest

from telcell.auxilliary_models.rare_pair.utils import Bin
from telcell.data.models import Track, RDPoint, Point
from telcell.data.parsers import parse_measurements_csv


@pytest.fixture
def testdata_path():
    # create an absolute reference to the "testdata.csv" in the tests folder
    return Path(__file__).parent / 'testdata.csv'


@pytest.fixture
def test_data(testdata_path) -> List[Track]:
    return parse_measurements_csv(testdata_path)


@pytest.fixture
def testdata_3days_path():
    # create an absolute reference to the "testdata_3days.csv" in the tests
    # folder
    return Path(__file__).parent / 'testdata_3days.csv'


@pytest.fixture
def test_data_3days(testdata_3days_path) -> List[Track]:
    return parse_measurements_csv(testdata_3days_path)


@pytest.fixture
def testdata_simple_path():
    # create an absolute reference to the "testdata_simple.csv" in the tests
    # folder
    return Path(__file__).parent / 'testdata_simple.csv'


@pytest.fixture
def test_data_simple(testdata_simple_path) -> List[Track]:
    return parse_measurements_csv(testdata_simple_path)


@pytest.fixture
def bins() -> List[Bin]:
    return [(0,0), (1,20), (21,40), (41,60), (61,120)]

@pytest.fixture
def test_rd_point():
    return RDPoint(x=155000, y=463000)

@pytest.fixture
def test_wgs_point():
    return Point(lat=52.045, lon=4.358)

@pytest.fixture
def max_delay() -> int:
    return 120
