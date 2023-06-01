from pathlib import Path
from typing import List

import pytest

from telcell.data.models import Track
from telcell.data.parsers import parse_measurements_csv


@pytest.fixture
def testdata_path():
    # create an absolute reference to the "testdata.csv" in the tests folder
    return Path(__file__).parent / 'testdata.csv'


@pytest.fixture
def test_data(testdata_path) -> List[Track]:
    return parse_measurements_csv(testdata_path)
