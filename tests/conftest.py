from pathlib import Path

import pytest


@pytest.fixture
def testdata_path():
    # create an absolute reference to the "testdata.csv" in the tests folder
    return Path(__file__).parent / 'testdata.csv'
