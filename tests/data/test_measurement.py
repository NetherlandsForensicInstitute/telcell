from datetime import datetime

import geopy

from telcell.data.models import Measurement


def test_measurement():
    assert Measurement


def test_equal():
    measurement1 = Measurement(
        coords=geopy.Point(latitude=50, longitude=4),
        timestamp=datetime(2023, 8, 24, 12, 30, 59),
        extra={"mnc": 8},
    )
    measurement2 = Measurement(
        coords=geopy.Point(latitude=50, longitude=4),
        timestamp=datetime(2023, 8, 24, 12, 30, 59),
        extra={"mnc": 8},
    )
    assert measurement1 == measurement2
    assert len({measurement1, measurement2}) == 1

    measurement3 = Measurement(
        coords=geopy.Point(latitude=51, longitude=5),
        timestamp=datetime(2023, 8, 24, 12, 30, 59),
        extra={"mnc": 8},
    )
    measurement4 = Measurement(
        coords=geopy.Point(latitude=50, longitude=4),
        timestamp=datetime(2022, 8, 24, 12, 30, 59),
        extra={"mnc": 8},
    )
    measurement5 = Measurement(
        coords=geopy.Point(latitude=50, longitude=4),
        timestamp=datetime(2023, 8, 24, 12, 30, 59),
        extra={"mnc": 16},
    )
    assert not measurement1 == measurement3
    assert not measurement1 == measurement4
    assert not measurement1 == measurement5
    assert (
        len({measurement1, measurement2, measurement3, measurement4, measurement5}) == 4
    )
