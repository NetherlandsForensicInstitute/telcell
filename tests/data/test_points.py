import pytest

from telcell.data.models import RDPoint, Point


def test_distance(test_rd_point, test_wgs_point):
    # test that the function works with both RDPoints as WGSPoints
    assert pytest.approx(test_rd_point.distance(test_wgs_point)) == 71577.11
    assert pytest.approx(test_rd_point.distance(test_wgs_point.convert_to_rd())) == 71577.11
    assert pytest.approx(test_rd_point.convert_to_wgs84().distance(test_wgs_point)) == 71577.11
    assert test_rd_point.distance(test_rd_point) == 0
    # test that the function is interchangeable
    assert test_rd_point.distance(test_wgs_point) == test_wgs_point.distance(test_rd_point)


def test_convert(test_rd_point, test_wgs_point):
    test_rd_point_to_wgs = test_rd_point.convert_to_wgs84()
    test_wgs_point_to_rd = test_wgs_point.convert_to_rd()
    assert pytest.approx(test_rd_point_to_wgs.latlon, 0.001) == (52.155, 5.387)
    assert pytest.approx(test_wgs_point_to_rd.xy) == (84395.14, 451242.54)
    assert pytest.approx(test_rd_point.xy, 0.001) ==  test_rd_point_to_wgs.convert_to_rd().xy


def test_rd_outside_range(test_rd_point):
    with pytest.raises(ValueError, match=r"Invalid rijksdriehoek coordinates"):
        RDPoint(x=0, y=0)
    with pytest.raises(ValueError, match=r"Invalid rijksdriehoek coordinates"):
        RDPoint(x=test_rd_point.x, y=0)
    with pytest.raises(ValueError, match=r"Invalid rijksdriehoek coordinates"):
        RDPoint(x=10e6, y=test_rd_point.x)


def test_wgs_outside_range(test_wgs_point):
    with pytest.raises(ValueError, match=r"Invalid wgs84 coordinates"):
        Point(lat=360, lon=360)
    with pytest.raises(ValueError, match=r"Invalid wgs84 coordinates"):
        Point(lat=test_wgs_point.lat, lon=360)
    with pytest.raises(ValueError, match=r"Invalid wgs84 coordinates"):
        Point(lat=10e6, lon=test_wgs_point.lon)

def test_approx_equal(test_rd_point, test_wgs_point):
    # test that the function is interchangeable
    assert not test_rd_point.approx_equal(test_wgs_point)
    assert not test_wgs_point.approx_equal(test_rd_point)

    # test that the function works with both RDPoint as WGSPoint
    assert not test_rd_point.approx_equal(test_wgs_point.convert_to_rd())
    assert not test_wgs_point.approx_equal(test_rd_point.convert_to_wgs84())

    test_wgs_point_close = Point(lat=test_wgs_point.lat,
                                 lon=test_wgs_point.lon + 10e-7)
    test_wgs_point_far = Point(lat=test_wgs_point.lat,
                               lon=test_wgs_point.lon + 1)
    assert test_wgs_point.approx_equal(test_wgs_point_close)
    assert not test_wgs_point.approx_equal(test_wgs_point_far)
    assert test_rd_point.approx_equal(test_rd_point)

    # approx_equal() uses WGS coordinates, so it parses RDPoint to WGSPoint. This means
    # that the tolerance between two RDPoints is actually bigger than the tolerance
    # between two WGSPoints
    test_rd_point_close = RDPoint(x=test_rd_point.x,
                                  y=test_rd_point.y + 10e-4)
    test_rd_point_to_wgs = test_rd_point.convert_to_wgs84()
    test_wgs_point_far = Point(lat=test_rd_point_to_wgs.lat,
                               lon=test_rd_point_to_wgs.lon + 10e-4)
    assert test_rd_point.approx_equal(test_rd_point_close)
    assert not test_rd_point_to_wgs.approx_equal(test_wgs_point_far)
