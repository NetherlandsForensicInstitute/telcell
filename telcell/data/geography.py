from __future__ import annotations

import math

import pyproj

from .coord import Point

GEOD_WGS84 = pyproj.Geod(ellps="WGS84")


def _degrees_to_radians(degrees: float) -> float:
    return degrees * math.pi / 180


def _radians_to_degrees(radians: float) -> float:
    return radians * 180 / math.pi


class Angle:
    """
    Class to represent an angle between two lines. Implements mathematical operators and can work with degrees and
    radians.
    """

    def __init__(self, degrees: float = None, radians: float = None):
        assert (degrees is None) != (
            radians is None
        ), "provide exactly one of degrees and radians"
        if degrees is not None:
            self._degrees = degrees
        if radians is not None:
            self._degrees = _radians_to_degrees(radians)

    def __eq__(self, other):
        if isinstance(other, Angle):
            return self.degrees == other.degrees
        else:
            return self.degrees == other

    def __gt__(self, other):
        if isinstance(other, Angle):
            return self.degrees > other.degrees
        else:
            return self.degrees > other

    def __ge__(self, other):
        if isinstance(other, Angle):
            return self.degrees >= other.degrees
        else:
            return self.degrees >= other

    def __lt__(self, other):
        if isinstance(other, Angle):
            return self.degrees < other.degrees
        else:
            return self.degrees < other

    def __le__(self, other):
        if isinstance(other, Angle):
            return self.degrees <= other.degrees
        else:
            return self.degrees <= other

    def __ne__(self, other):
        if isinstance(other, Angle):
            return self.degrees != other.degrees
        else:
            return self.degrees != other

    def __mul__(self, other):
        return Angle(degrees=self.degrees * other)

    def __truediv__(self, other):
        return Angle(degrees=self.degrees / other)

    def __add__(self, other):
        if isinstance(other, Angle):
            return Angle(degrees=self.degrees + other.degrees)
        else:
            raise TypeError(
                f"unsupported operand types for +: {type(self)} and {type(other)}"
            )

    def __sub__(self, other):
        if isinstance(other, Angle):
            return Angle(degrees=self.degrees - other.degrees)
        else:
            raise TypeError(
                f"unsupported operand types for -: {type(self)} and {type(other)}"
            )

    def __repr__(self):
        return f"Angle(degrees={self._degrees})"

    def isnan(self):
        return math.isnan(self.degrees)

    @property
    def degrees(self):
        return self._degrees

    @property
    def radians(self):
        return _degrees_to_radians(self._degrees)


def normalize_angle(angle: Angle) -> Angle:
    """
    Normalizes an angle to be in the range [-180,180> degrees
    """
    normalized = (angle.degrees + 180) % 360 - 180
    return Angle(degrees=normalized)


def azimuth_deg(coord1: Point, coord2: Point) -> float:
    geodesic = pyproj.Geod(ellps="WGS84")
    coord1 = coord1.wgs84()
    coord2 = coord2.wgs84()
    fwd_azimuth, back_azimuth, distance = geodesic.inv(*coord1.lonlat, *coord2.lonlat)
    return fwd_azimuth if distance > 0 else float("nan")


def azimuth(coord1: Point, coord2: Point) -> Angle:
    """
    Calculates the azimuth of the line between two points. That is, the angle between the line from the first point northward and
    the line from the first point to the second.

    @param coord1: the coordinates of the first point
    @param coord2: the coordinates of the second point
    @return: the azimuth of the line that connects the first point to the second
    """
    return Angle(degrees=azimuth_deg(coord1, coord2))
