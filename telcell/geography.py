from __future__ import annotations

import math

import geopy
import geopy.units
import pyproj


GEOD_WGS84 = pyproj.Geod(ellps="WGS84")


class Angle:
    """
    Class to represent an angle between two lines. Implements mathematical operators and can work with degrees and
    radians.
    """

    def __init__(self, degrees: float = None, radians: float = None):
        assert (
            degrees is None or radians is None
        ), "angle requires either degrees or radians; both provided"
        if degrees is not None:
            self._degrees = degrees
        elif radians is not None:
            self._degrees = geopy.units.degrees(radians=radians)
        else:
            raise ValueError("angle requires either degrees or radians; none provided")

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
        return geopy.units.radians(degrees=self._degrees)


def normalize_angle(angle: Angle) -> Angle:
    """
    Normalizes an angle to be in the range [-180,180> degrees.
    """
    normalized = (angle.degrees + 180) % 360 - 180
    return Angle(degrees=normalized)


def azimuth_deg(coord1: geopy.Point, coord2: geopy.Point) -> float:
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, back_azimuth, distance = geodesic.inv(
        coord1.longitude, coord1.latitude, coord2.longitude, coord2.latitude
    )
    return fwd_azimuth if distance > 0 else float("nan")


def azimuth(coord1: geopy.Point, coord2: geopy.Point) -> Angle:
    """
    Calculates the azimuth of the line between two points. That is, the angle between the line from the first point
    northward and the line from the first point to the second.

    @param coord1: the coordinates of the first point
    @param coord2: the coordinates of the second point
    @return: the azimuth of the line that connects the first point to the second
    """
    return Angle(degrees=azimuth_deg(coord1, coord2))
