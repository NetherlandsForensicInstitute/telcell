from __future__ import annotations
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Any, Mapping, Tuple, Sequence, Iterator, Optional

import geopy
import geopy.distance
import pyproj
from pyproj import Proj, Geod, Transformer

RD = (
    "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 "
    "+k=0.999908 +x_0=155000 +y_0=463000 +ellps=bessel "
    "+towgs84=565.237,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812 "
    "+units=m +no_defs"
)
GOOGLE = (
    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 "
    "+lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m "
    "+nadgrids=@null +no_defs +over"
)
WGS84 = "+proj=latlong +datum=WGS84"
rd_projection = Proj(RD)
google_projection = Proj(GOOGLE)
wgs84_projection = Proj(WGS84)
geodesic = Geod("+ellps=sphere")
WGS84_TO_RD = Transformer.from_proj(wgs84_projection, rd_projection)
RD_TO_WGS84 = Transformer.from_proj(rd_projection, wgs84_projection)
# TODO: more accurate range would be: x (7000, 300000) and y (289000, 629000)
RD_X_RANGE = (1000, 300000)
RD_Y_RANGE = (1000, 629000)
GEOD_WGS84 = pyproj.Geod(ellps="WGS84")


def approximately_equal(first, second, tolerance=0.0001):
    return abs(first - second) < tolerance


def rd_to_point(x: float, y: float) -> geopy.Point:
    """
    Converts an x y tuple of Rijksdriehoek coordinates to a `geopy.Point` object.

    @param x: the x value
    @param y: the y value
    @return: a `geopy.Point` object
    """
    if not (RD_Y_RANGE[0] <= y <= RD_Y_RANGE[1]) or not (
        RD_X_RANGE[0] <= x <= RD_X_RANGE[1]
    ):
        warnings.warn(
            f"rijksdriehoek coordinates {x}, {y} outside range: x={RD_X_RANGE}, y={RD_Y_RANGE}"
        )
    c = RD_TO_WGS84.transform(x, y)
    return geopy.Point(longitude=c[0], latitude=c[1])


def point_to_rd(point: geopy.Point) -> Tuple[float, float]:
    """
    Converts a `geopy.Point` object to an x, y tuple of Rijksdriehoek coordinates.

    @param point: the point
    @return: an x y tuple
    """
    return WGS84_TO_RD.transform(point.longitude, point.latitude)


@dataclass(eq=True, frozen=True)
class Measurement:
    """
    A single measurement of a device at a certain place and time.

    :param coords: The WGS84 latitude and longitude coordinates
    :param timestamp: The time of registration
    :param extra: Additional metadata related to the source that registered
            this measurement. These could for example inform the accuracy or
            uncertainty of the measured WGS84 coordinates.
    """

    coords: geopy.Point
    timestamp: Optional[datetime]
    extra: Mapping[str, Any]

    @property
    def lat(self):
        return self.coords.latitude

    @property
    def lon(self):
        return self.coords.longitude

    @property
    def latlon(self) -> Tuple[float, float]:
        return self.lat, self.lon

    @property
    def xy(self) -> Tuple[float, float]:
        return point_to_rd(self.coords)

    def __str__(self):
        return f"<{self.timestamp}: ({self.lat}, {self.lon})>"

    def __hash__(self):
        return hash(
            (
                self.lat,
                self.lon,
                self.timestamp.date(),
                *(_extra for _extra in self.extra.values()),
            )
        )


@dataclass
class Track:
    """
    A history of measurements for a single device.

    :param owner: The owner of the device. Can be anything with a simcard.
    :param device: The name of the device.
    :param measurements: A series of measurements ordered by timestamp.
    """

    owner: str
    device: str
    measurements: Sequence[Measurement]

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self.measurements)


@dataclass(order=False, frozen=True, eq=True)
class MeasurementPair:
    """
    A pair of two measurements. The pair can be made with different criteria,
    for example the time difference between the two measurements. It always
    contains the information from the two measurements it was created from.
    """

    measurement_a: Measurement
    measurement_b: Measurement

    @cached_property
    def time_difference(self):
        """Calculate the absolute time difference between the measurements."""
        return abs(self.measurement_a.timestamp - self.measurement_b.timestamp)

    @cached_property
    def distance(self):
        """Calculate the distance (in meters) between the two measurements of
        the pair."""
        return (
            geopy.distance.geodesic(
                self.measurement_a.coords, self.measurement_b.coords
            ).km
            * 1000
        )

    def __str__(self):
        return f"<{self.measurement_a}, ({self.measurement_b})>"
