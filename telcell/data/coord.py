from __future__ import annotations
import math
import warnings
from abc import ABC, abstractmethod

import geopy
import geopy.distance
from pyproj import Proj, Geod, Transformer

# rd = Rijksdriehoekscoördinaten

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

RD_X_RANGE = (7000, 300000)
RD_Y_RANGE = (289000, 629000)


class Point(ABC):
    @abstractmethod
    def rd(self) -> RdPoint:
        raise NotImplementedError

    @abstractmethod
    def wgs84(self) -> WgsPoint:
        raise NotImplementedError


class WgsPoint(Point):
    def __init__(self, lon: float, lat: float):
        super().__init__()

        self.lon = lon
        self.lat = lat
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError(f"wgs84 coordinates {self} outside range")

    @property
    def lonlat(self):
        return self.lon, self.lat

    @property
    def latlon(self):
        return self.lat, self.lon

    def rd(self):
        c = WGS84_TO_RD.transform(self.lon, self.lat)
        return RdPoint(x=c[0], y=c[1])

    def wgs84(self):
        return self

    def __eq__(self, obj) -> bool:
        return isinstance(obj, WgsPoint) and self.lon == obj.lon and self.lat == obj.lat

    def __hash__(self):
        return hash(self.lonlat)

    def __repr__(self) -> str:
        return "wgs84(lat={},lon={})".format(self.lat, self.lon)


class RdPoint(Point):
    def __init__(self, x: int, y: int):
        super().__init__()

        self.x = x
        self.y = y
        if not (RD_Y_RANGE[0] <= y <= RD_Y_RANGE[1]) or not (
            RD_X_RANGE[0] <= x <= RD_X_RANGE[1]
        ):
            warnings.warn(
                f"rijksdriehoek coordinates {self} outside range: x={RD_X_RANGE}, y={RD_Y_RANGE}"
            )

    @property
    def xy(self):
        return self.x, self.y

    def rd(self):
        return self

    def wgs84(self):
        c = RD_TO_WGS84.transform(self.x, self.y)
        return WgsPoint(lon=c[0], lat=c[1])

    def __eq__(self, obj) -> bool:
        return isinstance(obj, RdPoint) and self.x == obj.x and self.y == obj.y

    def __hash__(self):
        return hash(self.xy)

    def __repr__(self) -> str:
        return f"rd(x={self.x},y={self.y})"


def move_point(point: Point, east_m: float = None, north_m: float = None) -> Point:
    geopoint = geopy.Point(*point.wgs84().latlon)
    if north_m is not None:
        d = geopy.distance.geodesic(meters=north_m)
        geopoint = d.destination(point=geopoint, bearing=0)

    if east_m is not None:
        d = geopy.distance.geodesic(meters=east_m)
        geopoint = d.destination(point=geopoint, bearing=90)

    return WgsPoint(lat=geopoint.latitude, lon=geopoint.longitude)


def cosines(lat1, long1, lat2, long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) + math.cos(
        phi1
    ) * math.cos(phi2)
    arc = math.acos(cos)

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc * 6378
