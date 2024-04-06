from __future__ import annotations
from abc import abstractmethod
import datetime
from typing import Optional, Iterable, Sized

import geopy

from telcell.cell_identity import CellIdentity


class Properties(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            return Properties()


class CellCollection(Iterable[Properties], Sized):
    """
    Collection of cell towers in a cellular network, that may be queried by cell id (e.g. for geocoding) or by
    coordinates (e.g. reverse geocoding).

    The known fields for cells may vary by database and is generally a subset of:
    - cell: the cell identity
    - wgs84: the geolocation of the cell tower or a connected device
    - accuracy: the accuracy of the location of the connected device in meters
    - azimuth: the orientation of the cell tower in meters
    """

    @abstractmethod
    def get(self, date: datetime.datetime, ci: CellIdentity) -> Optional[Properties]:
        """
        Retrieve a specific antenna from database.

        :param date: Used to select active antennas
        :param ci: The cell identity
        :return: The retrieved antenna or None
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        coords: geopy.Point = None,
        distance_limit_m: float = None,
        distance_lower_limit_m: float = None,
        date: datetime.datetime = None,
        radio: Optional[str | Iterable[str]] = None,
        mcc: int = None,
        mnc: int = None,
        count_limit: Optional[int] = 10000,
        exclude: Optional[CellIdentity] = None,
    ) -> CellCollection:
        """
        Given a Point, find antennas that are in reach from this point sorted by the distance from the grid point.

        :param coords: Point for which nearby antennas are retrieved
        :param distance_limit_m: antennas should be within this range
        :param date: used to select active antennas
        :param radio: antennas should be limited to this radio technology, e.g.: LTE, UMTS, GSM
        :param mcc: antennas should be limited to this mcc
        :param mnc: antennas should be limited to this mnc
        :param count_limit: maximum number of antennas to return
        :param exclude: antenna that should be excluded from the retrieved antennas
        :return: retrieved antennas within reach from the Point
        """
        raise NotImplementedError

    def limit(self, count_limit: int) -> CellCollection:
        return self.search(None, count_limit=count_limit)
